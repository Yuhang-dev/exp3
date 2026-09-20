"""Paired Full/V1 evaluation on BFCL V4 multi-turn long-context tasks."""

import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import shutil
import statistics
import sys
import time
import traceback
import zipfile

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from attention import AttentionBackend
from evaluate import generate, gpu_metadata, prefill


BFCL_VERSION = "2025.12.17"
BFCL_COMMIT = "f7cf7359b7ac615a0b294831c5ba2bc95ee4a000"
BFCL_WHEEL_SHA256 = "8555bc9407a56682ceb7d969e87eb724f6b679deb0ef05114d9c6e786406b103"
BFCL_CATEGORY = "multi_turn_long_context"
ADAPTER_VERSION = "exp3-qwen25-bfcl-v2-2026-09-20"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--bfcl-root",
        type=Path,
        default=Path("third_party/bfcl_eval_2025_12_17"),
        help="Directory containing the extracted bfcl_eval package.",
    )
    parser.add_argument(
        "--bfcl-wheel",
        type=Path,
        default=Path("third_party/downloads/bfcl_eval-2025.12.17-py3-none-any.whl"),
    )
    parser.add_argument("--methods", nargs="+", choices=("dense", "fp_v1"), default=["dense", "fp_v1"])
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument(
        "--selected-cases",
        type=Path,
        help="Reuse a selected_cases.jsonl manifest from --prepare-only.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--context-limit", type=int, default=32768)
    parser.add_argument("--selector-chunk-tiles", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted run whose generations form an exact schedule prefix.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/bfcl_v4_long_context_pilot20"),
    )
    return parser.parse_args()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_hash(token_ids):
    return hashlib.sha256(np.asarray(token_ids, dtype=np.int32).tobytes()).hexdigest()


def json_hash(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def write_jsonl_line(output, value):
    output.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
    output.flush()


class CsvSink:
    def __init__(self, path, fields, mode="w"):
        self.file = Path(path).open(mode, newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        if mode == "w" or self.file.tell() == 0:
            self.writer.writeheader()

    def write(self, row):
        self.writer.writerow({key: row.get(key) for key in self.writer.fieldnames})
        self.file.flush()

    def close(self):
        self.file.close()


def candidate(method, alpha):
    if method == "dense":
        return {"method": "dense", "alpha": None, "config_id": "dense"}
    alpha_label = f"{alpha:g}".replace("-", "m").replace(".", "p")
    return {
        "method": "fp_v1",
        "alpha": alpha,
        "config_id": f"fp_v1__a{alpha_label}",
    }


def rotate(items, offset):
    offset %= len(items)
    return items[offset:] + items[:offset]


def select_entries(entries, count, seed):
    if count >= len(entries):
        return list(entries)
    grouped = {}
    for entry in entries:
        signature = "+".join(sorted(entry["involved_classes"]))
        order = hashlib.sha256(f"bfcl-v4:{seed}:{entry['id']}".encode()).hexdigest()
        grouped.setdefault(signature, []).append((order, entry))
    for values in grouped.values():
        values.sort(key=lambda item: item[0])
    signatures = sorted(grouped)
    selected = []
    position = 0
    while len(selected) < count:
        added = False
        for signature in signatures:
            values = grouped[signature]
            if position < len(values):
                selected.append(values[position][1])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        position += 1
    return selected


def load_bfcl(args):
    package = args.bfcl_root / "bfcl_eval"
    data_root = package / "data"
    data_path = data_root / "BFCL_v4_multi_turn_long_context.json"
    answer_path = data_root / "possible_answer" / "BFCL_v4_multi_turn_long_context.json"
    if not data_path.is_file() or not answer_path.is_file():
        raise FileNotFoundError(
            f"BFCL V4 files are missing under {args.bfcl_root}; "
            "run prepare_modern_benchmarks.sh first"
        )
    if not args.bfcl_wheel.is_file():
        raise FileNotFoundError(
            f"pinned BFCL wheel is missing at {args.bfcl_wheel}; "
            "run prepare_modern_benchmarks.sh first"
        )
    actual_wheel_hash = file_hash(args.bfcl_wheel)
    if actual_wheel_hash != BFCL_WHEEL_SHA256:
        raise ValueError(
            f"BFCL wheel hash mismatch: expected {BFCL_WHEEL_SHA256}, "
            f"got {actual_wheel_hash}"
        )
    verified_extracted_files = 0
    exact_members = {
        "bfcl_eval/data/BFCL_v4_multi_turn_long_context.json",
        "bfcl_eval/data/possible_answer/BFCL_v4_multi_turn_long_context.json",
        "bfcl_eval/constants/executable_backend_config.py",
    }
    with zipfile.ZipFile(args.bfcl_wheel) as archive:
        for member in archive.infolist():
            name = member.filename
            needs_verification = (
                name in exact_members
                or name.startswith("bfcl_eval/data/multi_turn_func_doc/")
                or name.startswith("bfcl_eval/eval_checker/multi_turn_eval/")
            )
            if not needs_verification or name.endswith("/"):
                continue
            extracted = args.bfcl_root / Path(name)
            if not extracted.is_file():
                raise FileNotFoundError(f"missing extracted BFCL wheel member: {extracted}")
            archived_hash = hashlib.sha256(archive.read(member)).hexdigest()
            if file_hash(extracted) != archived_hash:
                raise ValueError(f"extracted BFCL file differs from pinned wheel: {extracted}")
            verified_extracted_files += 1

    root = str(args.bfcl_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from bfcl_eval.constants.executable_backend_config import (
        MULTI_TURN_FUNC_DOC_FILE_MAPPING,
    )
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
        multi_turn_checker,
    )

    entries = load_jsonl(data_path)
    answers = {row["id"]: row["ground_truth"] for row in load_jsonl(answer_path)}
    selected_cases_path = getattr(args, "selected_cases", None)
    selected_manifest_hashes = None
    if selected_cases_path is not None:
        selected_manifest = load_jsonl(selected_cases_path)
        selected_ids = [row["id"] for row in selected_manifest]
        selected_manifest_hashes = {
            row["id"]: row["source_row_sha256"]
            for row in selected_manifest
        }
        if len(selected_ids) != args.samples or len(selected_ids) != len(set(selected_ids)):
            raise ValueError("selected-cases must contain exactly --samples unique IDs")
        entries_by_id = {entry["id"]: entry for entry in entries}
        absent = sorted(set(selected_ids) - set(entries_by_id))
        if absent:
            raise ValueError(f"selected BFCL IDs are absent from the pinned data: {absent}")
        selected = [entries_by_id[source_id] for source_id in selected_ids]
        selection_description = "exact ID order reused from selected_cases.jsonl"
    else:
        selected = select_entries(entries, args.samples, args.seed)
        selection_description = (
            "all rows in official order when samples >= 200; otherwise deterministic "
            "SHA256 ordering round-robin across involved-class signatures"
        )
    selected_rows = []
    function_file_hashes = {}
    for entry in selected:
        if entry["id"] not in answers:
            raise ValueError(f"missing ground truth for {entry['id']}")
        source_row_sha256 = json_hash(entry)
        if (
            selected_manifest_hashes is not None
            and selected_manifest_hashes[entry["id"]] != source_row_sha256
        ):
            raise ValueError(f"selected source row hash mismatch for {entry['id']}")
        functions = []
        for class_name in entry["involved_classes"]:
            function_path = data_root / "multi_turn_func_doc" / MULTI_TURN_FUNC_DOC_FILE_MAPPING[class_name]
            functions.extend(load_jsonl(function_path))
            function_file_hashes[function_path.name] = file_hash(function_path)
        row = deepcopy(entry)
        row["function"] = functions
        row["source_row_sha256"] = source_row_sha256
        row["ground_truth"] = answers[entry["id"]]
        selected_rows.append(row)

    sources = {
        "package_version": BFCL_VERSION,
        "leaderboard_commit": BFCL_COMMIT,
        "wheel_sha256_expected": BFCL_WHEEL_SHA256,
        "wheel_sha256_actual": actual_wheel_hash,
        "verified_extracted_files": verified_extracted_files,
        "data_file": str(data_path.resolve()),
        "data_file_sha256": file_hash(data_path),
        "answer_file": str(answer_path.resolve()),
        "answer_file_sha256": file_hash(answer_path),
        "function_file_sha256": function_file_hashes,
        "source_rows": len(entries),
        "selected_ids": [entry["id"] for entry in selected_rows],
        "selection": selection_description,
        "selected_cases_file": (
            str(selected_cases_path.resolve()) if selected_cases_path is not None else None
        ),
        "selected_cases_file_sha256": (
            file_hash(selected_cases_path) if selected_cases_path is not None else None
        ),
    }
    return selected_rows, sources, multi_turn_utils, multi_turn_checker


def format_qwen_prompt(messages, functions):
    """Mirror BFCL's QwenFCHandler prompt protocol."""
    prompt = ""
    if functions:
        prompt += "<|im_start|>system\n"
        if messages and messages[0]["role"] == "system":
            prompt += messages[0]["content"] + "\n\n"
        prompt += (
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        )
        for function in functions:
            prompt += "\n" + json.dumps(function)
        prompt += (
            "\n</tools>\n\nFor each function call, return a json object with function name "
            "and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call><|im_end|>\n"
        )
    elif messages and messages[0]["role"] == "system":
        prompt += f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n"

    last_query_index = len(messages) - 1
    for offset, message in enumerate(reversed(messages)):
        index = len(messages) - 1 - offset
        content = message.get("content", "")
        if (
            message["role"] == "user"
            and isinstance(content, str)
            and not (
                content.startswith("<tool_response>")
                and content.endswith("</tool_response>")
            )
        ):
            last_query_index = index
            break

    for index, message in enumerate(messages):
        role = message["role"]
        content = message.get("content", "")
        if role == "user" or (role == "system" and index != 0):
            prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        elif role == "assistant":
            reasoning = message.get("reasoning_content", "")
            if not reasoning and "</think>" in content:
                parts = content.split("</think>")
                reasoning = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                content = parts[-1].lstrip("\n")
            if index > last_query_index and (index == len(messages) - 1 or reasoning):
                prompt += (
                    "<|im_start|>assistant\n<think>\n"
                    + reasoning.strip("\n")
                    + "\n</think>\n\n"
                    + content.lstrip("\n")
                )
            else:
                prompt += f"<|im_start|>assistant\n{content}"
            for call_index, tool_call in enumerate(message.get("tool_calls", [])):
                if (call_index == 0 and content) or call_index > 0:
                    prompt += "\n"
                call = tool_call.get("function", tool_call)
                arguments = call["arguments"]
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments)
                prompt += (
                    '<tool_call>\n{"name": "'
                    + call["name"]
                    + '", "arguments": '
                    + arguments
                    + "}\n</tool_call>"
                )
            prompt += "<|im_end|>\n"
        elif role == "tool":
            previous_role = messages[index - 1]["role"] if index else None
            next_role = messages[index + 1]["role"] if index + 1 < len(messages) else None
            if previous_role != "tool":
                prompt += "<|im_start|>user"
            prompt += f"\n<tool_response>\n{content}\n</tool_response>"
            if next_role != "tool":
                prompt += "<|im_end|>\n"
    return prompt + "<|im_start|>assistant\n"


def parse_qwen_response(raw_text):
    matches = re.findall(r"<tool_call>\n(.*?)\n</tool_call>", raw_text, re.DOTALL)
    calls = []
    for match in matches:
        try:
            calls.append(json.loads(match))
        except Exception:
            pass
    cleaned = raw_text
    reasoning = ""
    if "</think>" in raw_text:
        parts = raw_text.split("</think>")
        reasoning = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
        cleaned = parts[-1].lstrip("\n")
    try:
        decoded = []
        for call in calls:
            name = call["name"]
            if not isinstance(name, str):
                raise TypeError("tool name must be a string")
            arguments = call["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise TypeError("tool arguments must be an object")
            decoded.append(
                f"{name}({','.join(f'{key}={repr(value)}' for key, value in arguments.items())})"
            )
        decode_error = None
    except Exception as error:
        decoded = []
        decode_error = f"{type(error).__name__}: {error}"
    assistant = {
        "role": "assistant",
        "content": "" if calls and decode_error is None else cleaned,
        "reasoning_content": reasoning,
    }
    if calls and decode_error is None:
        assistant["tool_calls"] = calls
    return cleaned, calls, decoded, decode_error, assistant


def instance_name(model_name, test_id, class_name):
    raw = f"{model_name}_{test_id}_{class_name}_instance"
    return re.sub(r"[-./]", "_", raw)


def drop_instances(module, model_names, entry):
    for model_name in model_names:
        for class_name in entry["involved_classes"]:
            module.__dict__.pop(instance_name(model_name, entry["id"], class_name), None)


def run_episode(
    args,
    entry,
    config,
    model,
    tokenizer,
    backend,
    multi_turn_utils,
    timing_sink,
    warmed_shapes,
    warmed_generation,
):
    model_name = f"exp3_agent_{config['config_id']}"
    test_category = entry["id"].rsplit("_", 1)[0]
    multi_turn_utils.execute_multi_turn_func_call(
        [],
        entry["initial_config"],
        entry["involved_classes"],
        model_name,
        entry["id"],
        long_context="long_context" in test_category,
        is_evaL_run=False,
    )
    messages = []
    raw_result = []
    decoded_result = []
    step_logs = []
    force_terminated = False
    termination_reason = None
    calls_executed = 0

    for turn_index, turn_messages in enumerate(entry["question"]):
        messages.extend(deepcopy(turn_messages))
        turn_raw = []
        turn_decoded = []
        executed_steps = 0
        step_index = 0
        while True:
            formatted_prompt = format_qwen_prompt(messages, entry["function"])
            token_ids = tokenizer(
                formatted_prompt,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
            prompt_tokens = len(token_ids)
            prompt_sha256 = token_hash(token_ids)
            if prompt_tokens + args.max_new_tokens > args.context_limit:
                force_terminated = True
                termination_reason = "native_context_overflow"
                step_logs.append({
                    "turn": turn_index,
                    "step": step_index,
                    "prompt_tokens": prompt_tokens,
                    "prompt_sha256": prompt_sha256,
                    "status": "context_overflow",
                })
                break

            input_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
            backend.configure(config["method"], config["alpha"])
            warm_key = (config["config_id"], prompt_tokens)
            if warm_key not in warmed_shapes:
                prefill(model, input_ids, measure=False)
                warmed_shapes.add(warm_key)
            if config["config_id"] not in warmed_generation:
                generate(model, tokenizer, input_ids, 2)
                warmed_generation.add(config["config_id"])

            measurements = []
            for repeat in range(args.repeats):
                metrics = prefill(model, input_ids, measure=True)
                measurements.append(metrics["prefill_ms"])
                timing_sink.write({
                    "method": config["method"],
                    "config_id": config["config_id"],
                    "alpha": config["alpha"],
                    "sample_id": entry["id"],
                    "turn": turn_index,
                    "step": step_index,
                    "prompt_tokens": prompt_tokens,
                    "prompt_sha256": prompt_sha256,
                    "repeat": repeat,
                    **metrics,
                })

            torch.cuda.synchronize()
            generation_started = time.perf_counter()
            generated_ids, raw_text, end_reason = generate(
                model,
                tokenizer,
                input_ids,
                args.max_new_tokens,
            )
            torch.cuda.synchronize()
            generation_ms = (time.perf_counter() - generation_started) * 1000
            cleaned, calls, decoded, decode_error, assistant = parse_qwen_response(raw_text)
            turn_raw.append(cleaned)
            messages.append(assistant)
            step_log = {
                "turn": turn_index,
                "step": step_index,
                "status": "generated",
                "prompt_tokens": prompt_tokens,
                "prompt_sha256": prompt_sha256,
                "prefill_ms": measurements,
                "prefill_ms_median": statistics.median(measurements),
                "generation_ms": generation_ms,
                "generated_token_ids": generated_ids,
                "raw_text": raw_text,
                "scorer_text": cleaned,
                "end_reason": end_reason,
                "extracted_tool_calls": calls,
                "decoded_calls": decoded,
                "decode_error": decode_error,
            }
            del input_ids

            if not decoded:
                step_logs.append(step_log)
                break
            turn_decoded.append(decoded)
            execution_results, _ = multi_turn_utils.execute_multi_turn_func_call(
                decoded,
                entry["initial_config"],
                entry["involved_classes"],
                model_name,
                entry["id"],
                long_context=True,
                is_evaL_run=False,
            )
            step_log["execution_results"] = execution_results
            step_logs.append(step_log)
            for execution_result, decoded_call in zip(execution_results, decoded):
                messages.append({
                    "role": "tool",
                    "name": decoded_call,
                    "content": execution_result,
                })
            calls_executed += len(decoded)
            executed_steps += 1
            step_index += 1
            if executed_steps > args.max_steps:
                force_terminated = True
                termination_reason = "maximum_step_limit"
                break

        raw_result.append(turn_raw)
        decoded_result.append(turn_decoded)
        if force_terminated:
            break

    drop_instances(multi_turn_utils, [model_name], entry)
    return {
        "id": entry["id"],
        "category": BFCL_CATEGORY,
        "method": config["method"],
        "config_id": config["config_id"],
        "alpha": config["alpha"],
        "source_row_sha256": entry["source_row_sha256"],
        "involved_classes": entry["involved_classes"],
        "expected_turns": len(entry["question"]),
        "completed_turns": len(raw_result),
        "force_terminated": force_terminated,
        "termination_reason": termination_reason,
        "tool_calls_executed": calls_executed,
        "result": raw_result,
        "decoded_result": decoded_result,
        "steps": step_logs,
        "adapter_version": ADAPTER_VERSION,
    }


def decode_saved_result(raw_result):
    decoded_result = []
    for turn in raw_result:
        decoded_turn = []
        for raw_text in turn:
            _, _, decoded, _, _ = parse_qwen_response(raw_text)
            if decoded:
                decoded_turn.append(decoded)
        decoded_result.append(decoded_turn)
    return decoded_result


def score_record(record, entry, multi_turn_utils, multi_turn_checker):
    score_model_name = f"exp3_score_{record['config_id']}"
    checker_decoded_result = decode_saved_result(record["result"])
    if record["force_terminated"] or record["completed_turns"] != record["expected_turns"]:
        result = {
            "valid": False,
            "error_type": f"multi_turn:{record['termination_reason'] or 'force_terminated'}",
            "error_message": "Agent trajectory did not complete every official turn.",
        }
    else:
        result = multi_turn_checker(
            checker_decoded_result,
            entry["ground_truth"],
            entry,
            BFCL_CATEGORY,
            score_model_name,
        )
    drop_instances(
        multi_turn_utils,
        [score_model_name + "_eval", score_model_name + "_ground_truth_eval"],
        entry,
    )
    return result, checker_decoded_result


def episode_row(record, valid):
    generated_steps = [step for step in record["steps"] if step["status"] == "generated"]
    prompt_tokens = [step["prompt_tokens"] for step in generated_steps]
    return {
        "method": record["method"],
        "config_id": record["config_id"],
        "alpha": record["alpha"],
        "sample_id": record["id"],
        "success": int(valid),
        "expected_turns": record["expected_turns"],
        "completed_turns": record["completed_turns"],
        "steps": len(generated_steps),
        "tool_calls_executed": record["tool_calls_executed"],
        "prompt_token_instances": sum(prompt_tokens),
        "max_prompt_tokens": max(prompt_tokens, default=0),
        "total_prefill_ms": sum(step["prefill_ms_median"] for step in generated_steps),
        "total_generation_ms": sum(step["generation_ms"] for step in generated_steps),
        "force_terminated": record["force_terminated"],
        "termination_reason": record["termination_reason"],
    }


def median(values):
    return statistics.median(values) if values else None


def make_report(folder, quality_rows, episode_rows, timing_rows):
    by_config = {}
    for row in episode_rows:
        by_config.setdefault(row["config_id"], []).append(row)
    lines = [
        "# BFCL V4 multi-turn long-context: Full vs V1",
        "",
        "能力分数使用 BFCL V4 官方 `multi_turn_checker`：执行模型工具调用后逐轮比较后端状态与返回结果。",
        "下表中的 `dense` 即 Full attention；`fp_v1__a0p08` 即 FlashPrefill V1。",
        "Qwen2.5 通过本项目适配器使用 BFCL 官方 Qwen `<tool_call>` 协议，因此这是固定模型的成对实验，"
        "不是可直接提交官方 leaderboard 的内置模型配置。",
        "",
        "| Config | n | Success | Success rate | Median steps | Median max prompt | Median summed prefill ms | Median generation ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for config_id, rows in sorted(by_config.items()):
        successes = sum(row["success"] for row in rows)
        lines.append(
            f"| {config_id} | {len(rows)} | {successes} | {successes / len(rows):.1%} | "
            f"{median([row['steps'] for row in rows]):.1f} | "
            f"{median([row['max_prompt_tokens'] for row in rows]):.0f} | "
            f"{median([row['total_prefill_ms'] for row in rows]):.1f} | "
            f"{median([row['total_generation_ms'] for row in rows]):.1f} |"
        )

    success_by_sample = {}
    for row in quality_rows:
        success_by_sample.setdefault(row["sample_id"], {})[row["method"]] = row["success"]
    regressions = [
        sample_id for sample_id, values in success_by_sample.items()
        if values.get("dense") == 1 and values.get("fp_v1") == 0
    ]
    recoveries = [
        sample_id for sample_id, values in success_by_sample.items()
        if values.get("dense") == 0 and values.get("fp_v1") == 1
    ]

    timing_groups = {}
    for row in timing_rows:
        key = (
            row["sample_id"],
            int(row["turn"]),
            int(row["step"]),
            row["prompt_sha256"],
            row["method"],
        )
        timing_groups.setdefault(key, []).append(float(row["prefill_ms"]))
    prompt_pairs = {}
    for key, values in timing_groups.items():
        sample_id, turn, step, prompt_sha256, method = key
        prompt_pairs.setdefault((sample_id, turn, step, prompt_sha256), {})[method] = median(values)
    ratios = [
        methods["dense"] / methods["fp_v1"]
        for methods in prompt_pairs.values()
        if "dense" in methods and "fp_v1" in methods
    ]
    v1_prompt_count = len([key for key in timing_groups if key[-1] == "fp_v1"])
    lines += [
        "",
        "## Paired differences",
        "",
        f"- Dense 成功、V1 失败：{len(regressions)} 个：{', '.join(regressions) if regressions else '无'}。",
        f"- Dense 失败、V1 成功：{len(recoveries)} 个：{', '.join(recoveries) if recoveries else '无'}。",
        (
            f"- 完全相同 prompt hash 的成对 prefill：{len(ratios)}/{v1_prompt_count} 个 V1 step；"
            f"中位 Full/V1 speedup = {median(ratios):.2f}×。"
            if ratios else
            "- 没有可比较的相同 prompt hash 成对 prefill。"
        ),
        "",
        "## Interpretation contract",
        "",
        "- 首个用户请求前 Full 与 V1 输入相同；一旦工具调用不同，后续工具结果和 prompt 也可能不同。",
        "- 因此成功率是端到端 agent 能力差异；只有相同 prompt hash 的 step 才用于纯 prefill speedup。",
        "- `summed prefill ms` 是一次 episode 内各 step 重复计时中位数之和；generation ms 还包含 dense decode，但不包含工具执行。",
        "- 超过 Qwen2.5 原生 32K（同时预留生成 token）的轨迹不截断，记为失败并单独保存原因。",
        "- 原始输出、解析后的调用、工具返回、每步 token 数/hash 和官方 checker 详情均已保存，可离线复核 scorer。",
        "",
    ]
    (folder / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def episode_schedule(entries, candidates):
    return [
        (entry, config)
        for sample_index, entry in enumerate(entries)
        for config in rotate(candidates, sample_index)
    ]


def write_csv_rows(path, fields, rows):
    sink = CsvSink(path, fields)
    for row in rows:
        sink.write(row)
    sink.close()


def prepare_resume_prefix(folder, schedule, timing_fields, attempt_dir):
    generations_path = folder / "generations.jsonl"
    timings_path = folder / "timings.csv"
    if not generations_path.is_file() or not timings_path.is_file():
        raise FileNotFoundError("resume requires generations.jsonl and timings.csv")

    shutil.copy2(generations_path, attempt_dir / "generations_before_resume.jsonl")
    shutil.copy2(timings_path, attempt_dir / "timings_before_resume.csv")
    records = load_jsonl(generations_path)
    if len(records) > len(schedule):
        raise ValueError("saved generations are longer than the requested schedule")
    expected_keys = [
        (entry["id"], config["config_id"])
        for entry, config in schedule
    ]
    actual_keys = [(record["id"], record["config_id"]) for record in records]
    if actual_keys != expected_keys[:len(actual_keys)]:
        raise ValueError("saved generations are not an exact prefix of the requested schedule")
    for record, (entry, config) in zip(records, schedule):
        if record["source_row_sha256"] != entry["source_row_sha256"]:
            raise ValueError(f"source hash changed for {entry['id']}")
        if record["method"] != config["method"] or record["alpha"] != config["alpha"]:
            raise ValueError(f"configuration changed for {entry['id']} / {config['config_id']}")

    with timings_path.open(newline="", encoding="utf-8") as source:
        timing_rows = list(csv.DictReader(source))
    complete_keys = set(actual_keys)
    kept_rows = [
        row for row in timing_rows
        if (row["sample_id"], row["config_id"]) in complete_keys
    ]
    partial_rows = [
        row for row in timing_rows
        if (row["sample_id"], row["config_id"]) not in complete_keys
    ]
    counts = {}
    for row in kept_rows:
        key = (row["sample_id"], row["config_id"])
        counts[key] = counts.get(key, 0) + 1
    for record in records:
        key = (record["id"], record["config_id"])
        expected_count = sum(
            len(step["prefill_ms"])
            for step in record["steps"]
            if step["status"] == "generated"
        )
        if counts.get(key, 0) != expected_count:
            raise ValueError(
                f"timing rows do not match saved episode {record['id']} / "
                f"{record['config_id']}: {counts.get(key, 0)} != {expected_count}"
            )
    if partial_rows:
        write_csv_rows(attempt_dir / "discarded_partial_timings.csv", timing_fields, partial_rows)
    timings_temp = timings_path.with_suffix(".resume.tmp")
    write_csv_rows(timings_temp, timing_fields, kept_rows)
    timings_temp.replace(timings_path)

    for config_id in sorted({config["config_id"] for _, config in schedule}):
        path = folder / "official_results" / config_id / "multi_turn"
        path.mkdir(parents=True, exist_ok=True)
        with (path / "BFCL_v4_multi_turn_long_context_result.json").open(
            "w", encoding="utf-8"
        ) as output:
            for record in records:
                if record["config_id"] == config_id:
                    write_jsonl_line(output, {
                        "id": record["id"],
                        "result": record["result"],
                        "metadata": {
                            "adapter_version": record["adapter_version"],
                            "force_terminated": record["force_terminated"],
                            "termination_reason": record["termination_reason"],
                        },
                    })

    next_key = expected_keys[len(records)] if len(records) < len(expected_keys) else None
    manifest = {
        "completed_records": len(records),
        "schedule_records": len(schedule),
        "kept_timing_rows": len(kept_rows),
        "discarded_partial_timing_rows": len(partial_rows),
        "next_episode": next_key,
        "generations_before_resume_sha256": file_hash(
            attempt_dir / "generations_before_resume.jsonl"
        ),
        "timings_before_resume_sha256": file_hash(
            attempt_dir / "timings_before_resume.csv"
        ),
    }
    write_json(attempt_dir / "resume_manifest.json", manifest)
    return records, manifest


def run(
    args,
    entries,
    multi_turn_utils,
    multi_turn_checker,
    metadata,
    resume_attempt_dir=None,
):
    timing_fields = [
        "method", "config_id", "alpha", "sample_id", "turn", "step",
        "prompt_tokens", "prompt_sha256", "repeat", "prefill_ms",
        "prefill_tokens_s", "peak_allocated_gib", "peak_reserved_gib",
        "first_token_id",
    ]
    candidates = [candidate(method, args.alpha) for method in args.methods]
    schedule = episode_schedule(entries, candidates)
    if args.resume:
        existing_records, resume_manifest = prepare_resume_prefix(
            args.out,
            schedule,
            timing_fields,
            resume_attempt_dir,
        )
        metadata["active_resume"] = resume_manifest
    else:
        existing_records = []
    metadata["completed_episode_prefix"] = len(existing_records)
    metadata["episode_schedule_size"] = len(schedule)
    write_json(args.out / "metadata.json", metadata)

    backend = AttentionBackend(
        alpha=args.alpha,
        selector_chunk_tiles=args.selector_chunk_tiles,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
    ).eval()
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    metadata["model_commit"] = model.config._commit_hash
    metadata["model_config"] = model.config.to_dict()
    metadata["gpu"] = gpu_metadata()
    write_json(args.out / "metadata.json", metadata)

    output_mode = "a" if args.resume else "w"
    timing_sink = CsvSink(args.out / "timings.csv", timing_fields, mode=output_mode)
    raw_output = (args.out / "generations.jsonl").open(output_mode, encoding="utf-8")
    official_outputs = {}
    for config in candidates:
        path = args.out / "official_results" / config["config_id"] / "multi_turn"
        path.mkdir(parents=True, exist_ok=True)
        official_outputs[config["config_id"]] = (
            path / "BFCL_v4_multi_turn_long_context_result.json"
        ).open(output_mode, encoding="utf-8")

    warmed_shapes = set()
    warmed_generation = set()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for entry, config in schedule[len(existing_records):]:
            print(
                f"Agent {config['config_id']:18s} {entry['id']} "
                f"turns={len(entry['question'])}",
                flush=True,
            )
            record = run_episode(
                args,
                entry,
                config,
                model,
                tokenizer,
                backend,
                multi_turn_utils,
                timing_sink,
                warmed_shapes,
                warmed_generation,
            )
            write_jsonl_line(raw_output, record)
            write_jsonl_line(official_outputs[config["config_id"]], {
                "id": record["id"],
                "result": record["result"],
                "metadata": {
                    "adapter_version": ADAPTER_VERSION,
                    "force_terminated": record["force_terminated"],
                    "termination_reason": record["termination_reason"],
                },
            })
    timing_sink.close()
    raw_output.close()
    for output in official_outputs.values():
        output.close()

    entries_by_id = {entry["id"]: entry for entry in entries}
    quality_fields = [
        "method", "config_id", "alpha", "sample_id", "success", "error_type",
        "error_message", "force_terminated", "termination_reason",
    ]
    episode_fields = [
        "method", "config_id", "alpha", "sample_id", "success", "expected_turns",
        "completed_turns", "steps", "tool_calls_executed", "prompt_token_instances",
        "max_prompt_tokens", "total_prefill_ms", "total_generation_ms",
        "force_terminated", "termination_reason",
    ]
    quality_sink = CsvSink(args.out / "quality.csv", quality_fields)
    episode_sink = CsvSink(args.out / "episodes.csv", episode_fields)
    scored_output = (args.out / "predictions.jsonl").open("w", encoding="utf-8")
    quality_rows = []
    episode_rows = []
    with (args.out / "generations.jsonl").open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            checker_result, checker_decoded_result = score_record(
                record,
                entries_by_id[record["id"]],
                multi_turn_utils,
                multi_turn_checker,
            )
            quality_row = {
                "method": record["method"],
                "config_id": record["config_id"],
                "alpha": record["alpha"],
                "sample_id": record["id"],
                "success": int(checker_result["valid"]),
                "error_type": checker_result.get("error_type"),
                "error_message": checker_result.get("error_message"),
                "force_terminated": record["force_terminated"],
                "termination_reason": record["termination_reason"],
            }
            episode = episode_row(record, checker_result["valid"])
            quality_rows.append(quality_row)
            episode_rows.append(episode)
            quality_sink.write(quality_row)
            episode_sink.write(episode)
            write_jsonl_line(scored_output, {
                **record,
                "checker": "bfcl_eval.multi_turn_checker",
                "checker_package_version": BFCL_VERSION,
                "checker_decoded_result": checker_decoded_result,
                "checker_result": checker_result,
            })
            print(
                f"Score {record['config_id']:18s} {record['id']} "
                f"valid={checker_result['valid']}",
                flush=True,
            )
    quality_sink.close()
    episode_sink.close()
    scored_output.close()

    timing_rows = []
    with (args.out / "timings.csv").open(encoding="utf-8") as source:
        timing_rows.extend(csv.DictReader(source))
    make_report(args.out, quality_rows, episode_rows, timing_rows)
    metadata["status"] = "complete"
    metadata["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_json(args.out / "metadata.json", metadata)


def main():
    args = arguments()
    if args.samples < 1 or args.repeats < 1 or args.max_new_tokens < 1:
        raise ValueError("samples, repeats, and max-new-tokens must be positive")
    if args.max_steps < 1 or args.context_limit < 1:
        raise ValueError("max-steps and context-limit must be positive")
    if args.alpha < 0:
        raise ValueError("alpha must be non-negative")
    if len(args.methods) != len(set(args.methods)):
        raise ValueError("methods must be unique")
    if args.prepare_only and args.resume:
        raise ValueError("--prepare-only and --resume cannot be combined")
    if args.resume:
        if not args.out.is_dir() or not (args.out / "metadata.json").is_file():
            raise FileNotFoundError(f"resume output is incomplete or missing: {args.out}")
    elif args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    entries, sources, multi_turn_utils, multi_turn_checker = load_bfcl(args)
    if len(entries) != args.samples:
        raise ValueError(f"requested {args.samples} rows, selected {len(entries)}")
    selected_case_rows = [
        {
            "id": entry["id"],
            "source_row_sha256": entry["source_row_sha256"],
            "involved_classes": entry["involved_classes"],
            "turn_count": len(entry["question"]),
            "question": entry["question"],
            "ground_truth": entry["ground_truth"],
        }
        for entry in entries
    ]
    resume_attempt_dir = None
    previous_metadata = None
    if args.resume:
        saved_case_rows = load_jsonl(args.out / "selected_cases.jsonl")
        if saved_case_rows != selected_case_rows:
            raise ValueError("saved selected_cases.jsonl does not match the requested cases")
        previous_metadata = json.loads(
            (args.out / "metadata.json").read_text(encoding="utf-8")
        )
        attempts_root = args.out / "resume_attempts"
        attempts_root.mkdir(parents=True, exist_ok=True)
        attempt_number = 1
        while (attempts_root / f"{attempt_number:03d}").exists():
            attempt_number += 1
        resume_attempt_dir = attempts_root / f"{attempt_number:03d}"
        resume_attempt_dir.mkdir()
        shutil.copy2(
            args.out / "metadata.json",
            resume_attempt_dir / "metadata_before_resume.json",
        )
    metadata = {
        "status": "resuming" if args.resume else "running",
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "sources": sources,
        "adapter": {
            "version": ADAPTER_VERSION,
            "protocol": "BFCL QwenFCHandler <tool_call>/<tool_response> prompt protocol",
            "status": (
                "project adapter for Qwen/Qwen2.5-7B-Instruct; official BFCL package "
                "does not register this exact model as a built-in leaderboard handler"
            ),
            "generation": "greedy; sparse/full prompt prefill followed by dense decode",
            "scorer": "official BFCL V4 multi_turn_checker",
        },
        "comparison_contract": (
            "same source episodes and initial state; trajectories may diverge after model output. "
            "Pure prefill speedup is computed only for steps with identical prompt SHA256."
        ),
        "candidates": [candidate(method, args.alpha) for method in args.methods],
    }
    if previous_metadata is not None:
        history = list(previous_metadata.get("resume_history", []))
        history.append({
            "attempt": len(history) + 1,
            "resumed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "previous_status": previous_metadata.get("status"),
            "previous_failure": previous_metadata.get("failure"),
            "artifact_directory": str(resume_attempt_dir),
        })
        metadata["resume_history"] = history
    write_json(args.out / "metadata.json", metadata)
    if not args.resume:
        with (args.out / "selected_cases.jsonl").open("w", encoding="utf-8") as output:
            for row in selected_case_rows:
                write_jsonl_line(output, row)
    if args.prepare_only:
        metadata["status"] = "prepared"
        write_json(args.out / "metadata.json", metadata)
        print(f"Prepared {len(entries)} fixed BFCL cases in {args.out}", flush=True)
        return
    try:
        run(
            args,
            entries,
            multi_turn_utils,
            multi_turn_checker,
            metadata,
            resume_attempt_dir=resume_attempt_dir,
        )
    except Exception as error:
        metadata["status"] = "failed"
        metadata["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(args.out / "metadata.json", metadata)
        raise
    print(f"Completed: {args.out / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
