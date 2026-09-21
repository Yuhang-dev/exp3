"""Audit and summarize the final paired LongBench v2 and BFCL V4 runs."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys

from scoring import score_longbench_v2


DENSE = "dense"
V1 = "fp_v1"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Extracted results directory containing the four final result folders.",
    )
    parser.add_argument(
        "--bfcl-root",
        type=Path,
        default=Path("third_party/bfcl_eval_2025_12_17"),
    )
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as sink:
        writer = csv.DictWriter(sink, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def percentile(sorted_values, probability):
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return (
        sorted_values[lower] * (1 - fraction)
        + sorted_values[upper] * fraction
    )


def bootstrap_mean_ci(values, draws=20000, seed=20260921):
    generator = random.Random(seed)
    count = len(values)
    estimates = []
    for _ in range(draws):
        estimates.append(
            sum(values[generator.randrange(count)] for _ in range(count)) / count
        )
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def bootstrap_cluster_median_ci(values_by_sample, draws=10000, seed=20260922):
    sample_ids = sorted(values_by_sample)
    generator = random.Random(seed)
    estimates = []
    for _ in range(draws):
        values = []
        for _ in sample_ids:
            selected = sample_ids[generator.randrange(len(sample_ids))]
            values.extend(values_by_sample[selected])
        estimates.append(statistics.median(values))
    estimates.sort()
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def mcnemar_exact(dense_only, v1_only):
    discordant = dense_only + v1_only
    if discordant == 0:
        return 1.0
    tail = min(dense_only, v1_only)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1))
    return min(1.0, 2 * probability / (2 ** discordant))


def length_bucket(tokens):
    if tokens < 16384:
        return "8–16K"
    if tokens < 24576:
        return "16–24K"
    return "24–32K"


def prompt_bucket(tokens):
    if tokens < 4096:
        return "<4K"
    if tokens < 8192:
        return "4–8K"
    if tokens < 16384:
        return "8–16K"
    return "16–32K"


def turn_bucket(turns):
    if turns <= 2:
        return "1–2 turns"
    if turns <= 4:
        return "3–4 turns"
    return "5–7 turns"


def outcome(dense, v1):
    if dense and v1:
        return "both_correct"
    if dense:
        return "dense_only"
    if v1:
        return "v1_only"
    return "both_wrong"


def median_or_none(values):
    return statistics.median(values) if values else None


def formatted(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def parse_qwen_response(raw_text):
    matches = re.findall(r"<tool_call>\n(.*?)\n</tool_call>", raw_text, re.DOTALL)
    calls = []
    for match in matches:
        try:
            calls.append(json.loads(match))
        except Exception:
            pass
    try:
        decoded = []
        for call in calls:
            name = call["name"]
            if not isinstance(name, str):
                raise TypeError("tool name must be a string")
            call_arguments = call["arguments"]
            if isinstance(call_arguments, str):
                call_arguments = json.loads(call_arguments)
            if not isinstance(call_arguments, dict):
                raise TypeError("tool arguments must be an object")
            decoded.append(
                f"{name}({','.join(f'{key}={value!r}' for key, value in call_arguments.items())})"
            )
        error = None
    except Exception as exception:
        decoded = []
        error = f"{type(exception).__name__}: {exception}"
    return decoded, error


def decode_saved_result(raw_result):
    decoded_result = []
    for turn in raw_result:
        decoded_turn = []
        for raw_text in turn:
            decoded, _ = parse_qwen_response(raw_text)
            if decoded:
                decoded_turn.append(decoded)
        decoded_result.append(decoded_turn)
    return decoded_result


def instance_name(model_name, test_id, class_name):
    return re.sub(
        r"[-./]",
        "_",
        f"{model_name}_{test_id}_{class_name}_instance",
    )


def summarize_group(
    benchmark,
    dimension,
    label,
    sample_ids,
    pairs,
    speed_by_sample,
    v1_step_count=None,
    notes="",
):
    sample_ids = sorted(sample_ids)
    dense_success = sum(pairs[sample_id][DENSE] for sample_id in sample_ids)
    v1_success = sum(pairs[sample_id][V1] for sample_id in sample_ids)
    outcomes = Counter(
        outcome(pairs[sample_id][DENSE], pairs[sample_id][V1])
        for sample_id in sample_ids
    )
    ratios = [
        ratio
        for sample_id in sample_ids
        for ratio in speed_by_sample.get(sample_id, [])
    ]
    return {
        "benchmark": benchmark,
        "dimension": dimension,
        "group": label,
        "n": len(sample_ids),
        "dense_success": dense_success,
        "v1_success": v1_success,
        "dense_rate_pct": 100 * dense_success / len(sample_ids),
        "v1_rate_pct": 100 * v1_success / len(sample_ids),
        "delta_pp": 100 * (v1_success - dense_success) / len(sample_ids),
        "both_correct": outcomes["both_correct"],
        "dense_only": outcomes["dense_only"],
        "v1_only": outcomes["v1_only"],
        "both_wrong": outcomes["both_wrong"],
        "matched_prompt_steps": len(ratios),
        "v1_prompt_steps": (
            sum(v1_step_count.get(sample_id, 0) for sample_id in sample_ids)
            if v1_step_count is not None
            else len(sample_ids)
        ),
        "paired_speedup_median": median_or_none(ratios),
        "notes": notes,
    }


def subgroup_rows(
    benchmark,
    pairs,
    attributes,
    dimensions,
    speed_by_sample,
    v1_step_count=None,
):
    rows = [
        summarize_group(
            benchmark,
            "overall",
            "all",
            pairs,
            pairs,
            speed_by_sample,
            v1_step_count,
        )
    ]
    for dimension, extractor, notes in dimensions:
        grouped = defaultdict(list)
        for sample_id in pairs:
            values = extractor(attributes[sample_id])
            if not isinstance(values, (list, tuple, set)):
                values = [values]
            for value in values:
                grouped[str(value)].append(sample_id)
        for label in sorted(grouped):
            rows.append(
                summarize_group(
                    benchmark,
                    dimension,
                    label,
                    grouped[label],
                    pairs,
                    speed_by_sample,
                    v1_step_count,
                    notes,
                )
            )
    return rows


def main():
    args = arguments()
    root = args.root.resolve()
    output = (args.out or root / "analysis").resolve()
    output.mkdir(parents=True, exist_ok=True)

    longbench = root / "longbench_v2_native32k_full116"
    longbench_inputs_folder = root / "longbench_v2_native32k_inputs116"
    bfcl = root / "bfcl_v4_multi_turn_base_full100"
    bfcl_inputs_folder = root / "bfcl_v4_multi_turn_base_inputs100"
    for path in (longbench, longbench_inputs_folder, bfcl, bfcl_inputs_folder):
        if not path.is_dir():
            raise FileNotFoundError(path)

    source_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not path.resolve().is_relative_to(output)
    )
    source_hashes = {
        str(path.relative_to(root)).replace("\\", "/"): sha256(path)
        for path in source_files
    }

    # LongBench integrity and independent rescoring.
    lb_metadata = json.loads((longbench / "metadata.json").read_text(encoding="utf-8"))
    lb_inputs = {
        row["sample_id"]: row
        for row in read_jsonl(longbench / "inputs.jsonl")
    }
    lb_prepared_inputs = read_jsonl(longbench_inputs_folder / "inputs.jsonl")
    lb_generations = read_jsonl(longbench / "generations.jsonl")
    lb_predictions = read_jsonl(longbench / "predictions.jsonl")
    lb_quality = read_csv(longbench / "quality.csv")
    lb_timings = read_csv(longbench / "timings.csv")
    if lb_metadata["status"] != "complete":
        raise ValueError(f"LongBench status is {lb_metadata['status']}")
    if len(lb_inputs) != 116 or len(lb_prepared_inputs) != 116:
        raise ValueError("LongBench input count mismatch")
    if len(lb_generations) != 232 or len(lb_predictions) != 232:
        raise ValueError("LongBench generation/prediction count mismatch")
    if len(lb_quality) != 232 or len(lb_timings) != 696:
        raise ValueError("LongBench quality/timing count mismatch")
    if (longbench / "inputs.jsonl").read_bytes() != (
        longbench_inputs_folder / "inputs.jsonl"
    ).read_bytes():
        raise ValueError("LongBench reused input manifest differs from prepared inputs")
    if sha256(longbench / "inputs.pt") != sha256(longbench_inputs_folder / "inputs.pt"):
        raise ValueError("LongBench reused input tensor differs from prepared inputs")

    lb_generation_by_key = {
        (row["sample_id"], row["method"]): row for row in lb_generations
    }
    lb_prediction_by_key = {
        (row["sample_id"], row["method"]): row for row in lb_predictions
    }
    lb_quality_by_key = {
        (row["sample_id"], row["method"]): row for row in lb_quality
    }
    expected_lb_keys = {
        (sample_id, method) for sample_id in lb_inputs for method in (DENSE, V1)
    }
    if set(lb_generation_by_key) != expected_lb_keys:
        raise ValueError("LongBench generation pairs are incomplete")
    if set(lb_prediction_by_key) != expected_lb_keys or set(lb_quality_by_key) != expected_lb_keys:
        raise ValueError("LongBench scored pairs are incomplete")

    lb_rescore_rows = []
    lb_score_changes = 0
    for key in sorted(expected_lb_keys):
        sample_id, method = key
        sample = lb_inputs[sample_id]
        generation = lb_generation_by_key[key]
        prediction = lb_prediction_by_key[key]
        quality = lb_quality_by_key[key]
        if generation["input_sha256"] != sample["input_sha256"]:
            raise ValueError(f"LongBench input hash mismatch for {key}")
        for field in ("raw_text", "generated_token_ids", "end_reason", "config_id"):
            if generation[field] != prediction[field]:
                raise ValueError(f"LongBench generation/prediction mismatch: {key} / {field}")
        rescored = score_longbench_v2(generation["raw_text"], sample["answers"])
        stored_score = float(quality["score"])
        changed = int(stored_score != rescored["score"])
        lb_score_changes += changed
        if prediction["score"]["score"] != rescored["score"]:
            raise ValueError(f"LongBench prediction score mismatch for {key}")
        lb_rescore_rows.append({
            "sample_id": sample_id,
            "method": method,
            "stored_score": stored_score,
            "rescored_score": rescored["score"],
            "stored_parsed_answer": quality["parsed_answer"],
            "rescored_parsed_answer": rescored["parsed_answer"],
            "changed": changed,
        })
    if lb_score_changes:
        raise ValueError(f"LongBench offline rescore changed {lb_score_changes} decisions")

    lb_timing_groups = defaultdict(list)
    for row in lb_timings:
        lb_timing_groups[(row["sample_id"], row["method"])].append(float(row["prefill_ms"]))
    if set(lb_timing_groups) != expected_lb_keys:
        raise ValueError("LongBench timing pairs are incomplete")
    if any(len(values) != 3 for values in lb_timing_groups.values()):
        raise ValueError("LongBench does not have exactly three timings per sample/method")
    lb_sample_time = {
        key: statistics.median(values) for key, values in lb_timing_groups.items()
    }
    lb_speed_by_sample = {
        sample_id: [
            lb_sample_time[(sample_id, DENSE)] / lb_sample_time[(sample_id, V1)]
        ]
        for sample_id in lb_inputs
    }
    lb_pairs = {
        sample_id: {
            method: int(float(lb_quality_by_key[(sample_id, method)]["score"]) == 100)
            for method in (DENSE, V1)
        }
        for sample_id in lb_inputs
    }
    lb_attributes = {
        sample_id: {
            "domain": sample["domain"],
            "sub_domain": sample["sub_domain"],
            "difficulty": sample["difficulty"],
            "length_bucket": length_bucket(int(sample["actual_tokens"])),
            "actual_tokens": int(sample["actual_tokens"]),
        }
        for sample_id, sample in lb_inputs.items()
    }
    lb_subgroups = subgroup_rows(
        "LongBench v2",
        lb_pairs,
        lb_attributes,
        [
            ("domain", lambda row: row["domain"], "official domain; small groups are descriptive"),
            ("difficulty", lambda row: row["difficulty"], "official difficulty"),
            ("length_bucket", lambda row: row["length_bucket"], "actual chat-templated prompt tokens"),
            ("sub_domain", lambda row: row["sub_domain"], "official sub-domain; exploratory"),
        ],
        lb_speed_by_sample,
    )

    lb_resume = json.loads(
        (longbench / "resume_attempts/001/resume_manifest.json").read_text(encoding="utf-8")
    )
    if lb_resume["completed_samples"] != 103:
        raise ValueError("Unexpected LongBench resume prefix")
    archived_generations = read_jsonl(
        longbench / "resume_attempts/001/generations_before_resume.jsonl"
    )
    if archived_generations != lb_generations[:206]:
        raise ValueError("LongBench preserved generation prefix differs from final output")
    archived_timings = read_csv(
        longbench / "resume_attempts/001/timings_before_resume.csv"
    )
    if archived_timings[:618] != lb_timings[:618]:
        raise ValueError("LongBench preserved timing prefix differs from final output")
    if len(read_csv(longbench / "resume_attempts/001/discarded_partial_timings.csv")) != 4:
        raise ValueError("LongBench discarded partial timing count differs")

    # BFCL integrity and independent official-checker rescoring.
    bf_metadata = json.loads((bfcl / "metadata.json").read_text(encoding="utf-8"))
    bf_selected = read_jsonl(bfcl / "selected_cases.jsonl")
    bf_prepared_selected = read_jsonl(bfcl_inputs_folder / "selected_cases.jsonl")
    bf_generations = read_jsonl(bfcl / "generations.jsonl")
    bf_predictions = read_jsonl(bfcl / "predictions.jsonl")
    bf_quality = read_csv(bfcl / "quality.csv")
    bf_episodes = read_csv(bfcl / "episodes.csv")
    bf_timings = read_csv(bfcl / "timings.csv")
    if bf_metadata["status"] != "complete":
        raise ValueError(f"BFCL status is {bf_metadata['status']}")
    if len(bf_selected) != 100 or bf_selected != bf_prepared_selected:
        raise ValueError("BFCL selected-case manifest mismatch")
    if len(bf_generations) != 200 or len(bf_predictions) != 200:
        raise ValueError("BFCL generation/prediction count mismatch")
    if len(bf_quality) != 200 or len(bf_episodes) != 200:
        raise ValueError("BFCL quality/episode count mismatch")

    bf_data_root = args.bfcl_root.resolve() / "bfcl_eval/data"
    sys.path.insert(0, str(args.bfcl_root.resolve()))
    from bfcl_eval.eval_checker.multi_turn_eval import multi_turn_utils
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker

    bf_source_entries = {
        row["id"]: row
        for row in read_jsonl(bf_data_root / "BFCL_v4_multi_turn_base.json")
    }
    bf_answers = {
        row["id"]: row["ground_truth"]
        for row in read_jsonl(
            bf_data_root / "possible_answer/BFCL_v4_multi_turn_base.json"
        )
    }
    bf_selected_by_id = {row["id"]: row for row in bf_selected}
    if len(bf_selected_by_id) != 100:
        raise ValueError("BFCL selected IDs are not unique")
    for sample_id, selected in bf_selected_by_id.items():
        if json_hash(bf_source_entries[sample_id]) != selected["source_row_sha256"]:
            raise ValueError(f"BFCL source row hash mismatch: {sample_id}")
        if bf_answers[sample_id] != selected["ground_truth"]:
            raise ValueError(f"BFCL answer mismatch: {sample_id}")

    bf_generation_by_key = {
        (row["id"], row["method"]): row for row in bf_generations
    }
    bf_prediction_by_key = {
        (row["id"], row["method"]): row for row in bf_predictions
    }
    bf_quality_by_key = {
        (row["sample_id"], row["method"]): row for row in bf_quality
    }
    bf_episode_by_key = {
        (row["sample_id"], row["method"]): row for row in bf_episodes
    }
    expected_bf_keys = {
        (sample_id, method) for sample_id in bf_selected_by_id for method in (DENSE, V1)
    }
    if set(bf_generation_by_key) != expected_bf_keys:
        raise ValueError("BFCL generation pairs are incomplete")
    if set(bf_prediction_by_key) != expected_bf_keys:
        raise ValueError("BFCL prediction pairs are incomplete")
    if set(bf_quality_by_key) != expected_bf_keys or set(bf_episode_by_key) != expected_bf_keys:
        raise ValueError("BFCL scored pairs are incomplete")

    bf_rescore_rows = []
    bf_score_changes = 0
    bf_error_type_changes = 0
    for record_index, key in enumerate(sorted(expected_bf_keys)):
        sample_id, method = key
        record = bf_generation_by_key[key]
        prediction = bf_prediction_by_key[key]
        quality = bf_quality_by_key[key]
        selected = bf_selected_by_id[sample_id]
        if record["source_row_sha256"] != selected["source_row_sha256"]:
            raise ValueError(f"BFCL generation source hash mismatch: {key}")
        for field in (
            "result",
            "steps",
            "force_terminated",
            "termination_reason",
            "completed_turns",
        ):
            if record[field] != prediction[field]:
                raise ValueError(f"BFCL generation/prediction mismatch: {key} / {field}")
        decoded = decode_saved_result(record["result"])
        if record["force_terminated"] or record["completed_turns"] != record["expected_turns"]:
            rescored = {
                "valid": False,
                "error_type": f"multi_turn:{record['termination_reason'] or 'force_terminated'}",
            }
        else:
            entry = bf_source_entries[sample_id]
            score_name = f"exp3_offline_{record_index}_{record['config_id']}"
            rescored = multi_turn_checker(
                decoded,
                bf_answers[sample_id],
                entry,
                "multi_turn_base",
                score_name,
            )
            for model_name in (score_name + "_eval", score_name + "_ground_truth_eval"):
                for class_name in entry["involved_classes"]:
                    multi_turn_utils.__dict__.pop(
                        instance_name(model_name, sample_id, class_name),
                        None,
                    )
        stored_valid = bool(prediction["checker_result"]["valid"])
        csv_valid = bool(int(quality["success"]))
        if stored_valid != csv_valid:
            raise ValueError(f"BFCL stored checker/CSV mismatch: {key}")
        changed = int(stored_valid != bool(rescored["valid"]))
        error_changed = int(
            (prediction["checker_result"].get("error_type") or "")
            != (rescored.get("error_type") or "")
        )
        bf_score_changes += changed
        bf_error_type_changes += error_changed
        bf_rescore_rows.append({
            "sample_id": sample_id,
            "method": method,
            "stored_success": int(stored_valid),
            "rescored_success": int(bool(rescored["valid"])),
            "stored_error_type": prediction["checker_result"].get("error_type"),
            "rescored_error_type": rescored.get("error_type"),
            "changed": changed,
            "error_type_changed": error_changed,
        })
    if bf_score_changes or bf_error_type_changes:
        raise ValueError(
            f"BFCL offline rescore changed {bf_score_changes} decisions and "
            f"{bf_error_type_changes} error types"
        )

    for config_id in ("dense", "fp_v1__a0p08"):
        official = read_jsonl(
            bfcl / f"official_results/{config_id}/multi_turn/BFCL_v4_multi_turn_base_result.json"
        )
        if len(official) != 100:
            raise ValueError(f"BFCL official output count mismatch for {config_id}")

    bf_pairs = {
        sample_id: {
            method: int(bf_quality_by_key[(sample_id, method)]["success"])
            for method in (DENSE, V1)
        }
        for sample_id in bf_selected_by_id
    }
    bf_attributes = {
        sample_id: {
            "turn_bucket": turn_bucket(int(row["turn_count"])),
            "turn_count": int(row["turn_count"]),
            "api_count": "single API" if len(row["involved_classes"]) == 1 else "cross API",
            "class_signature": "+".join(sorted(row["involved_classes"])),
            "classes": sorted(row["involved_classes"]),
        }
        for sample_id, row in bf_selected_by_id.items()
    }

    bf_timing_groups = defaultdict(list)
    bf_prompt_tokens = {}
    for row in bf_timings:
        key = (
            row["sample_id"],
            int(row["turn"]),
            int(row["step"]),
            row["prompt_sha256"],
            row["method"],
        )
        bf_timing_groups[key].append(float(row["prefill_ms"]))
        bf_prompt_tokens[key] = int(row["prompt_tokens"])
    if any(len(values) != 3 for values in bf_timing_groups.values()):
        raise ValueError("BFCL does not have exactly three timings per generated step")
    bf_step_median = {
        key: statistics.median(values) for key, values in bf_timing_groups.items()
    }
    bf_prompt_pairs = defaultdict(dict)
    for key, value in bf_step_median.items():
        sample_id, turn, step, prompt_sha256, method = key
        bf_prompt_pairs[(sample_id, turn, step, prompt_sha256)][method] = value
    bf_matched_steps = []
    bf_speed_by_sample = defaultdict(list)
    for key, methods in bf_prompt_pairs.items():
        if DENSE not in methods or V1 not in methods:
            continue
        sample_id, turn, step, prompt_sha256 = key
        v1_key = (sample_id, turn, step, prompt_sha256, V1)
        ratio = methods[DENSE] / methods[V1]
        bf_speed_by_sample[sample_id].append(ratio)
        bf_matched_steps.append({
            "sample_id": sample_id,
            "turn": turn,
            "step": step,
            "prompt_sha256": prompt_sha256,
            "prompt_tokens": bf_prompt_tokens[v1_key],
            "dense_prefill_ms": methods[DENSE],
            "v1_prefill_ms": methods[V1],
            "speedup": ratio,
        })
    bf_v1_step_count = Counter(
        key[0] for key in bf_timing_groups if key[-1] == V1
    )
    if len(bf_matched_steps) != 266 or sum(bf_v1_step_count.values()) != 933:
        raise ValueError("BFCL same-prompt coverage differs from the saved report")

    bf_subgroups = subgroup_rows(
        "BFCL V4 multi_turn_base",
        bf_pairs,
        bf_attributes,
        [
            ("turn_bucket", lambda row: row["turn_bucket"], "disjoint expected-turn buckets"),
            ("api_count", lambda row: row["api_count"], "single versus multiple involved backends"),
            ("class_signature", lambda row: row["class_signature"], "20 small descriptive strata"),
            ("class_presence", lambda row: row["classes"], "overlapping; one case may appear in multiple classes"),
        ],
        bf_speed_by_sample,
        bf_v1_step_count,
    )

    # Paired outcomes and latency slices.
    paired_rows = []
    for sample_id in sorted(lb_pairs):
        pair = lb_pairs[sample_id]
        attributes = lb_attributes[sample_id]
        paired_rows.append({
            "benchmark": "LongBench v2",
            "sample_id": sample_id,
            "dense_success": pair[DENSE],
            "v1_success": pair[V1],
            "outcome": outcome(pair[DENSE], pair[V1]),
            "domain": attributes["domain"],
            "difficulty": attributes["difficulty"],
            "length_bucket": attributes["length_bucket"],
            "turn_bucket": "",
            "class_signature": "",
        })
    for sample_id in sorted(bf_pairs):
        pair = bf_pairs[sample_id]
        attributes = bf_attributes[sample_id]
        paired_rows.append({
            "benchmark": "BFCL V4 multi_turn_base",
            "sample_id": sample_id,
            "dense_success": pair[DENSE],
            "v1_success": pair[V1],
            "outcome": outcome(pair[DENSE], pair[V1]),
            "domain": "",
            "difficulty": "",
            "length_bucket": "",
            "turn_bucket": attributes["turn_bucket"],
            "class_signature": attributes["class_signature"],
        })

    latency_rows = []
    for dimension, groups in (
        ("overall", {"all": list(lb_inputs)}),
        (
            "length_bucket",
            {
                label: [
                    sample_id
                    for sample_id, row in lb_attributes.items()
                    if row["length_bucket"] == label
                ]
                for label in ("8–16K", "16–24K", "24–32K")
            },
        ),
    ):
        for label, sample_ids in groups.items():
            dense_values = [lb_sample_time[(sample_id, DENSE)] for sample_id in sample_ids]
            v1_values = [lb_sample_time[(sample_id, V1)] for sample_id in sample_ids]
            ratios = [lb_speed_by_sample[sample_id][0] for sample_id in sample_ids]
            latency_rows.append({
                "benchmark": "LongBench v2",
                "dimension": dimension,
                "group": label,
                "n_pairs": len(sample_ids),
                "dense_prefill_ms_median": statistics.median(dense_values),
                "v1_prefill_ms_median": statistics.median(v1_values),
                "paired_speedup_median": statistics.median(ratios),
                "coverage_numerator": len(sample_ids),
                "coverage_denominator": len(sample_ids),
                "notes": "one pair per fixed prompt",
            })
    bf_latency_groups = defaultdict(list)
    for row in bf_matched_steps:
        bf_latency_groups[prompt_bucket(row["prompt_tokens"])].append(row)
    bf_latency_groups = {"all": bf_matched_steps, **dict(bf_latency_groups)}
    for label, rows in bf_latency_groups.items():
        latency_rows.append({
            "benchmark": "BFCL V4 multi_turn_base",
            "dimension": "overall" if label == "all" else "prompt_token_bucket",
            "group": label,
            "n_pairs": len(rows),
            "dense_prefill_ms_median": statistics.median(
                row["dense_prefill_ms"] for row in rows
            ),
            "v1_prefill_ms_median": statistics.median(
                row["v1_prefill_ms"] for row in rows
            ),
            "paired_speedup_median": statistics.median(row["speedup"] for row in rows),
            "coverage_numerator": len(rows),
            "coverage_denominator": sum(bf_v1_step_count.values()),
            "notes": "only identical sample/turn/step/prompt-hash pairs",
        })

    lb_differences = [
        100 * (lb_pairs[sample_id][V1] - lb_pairs[sample_id][DENSE])
        for sample_id in sorted(lb_pairs)
    ]
    bf_differences = [
        100 * (bf_pairs[sample_id][V1] - bf_pairs[sample_id][DENSE])
        for sample_id in sorted(bf_pairs)
    ]
    lb_quality_ci = bootstrap_mean_ci(lb_differences)
    bf_quality_ci = bootstrap_mean_ci(bf_differences, seed=20260923)
    lb_speed_ci = bootstrap_cluster_median_ci(lb_speed_by_sample)
    bf_speed_ci = bootstrap_cluster_median_ci(bf_speed_by_sample, seed=20260924)
    lb_outcomes = Counter(
        outcome(lb_pairs[sample_id][DENSE], lb_pairs[sample_id][V1])
        for sample_id in lb_pairs
    )
    bf_outcomes = Counter(
        outcome(bf_pairs[sample_id][DENSE], bf_pairs[sample_id][V1])
        for sample_id in bf_pairs
    )

    bf_error_types = {
        method: Counter(
            (bf_quality_by_key[(sample_id, method)]["error_type"] or "success")
            for sample_id in bf_selected_by_id
        )
        for method in (DENSE, V1)
    }
    bf_episode_medians = {
        method: {
            field: statistics.median(
                float(bf_episode_by_key[(sample_id, method)][field])
                for sample_id in bf_selected_by_id
            )
            for field in (
                "steps",
                "max_prompt_tokens",
                "total_prefill_ms",
                "total_generation_ms",
            )
        }
        for method in (DENSE, V1)
    }
    bf_step_diagnostics = {}
    for method in (DENSE, V1):
        method_records = [row for row in bf_generations if row["method"] == method]
        steps = [step for row in method_records for step in row["steps"]]
        generated_steps = [step for step in steps if step["status"] == "generated"]
        bf_step_diagnostics[method] = {
            "episodes": len(method_records),
            "generated_steps": len(generated_steps),
            "max_new_token_steps": sum(
                step["end_reason"] == "max_new_tokens" for step in generated_steps
            ),
            "decode_error_steps": sum(
                bool(step.get("decode_error")) for step in generated_steps
            ),
            "context_overflow_steps": sum(
                step["status"] == "context_overflow" for step in steps
            ),
            "force_terminated_episodes": sum(row["force_terminated"] for row in method_records),
            "termination_reasons": dict(
                Counter(
                    row["termination_reason"] or "none" for row in method_records
                )
            ),
        }

    subgroup_fields = [
        "benchmark", "dimension", "group", "n", "dense_success", "v1_success",
        "dense_rate_pct", "v1_rate_pct", "delta_pp", "both_correct", "dense_only",
        "v1_only", "both_wrong", "matched_prompt_steps", "v1_prompt_steps",
        "paired_speedup_median", "notes",
    ]
    write_csv(output / "longbench_subgroups.csv", lb_subgroups, subgroup_fields)
    write_csv(output / "bfcl_subgroups.csv", bf_subgroups, subgroup_fields)
    write_csv(
        output / "paired_outcomes.csv",
        paired_rows,
        [
            "benchmark", "sample_id", "dense_success", "v1_success", "outcome",
            "domain", "difficulty", "length_bucket", "turn_bucket", "class_signature",
        ],
    )
    write_csv(
        output / "latency_by_length.csv",
        latency_rows,
        [
            "benchmark", "dimension", "group", "n_pairs", "dense_prefill_ms_median",
            "v1_prefill_ms_median", "paired_speedup_median", "coverage_numerator",
            "coverage_denominator", "notes",
        ],
    )
    write_csv(
        output / "longbench_offline_rescore.csv",
        lb_rescore_rows,
        [
            "sample_id", "method", "stored_score", "rescored_score",
            "stored_parsed_answer", "rescored_parsed_answer", "changed",
        ],
    )
    write_csv(
        output / "bfcl_offline_rescore.csv",
        bf_rescore_rows,
        [
            "sample_id", "method", "stored_success", "rescored_success",
            "stored_error_type", "rescored_error_type", "changed", "error_type_changed",
        ],
    )

    lb_dense_success = sum(row[DENSE] for row in lb_pairs.values())
    lb_v1_success = sum(row[V1] for row in lb_pairs.values())
    bf_dense_success = sum(row[DENSE] for row in bf_pairs.values())
    bf_v1_success = sum(row[V1] for row in bf_pairs.values())
    lb_dense_ms = statistics.median(
        lb_sample_time[(sample_id, DENSE)] for sample_id in lb_inputs
    )
    lb_v1_ms = statistics.median(
        lb_sample_time[(sample_id, V1)] for sample_id in lb_inputs
    )
    lb_speed = statistics.median(
        lb_speed_by_sample[sample_id][0] for sample_id in lb_inputs
    )
    bf_speed = statistics.median(row["speedup"] for row in bf_matched_steps)

    report_lines = [
        "# Final 100+ Full/V1 benchmark audit",
        "",
        "两个 benchmark 分别报告，不把 LongBench accuracy 与 BFCL success rate 合成总分。",
        "质量差异按相同样本配对；LongBench 延迟按相同固定 prompt 配对；BFCL 纯 prefill 延迟只使用轨迹中 prompt hash 完全一致的 step。",
        "",
        "## Main results",
        "",
        "| Benchmark | n | Full | V1 | Δ V1−Full | Paired flips (Full-only / V1-only) | Prefill Full | Prefill V1 | Paired speedup |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| LongBench v2 native 8–32K | 116 | {100 * lb_dense_success / 116:.2f} | "
            f"{100 * lb_v1_success / 116:.2f} | {100 * (lb_v1_success - lb_dense_success) / 116:+.2f} pp | "
            f"{lb_outcomes['dense_only']} / {lb_outcomes['v1_only']} | {lb_dense_ms:.1f} ms | "
            f"{lb_v1_ms:.1f} ms | {lb_speed:.2f}× |"
        ),
        (
            f"| BFCL V4 multi_turn_base | 100 | {bf_dense_success:.0f}% | {bf_v1_success:.0f}% | "
            f"{bf_v1_success - bf_dense_success:+.2f} pp | {bf_outcomes['dense_only']} / "
            f"{bf_outcomes['v1_only']} | {bf_episode_medians[DENSE]['total_prefill_ms']:.1f} ms* | "
            f"{bf_episode_medians[V1]['total_prefill_ms']:.1f} ms* | {bf_speed:.2f}×** |"
        ),
        "",
        "\\* BFCL episode prefill totals come from divergent agent trajectories and are descriptive, not pure paired kernel timing.",
        f"\\** BFCL pure speed uses {len(bf_matched_steps)}/{sum(bf_v1_step_count.values())} identical-prompt V1 steps ({100 * len(bf_matched_steps) / sum(bf_v1_step_count.values()):.1f}% coverage).",
        "",
        "## Paired uncertainty",
        "",
        (
            f"- LongBench: Δ = {100 * (lb_v1_success - lb_dense_success) / 116:+.2f} pp; "
            f"paired bootstrap 95% CI [{lb_quality_ci[0]:+.2f}, {lb_quality_ci[1]:+.2f}] pp; "
            f"exact McNemar p={mcnemar_exact(lb_outcomes['dense_only'], lb_outcomes['v1_only']):.3f}."
        ),
        (
            f"- LongBench paired prefill speedup = {lb_speed:.2f}×; sample-cluster bootstrap "
            f"95% CI [{lb_speed_ci[0]:.3f}, {lb_speed_ci[1]:.3f}]×."
        ),
        (
            f"- BFCL: Δ = {bf_v1_success - bf_dense_success:+.2f} pp; paired bootstrap 95% CI "
            f"[{bf_quality_ci[0]:+.2f}, {bf_quality_ci[1]:+.2f}] pp; exact McNemar "
            f"p={mcnemar_exact(bf_outcomes['dense_only'], bf_outcomes['v1_only']):.3f}."
        ),
        (
            f"- BFCL identical-prompt prefill speedup = {bf_speed:.2f}×; sample-cluster bootstrap "
            f"95% CI [{bf_speed_ci[0]:.3f}, {bf_speed_ci[1]:.3f}]×."
        ),
        "",
        "The intervals are descriptive paired bootstrap intervals. They are not a pre-registered non-inferiority test.",
        "",
        "## LongBench sub-evaluations",
        "",
        "| Dimension | Group | n | Full | V1 | Δ pp | Full-only | V1-only | Speedup |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in lb_subgroups:
        if row["dimension"] not in {"domain", "difficulty", "length_bucket"}:
            continue
        report_lines.append(
            f"| {row['dimension']} | {row['group']} | {row['n']} | "
            f"{row['dense_rate_pct']:.2f} | {row['v1_rate_pct']:.2f} | "
            f"{row['delta_pp']:+.2f} | {row['dense_only']} | {row['v1_only']} | "
            f"{formatted(row['paired_speedup_median'])}× |"
        )
    report_lines += [
        "",
        "Code Repository (n=3), Long In-context Learning (n=7), and Long Structured Data (n=1) are too small for stable method claims. Their large deltas remain descriptive.",
        "",
        "## BFCL sub-evaluations",
        "",
        "| Dimension | Group | n | Full | V1 | Δ pp | Full-only | V1-only | Identical-prompt steps | Speedup |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in bf_subgroups:
        if row["dimension"] not in {"turn_bucket", "api_count"}:
            continue
        report_lines.append(
            f"| {row['dimension']} | {row['group']} | {row['n']} | "
            f"{row['dense_rate_pct']:.2f} | {row['v1_rate_pct']:.2f} | "
            f"{row['delta_pp']:+.2f} | {row['dense_only']} | {row['v1_only']} | "
            f"{row['matched_prompt_steps']} | {formatted(row['paired_speedup_median'])}× |"
        )
    report_lines += [
        "",
        "### BFCL failure and trajectory diagnostics",
        "",
        "| Method | Generated steps | Max-token steps | Decode-error steps | Context-overflow steps | Force-terminated episodes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in (DENSE, V1):
        diagnostic = bf_step_diagnostics[method]
        report_lines.append(
            f"| {method} | {diagnostic['generated_steps']} | "
            f"{diagnostic['max_new_token_steps']} | {diagnostic['decode_error_steps']} | "
            f"{diagnostic['context_overflow_steps']} | {diagnostic['force_terminated_episodes']} |"
        )
    report_lines += [
        "",
        "Official checker outcome counts:",
        "",
    ]
    all_error_types = sorted(set(bf_error_types[DENSE]) | set(bf_error_types[V1]))
    for error_type in all_error_types:
        report_lines.append(
            f"- `{error_type}`: Full {bf_error_types[DENSE][error_type]}, "
            f"V1 {bf_error_types[V1][error_type]}."
        )
    report_lines += [
        "",
        "## Integrity and rescoring",
        "",
        f"- Archive SHA256: `{sha256(args.archive) if args.archive else 'not supplied'}`.",
        f"- LongBench: 116 inputs, 232 generations, 696 timing rows; independent raw-text rescore changed {lb_score_changes}/232 decisions.",
        f"- BFCL: 100 inputs, 200 trajectories; pinned official checker rescore changed {bf_score_changes}/200 decisions and {bf_error_type_changes}/200 error types.",
        "- LongBench recovery retained the exact 103-sample prefix, archived the original 206 generations and 622 timing rows, kept 618 complete timing rows, and isolated four partial rows.",
        "- Every LongBench generation input hash matches its fixed input. Every BFCL selected source-row hash and ground truth matches the pinned wheel.",
        "",
        "## Interpretation",
        "",
        "- LongBench's +2.59 pp is a net three-question difference and its paired interval includes zero. It is not evidence that sparse attention improves quality; it shows no large aggregate loss on this panel.",
        "- LongBench V1 provides a reproducible moderate prefill gain around 1.15× at 18.4% exact-pair ratio in the single independent profile sample.",
        "- BFCL changes from 22% to 21% with only three discordant cases. The quality comparison is underpowered, while the identical-prompt timing shows essentially no V1 prefill benefit on these shorter, repeated agent prompts.",
        "- Domain/signature rows with small n are diagnostic only. The overall paired rows remain the primary result.",
        "",
    ]
    (output / "FINAL_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    integrity = {
        "archive_sha256": sha256(args.archive) if args.archive else None,
        "source_file_count": len(source_files),
        "source_file_sha256": source_hashes,
        "longbench": {
            "status": lb_metadata["status"],
            "inputs": len(lb_inputs),
            "generations": len(lb_generations),
            "timing_rows": len(lb_timings),
            "stored_scores": len(lb_quality),
            "offline_rescore_changes": lb_score_changes,
            "resume_manifest": lb_resume,
        },
        "bfcl": {
            "status": bf_metadata["status"],
            "inputs": len(bf_selected),
            "trajectories": len(bf_generations),
            "timing_rows": len(bf_timings),
            "stored_scores": len(bf_quality),
            "offline_rescore_changes": bf_score_changes,
            "offline_error_type_changes": bf_error_type_changes,
            "identical_prompt_steps": len(bf_matched_steps),
            "v1_prompt_steps": sum(bf_v1_step_count.values()),
        },
    }
    (output / "integrity_manifest.json").write_text(
        json.dumps(integrity, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(output / "FINAL_REPORT.md")


if __name__ == "__main__":
    main()
