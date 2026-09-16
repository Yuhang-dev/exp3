"""Deterministic synthetic retrieval and complete-input LongBench hotpotqa subsets."""

import hashlib
import json
from pathlib import Path
import random
import zipfile

import numpy as np
from huggingface_hub import hf_hub_download


LONGBENCH_REPO = "zai-org/LongBench"
LONGBENCH_CODE_COMMIT = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
HOTPOT_PROMPT = (
    "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
    "The following are given passages.\n{context}\n\n"
    "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
    "Question: {input}\nAnswer:"
)


def input_hash(token_ids):
    return hashlib.sha256(np.asarray(token_ids, dtype=np.int32).tobytes()).hexdigest()


def _chat_encoding(tokenizer, content, offsets=False):
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=offsets,
    )
    return rendered, encoded


def _token_span(offset_mapping, char_start, char_end):
    touched = [
        index
        for index, (left, right) in enumerate(offset_mapping)
        if right > char_start and left < char_end
    ]
    if not touched:
        raise ValueError("target evidence did not map to any input token")
    return [touched[0], touched[-1] + 1]


def _split_seed(seed, split):
    offsets = {"quick": 11_000_003, "calibration": 23_000_033, "holdout": 47_000_059}
    return seed + offsets[split]


def _synthetic_content(filler, targets, variant, filler_count, extra_words):
    fractions = (0.08, 0.50, 0.86)
    slots = {}
    for target_index, fraction in enumerate(fractions[:len(targets)]):
        slots.setdefault(round(filler_count * fraction), []).append(target_index)
    records = []
    target_lines = []
    for position in range(filler_count + 1):
        for target_index in slots.get(position, []):
            key, value = targets[target_index]
            line = f"Record {key}: {value}."
            records.append(line)
            target_lines.append(line)
        if position < filler_count:
            records.append(filler[position])
    notes = " ".join(extra_words)
    if variant == "multi_key":
        question = (
            "Return the value for each requested key in exactly this format: "
            + "; ".join(f"{key}=VALUE" for key, _ in targets)
            + ". Requested keys: "
            + ", ".join(key for key, _ in targets)
            + "."
        )
    else:
        questions = " ".join(
            f"Query {index + 1}: What is the value of {key}?"
            for index, (key, _) in enumerate(targets)
        )
        question = (
            questions
            + " Return all answers in exactly this format: "
            + "; ".join(f"{key}=VALUE" for key, _ in targets)
            + "."
        )
    content = (
        "Use only the key-value records below. Ignore unrelated records. Do not explain your answer.\n\n"
        "Records:\n"
        + "\n".join(records)
    )
    if notes:
        content += "\nIrrelevant notes: " + notes
    content += "\n\n" + question
    return content, target_lines, question


def make_synthetic_sample(tokenizer, split, sample_index, seed, prompt_budget, total_budget, variant):
    sample_seed = _split_seed(seed, split) + sample_index * 104_729 + total_budget * 17
    rng = random.Random(sample_seed)
    namespace = hashlib.sha256(
        f"{seed}:{split}:{total_budget}:{sample_index}:{variant}".encode()
    ).hexdigest()[:10].upper()
    targets = [
        (f"KEY{namespace}{index}", f"VAL{namespace}{index}{rng.getrandbits(32):08X}")
        for index in range(3)
    ]
    filler = [
        f"Record FKEY{namespace}{index:05d}: FVAL{namespace}{index:05d}{rng.getrandbits(24):06X}."
        for index in range(8192)
    ]
    note_words = [
        f"note{rng.getrandbits(20):05x}"
        for _ in range(2048)
    ]

    def length_for(filler_count, word_count=0):
        content, _, _ = _synthetic_content(
            filler,
            targets,
            variant,
            filler_count,
            note_words[:word_count],
        )
        _, encoded = _chat_encoding(tokenizer, content)
        return len(encoded["input_ids"])

    if length_for(0) > prompt_budget:
        raise ValueError(f"prompt budget {prompt_budget} is too small for the synthetic task")
    low, high = 0, 64
    while high < len(filler) and length_for(high) <= prompt_budget:
        low, high = high, min(high * 2, len(filler))
    if high == len(filler) and length_for(high) <= prompt_budget:
        raise ValueError("synthetic filler pool is too small for the requested prompt budget")
    while low + 1 < high:
        middle = (low + high) // 2
        if length_for(middle) <= prompt_budget:
            low = middle
        else:
            high = middle

    words_low, words_high = 0, 64
    while words_high < len(note_words) and length_for(low, words_high) <= prompt_budget:
        words_low, words_high = words_high, min(words_high * 2, len(note_words))
    while words_low + 1 < words_high:
        middle = (words_low + words_high) // 2
        if length_for(low, middle) <= prompt_budget:
            words_low = middle
        else:
            words_high = middle

    content, target_lines, question = _synthetic_content(
        filler,
        targets,
        variant,
        low,
        note_words[:words_low],
    )
    rendered, encoded = _chat_encoding(tokenizer, content, offsets=True)
    token_ids = encoded["input_ids"]
    evidence_positions = []
    for line in target_lines:
        char_start = rendered.index(line)
        evidence_positions.append({
            "record": line,
            "char_span": [char_start, char_start + len(line)],
            "token_span": _token_span(encoded["offset_mapping"], char_start, char_start + len(line)),
        })
    sample_id = f"synthetic-{split}-{total_budget}-{sample_index}-{variant}"
    return {
        "task": "synthetic_kv_retrieval",
        "task_label": f"synthetic_kv_retrieval/{variant}",
        "variant": variant,
        "split": split,
        "sample_id": sample_id,
        "source_id": sample_id,
        "sample_index": sample_index,
        "seed": sample_seed,
        "total_context_budget": total_budget,
        "prompt_budget": prompt_budget,
        "actual_tokens": len(token_ids),
        "question": question,
        "answers": {key: value for key, value in targets},
        "target_keys": [key for key, _ in targets],
        "evidence_positions": evidence_positions,
        "expected_format": "; ".join(f"{key}=VALUE" for key, _ in targets),
        "max_new_tokens": min(128, total_budget - len(token_ids)),
        "input_sha256": input_hash(token_ids),
        "input_ids": token_ids,
    }


def prepare_synthetic(
    tokenizer,
    split,
    seed,
    total_budgets,
    prompt_budgets,
    samples,
    max_new_tokens,
):
    inputs = []
    if prompt_budgets:
        budgets = [(budget + min(max_new_tokens, 128), budget) for budget in prompt_budgets]
    else:
        budgets = [
            (total, total - min(max_new_tokens, 128))
            for total in total_budgets
        ]
    for total_budget, prompt_budget in budgets:
        if prompt_budget <= 0 or total_budget > 32768:
            raise ValueError("synthetic prompt plus generation must remain within the native 32768-token range")
        for variant_index, variant in enumerate(("multi_key", "multi_query")):
            for within_variant in range(samples):
                sample_index = variant_index * samples + within_variant
                sample = make_synthetic_sample(
                    tokenizer,
                    split,
                    sample_index,
                    seed,
                    prompt_budget,
                    total_budget,
                    variant,
                )
                sample["max_new_tokens"] = min(max_new_tokens, 128)
                if sample["actual_tokens"] + sample["max_new_tokens"] > total_budget:
                    raise AssertionError("synthetic construction exceeded its total context budget")
                inputs.append(sample)
    return inputs


def _hotpot_split(source_id, seed):
    digest = hashlib.sha256(f"{seed}:{source_id}".encode()).digest()
    return "calibration" if digest[0] % 4 == 0 else "holdout"


def _archive_revision(path):
    parts = Path(path).parts
    if "snapshots" in parts:
        position = parts.index("snapshots")
        return parts[position + 1]
    return Path(path).parent.name


def prepare_hotpotqa(tokenizer, split, seed, samples, max_new_tokens, native_limit=32768):
    if split == "quick":
        selected_split = "calibration"
    else:
        selected_split = split
    archive = hf_hub_download(LONGBENCH_REPO, "data.zip", repo_type="dataset")
    with zipfile.ZipFile(archive) as zipped:
        rows = [json.loads(line) for line in zipped.read("data/hotpotqa.jsonl").splitlines()]
    candidates = []
    for row in rows:
        source_id = str(row.get("_id", row.get("id")))
        if _hotpot_split(source_id, seed) != selected_split:
            continue
        order = hashlib.sha256(f"order:{seed}:{source_id}".encode()).hexdigest()
        candidates.append((order, source_id, row))
    candidates.sort(key=lambda item: item[0])

    task_max_new = min(max_new_tokens, 32)
    inputs = []
    for _, source_id, row in candidates:
        content = HOTPOT_PROMPT.format(context=row["context"], input=row["input"])
        _, encoded = _chat_encoding(tokenizer, content)
        token_ids = encoded["input_ids"]
        if len(token_ids) + task_max_new > native_limit:
            continue
        sample_index = len(inputs)
        inputs.append({
            "task": "hotpotqa",
            "task_label": "LongBench hotpotqa subset",
            "variant": "official_prompt",
            "split": split,
            "sample_id": f"hotpotqa-{split}-{source_id}",
            "source_id": source_id,
            "sample_index": sample_index,
            "seed": seed,
            "total_context_budget": native_limit,
            "prompt_budget": native_limit - task_max_new,
            "actual_tokens": len(token_ids),
            "question": row["input"],
            "answers": row["answers"],
            "all_classes": row.get("all_classes", []),
            "source_length": row.get("length"),
            "expected_format": "concise answer only",
            "max_new_tokens": task_max_new,
            "input_sha256": input_hash(token_ids),
            "input_ids": token_ids,
        })
        if len(inputs) == samples:
            break
    if len(inputs) != samples:
        raise ValueError(
            f"only {len(inputs)} complete hotpotqa prompts fit the native context budget for split {split}"
        )
    return inputs, _archive_revision(archive)


def prepare_inputs(
    tokenizer,
    tasks,
    split,
    seed,
    total_budgets,
    prompt_budgets,
    samples,
    synthetic_samples,
    hotpot_samples,
    max_new_tokens,
):
    inputs = []
    revisions = {
        "longbench_code_commit": LONGBENCH_CODE_COMMIT,
        "longbench_archive_revision": None,
    }
    if "synthetic_kv_retrieval" in tasks:
        inputs.extend(prepare_synthetic(
            tokenizer,
            split,
            seed,
            total_budgets,
            prompt_budgets,
            synthetic_samples or samples,
            max_new_tokens,
        ))
    if "hotpotqa" in tasks:
        hotpot, revision = prepare_hotpotqa(
            tokenizer,
            split,
            seed,
            hotpot_samples or samples,
            max_new_tokens,
        )
        inputs.extend(hotpot)
        revisions["longbench_archive_revision"] = revision
    return inputs, revisions
