"""Deterministic synthetic, LongBench, and pinned RULER inputs."""

import hashlib
import json
from pathlib import Path
import random
import zipfile

import numpy as np
from huggingface_hub import hf_hub_download


LONGBENCH_REPO = "zai-org/LongBench"
LONGBENCH_CODE_COMMIT = "2e00731f8d0bff23dc4325161044d0ed8af94c1e"
LONGBENCH_V2_REPO = "THUDM/LongBench-v2"
LONGBENCH_V2_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
LONGBENCH_V2_DATA_SHA256 = "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"
RULER_REPO = "aldjalkdf/ruler"
RULER_REVISION = "2a9d66ecfcdbcaa72d692b6e89d1fb3325e7d634"
RULER_FLASH_PREFILL_COMMIT = "baa612047433a992a00d07dc178205eed065ae14"
RULER_TASK_SPECS = {
    # FlashPrefill ruler/configs/ruler_32k.yaml (baa6120): dataset and generation_max_length.
    "ruler_niah_mk_1": {"dataset": "niah_multikey_1", "generation_max_tokens": 50, "template": "niah"},
    "ruler_niah_mk_2": {"dataset": "niah_multikey_2", "generation_max_tokens": 50, "template": "niah"},
    "ruler_niah_mk_3": {"dataset": "niah_multikey_3", "generation_max_tokens": 100, "template": "niah"},
    "ruler_niah_mq": {"dataset": "niah_multiquery", "generation_max_tokens": 100, "template": "niah_plural"},
    "ruler_niah_mv": {"dataset": "niah_multivalue", "generation_max_tokens": 50, "template": "niah_plural"},
    "ruler_niah_s_1": {"dataset": "niah_single_1", "generation_max_tokens": 50, "template": "niah"},
    "ruler_niah_s_2": {"dataset": "niah_single_2", "generation_max_tokens": 50, "template": "niah"},
    "ruler_niah_s_3": {"dataset": "niah_single_3", "generation_max_tokens": 50, "template": "niah"},
    "ruler_cwe": {"dataset": "cwe", "generation_max_tokens": 100, "template": "cwe"},
    "ruler_fwe": {"dataset": "fwe", "generation_max_tokens": 50, "template": "fwe"},
    "ruler_vt": {"dataset": "vt", "generation_max_tokens": 50, "template": "vt"},
    "ruler_qa_1": {"dataset": "qa_1", "generation_max_tokens": 50, "template": "qa"},
    "ruler_qa_2": {"dataset": "qa_2", "generation_max_tokens": 50, "template": "qa"},
}
# FlashPrefill ruler/data.py load_ruler (baa6120): (user_template, system_template).
RULER_TEMPLATES = {
    "niah_plural": (
        "Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic {type_needle_v} for {query} mentioned in the provided text?",
        "The special magic {type_needle_v} for {query} mentioned in the provided text are",
    ),
    "niah": (
        "A special magic {type_needle_v} is hidden within the following text. Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat is the special magic {type_needle_v} for {query} mentioned in the provided text?",
        "The special magic {type_needle_v} for {query} mentioned in the provided text is",
    ),
    "vt": (
        "{example}Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\nQuestion: Find all variables that are assigned the value {query} in the text above.",
        "Answer: According to the chain(s) of variable assignment in the text above, {num_v} variables are assigned the value {query}, they are:",
    ),
    "cwe": (
        "{example}Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.\n{context}\nQuestion: What are the 10 most common words in the above list?",
        "Answer: The top 10 words that appear most often in the list are:",
    ),
    "fwe": (
        "Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words.\n{context}\nQuestion: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text?",
        "Answer: According to the coded text above, the three most frequently appeared words are:",
    ),
    "qa": (
        "Answer the question based on the given documents. Only give me the answer and do not output any other words.\n\nThe following are given documents.\n\n{context}\n\nAnswer the question based on the given documents. Only give me the answer and do not output any other words.\n\nQuestion: {question}",
        "Answer:",
    ),
}
RULER_TASKS = tuple(RULER_TASK_SPECS)
HOTPOT_PROMPT = (
    "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
    "The following are given passages.\n{context}\n\n"
    "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\n"
    "Question: {input}\nAnswer:"
)
LONGBENCH_V2_PROMPT = """Please read the following text and answer the question below.

<text>
$DOC$
</text>

What is the correct answer to this question: $Q$
Choices:
(A) $C_A$
(B) $C_B$
(C) $C_C$
(D) $C_D$

Format your response as follows: \"The correct answer is (insert answer here)\".
"""


def input_hash(token_ids):
    return hashlib.sha256(np.asarray(token_ids, dtype=np.int32).tobytes()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _ruler_prompt(spec, row, context):
    """Upstream process_example + prompt_template.format(**example)."""
    fields = {
        **row,
        "context": context,
        "question": row["query"] if "query" in row else row.get("question", ""),
        "example": row["example"] + "\n\n" if row.get("example") else "",
    }
    user, system = RULER_TEMPLATES[spec["template"]]
    prefix = system.format(**fields)
    return user.format(**fields) + "\n" + prefix, prefix


def _tokenize_ruler_prompt(tokenizer, spec, row, prompt_budget):
    context = row["context"]
    prompt, prefix = _ruler_prompt(spec, row, context)
    token_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    original_tokens = len(token_ids)
    original_context_chars = len(context)
    if len(token_ids) > prompt_budget:
        truncate_length = len(token_ids) - prompt_budget
        encoded_context = tokenizer(
            context,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        cut = encoded_context["offset_mapping"][-truncate_length][0]
        context = context[:cut]
        prompt, prefix = _ruler_prompt(spec, row, context)
        token_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    if len(token_ids) > prompt_budget:
        raise ValueError(
            f"official RULER truncation left {len(token_ids)} tokens for a {prompt_budget}-token budget"
        )
    return {
        "prompt": prompt,
        "prefix": prefix,
        "token_ids": token_ids,
        "original_tokens": original_tokens,
        "context_chars_removed": original_context_chars - len(context),
    }


def prepare_ruler(tokenizer, tasks, split, seed, total_budgets, samples, max_new_tokens):
    if split != "ruler":
        raise ValueError("official RULER tasks require --split ruler")
    if samples < 1:
        raise ValueError("ruler sample count must be positive")

    inputs = []
    files = {}
    for total_budget in total_budgets:
        if total_budget not in (4096, 8192, 16384, 32768):
            raise ValueError("RULER parity supports the official 4K/8K/16K/32K files")
        for task in tasks:
            spec = RULER_TASK_SPECS[task]
            generation_max = spec["generation_max_tokens"]
            if max_new_tokens < generation_max:
                raise ValueError(
                    f"{task} requires at least --max-new-tokens {generation_max} for official parity"
                )
            filename = f"{spec['dataset']}/validation_{total_budget}.jsonl"
            path = Path(hf_hub_download(
                RULER_REPO,
                filename,
                repo_type="dataset",
                revision=RULER_REVISION,
            ))
            source_rows = []
            with path.open("rb") as source:
                for source_row_index, raw_line in enumerate(source):
                    if raw_line.strip():
                        source_rows.append((
                            source_row_index,
                            hashlib.sha256(raw_line).hexdigest(),
                            json.loads(raw_line),
                        ))
            if samples > len(source_rows):
                raise ValueError(
                    f"requested {samples} {task} samples, but {filename} contains {len(source_rows)}"
                )
            selected = np.random.default_rng(seed).permutation(len(source_rows))[:samples]
            source_file_sha256 = file_hash(path)
            files[filename] = {
                "sha256": source_file_sha256,
                "rows": len(source_rows),
                "selected_source_rows": [int(source_rows[index][0]) for index in selected],
            }
            prompt_budget = total_budget - generation_max
            for shuffle_rank, selected_index in enumerate(selected):
                source_row_index, source_row_sha256, row = source_rows[int(selected_index)]
                prepared = _tokenize_ruler_prompt(tokenizer, spec, row, prompt_budget)
                token_ids = prepared["token_ids"]
                answers = [str(answer) for answer in (row["answer"] if "answer" in row else row["outputs"])]
                source_id = f"{task}:{total_budget}:{source_row_index}"
                inputs.append({
                    "task": "ruler",
                    "task_label": f"RULER/{spec['dataset']}",
                    "variant": task,
                    "split": split,
                    "source_split": "validation",
                    "sample_id": source_id,
                    "source_id": source_id,
                    "sample_index": shuffle_rank,
                    "seed": seed,
                    "total_context_budget": total_budget,
                    "prompt_budget": prompt_budget,
                    "actual_tokens": len(token_ids),
                    "question": str(row["query"] if "query" in row else row["question"]),
                    "answers": answers,
                    "expected_format": prepared["prefix"] + " " + ", ".join(answers),
                    "scorer_prefix": prepared["prefix"],
                    "prompt_text": prepared["prompt"],
                    "prompt_sha256": hashlib.sha256(
                        prepared["prompt"].encode("utf-8")
                    ).hexdigest(),
                    "source_repository": RULER_REPO,
                    "source_revision": RULER_REVISION,
                    "source_file": filename,
                    "source_file_sha256": source_file_sha256,
                    "source_row_index": source_row_index,
                    "source_example_index": row.get("index"),
                    "source_row_sha256": source_row_sha256,
                    "source_declared_length": row.get("length"),
                    "official_shuffle_rank": shuffle_rank,
                    "original_prompt_tokens": prepared["original_tokens"],
                    "context_chars_removed": prepared["context_chars_removed"],
                    "max_new_tokens": generation_max,
                    "input_sha256": input_hash(token_ids),
                    "input_ids": token_ids,
                })
    revisions = {
        "repository": RULER_REPO,
        "revision": RULER_REVISION,
        "flashprefill_repository_commit": RULER_FLASH_PREFILL_COMMIT,
        "selection": "numpy.default_rng(seed).permutation; mirrors datasets.Dataset.shuffle(seed)",
        "prompt": "FlashPrefill ruler/data.py load_ruler templates; use_chat_template=false",
        "files": files,
    }
    return inputs, revisions


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


def _longbench_v2_prompt(row):
    replacements = {
        "$DOC$": row["context"].strip(),
        "$Q$": row["question"].strip(),
        "$C_A$": row["choice_A"].strip(),
        "$C_B$": row["choice_B"].strip(),
        "$C_C$": row["choice_C"].strip(),
        "$C_D$": row["choice_D"].strip(),
    }
    prompt = LONGBENCH_V2_PROMPT
    for marker, value in replacements.items():
        prompt = prompt.replace(marker, value)
    return prompt


def _balanced_longbench_v2_candidates(candidates, samples):
    by_domain = {}
    for order, row, prepared in candidates:
        by_domain.setdefault(str(row["domain"]), []).append((order, row, prepared))
    for rows in by_domain.values():
        rows.sort(key=lambda item: item[0])
    selected = []
    position = 0
    domains = sorted(by_domain)
    while len(selected) < samples:
        added = False
        for domain in domains:
            rows = by_domain[domain]
            if position < len(rows):
                selected.append(rows[position])
                added = True
                if len(selected) == samples:
                    break
        if not added:
            break
        position += 1
    return selected


def prepare_longbench_v2(
    tokenizer,
    split,
    seed,
    samples,
    max_new_tokens,
    data_file,
    min_tokens=16384,
    native_limit=32768,
):
    if split != "modern":
        raise ValueError("LongBench v2 requires --split modern")
    path = Path(data_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"LongBench v2 data not found at {path}; run prepare_modern_benchmarks.sh download"
        )
    source_sha256 = file_hash(path)
    if source_sha256 != LONGBENCH_V2_DATA_SHA256:
        raise ValueError(
            f"LongBench v2 data hash mismatch: expected {LONGBENCH_V2_DATA_SHA256}, "
            f"got {source_sha256}"
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    task_max_new = min(max_new_tokens, 128)
    max_prompt_tokens = native_limit - task_max_new
    candidates = []
    excluded_non_short = 0
    excluded_too_short = 0
    excluded_too_long = 0
    for source_row_index, row in enumerate(rows):
        if str(row["length"]).lower() != "short":
            excluded_non_short += 1
            continue
        prompt = _longbench_v2_prompt(row)
        rendered, encoded = _chat_encoding(tokenizer, prompt)
        token_ids = encoded["input_ids"]
        if len(token_ids) < min_tokens:
            excluded_too_short += 1
            continue
        if len(token_ids) > max_prompt_tokens:
            excluded_too_long += 1
            continue
        source_id = str(row["_id"])
        order = hashlib.sha256(f"longbench-v2:{seed}:{source_id}".encode()).hexdigest()
        source_row_sha256 = hashlib.sha256(
            json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        candidates.append((order, row, {
            "source_row_index": source_row_index,
            "source_row_sha256": source_row_sha256,
            "prompt": prompt,
            "rendered": rendered,
            "token_ids": token_ids,
        }))
    selected = _balanced_longbench_v2_candidates(candidates, samples)
    if len(selected) != samples:
        raise ValueError(
            f"requested {samples} LongBench v2 samples but only {len(candidates)} fit "
            f"[{min_tokens}, {max_prompt_tokens}] prompt tokens"
        )

    inputs = []
    for sample_index, (_, row, prepared) in enumerate(selected):
        token_ids = prepared["token_ids"]
        source_id = str(row["_id"])
        inputs.append({
            "task": "longbench_v2",
            "task_label": "LongBench v2 native-32K subset",
            "variant": "official_0shot",
            "split": split,
            "sample_id": f"longbench-v2-{source_id}",
            "source_id": source_id,
            "sample_index": sample_index,
            "seed": seed,
            "domain": str(row["domain"]),
            "sub_domain": str(row["sub_domain"]),
            "difficulty": str(row["difficulty"]),
            "source_length_category": str(row["length"]),
            "total_context_budget": native_limit,
            "prompt_budget": max_prompt_tokens,
            "length_label": f"{min_tokens // 1024}K-native32K",
            "actual_tokens": len(token_ids),
            "question": row["question"],
            "answers": [str(row["answer"])],
            "choices": {letter: row[f"choice_{letter}"] for letter in "ABCD"},
            "expected_format": "The correct answer is (A|B|C|D)",
            "prompt_text": prepared["prompt"],
            "rendered_prompt_sha256": hashlib.sha256(
                prepared["rendered"].encode("utf-8")
            ).hexdigest(),
            "source_repository": LONGBENCH_V2_REPO,
            "source_revision": LONGBENCH_V2_REVISION,
            "source_file": str(path),
            "source_file_sha256": source_sha256,
            "source_row_index": prepared["source_row_index"],
            "source_row_sha256": prepared["source_row_sha256"],
            "max_new_tokens": task_max_new,
            "input_sha256": input_hash(token_ids),
            "input_ids": token_ids,
        })
    revisions = {
        "repository": LONGBENCH_V2_REPO,
        "revision": LONGBENCH_V2_REVISION,
        "data_file_sha256": source_sha256,
        "code_commit": LONGBENCH_CODE_COMMIT,
        "prompt": "prompts/0shot.txt",
        "scorer": "pred.py extract_answer exact A/B/C/D accuracy",
        "decoding": "project-standard greedy paired decoding; official script uses temperature=0.1",
        "selection": (
            "pre-filter official Short (<32K words) rows; tokenize the full official prompt "
            "with the model chat template; exclude overflow without truncation; deterministic "
            "hash order round-robin across domain"
        ),
        "source_rows": len(rows),
        "eligible_rows": len(candidates),
        "excluded_non_short": excluded_non_short,
        "excluded_too_short": excluded_too_short,
        "excluded_too_long": excluded_too_long,
        "selected_ids": [sample["source_id"] for sample in inputs],
    }
    return inputs, revisions


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
    ruler_samples,
    max_new_tokens,
    longbench_v2_samples=None,
    longbench_v2_file=Path("datasets/longbench_v2/data.json"),
    longbench_v2_min_tokens=16384,
):
    inputs = []
    revisions = {
        "longbench_code_commit": LONGBENCH_CODE_COMMIT,
        "longbench_archive_revision": None,
        "ruler": None,
        "longbench_v2": None,
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
    if "longbench_v2" in tasks:
        longbench_v2, longbench_v2_revision = prepare_longbench_v2(
            tokenizer,
            split,
            seed,
            longbench_v2_samples or samples,
            max_new_tokens,
            longbench_v2_file,
            longbench_v2_min_tokens,
        )
        inputs.extend(longbench_v2)
        revisions["longbench_v2"] = longbench_v2_revision
    ruler_tasks = [task for task in tasks if task in RULER_TASKS]
    if ruler_tasks:
        ruler, ruler_revision = prepare_ruler(
            tokenizer,
            ruler_tasks,
            split,
            seed,
            total_budgets,
            ruler_samples or samples,
            max_new_tokens,
        )
        inputs.extend(ruler)
        revisions["ruler"] = ruler_revision
    return inputs, revisions
