"""Build exact-copy prompts from real text: log identifiers (logpai/loghub HDFS block ids, OpenStack request
ids) and SHA-256 digests from installed wheels' RECORD manifests, rendered as a sha256sum-style list.
Each prompt asks for three entries near 8%, 50%, and 86% of the context whose value is unique in the prompt;
neighbouring entries are the competitors."""

import argparse
import base64
from collections import Counter
from pathlib import Path
import random
import re

import torch
from transformers import AutoTokenizer

from data import _chat_encoding, _token_span, input_hash


FRACTIONS = (0.08, 0.50, 0.86)
SOURCES = {
    "hdfs": ("log_copy", r"blk_-?\d+", "Use only the log lines below. Each block id starts with blk_.",
             "Return the block id of each requested line", "Log"),
    "openstack": ("log_copy", r"req-[0-9a-f-]{36}", "Use only the log lines below. Each request id starts with req-.",
                  "Return the request id of each requested line", "Log"),
    "manifest": ("manifest_copy", r"[0-9a-f]{64}",
                 "Use only the checksum list below. Each line is a SHA-256 digest followed by a file path.",
                 "Return the SHA-256 digest of each requested file", "Checksums"),
    "manifest_pf": ("manifest_copy", r"[0-9a-f]{64}",
                    "Use only the checksum list below. Each line is a file path followed by its SHA-256 digest.",
                    "Return the SHA-256 digest of each requested file", "Checksums"),
}
MANIFEST_PACKAGES = ("torch", "transformers", "sympy", "pandas", "numpy")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--path", type=Path, required=True, help="log file, or site-packages for manifest")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--prompt-budget", type=int, default=32640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id-prefix", default="copy")
    return parser.parse_args()


def log_pool(path, pattern):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [(f"L{index:05d}", f"L{index:05d} {text}", (re.search(pattern, text) or [None])[0])
            for index, text in enumerate(lines)]


def manifest_pools(site_packages, path_first):
    pools = {}
    for package in MANIFEST_PACKAGES:
        (record,) = site_packages.glob(f"{package}-*.dist-info/RECORD")
        entries = []
        for line in record.read_text(encoding="utf-8").splitlines():
            path, digest, _ = line.rsplit(",", 2)
            if digest.startswith("sha256="):
                hexdigest = base64.urlsafe_b64decode(digest[7:] + "=").hex()
                line = f"{path}  {hexdigest}" if path_first else f"{hexdigest}  {path}"
                entries.append((path, line, hexdigest))
        pools[package] = entries
    return pools


def content_for(source, window):
    _, _, intro, request, header = SOURCES[source]
    counts = Counter(answer for _, _, answer in window)
    keys = []
    for fraction in FRACTIONS:
        position = round((len(window) - 1) * fraction)
        while window[position][2] is None or counts[window[position][2]] > 1:
            position += 1
        keys.append(window[position])
    question = (
        f"{request} in exactly this format: " + "; ".join(f"{key}=VALUE" for key, _, _ in keys)
        + ". Requested: " + ", ".join(key for key, _, _ in keys) + "."
    )
    records = "\n".join(line for _, line, _ in window)
    return f"{intro} Do not explain your answer.\n\n{header}:\n{records}\n\n{question}", keys, question


def main():
    args = arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    benchmark, pattern, _, _, _ = SOURCES[args.source]
    if args.source.startswith("manifest"):
        pools = manifest_pools(args.path, args.source == "manifest_pf")
    else:
        pools = {args.source: log_pool(args.path, pattern)}
    rng = random.Random(args.seed)

    def tokens(pool, first, count):
        return len(_chat_encoding(tokenizer, content_for(args.source, pool[first:first + count])[0])[1]["input_ids"])

    output = []
    for sample_index in range(args.samples):
        name = rng.choice(sorted(pools))
        pool = pools[name]
        first = rng.randrange(0, len(pool) // 3)
        low, high = 16, len(pool) - first
        while low + 1 < high:
            middle = (low + high) // 2
            if tokens(pool, first, middle) <= args.prompt_budget:
                low = middle
            else:
                high = middle
        window = pool[first:first + low]
        content, keys, question = content_for(args.source, window)
        rendered, encoded = _chat_encoding(tokenizer, content, offsets=True)
        token_ids = encoded["input_ids"]
        evidence = []
        for _, line, _ in keys:
            start = rendered.index(line)
            evidence.append({
                "record": line,
                "char_span": [start, start + len(line)],
                "token_span": _token_span(encoded["offset_mapping"], start, start + len(line)),
            })
        identifier = f"{args.id_prefix}-{args.source}-{sample_index}"
        output.append({
            "task": "synthetic_kv_retrieval",
            "task_label": f"{benchmark}/{args.source}",
            "variant": args.source,
            "split": "holdout",
            "sample_id": identifier,
            "source_id": f"{identifier}-{name}-{first}",
            "sample_index": sample_index,
            "seed": args.seed,
            "total_context_budget": args.prompt_budget + 256,
            "prompt_budget": args.prompt_budget,
            "actual_tokens": len(token_ids),
            "question": question,
            "answers": {key: answer for key, _, answer in keys},
            "target_keys": [key for key, _, _ in keys],
            "evidence_positions": evidence,
            "context_lines": [line for _, line, _ in window],
            "expected_format": "; ".join(f"{key}=VALUE" for key, _, _ in keys),
            "max_new_tokens": 256,
            "input_sha256": input_hash(token_ids),
            "input_ids": token_ids,
        })
        print(identifier, name, "entries", low, "first", first, "tokens", len(token_ids), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.out)


if __name__ == "__main__":
    main()
