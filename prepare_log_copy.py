"""Build exact-copy prompts from real HDFS logs (logpai/loghub): return each requested line's block id."""

import argparse
from pathlib import Path
import random
import re

import torch
from transformers import AutoTokenizer

from data import _chat_encoding, _token_span, input_hash


FRACTIONS = (0.08, 0.50, 0.86)
BLOCK = re.compile(r"blk_-?\d+")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--prompt-budget", type=int, default=32640)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def content_for(lines, first, count):
    window = [(f"L{index:05d}", lines[index]) for index in range(first, first + count)]
    keys = []
    for fraction in FRACTIONS:
        position = round((count - 1) * fraction)
        while not BLOCK.search(window[position][1]):
            position += 1
        keys.append(window[position])
    question = (
        "Return the block id of each requested line in exactly this format: "
        + "; ".join(f"{key}=VALUE" for key, _ in keys)
        + ". Requested lines: "
        + ", ".join(key for key, _ in keys)
        + "."
    )
    records = "\n".join(f"{key} {text}" for key, text in window)
    content = (
        "Use only the log lines below. Each block id starts with blk_. Do not explain your answer.\n\n"
        f"Log:\n{records}\n\n{question}"
    )
    return content, keys, question


def main():
    args = arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    lines = args.log.read_text(encoding="utf-8").splitlines()
    rng = random.Random(args.seed)
    output = []

    def tokens(first, count):
        content = content_for(lines, first, count)[0]
        return len(_chat_encoding(tokenizer, content)[1]["input_ids"])

    low, high = 16, len(lines)
    while low + 1 < high:
        middle = (low + high) // 2
        if tokens(0, middle) <= args.prompt_budget:
            low = middle
        else:
            high = middle
    typical = low
    for sample_index in range(args.samples):
        first = rng.randrange(0, len(lines) - typical - 64)
        low, high = 16, typical + 64
        while low + 1 < high:
            middle = (low + high) // 2
            if tokens(first, middle) <= args.prompt_budget:
                low = middle
            else:
                high = middle
        content, keys, question = content_for(lines, first, low)
        rendered, encoded = _chat_encoding(tokenizer, content, offsets=True)
        token_ids = encoded["input_ids"]
        evidence = []
        for key, text in keys:
            record = f"{key} {text}"
            start = rendered.index(record)
            evidence.append({
                "record": record,
                "char_span": [start, start + len(record)],
                "token_span": _token_span(encoded["offset_mapping"], start, start + len(record)),
            })
        identifier = f"log-hdfs-{sample_index}"
        output.append({
            "task": "synthetic_kv_retrieval",
            "task_label": "log_copy/hdfs",
            "variant": "hdfs",
            "split": "log",
            "sample_id": identifier,
            "source_id": identifier,
            "sample_index": sample_index,
            "seed": args.seed,
            "total_context_budget": args.prompt_budget + 128,
            "prompt_budget": args.prompt_budget,
            "actual_tokens": len(token_ids),
            "question": question,
            "answers": {key: BLOCK.search(text).group(0) for key, text in keys},
            "target_keys": [key for key, _ in keys],
            "evidence_positions": evidence,
            "expected_format": "; ".join(f"{key}=VALUE" for key, _ in keys),
            "max_new_tokens": 128,
            "input_sha256": input_hash(token_ids),
            "input_ids": token_ids,
        })
        print(identifier, "lines", low, "first", first, "tokens", len(token_ids), output[-1]["answers"], flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.out)


if __name__ == "__main__":
    main()
