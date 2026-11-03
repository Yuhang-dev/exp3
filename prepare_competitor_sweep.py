"""Vary the share of filler records that share the target namespace, at fixed prompts and length.

A level p keeps the namespace on a seeded fraction p of filler records and moves the rest to a
digit/letter-preserving rotated namespace; targets, positions, and the question are unchanged.
"""

import argparse
from pathlib import Path
import random
import re

import torch
from transformers import AutoTokenizer

from prepare_pointer_factors import from_text, namespace_of


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--levels", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    return parser.parse_args()


def shared_level(base, tokenizer, rendered, level):
    namespace = namespace_of(base)
    other = namespace.translate(str.maketrans("0123456789ABCDEF", "3456789012DEFABC"))
    rng = random.Random(f"{base['sample_id']}:{level}")

    def relabel(match):
        return match.group(0) if rng.random() < level else match.group(0).replace(namespace, other)

    pattern = rf"Record FKEY{namespace}\d{{5}}: FVAL{namespace}\d{{5}}[0-9A-F]{{6}}\."
    return from_text(base, tokenizer, f"shared_{level:g}", re.sub(pattern, relabel, rendered))


def main():
    args = arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    output = []
    for base in torch.load(args.inputs, map_location="cpu", weights_only=False):
        rendered = tokenizer.decode(base["input_ids"], skip_special_tokens=False)
        output.extend(shared_level(base, tokenizer, rendered, level) for level in args.levels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.out)
    lengths = {}
    for sample in output:
        lengths.setdefault(sample["factor"], []).append(sample["actual_tokens"])
    print("prompts", len(output), {level: (min(v), max(v)) for level, v in lengths.items()})


if __name__ == "__main__":
    main()
