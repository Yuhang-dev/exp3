"""Build counterfactual synthetic prompts from saved calibration inputs, for evaluate.py --inputs.

Each factor separates one explanation of the sparse copying failures:
filler_index_7   does the index collapse follow the filler prior (0 -> 7)?
filler_other_ns  does the failure need fillers sharing the target namespace?
order_cyclic     does the collapse follow the first emitted line rather than the filler prior?
position_swap    index or position: key 0 at 86% and key 2 at 8%.
shift_K          same content, every needle moved K tokens against the block grid.
"""

import argparse
from pathlib import Path
import random
import re

import torch
from transformers import AutoTokenizer

from data import _token_span, input_hash


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--shifts", type=int, nargs="+", default=[16, 32, 48, 64, 80, 96, 112])
    return parser.parse_args()


def namespace_of(sample):
    return sample["target_keys"][0][3:-1]


def from_text(base, tokenizer, factor, rendered, question=None):
    question = question or base["question"]
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    evidence = []
    for item in base["evidence_positions"]:
        start = rendered.index(item["record"])
        end = start + len(item["record"])
        evidence.append({
            "record": item["record"],
            "char_span": [start, end],
            "token_span": _token_span(encoded["offset_mapping"], start, end),
        })
    return finish(base, factor, encoded["input_ids"], evidence, question)


def finish(base, factor, token_ids, evidence, question):
    identifier = f"{factor}-{base['sample_id']}"
    return {
        **base,
        "sample_id": identifier,
        "source_id": identifier,
        "base_sample_id": base["sample_id"],
        "factor": factor,
        "question": question,
        "evidence_positions": evidence,
        "actual_tokens": len(token_ids),
        "input_sha256": input_hash(token_ids),
        "input_ids": token_ids,
    }


def filler_index_7(base, tokenizer, rendered):
    pattern = rf"(F(?:KEY|VAL){namespace_of(base)})0(\d{{4}})"
    return from_text(base, tokenizer, "filler_index_7", re.sub(pattern, r"\g<1>7\2", rendered))


def filler_other_ns(base, tokenizer, rendered):
    namespace = namespace_of(base)
    # Keep the digit/letter pattern: digits tokenize one by one, so a free namespace changes the length.
    rotate = str.maketrans("0123456789ABCDEF", "3456789012DEFABC")
    other = namespace.translate(rotate)
    changed = rendered.replace(f"FKEY{namespace}", f"FKEY{other}").replace(f"FVAL{namespace}", f"FVAL{other}")
    return from_text(base, tokenizer, "filler_other_ns", changed)


def order_cyclic(base, tokenizer, rendered):
    keys = base["target_keys"]
    mapping = {keys[0]: keys[1], keys[1]: keys[2], keys[2]: keys[0]}
    question = re.sub("|".join(keys), lambda match: mapping[match.group(0)], base["question"])
    start = rendered.rindex(base["question"])
    changed = rendered[:start] + question + rendered[start + len(base["question"]):]
    return from_text(base, tokenizer, "order_cyclic", changed, question)


def position_swap(base, tokenizer, rendered):
    first = base["evidence_positions"][0]["record"]
    last = base["evidence_positions"][2]["record"]
    changed = rendered.replace(first, "\0").replace(last, first).replace("\0", last)
    return from_text(base, tokenizer, "position_swap", changed)


def shifted(base, tokenizer, rendered, shift):
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    assert encoded["input_ids"] == list(base["input_ids"])
    records_char = rendered.index("Records:\n")
    insert_at = next(index for index, (_, right) in enumerate(encoded["offset_mapping"]) if right > records_char)
    rng = random.Random(f"{base['sample_id']}:{shift}")
    padding = " ".join(f"note{rng.getrandbits(20):05x}" for _ in range(shift))
    padding_ids = tokenizer(padding, add_special_tokens=False)["input_ids"][:shift]
    token_ids = encoded["input_ids"][:insert_at] + padding_ids + encoded["input_ids"][insert_at:]
    evidence = [
        {**item, "token_span": [item["token_span"][0] + shift, item["token_span"][1] + shift]}
        for item in base["evidence_positions"]
    ]
    return finish(base, f"shift_{shift}", token_ids, evidence, base["question"])


def main():
    args = arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    inputs = torch.load(args.inputs, map_location="cpu", weights_only=False)
    output = []
    for base in inputs:
        rendered = tokenizer.decode(base["input_ids"], skip_special_tokens=False)
        output.append(finish(base, "baseline", list(base["input_ids"]), base["evidence_positions"], base["question"]))
        for build in (filler_index_7, filler_other_ns, order_cyclic, position_swap):
            output.append(build(base, tokenizer, rendered))
        output.extend(shifted(base, tokenizer, rendered, shift) for shift in args.shifts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.out)
    for sample in output[:12]:
        print(sample["sample_id"], sample["actual_tokens"], [item["token_span"][0] % 128 for item in sample["evidence_positions"]])
    print("prompts", len(output), "distinct lengths", len({sample["actual_tokens"] for sample in output}))


if __name__ == "__main__":
    main()
