"""C4-validation perplexity of a (compressed) model on windows disjoint from the Wanda calibration."""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from weight_compression import load_model, perplexity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--weight-format", choices=("bf16", "int8", "nf4", "nf4sim", "nf4sim_mlp", "rtn4", "rtn3"), default="bf16")
    parser.add_argument("--wanda", choices=("none", "unstructured", "2:4"), default="none")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    model = load_model(args.model, args.weight_format, args.wanda, args.calibration)
    value = perplexity(model, AutoTokenizer.from_pretrained(args.model), args.calibration)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "weight_format": args.weight_format, "wanda": args.wanda, "perplexity": value,
    }), encoding="utf-8")
    print(args.weight_format, args.wanda, "perplexity", value)


if __name__ == "__main__":
    main()
