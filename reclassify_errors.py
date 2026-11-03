"""Strict error classes: whole-value matches only, truncation/format checked before 'copied'."""
import collections, glob, json, re, sys, torch
from transformers import AutoTokenizer
from scoring import parse_synthetic
root = "results/pointer"; tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
samples = {}
for path in ["holdout/inputs.pt", "batch2_inputs/competitors.pt"]:
    for s in torch.load(f"{root}/{path}", weights_only=False): samples[s["sample_id"]] = s
VALUE = re.compile(r"F?VAL[0-9A-F]+")
def label(got, exp, s, values):
    ns = exp[3:-9]
    if got == exp: return "correct"
    if not got: return "missing"
    if "VAL" + got == exp: return "format_only"
    if exp.startswith(got) or exp.startswith("VAL" + got): return "truncated"
    full = got if got.startswith(("VAL", "FVAL")) else "VAL" + got
    if full in s["answers"].values(): return "copy_other_target"
    if full in values or "F" + full in values: return "copy_filler_same_ns" if ns in full else "copy_filler_other_ns"
    if full[-8:] == exp[-8:]: return "right_suffix_wrong_index"
    if full[:-8] == exp[:-8]: return "right_index_foreign_suffix" if any(v.endswith(full[-8:]) for v in values) else "right_index_wrong_suffix"
    return "other"
runs = {"holdout": "holdout", "quant": "quant", "small_1.5b": "small/Qwen2.5-1.5B-Instruct", "small_0.5b": "small/Qwen2.5-0.5B-Instruct",
        "temperature": "temperature", "competitors_7b": "competitors_7b", "competitors_0.5b": "competitors_0p5b"}
runs.update({f"weights_{p.split('/')[-1]}": p[len(root)+1:] for p in glob.glob(f"{root}/weights/*")})
cache = {}; table = collections.defaultdict(collections.Counter)
for run, folder in runs.items():
    for line in open(f"{root}/{folder}/generations.jsonl"):
        r = json.loads(line); s = samples.get(r["sample_id"])
        if s is None or s["task"] != "synthetic_kv_retrieval": continue
        if s["sample_id"] not in cache: cache[s["sample_id"]] = set(VALUE.findall(tok.decode(s["input_ids"])))
        parsed = parse_synthetic(r["raw_text"], s["target_keys"])
        config = r.get("config_id") or r.get("condition") or "dense"
        level = r["sample_id"].split("-synthetic")[0] if r["sample_id"].startswith("shared") else ""
        for key in s["target_keys"]:
            table[(run, config + ("|" + level if level else ""))][label(parsed.get(key, ""), s["answers"][key], s, cache[s["sample_id"]])] += 1
cols = ["correct", "format_only", "truncated", "missing", "copy_other_target", "copy_filler_same_ns", "copy_filler_other_ns",
        "right_suffix_wrong_index", "right_index_foreign_suffix", "right_index_wrong_suffix", "other"]
print("| run | config | " + " | ".join(cols) + " |"); print("|" + "---|" * (len(cols) + 2))
for (run, config), c in sorted(table.items()):
    print(f"| {run} | {config} | " + " | ".join(str(c[k]) for k in cols) + " |")
