"""Audit a completed Qwen3.5 CL-bench archive without CUDA or a judge API."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import random
import statistics as st
import struct
import zipfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


class PlainInputsUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("inputs.pt must contain only plain input records, without Python classes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    meta = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((run / "judge_manifest.json").read_text())
    public_inputs = read_rows(run / "inputs.jsonl")
    with zipfile.ZipFile(run / "inputs.pt") as bundle:
        name = next(n for n in bundle.namelist() if n.endswith("/data.pkl"))
        inputs = PlainInputsUnpickler(io.BytesIO(bundle.read(name))).load()
    generations = read_rows(run / "generations.jsonl")
    with (run / "timings.csv").open() as source:
        timings = list(csv.DictReader(source))
    with (run / "profile.csv").open() as source:
        profiles = list(csv.DictReader(source))
    methods = ["dense", "fp_v1__a0p08"]
    assert meta["status"] == "generated" and len(inputs) == len(public_inputs) == 100
    by_id = {r["sample_id"]: r for r in inputs}
    assert len(by_id) == 100
    assert meta["input_signatures"] == [[r["sample_id"], r["input_sha256"]] for r in inputs]
    for i, (row, public) in enumerate(zip(inputs, public_inputs)):
        assert row["sample_index"] == i and row["max_new_tokens"] == 32768
        assert len(row["input_ids"]) == row["prompt_tokens"]
        digest = hashlib.sha256(struct.pack(f"<{len(row['input_ids'])}i", *row["input_ids"])).hexdigest()
        assert digest == row["input_sha256"]
        assert canonical(row["messages"]) == row["messages_sha256"]
        assert canonical(row["rubrics"]) == row["rubrics_sha256"]
        assert all(public[k] == v for k, v in row.items() if k != "input_ids")
    expected = {(sid, method) for sid in by_id for method in methods}
    generated = {(r["sample_id"], r["config_id"]): r for r in generations}
    assert len(generations) == len(generated) == 200 and set(generated) == expected
    assert sha(run / "generations.jsonl") == manifest["generation_sha256"]
    for row in generations:
        source = by_id[row["sample_id"]]
        for key in ["sample_index", "prompt_tokens", "input_sha256", "messages_sha256", "rubrics_sha256", "source_row_sha256"]:
            assert row[key] == source[key]
    timing_groups = defaultdict(list)
    for row in timings:
        assert row["input_sha256"] == by_id[row["sample_id"]]["input_sha256"]
        timing_groups[(row["sample_id"], row["config_id"])].append(row)
    assert len(timings) == 600 and set(timing_groups) == expected
    assert all(sorted(int(r["repeat"]) for r in group) == [0, 1, 2] for group in timing_groups.values())
    for method in methods:
        path = run / "official_inputs" / f"{method}.jsonl"
        assert sha(path) == manifest["files"][method]["sha256"]
        official = read_rows(path)
        assert len(official) == 100 and {r["metadata"]["task_id"] for r in official} == set(by_id)
        for row in official:
            sid = row["metadata"]["task_id"]
            assert row["model_output"] == generated[(sid, method)]["final_text"]
            assert row["messages"] == by_id[sid]["messages"] and row["rubrics"] == by_id[sid]["rubrics"]
    medians = {key: st.median(float(r["prefill_ms"]) for r in rows) for key, rows in timing_groups.items()}
    paired = [{"sample_index": r["sample_index"], "sample_id": r["sample_id"],
               "prompt_tokens": r["prompt_tokens"], "category": r["context_category"],
               "dense_ms": medians[(r["sample_id"], methods[0])],
               "v1_ms": medians[(r["sample_id"], methods[1])],
               "speedup": medians[(r["sample_id"], methods[0])] / medians[(r["sample_id"], methods[1])]}
              for r in inputs]
    ratios = [r["speedup"] for r in paired]
    rng = random.Random(42)
    boots = sorted(st.median(rng.choices(ratios, k=100)) for _ in range(10000))
    summary = {}
    for method in methods:
        rows = [r for r in generations if r["config_id"] == method]
        summary[method] = {
            "outputs": len(rows), "eos": sum(r["end_reason"] == "eos" for r in rows),
            "cap_hits": sum(r["end_reason"] == "max_new_tokens" for r in rows),
            "unclosed_thinking": sum(r["thinking_status"] != "closed" for r in rows),
            "empty_final": sum(not r["final_text"].strip() for r in rows),
            "median_output_tokens": st.median(len(r["generated_token_ids"]) for r in rows),
            "max_output_tokens": max(len(r["generated_token_ids"]) for r in rows),
            "median_prefill_ms": st.median(medians[(sid, method)] for sid in by_id),
            "sum_generation_seconds": sum(r["generation_seconds"] for r in rows),
            "generation_peak_allocated_gib": max(r["generation_peak_allocated_gib"] for r in rows),
            "generation_peak_reserved_gib": max(r["generation_peak_reserved_gib"] for r in rows),
        }
    subgroups = []
    for label, lo, hi in [("<8K", 0, 8192), ("8K-16K", 8192, 16384),
                          ("16K-32K", 16384, 32768), (">=32K", 32768, 10**9)]:
        group = [r for r in paired if lo <= r["prompt_tokens"] < hi]
        subgroups.append({"length": label, "n": len(group), "median_speedup": st.median(r["speedup"] for r in group)})
    profile_summary = {}
    for method in methods:
        rows = [r for r in profiles if r["config_id"] == method]
        assert [int(r["layer"]) for r in rows] == [3, 7, 11, 15, 19, 23, 27, 31]
        assert len({r["sample_id"] for r in rows}) == 1
        profile_summary[method] = {
            "sample_id": rows[0]["sample_id"],
            "sum_attention_ms": sum(float(r["attention_ms"]) for r in rows),
            "median_exact_ratio": st.median(float(r["effective_exact_token_pair_ratio"]) for r in rows),
        }
    assert all(r["generated_token_ids"][-3:] == [248046, 198, 248044] for r in generations)
    assert all(r["generated_token_ids"].index(248046) == len(r["generated_token_ids"]) - 3 for r in generations)
    exceptions = [{k: r[k] for k in ["sample_index", "sample_id", "config_id", "end_reason", "thinking_status"]}
                  for r in generations if not r["final_text"].strip()]
    result = {
        "archive_sha256": sha(args.archive), "integrity": "PASS",
        "generated_at": meta["generated_at"], "quality_status": meta["quality_status"],
        "summary": summary, "subgroups": subgroups, "empty_answer_samples": exceptions,
        "median_paired_speedup": st.median(ratios),
        "descriptive_bootstrap_95ci_median_speedup": [boots[249], boots[9749]],
        "geometric_mean_speedup": math.exp(st.mean(math.log(x) for x in ratios)),
        "ratio_of_total_median_prefill_times": sum(r["dense_ms"] for r in paired) / sum(r["v1_ms"] for r in paired),
        "v1_faster_samples": sum(x > 1 for x in ratios), "profile_single_sample": profile_summary,
        "stop_token_audit": "All 200 records generate im_end, newline, endoftext; two tokens follow first im_end. Nonempty final exports retain im_end. Original artifacts preserved.",
    }
    out = run.parent / "audit"
    out.mkdir(exist_ok=True)
    (out / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "paired_prefill.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
