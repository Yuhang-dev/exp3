"""Write PREDECODE.md: how well prompt-only quantities from predecode_probe.py predict which targets flip under
compression, against teacher-forced flips (mechanism probes) and free-generation flips (batch 3 / 4 runs).
Usage: summarize_predecode.py ROOT; AUC confidence intervals resample prompts."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from scoring import parse_synthetic
from summarize_mechanism import auc
from summarize_pointer_study import read_jsonl, table


TEACHER_FORCED = {"nf4sim:all": ("mechanism", "nf4sim:all"), "nf4sim:all@prefill": ("mechanism_phase", "nf4sim:all@prefill"),
                  "wanda:all": ("mechanism", "wanda:all"), "nf4sim:!attn": ("repair", "nf4sim:!attn"),
                  "rtn4:all": ("mechanism_rtn4", "rtn4:all"), "rtn4:all@prefill": ("mechanism_rtn4", "rtn4:all@prefill"),
                  "rtn4:all@decode": ("mechanism_rtn4", "rtn4:all@decode")}
FREE = {"nf4sim:all": "batch4_formats/nf4sim_none", "nf4 (bitsandbytes)": "batch3/nf4_none",
        "rtn4:all": "batch4_formats/rtn4_none", "wanda:all": "batch3/bf16_unstructured",
        "nf4sim:!attn": "batch4_formats/nf4sim_mlp_none"}
PROBE = {"nf4sim:all@prefill": "nf4sim:all", "nf4 (bitsandbytes)": "nf4sim:all",
         "rtn4:all@prefill": "rtn4:all", "rtn4:all@decode": "rtn4:all"}


def free_correct(root, run, samples):
    result = {}
    for record in read_jsonl(root / run / "generations.jsonl"):
        if record["config_id"] == "dense" and record["sample_id"] in samples:
            sample = samples[record["sample_id"]]
            parsed = parse_synthetic(record["raw_text"], sample["target_keys"])
            for key in sample["target_keys"]:
                result[f"{record['sample_id']}|{key}"] = parsed.get(key, "") == sample["answers"][key]
    return pd.Series(result)


def teacher_correct(root, folder, condition):
    rows = pd.DataFrame(read_jsonl(root / folder / "targets.jsonl"))
    rows = rows[rows.condition == condition]
    return pd.Series(rows.teacher_forced_correct.values, index=rows.sample_id + "|" + rows.key)


def bootstrap(score, label, prompt, draws=1000, seed=0):
    rng = np.random.default_rng(seed)
    prompts = np.unique(prompt)
    members = {name: np.flatnonzero(prompt == name) for name in prompts}
    values = []
    for _ in range(draws):
        index = np.concatenate([members[name] for name in rng.choice(prompts, len(prompts))])
        values.append(auc(score[index], label[index]))
    return np.nanpercentile(values, [2.5, 97.5])


def main():
    root = Path(sys.argv[1])
    rows = pd.DataFrame(read_jsonl(root / "predecode" / "targets.jsonl"))
    rows["target"] = rows.sample_id + "|" + rows.key
    samples = {s["sample_id"]: s for s in torch.load(root / "batch3_inputs/copy.pt", weights_only=False)
               if s["sample_id"] in set(rows.sample_id)}
    base = rows[rows.condition == "bf16"].set_index("target")
    layers = len(base.native_equals_gold.iloc[0])
    profile = np.array(base.native_equals_gold.tolist()).mean(0)
    retrieval = sorted(np.argsort(-profile)[:5].tolist())
    named_retrieval = sorted(np.argsort(-np.array(base.native_named_gold.tolist()).mean(0))[:5].tolist())
    start_retrieval = sorted(np.argsort(-np.array(base.start_gold.tolist()).mean(0))[:5].tolist())
    late = list(range(19, layers))

    def pick(series, chosen):
        return np.array(series.tolist())[:, chosen].mean(1)

    def lost(frame, column, reference, chosen):
        return 1 - pick(frame[column], chosen) / pick(reference.loc[frame.target, reference_column(column)], chosen)

    def reference_column(column):
        return column.replace("bf16_query_", "native_").replace("bf16_keys_", "native_")

    bf16_scores = {
        "shared characters with the closest rival": lambda f, r: r.loc[f.target, "shared_chars"].values,
        "BF16 gold attention from '=' (low = risky)": lambda f, r: -pick(r.loc[f.target, "native_equals_gold"], retrieval),
        "BF16 gold attention from the key (low = risky)": lambda f, r: -pick(r.loc[f.target, "native_named_gold"], named_retrieval),
        "BF16 gold-rival hidden similarity, L19-27": lambda f, r: pick(r.loc[f.target, "gold_rival_cos"], late),
    }
    compressed_scores = {
        "gold attention lost from '='": lambda f, r: lost(f, "native_equals_gold", r, retrieval),
        "gold attention lost from the key": lambda f, r: lost(f, "native_named_gold", r, named_retrieval),
        "gold attention lost at answer start (KEY= appended)": lambda f, r: lost(f, "start_gold", r, start_retrieval),
        "... BF16 queries, compressed keys (context side)": lambda f, r: lost(f, "bf16_query_equals_gold", r, retrieval),
        "... compressed queries, BF16 keys (question side)": lambda f, r: lost(f, "bf16_keys_equals_gold", r, retrieval),
        "gold key drift, all layers": lambda f, r: pick(f.gold_k_drift, slice(None)),
        "gold value drift, all layers": lambda f, r: pick(f.gold_v_drift, slice(None)),
        "gold hidden drift, all layers": lambda f, r: pick(f.gold_h_drift, slice(None)),
        "gold / context key drift": lambda f, r: pick(f.gold_k_drift, slice(None)) / pick(f.context_k_drift, slice(None)),
        "gold-rival similarity rise, L19-27": lambda f, r: pick(f.gold_rival_cos, late) - pick(r.loc[f.target, "gold_rival_cos"], late),
        "context key drift (global)": lambda f, r: pick(f.context_k_drift, slice(None)),
        "context hidden drift (global)": lambda f, r: pick(f.context_h_drift, slice(None)),
    }
    scores = {**bf16_scores, **compressed_scores}

    bf16_free = free_correct(root, "batch3/bf16", samples)
    bf16_teacher = teacher_correct(root, "mechanism", "bf16")
    labels = []
    for condition, (folder, name) in TEACHER_FORCED.items():
        labels.append(("teacher-forced", condition, bf16_teacher, teacher_correct(root, folder, name)))
    for condition, run in FREE.items():
        labels.append(("free generation", condition, bf16_free, free_correct(root, run, samples)))

    result, decomposition = [], []
    for kind, condition, reference_ok, correct in labels:
        frame = rows[rows.condition == PROBE.get(condition, condition)]
        frame = frame[frame.target.map(reference_ok).fillna(False).astype(bool)]
        flipped = ~frame.target.map(correct).astype(bool).values
        prompt = frame.sample_id.values
        entry = {"label": kind, "condition": condition, "targets": len(frame), "flips": int(flipped.sum())}
        for name, score in scores.items():
            value = np.asarray(score(frame, base), float)
            low, high = bootstrap(value, flipped, prompt)
            entry[name] = f"{auc(value, flipped):.2f} [{low:.2f}, {high:.2f}]"
        result.append(entry)
        for side, column in (("native", "native_equals_gold"), ("BF16 queries", "bf16_query_equals_gold"),
                             ("BF16 keys", "bf16_keys_equals_gold")):
            kept = 1 - lost(frame, column, base, retrieval)
            decomposition.append({"label": kind, "condition": condition, "attention": side,
                                  "gold kept, flipped": np.median(kept[flipped]),
                                  "gold kept, unchanged": np.median(kept[~flipped])})

    own = base[base.index.isin(bf16_free.index)]
    wrong = ~own.index.map(bf16_free).astype(bool).values
    own_rows = {"label": "BF16 own errors (free generation)", "targets": len(own), "errors": int(wrong.sum())}
    for name, score in bf16_scores.items():
        value = np.asarray(score(own.reset_index(), base), float)
        own_rows[name] = f"{auc(value, wrong):.2f}"

    result = pd.DataFrame(result)
    result.index = result.label.str.split().str[0] + " " + result.condition
    transposed = result.drop(columns=["label", "condition"]).T.reset_index()
    sections = [
        f"# Pre-decoding flip prediction ({rows.sample_id.nunique()} prompts, {len(base)} targets)\n",
        f"Query layers: '=' {retrieval}, key {named_retrieval}, answer start {start_retrieval} "
        "(top five by BF16 gold-line attention). "
        "Each cell is AUC [95% prompt-bootstrap interval] for predicting that a target BF16 gets right turns wrong.\n",
        table(transposed),
        "## Where the gold-line attention is lost (median share of BF16 attention kept, '=' query, query layers)\n",
        table(pd.DataFrame(decomposition)),
        "## BF16's own errors from BF16-only quantities (AUC)\n",
        table(pd.DataFrame([own_rows])),
        "## BF16 gold-line attention by layer (from '=' / from the key)\n",
        " ".join(f"L{l}:{profile[l]:.3f}/{np.array(base.native_named_gold.tolist()).mean(0)[l]:.3f}" for l in range(layers)),
    ]
    text = "\n".join(sections)
    (root / "predecode" / "PREDECODE.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
