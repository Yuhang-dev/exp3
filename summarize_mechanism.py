"""Write MECHANISM.md from copy_mechanism.py output: damage localization, which metric predicts flips,
and whether flipped targets lose attention to the gold line (addressing) or keep it (readout)."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from summarize_pointer_study import read_jsonl, table


def auc(score, label):
    """Probability that a flipped target scores higher than a kept one (ties count half)."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    positive, negative = score[label], score[~label]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    greater = (positive[:, None] > negative[None, :]).mean()
    equal = (positive[:, None] == negative[None, :]).mean()
    return greater + equal / 2


def main():
    folder = Path(sys.argv[1])
    rows = pd.DataFrame(read_jsonl(folder / "targets.jsonl"))
    rows["target"] = rows.sample_id + "|" + rows.key
    reference = rows[rows.condition == "bf16"].set_index("target")
    for column in ("divergence_margin", "value_min_margin", "context_nll_mean", "teacher_forced_correct"):
        rows[f"bf16_{column}"] = rows.target.map(reference[column])
    rows["flipped"] = rows.bf16_teacher_forced_correct & ~rows.teacher_forced_correct
    rows["margin_drop"] = rows.bf16_divergence_margin - rows.divergence_margin
    rows["min_margin_drop"] = rows.bf16_value_min_margin - rows.value_min_margin
    rows["context_nll_rise"] = rows.context_nll_mean - rows.bf16_context_nll_mean
    sections = [f"# Copy mechanism probe ({rows.sample_id.nunique()} prompts, {reference.shape[0]} targets)\n"]

    summary = rows.groupby(["condition", "source"]).agg(
        teacher_forced_accuracy=("teacher_forced_correct", "mean"), flips=("flipped", "sum"),
        median_divergence_margin=("divergence_margin", "median"), median_margin_drop=("margin_drop", "median"),
        median_min_margin_drop=("min_margin_drop", "median"), context_kl=("context_kl_mean", "mean"),
        context_nll_rise=("context_nll_rise", "mean"), value_kl=("value_kl_mean", "mean"),
    ).reset_index()
    summary["teacher_forced_accuracy"] *= 100
    sections += ["## 1. Which weights carry the damage (compress only the named group)\n", table(summary)]

    candidates = {
        "bf16 divergence margin (low = risky)": lambda frame: -frame.bf16_divergence_margin,
        "bf16 min value margin (low = risky)": lambda frame: -frame.bf16_value_min_margin,
        "divergence margin drop": lambda frame: frame.margin_drop,
        "min value margin drop": lambda frame: frame.min_margin_drop,
        "KL at divergence token": lambda frame: frame.divergence_kl,
        "mean KL over value tokens": lambda frame: frame.value_kl_mean,
        "mean KL over whole answer": lambda frame: frame.answer_kl_mean,
        "context KL (500 log tokens)": lambda frame: frame.context_kl_mean,
        "context NLL rise": lambda frame: frame.context_nll_rise,
    }
    at_risk = rows[(rows.condition != "bf16") & rows.bf16_teacher_forced_correct]
    result = []
    for condition, frame in [("pooled", at_risk)] + list(at_risk.groupby("condition")):
        if frame.flipped.sum() < 3:
            continue
        result.append({"condition": condition, "targets": len(frame), "flips": int(frame.flipped.sum()),
                       **{name: auc(score(frame), frame.flipped) for name, score in candidates.items()}})
    if result:
        sections += ["## 2. Which metric predicts a flip (AUC; targets BF16 copies correctly; conditions with >=3 flips)\n",
                     table(pd.DataFrame(result))]

    layers = len(rows.gold_line_mass.iloc[0])
    profile = []
    for condition in ("bf16", "nf4sim:all", "wanda:all"):
        frame = rows[rows.condition == condition]
        if frame.empty:
            continue
        for kept, part in frame.groupby(frame.flipped):
            gold = np.array(part.gold_line_mass.tolist())
            rival = np.array(part.rival_line_mass.tolist())
            profile.append({"condition": condition, "flipped": kept, "targets": len(part),
                            **{f"L{layer}": f"{gold[:, layer].mean():.3f}/{rival[:, layer].mean():.3f}"
                               for layer in range(0, layers, 3)}})
    retrieval = np.argsort(-np.array(reference.gold_line_mass.tolist()).mean(0))[:5]
    verdicts = []
    for condition in ("nf4sim:all", "wanda:all"):
        frame = rows[(rows.condition == condition) & rows.flipped]
        for _, row in frame.iterrows():
            gold = np.array(row.gold_line_mass)[retrieval].sum()
            others = np.array(row.neighbour_mass)[retrieval].sum()
            base = np.array(reference.loc[row.target].gold_line_mass)[retrieval].sum()
            verdicts.append({"condition": condition, "source": row.source,
                             "kind": "addressing" if others > gold else "readout",
                             "gold_mass_kept": gold / base})
    verdicts = pd.DataFrame(verdicts)
    sections += [f"## 3. Attention from the divergence step to the gold / rival line (mean head-averaged mass, every third layer; "
                 f"retrieval layers by BF16 gold mass: {sorted(retrieval.tolist())})\n", table(pd.DataFrame(profile))]
    if not verdicts.empty:
        kinds = verdicts.groupby(["condition", "source", "kind"]).agg(
            targets=("kind", "size"), gold_mass_kept=("gold_mass_kept", "median")).reset_index()
        sections += ["### Flipped targets: neighbours out-attend the gold line in retrieval layers (addressing) or not (readout)\n",
                     table(kinds)]
    text = "\n".join(sections)
    (folder / "MECHANISM.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
