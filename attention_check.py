"""Attention retention in BF16 retrieval layers for targets that flip under compression, from copy_mechanism.py output."""
import json, sys, numpy as np, pandas as pd
rows = pd.DataFrame([json.loads(l) for l in open(sys.argv[1])]); rows["target"] = rows.sample_id + "|" + rows.key
ref = rows[rows.condition == "bf16"].set_index("target")
R = np.argsort(-np.array(ref.gold_line_mass.tolist()).mean(0))[:5]
def g(series): return np.array(series.tolist())[:, R].sum(1)
print("retrieval layers", sorted(R.tolist()))
print("bf16 rule rate (neighbours > gold):", (g(ref.neighbour_mass) > g(ref.gold_line_mass)).mean().round(3),
      " rival > gold:", (g(ref.rival_line_mass) > g(ref.gold_line_mass)).mean().round(3))
def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool); p, n = s[y], s[~y]
    return ((p[:, None] > n[None]).mean() + (p[:, None] == n[None]).mean() / 2) if len(p) and len(n) else float("nan")
base_ok = rows.target.map(ref.teacher_forced_correct)
for cond in sorted(set(rows.condition) - {"bf16"}):
    f = rows[(rows.condition == cond) & base_ok]
    flip = ~f.teacher_forced_correct.values
    ratio = g(f.gold_line_mass) / g(ref.loc[f.target].gold_line_mass)
    rival = g(f.rival_line_mass) > g(f.gold_line_mass)
    att = np.array(f.attended_offset.tolist())[:, R]
    on_rival = (att == f.rival_offset.values[:, None]).mean(1)
    print(f"{cond:16s} flips {flip.sum():3d}/{len(f)}  gold-mass kept: flipped {np.median(ratio[flip]) if flip.any() else float('nan'):.2f} "
          f"kept {np.median(ratio[~flip]):.2f}  AUC(1-kept) {auc(-ratio, flip):.3f}  rival>gold: flipped {rival[flip].mean() if flip.any() else float('nan'):.2f} kept {rival[~flip].mean():.2f}  "
          f"argmax=rival layers: flipped {on_rival[flip].mean() if flip.any() else float('nan'):.2f} kept {on_rival[~flip].mean():.2f}")
