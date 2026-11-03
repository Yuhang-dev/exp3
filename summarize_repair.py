"""Write REPAIR.md: free-generation exact copying at 8K / 16K (HDFS) and 32K (HDFS, SHA-256 manifests) for BF16,
simulated NF4 and RTN4, and the same quantizers with attention projections (_mlp) or only q/k projections (_noqk)
kept in BF16. Recovery = (repaired - compressed) / (BF16 - compressed); sign tests pair prompts.
_mlp03 / _rand / _err keep the same 12.6% of parameters as _mlp, chosen without the noise picture; rtn5 / rtn6
spend about the same average bits uniformly. Usage: summarize_repair.py ROOT."""

import json
from pathlib import Path
import sys

import pandas as pd

from summarize_pointer_study import sign_test, table


RUNS = {"batch4_length": "bf16_none", "batch4_formats": "bf16"}
FORMATS = ["nf4sim", "nf4sim_mlp", "nf4sim_noqk", "nf4sim_mlp03", "nf4sim_rand", "nf4sim_err", "rtn4", "rtn4_mlp",
           "rtn5", "rtn6"]
COMPRESSED = {"nf4sim_mlp": "nf4sim", "nf4sim_noqk": "nf4sim", "nf4sim_mlp03": "nf4sim", "nf4sim_rand": "nf4sim",
              "nf4sim_err": "nf4sim", "rtn4_mlp": "rtn4", "rtn5": "rtn4", "rtn6": "rtn4"}


def main():
    root = Path(sys.argv[1])
    scores = []
    for folder, reference in RUNS.items():
        for name in [reference] + [f"{fmt}_none" for fmt in FORMATS]:
            frame = pd.read_csv(root / folder / name / "quality.csv")
            frame = frame[frame.config_id == "dense"]
            scores.append(frame.assign(format="bf16" if name == reference else name[:-5])[["sample_id", "format", "score"]])
    scores = pd.concat(scores)
    scores = scores[~scores.sample_id.str.contains("openstack")]
    scores["source"] = scores.sample_id.str.rsplit("-", n=1).str[0]
    wide = scores.pivot_table(index=["source", "sample_id"], columns="format", values="score")

    accuracy = wide.groupby("source").mean().reindex(columns=["bf16"] + FORMATS).reset_index()
    tests = []
    for source, part in wide.groupby("source"):
        for repaired, compressed in COMPRESSED.items():
            gap = part.bf16.mean() - part[compressed].mean()
            tests.append({
                "source": source, "repair": repaired, "compressed": part[compressed].mean(),
                "repaired": part[repaired].mean(), "bf16": part.bf16.mean(),
                "recovery": (part[repaired].mean() - part[compressed].mean()) / gap if gap > 0 else float("nan"),
                "vs compressed: better/worse": f"{(part[repaired] > part[compressed]).sum()}/{(part[repaired] < part[compressed]).sum()}",
                "p vs compressed": sign_test((part[repaired] > part[compressed]).sum(), (part[repaired] < part[compressed]).sum()),
                "vs bf16: better/worse": f"{(part[repaired] > part.bf16).sum()}/{(part[repaired] < part.bf16).sum()}",
                "p vs bf16": sign_test((part[repaired] > part.bf16).sum(), (part[repaired] < part.bf16).sum()),
            })
    pooled = wide[wide.index.get_level_values("source").isin(["copy-hdfs", "len16k-hdfs", "copy-manifest"])]
    for repaired, compressed in COMPRESSED.items():
        tests.append({
            "source": "16K+32K pooled", "repair": repaired, "compressed": pooled[compressed].mean(),
            "repaired": pooled[repaired].mean(), "bf16": pooled.bf16.mean(),
            "recovery": (pooled[repaired].mean() - pooled[compressed].mean()) / (pooled.bf16.mean() - pooled[compressed].mean()),
            "vs compressed: better/worse": f"{(pooled[repaired] > pooled[compressed]).sum()}/{(pooled[repaired] < pooled[compressed]).sum()}",
            "p vs compressed": sign_test((pooled[repaired] > pooled[compressed]).sum(), (pooled[repaired] < pooled[compressed]).sum()),
            "vs bf16: better/worse": f"{(pooled[repaired] > pooled.bf16).sum()}/{(pooled[repaired] < pooled.bf16).sum()}",
            "p vs bf16": sign_test((pooled[repaired] > pooled.bf16).sum(), (pooled[repaired] < pooled.bf16).sum()),
        })
    perplexity = {path.stem: json.loads(path.read_text())["perplexity"] for path in (root / "ppl").glob("*.json")}
    sections = ["# Free-generation repair: attention projections kept in BF16\n",
                "## Exact-copy accuracy (%) by source\n", table(accuracy),
                "## Recovery of the compression gap (prompt-paired sign tests)\n", table(pd.DataFrame(tests)),
                "## C4 perplexity\n",
                table(pd.DataFrame(sorted(perplexity.items()), columns=["format", "perplexity"]))]
    text = "\n".join(sections)
    (root / "REPAIR.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
