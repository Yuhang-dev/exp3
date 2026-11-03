"""Re-score copy outputs accepting 'key value' as well as 'key=value', separating format drift from copy errors."""
import json, re, sys, collections, torch
inputs, *folders = sys.argv[1:]
S = {s["sample_id"]: s for s in torch.load(inputs, weights_only=False)}
for folder in folders:
    c = collections.Counter()
    for line in open(f"{folder}/generations.jsonl"):
        r = json.loads(line); s = S.get(r["sample_id"])
        if s is None: continue
        values = {v for l in s["context_lines"] for v in re.findall(r"[0-9a-f]{64}|blk_-?\d+", l)}
        for key in s["target_keys"]:
            m = re.search(rf"(?<![A-Za-z0-9_/]){re.escape(key)}\s*[:=]?\s*([A-Za-z0-9_-]+)", r["raw_text"])
            got = m.group(1) if m else ""
            c["correct" if got == s["answers"][key] else "missing" if not got else "copy_other_line" if got in values else "other"] += 1
    total = sum(c.values()); print(folder.split("/")[-1], f"lenient accuracy {100*c['correct']/total:.1f}", dict(c))
