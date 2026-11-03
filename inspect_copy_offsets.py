import collections, json, re, torch
from scoring import parse_synthetic
root = "results/pointer"
S = {s["sample_id"]: s for s in torch.load(f"{root}/batch3_inputs/copy.pt", weights_only=False)}
pat = {"hdfs": r"blk_-?\d+", "openstack": r"req-[0-9a-f-]{36}", "manifest": r"[0-9a-f]{64}"}
for folder in ("bf16", "nf4_none", "bf16_unstructured"):
    shown = collections.Counter(); signs = collections.defaultdict(collections.Counter)
    for line in open(f"{root}/batch3/{folder}/generations.jsonl"):
        r = json.loads(line); s = S[r["sample_id"]]; src = s["variant"]; lines = s["context_lines"]
        parsed = parse_synthetic(r["raw_text"], s["target_keys"])
        if src == "openstack" and shown[(r["config_id"], src)] < 2:
            shown[(r["config_id"], src)] += 1; print(folder, r["config_id"], repr(r["raw_text"][:300])); print("   exp", s["answers"])
        if src == "manifest" and folder == "bf16_unstructured" and shown["m"] < 2:
            shown["m"] += 1; print(folder, repr(r["raw_text"][:300]))
        for key, ev in zip(s["target_keys"], s["evidence_positions"]):
            got = parsed.get(key, ""); t = lines.index(ev["record"])
            if got and got != s["answers"][key]:
                hits = [i - t for i, l in enumerate(lines) if got in re.findall(pat[src], l)]
                if hits: signs[(r["config_id"], src)][min(hits, key=abs)] += 1
    for k, c in sorted(signs.items()): print(folder, k, sorted(c.items())[:12], "... +1:", c[1], "-1:", c[-1], "total", sum(c.values()))
