import collections, json, re, torch
from transformers import AutoTokenizer
root = "results/pointer"; tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
samples = {s["sample_id"]: s for s in torch.load(f"{root}/batch2_inputs/combined.pt", weights_only=False) if "log" in s["task_label"]}
BLK = re.compile(r"blk_-?\d+")
for folder in ["temperature", "weights/int8_none", "weights/nf4_none", "weights/bf16_unstructured", "weights/bf16_2_4"]:
    c = collections.Counter()
    for line in open(f"{root}/{folder}/generations.jsonl"):
        r = json.loads(line); s = samples.get(r["sample_id"])
        if s is None or (folder == "temperature" and r["config_id"] != "dense"): continue
        ids = set(BLK.findall(tok.decode(s["input_ids"])))
        out = r["raw_text"]
        for key, exp in s["answers"].items():
            m = re.search(re.escape(key) + r"\D[^\n]*?(blk_-?\d+)", out)
            got = m.group(1) if m else ""
            c["correct" if got == exp else "missing" if not got else "copy_other_line" if got in ids else "not_in_prompt"] += 1
    print(folder, dict(c))
