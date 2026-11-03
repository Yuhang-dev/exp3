"""Teacher-forced mechanism probe for exact copying under weight compression.

For each target value the gold answer is forced and, under every weight condition, we record the logit margin
at the divergence token (the first answer token where the gold value differs from the closest value among the
ten lines on either side), the margin over all value tokens, whether greedy decoding would reproduce the value,
KL from BF16 on answer tokens and on 500 context tokens, and per-layer attention from the divergence step to
the gold line and to its neighbours. Conditions compress all decoder Linears or only one group (layer range or
attention / MLP), with simulated NF4 or Wanda 50% weights, so the damage can be localized. A suffix @prefill
or @decode applies the compressed weights only while encoding the prompt or only while reading out the answer,
separating damage to the stored context from damage to the query side.
"""

import argparse
import json
from pathlib import Path
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from attention import AttentionBackend
from data import _token_span
from prepare_copy_tasks import SOURCES
from weight_compression import CALIBRATION_WINDOWS, decoder_linears, prune_wanda, simulate, text_windows


GROUPS = {
    "all": lambda name: True,
    "L0-9": lambda name: int(name.split(".")[0]) <= 9,
    "L10-18": lambda name: 10 <= int(name.split(".")[0]) <= 18,
    "L19-27": lambda name: int(name.split(".")[0]) >= 19,
    "attn": lambda name: ".self_attn." in name,
    "mlp": lambda name: ".mlp." in name,
    "attnL0-18": lambda name: ".self_attn." in name and int(name.split(".")[0]) <= 18,
}


def selected(group, name):
    """A leading ! compresses everything except the group, i.e. keeps the group in BF16."""
    return not GROUPS[group[1:]](name) if group.startswith("!") else GROUPS[group](name)
NEIGHBOURS = 10
TOP = 20
FP_ALPHA = 0.08  # fp_v1 sparsifies the prompt encoding only; later chunks have fewer queries than keys and stay dense
HEAD_LAYERS = 14  # per-head gold and context mass are stored from this layer on


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--samples", required=True, help="regular expression the sample_id must fully match")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--calibration", type=Path, default=Path("datasets/c4/c4-validation.00000-of-00008.json.gz"))
    parser.add_argument("--conditions", nargs="+", default=[
        "bf16", "nf4sim:all", "wanda:all",
        *(f"nf4sim:{group}" for group in GROUPS if group != "all"),
        *(f"wanda:{group}" for group in GROUPS if group != "all"),
    ], help="base:group[@prefill|@decode]; base is bf16, nf4sim, rtn4, wanda or fp_v1 (FlashPrefill prefill, "
                             "BF16 weights); group may be negated with !")
    return parser.parse_args()


def targets_of(sample, tokenizer):
    answer = "; ".join(f"{key}={sample['answers'][key]}" for key in sample["target_keys"])
    encoded = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
    pattern = SOURCES[sample["variant"]][1]
    lines = sample["context_lines"]
    targets, cursor = [], 0
    for key, evidence in zip(sample["target_keys"], sample["evidence_positions"]):
        value = sample["answers"][key]
        start = answer.index(f"{key}={value}", cursor) + len(key) + 1
        cursor = start + len(value)
        line = lines.index(evidence["record"])
        rivals = []
        for offset in range(-NEIGHBOURS, NEIGHBOURS + 1):
            if offset and 0 <= line + offset < len(lines):
                found = re.search(pattern, lines[line + offset])
                if found:
                    rival = found.group(0)
                    shared = next((i for i, (a, b) in enumerate(zip(value, rival)) if a != b), len(value))
                    rivals.append((shared, -abs(offset), offset, rival))
        shared, _, offset, rival = max(rivals)
        tokens = _token_span(encoded["offset_mapping"], start, start + len(value))
        divergence = _token_span(encoded["offset_mapping"], start + shared, start + shared + 1)[0]
        targets.append({"key": key, "line": line, "rival_offset": offset, "rival": rival, "shared_chars": shared,
                        "value_tokens": tokens, "divergence": divergence})
    return encoded["input_ids"], targets


def line_spans(sample, tokenizer):
    rendered = tokenizer.decode(sample["input_ids"])
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    assert encoded["input_ids"] == sample["input_ids"]
    spans, cursor = [], 0
    for line in sample["context_lines"]:
        start = rendered.index(line, cursor)
        cursor = start + len(line)
        spans.append(_token_span(encoded["offset_mapping"], start, cursor))
    return spans


@torch.no_grad()
def probe(model, sample, answer_ids, targets, spans, reference, switch=None):
    prompt = torch.tensor([sample["input_ids"]], device="cuda")
    cache = DynamicCache()
    torch.compiler.cudagraph_mark_step_begin()
    output = model.model(input_ids=prompt, past_key_values=cache, use_cache=True)
    context = model.lm_head(output.last_hidden_state[0, -700:-200]).float().log_softmax(-1)
    context_gold = torch.tensor(sample["input_ids"][-699:-199], device="cuda")
    hidden = [output.last_hidden_state[0, -1:]]
    if switch:
        switch()
    attention = {}
    steps = sorted(target["divergence"] - 1 for target in targets)
    start = 0
    for step in steps + [len(answer_ids) - 1]:
        if step > start:
            chunk = torch.tensor([answer_ids[start:step]], device="cuda")
            hidden.append(model.model(input_ids=chunk, past_key_values=cache, use_cache=True).last_hidden_state[0])
        if step == len(answer_ids) - 1:
            break
        model.config._attn_implementation = "eager"
        result = model.model(input_ids=torch.tensor([[answer_ids[step]]], device="cuda"), past_key_values=cache,
                             use_cache=True, output_attentions=True)
        model.config._attn_implementation = "sdpa"
        hidden.append(result.last_hidden_state[0])
        attention[step + 1] = torch.stack([layer[0, :, 0].float() for layer in result.attentions])
        start = step + 1
    logits = model.lm_head(torch.cat(hidden)).float()
    logprobs = logits.log_softmax(-1)
    gold = torch.tensor(answer_ids, device="cuda")
    gold_logit = logits.gather(1, gold[:, None])[:, 0]
    margin = gold_logit - logits.scatter(1, gold[:, None], float("-inf")).amax(1)
    argmax_ok = logits.argmax(1) == gold
    if reference is None:
        reference = {"answer": logprobs.topk(TOP), "context": context.topk(TOP)}
    answer_kl = kl(reference["answer"], logprobs)
    context_kl = kl(reference["context"], context)
    context_nll = -context.gather(1, context_gold[:, None])[:, 0]
    rows = []
    for target in targets:
        value = list(range(*target["value_tokens"]))
        division = target["divergence"]
        weights = attention[division]
        line, offset = target["line"], target["rival_offset"]
        mass = torch.stack([weights[:, :, slice(*spans[index])].sum(-1) for index in
                            range(max(0, line - NEIGHBOURS), min(len(spans), line + NEIGHBOURS + 1))], dim=-1)
        local = line - max(0, line - NEIGHBOURS)
        others = torch.cat([mass[..., :local], mass[..., local + 1:]], dim=-1)
        context = weights[HEAD_LAYERS:, :, spans[0][0]:spans[-1][1]].sum(-1)
        rows.append({
            "sample_id": sample["sample_id"], "source": sample["variant"], "key": target["key"],
            "rival_offset": offset, "shared_chars": target["shared_chars"],
            "divergence_margin": margin[division].item(), "divergence_gold_logprob": logprobs[division, gold[division]].item(),
            "value_min_margin": margin[value].min().item(), "value_mean_margin": margin[value].mean().item(),
            "teacher_forced_correct": bool(argmax_ok[value].all()),
            "first_wrong_value_token": next((i for i, index in enumerate(value) if not argmax_ok[index]), -1),
            "divergence_kl": answer_kl[division].item(), "value_kl_mean": answer_kl[value].mean().item(),
            "value_kl_max": answer_kl[value].max().item(), "answer_kl_mean": answer_kl.mean().item(),
            "context_kl_mean": context_kl.mean().item(), "context_nll_mean": context_nll.mean().item(),
            "gold_line_mass": mass[..., local].mean(1).tolist(),
            "rival_line_mass": mass[..., local + offset].mean(1).tolist(),
            "neighbour_mass": others.sum(-1).mean(1).tolist(),
            "gold_line_head_max": mass[..., local].amax(1).tolist(),
            "rival_line_head_max": mass[..., local + offset].amax(1).tolist(),
            "attended_offset": (mass.mean(1).argmax(-1) - local).tolist(),
            "gold_head_mass": mass[HEAD_LAYERS:, :, local].tolist(), "context_head_mass": context.tolist(),
            "context_tokens": spans[-1][1] - spans[0][0], "gold_tokens": spans[line][1] - spans[line][0],
        })
    return rows, reference


def kl(reference, logprobs):
    values, indices = reference
    return (values.exp() * (values - logprobs.gather(1, indices))).sum(1)


def main():
    args = arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    samples = [s for path in args.inputs for s in torch.load(path, weights_only=False)
               if re.fullmatch(args.samples, s["sample_id"])]
    prepared = [(sample, *targets_of(sample, tokenizer), line_spans(sample, tokenizer)) for sample in samples]
    print("prompts", len(prepared), flush=True)
    backend = AttentionBackend() if any(condition.startswith("fp_v1") for condition in args.conditions) else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()
    model.requires_grad_(False)
    linears = decoder_linears(model)
    weights = {"bf16": {name: module.weight.detach().cpu() for name, module in linears.items()}}
    for base in ("nf4sim", "rtn4"):
        if any(condition.startswith(base) for condition in args.conditions):
            weights[base] = {name: simulate(module.weight, base).cpu() for name, module in linears.items()}
    if any(condition.startswith("wanda") for condition in args.conditions):
        prune_wanda(model, text_windows(tokenizer, args.calibration, CALIBRATION_WINDOWS), "unstructured")
        weights["wanda"] = {name: module.weight.detach().cpu() for name, module in linears.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    references = {}
    with (args.out / "targets.jsonl").open("w", encoding="utf-8") as handle:
        for condition in args.conditions:
            spec, _, phase = condition.partition("@")
            base, _, group = spec.partition(":")
            if backend:
                backend.configure("fp_v1" if base == "fp_v1" else "dense", FP_ALPHA)
            chosen = {name: weights[base if base in weights and base != "bf16" and selected(group, name) else "bf16"][name]
                      for name in linears}
            with torch.no_grad():
                for name, module in linears.items():
                    module.weight.copy_(weights["bf16"][name] if phase else chosen[name])
            spare = {name: tensor.cuda() for name, tensor in chosen.items()} if phase else {}

            def swap():
                for name, module in linears.items():
                    module.weight.data, spare[name] = spare[name], module.weight.data

            for sample, answer_ids, targets, spans in prepared:
                if phase == "prefill":
                    swap()
                rows, reference = probe(model, sample, answer_ids, targets, spans, references.get(sample["sample_id"]),
                                        swap if phase else None)
                if phase == "decode":
                    swap()
                references.setdefault(sample["sample_id"], reference)
                for row in rows:
                    handle.write(json.dumps({"condition": condition, **row}) + "\n")
            spare.clear()
            handle.flush()
            print("done", condition, flush=True)


if __name__ == "__main__":
    main()
