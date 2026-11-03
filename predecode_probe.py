"""Pre-decoding probe: can flips under weight compression be predicted from the prompt alone?

Only the prompt is encoded (no answer token is forced or generated). For every target value we record, per layer:
attention from two query positions inside the question that name the key (the "=" in "KEY=VALUE" and the key in
"Requested: ...") to the gold line, the most similar rival line within ten lines, and all ten-line neighbours;
the same attention with BF16 queries against the condition's keys and with the condition's queries against BF16
keys, separating damage to the stored context from damage to the question encoding; the drift of the gold value
tokens' keys, values and hidden states from BF16 and the same drift over 500 context tokens; and the cosine
similarity between the gold and rival value representations. Labels come from earlier runs.
"""

import argparse
import json
from pathlib import Path
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from copy_mechanism import NEIGHBOURS, selected, targets_of
from data import _token_span
from prepare_copy_tasks import SOURCES
from weight_compression import CALIBRATION_WINDOWS, decoder_linears, prune_wanda, simulate, text_windows


KEPT = 14  # BF16 keys are kept from this layer on (query layers are 19-25); earlier layers get NaN for bf16_keys


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--samples", required=True, help="regular expression the sample_id must fully match")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--calibration", type=Path, default=Path("datasets/c4/c4-validation.00000-of-00008.json.gz"))
    parser.add_argument("--conditions", nargs="+",
                        default=["bf16", "nf4sim:all", "nf4sim:!attn", "rtn4:all", "wanda:all"])
    return parser.parse_args()


def layout(sample, tokenizer, targets):
    """Token spans of every line and of every line's value, and the two question positions of each target."""
    rendered = tokenizer.decode(sample["input_ids"])
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    assert encoded["input_ids"] == sample["input_ids"]
    offsets, pattern = encoded["offset_mapping"], SOURCES[sample["variant"]][1]
    lines, values, cursor = [], [], 0
    for line in sample["context_lines"]:
        start = rendered.index(line, cursor)
        cursor = start + len(line)
        lines.append(_token_span(offsets, start, cursor))
        found = re.search(pattern, line)
        values.append(_token_span(offsets, start + found.start(), start + found.end()) if found else None)
    question = rendered.rindex(sample["question"])
    requested = sample["question"].index("Requested:")
    for target in targets:
        key = target["key"]
        equals = question + sample["question"].index(f"{key}=VALUE") + len(key)
        named = question + sample["question"].index(key, requested) + len(key) - 1
        target["queries"] = [_token_span(offsets, equals, equals + 1)[0], _token_span(offsets, named, named + 1)[0]]
        prefix = tokenizer(f"{key}={sample['answers'][key]}", add_special_tokens=False, return_offsets_mapping=True)
        target["prefix"] = [token for token, (_, end) in zip(prefix["input_ids"], prefix["offset_mapping"])
                            if end <= len(key) + 1]
    return lines, values


@torch.no_grad()
def encode(model, sample):
    prompt = torch.tensor([sample["input_ids"]], device="cuda")
    cache = DynamicCache()
    output = model.model(input_ids=prompt, past_key_values=cache, use_cache=True, output_hidden_states=True)
    return output.hidden_states, cache


@torch.no_grad()
def queries(model, hidden, positions):
    """Rotated query vectors [layer, head, position, dim] at the given prompt positions."""
    index = torch.tensor(positions, device="cuda")
    result = []
    for layer, block in enumerate(model.model.layers):
        x = block.input_layernorm(hidden[layer][0, index])
        q = block.self_attn.q_proj(x).view(len(positions), -1, block.self_attn.head_dim).transpose(0, 1)[None]
        cos, sin = model.model.rotary_emb(x[None], index[None])
        result.append(apply_rotary_pos_emb(q, q, cos, sin)[0][0])
    return torch.stack(result)


def attention(query, keys, groups, positions, first=0):
    """Head-averaged causal attention [layer, position, token] of rotated queries over the prompt; keys start at
    layer `first` and earlier layers are NaN."""
    future = torch.arange(keys[0].shape[2], device="cuda")[None] > torch.tensor(positions, device="cuda")[:, None]
    rows = [torch.full(future.shape, float("nan"), device="cuda")] * first
    for layer, key in enumerate(keys, start=first):
        key = key[0].cuda().float().repeat_interleave(groups, dim=0)
        scores = query[layer].float() @ key.transpose(1, 2) / key.shape[-1] ** 0.5
        rows.append(scores.masked_fill(future, float("-inf")).softmax(-1).mean(0))
    return torch.stack(rows)


def masses(weights, spans, line, rival):
    window = range(max(0, line - NEIGHBOURS), min(len(spans), line + NEIGHBOURS + 1))
    per_line = {index: weights[..., slice(*spans[index])].sum(-1) for index in window}
    neighbours = sum(mass for index, mass in per_line.items() if index != line)
    return per_line[line], per_line[line + rival], neighbours


def drift(current, reference):
    current, reference = current.float(), reference.float()
    return ((current - reference).norm(dim=-1) / reference.norm(dim=-1)).mean().item()


def cosine_gap(current, reference):
    return 1 - torch.nn.functional.cosine_similarity(current.float(), reference.float(), dim=-1).mean().item()


@torch.no_grad()
def probe(model, sample, targets, spans, values, reference):
    hidden, cache = encode(model, sample)
    keys, cache_values = list(cache.key_cache), list(cache.value_cache)
    groups = model.config.num_attention_heads // model.config.num_key_value_heads
    positions = [position for target in targets for position in target["queries"]]
    query = queries(model, hidden, positions)
    context = torch.arange(len(sample["input_ids"]) - 700, len(sample["input_ids"]) - 200, device="cuda")
    own = {
        "query": query.cpu(), "keys": [key.cpu() for key in keys[KEPT:]],
        "context_k": torch.stack([key[0][:, context] for key in keys]).cpu(),
        "context_v": torch.stack([value[0][:, context] for value in cache_values]).cpu(),
        "context_h": torch.stack([state[0, context] for state in hidden[1:]]).cpu(),
    }
    if reference is None:
        reference = own
    native = attention(query, keys, groups, positions)
    bf16_query = attention(reference["query"].cuda(), keys, groups, positions)
    bf16_keys = attention(query, reference["keys"], groups, positions, KEPT)
    layers = len(keys)
    global_drift = {
        "context_k_drift": [drift(own["context_k"][l], reference["context_k"][l]) for l in range(layers)],
        "context_v_drift": [drift(own["context_v"][l], reference["context_v"][l]) for l in range(layers)],
        "context_h_drift": [cosine_gap(own["context_h"][l], reference["context_h"][l]) for l in range(layers)],
    }
    rows = []
    for number, target in enumerate(targets):
        line, rival = target["line"], target["rival_offset"]
        gold_span, rival_span = slice(*values[line]), slice(*values[line + rival])
        row = {"sample_id": sample["sample_id"], "source": sample["variant"], "key": target["key"], "line": line,
               "depth": line / len(spans), "rival_offset": rival, "shared_chars": target["shared_chars"], **global_drift}
        for name, weights in (("native", native), ("bf16_query", bf16_query), ("bf16_keys", bf16_keys)):
            for kind, column in (("equals", 2 * number), ("named", 2 * number + 1)):
                gold, competitor, neighbours = masses(weights[:, column], spans, line, rival)
                row[f"{name}_{kind}_gold"] = gold.tolist()
                if name == "native":
                    row[f"{kind}_rival"] = competitor.tolist()
                    row[f"{kind}_neighbours"] = neighbours.tolist()
        gold_k = [key[0][:, gold_span] for key in keys]
        gold_v = [value[0][:, gold_span] for value in cache_values]
        gold_h = [state[0, gold_span] for state in hidden[1:]]
        own[f"gold_{number}"] = ([k.cpu() for k in gold_k], [v.cpu() for v in gold_v], [h.cpu() for h in gold_h])
        base_k, base_v, base_h = reference[f"gold_{number}"]
        row["gold_k_drift"] = [drift(gold_k[l].cpu(), base_k[l]) for l in range(layers)]
        row["gold_v_drift"] = [drift(gold_v[l].cpu(), base_v[l]) for l in range(layers)]
        row["gold_h_drift"] = [cosine_gap(gold_h[l].cpu(), base_h[l]) for l in range(layers)]
        row["gold_rival_cos"] = [torch.nn.functional.cosine_similarity(
            state[0, gold_span].float().mean(0), state[0, rival_span].float().mean(0), dim=0).item()
            for state in hidden[1:]]
        rows.append(row)
    model.config._attn_implementation = "eager"
    for row, target in zip(rows, targets):
        result = model.model(input_ids=torch.tensor([target["prefix"]], device="cuda"), past_key_values=cache,
                             use_cache=True, output_attentions=True)
        weights = torch.stack([layer[0, :, -1].float().mean(0) for layer in result.attentions])
        gold, competitor, neighbours = masses(weights, spans, target["line"], target["rival_offset"])
        row.update(start_gold=gold.tolist(), start_rival=competitor.tolist(), start_neighbours=neighbours.tolist())
        cache.crop(len(sample["input_ids"]))
    model.config._attn_implementation = "sdpa"
    return rows, reference


def main():
    args = arguments()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    samples = [s for s in torch.load(args.inputs, weights_only=False) if re.fullmatch(args.samples, s["sample_id"])]
    prepared = []
    for sample in samples:
        _, targets = targets_of(sample, tokenizer)
        prepared.append((sample, targets, *layout(sample, tokenizer, targets)))
    print("prompts", len(prepared), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()
    model.requires_grad_(False)
    linears = decoder_linears(model)
    weights = {"bf16": {name: module.weight.detach().cpu() for name, module in linears.items()}}
    args.out.mkdir(parents=True, exist_ok=True)
    references = {}
    with (args.out / "targets.jsonl").open("w", encoding="utf-8") as handle:
        for condition in args.conditions:
            base, _, group = condition.partition(":")
            if base not in weights:
                weights = {"bf16": weights["bf16"]}
                with torch.no_grad():
                    for name, module in linears.items():
                        module.weight.copy_(weights["bf16"][name])
                if base == "wanda":
                    prune_wanda(model, text_windows(tokenizer, args.calibration, CALIBRATION_WINDOWS), "unstructured")
                    weights[base] = {name: module.weight.detach().cpu() for name, module in linears.items()}
                else:
                    weights[base] = {name: simulate(module.weight, base).cpu() for name, module in linears.items()}
            with torch.no_grad():
                for name, module in linears.items():
                    module.weight.copy_(weights[base if base != "bf16" and selected(group, name) else "bf16"][name])
            for sample, targets, spans, values in prepared:
                rows, reference = probe(model, sample, targets, spans, values, references.get(sample["sample_id"]))
                references.setdefault(sample["sample_id"], reference)
                for row in rows:
                    handle.write(json.dumps({"condition": condition, **row}) + "\n")
            handle.flush()
            print("done", condition, flush=True)


if __name__ == "__main__":
    main()
