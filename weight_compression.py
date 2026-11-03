"""Weight-compressed Qwen2.5 for the pointer study: bitsandbytes int8 / NF4 loading and Wanda pruning
(Sun et al., 2023: |W| * ||X|| per input channel, compared within each output row, or within groups of four
for 2:4), with 128 C4-validation windows of 2048 tokens as calibration and the next 32 for perplexity.
Simulated formats keep BF16 storage so single modules can be restored: nf4sim quantizes and dequantizes each
decoder Linear with bitsandbytes NF4 (blocksize 64, as BitsAndBytesConfig loads it); rtn4 / rtn3 round each
group of 128 input weights to 4 / 3 bits with an asymmetric min-max grid. nf4sim_mlp quantizes only the MLP
Linears and keeps attention projections in BF16."""

import gzip
import json
import math
import random

import bitsandbytes.functional as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


WINDOW = 2048
CALIBRATION_WINDOWS = 128
PERPLEXITY_WINDOWS = 32


def text_windows(tokenizer, path, count, skip=0, seed=0):
    rng = random.Random(seed)
    windows = []
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            ids = tokenizer(json.loads(line)["text"], add_special_tokens=False)["input_ids"]
            if len(ids) <= WINDOW:
                continue
            start = rng.randrange(0, len(ids) - WINDOW)
            if skip:
                skip -= 1
                continue
            windows.append(torch.tensor([ids[start:start + WINDOW]]))
            if len(windows) == count:
                return windows
    raise ValueError(f"only {len(windows)} windows of {WINDOW} tokens in {path}")


@torch.no_grad()
def prune_wanda(model, windows, mode):
    layers = model.model.layers
    position_ids = torch.arange(WINDOW, device="cuda").unsqueeze(0)
    hidden = [model.model.embed_tokens(window.cuda()) for window in windows]
    position_embeddings = model.model.rotary_emb(hidden[0], position_ids)
    for layer in layers:
        linears = {name: module for name, module in layer.named_modules() if isinstance(module, torch.nn.Linear)}
        squares = {name: torch.zeros(module.in_features, device="cuda") for name, module in linears.items()}
        def accumulate(name):
            def hook(_, inputs, __):
                squares[name].add_(inputs[0].float().square().sum(dim=(0, 1)))
            return hook

        hooks = [module.register_forward_hook(accumulate(name)) for name, module in linears.items()]
        for state in hidden:
            layer(state, position_ids=position_ids, position_embeddings=position_embeddings)
        for hook in hooks:
            hook.remove()
        for name, module in linears.items():
            norms = (squares[name] / (len(hidden) * WINDOW)).sqrt()
            metric = module.weight.float().abs() * norms
            rows, columns = metric.shape
            if mode == "unstructured":
                pruned = metric.topk(columns // 2, dim=1, largest=False).indices
                module.weight.scatter_(1, pruned, 0)
            else:
                pruned = metric.view(rows, columns // 4, 4).topk(2, dim=2, largest=False).indices
                keep = torch.ones(rows, columns // 4, 4, dtype=torch.bool, device="cuda").scatter_(2, pruned, False)
                module.weight.mul_(keep.view(rows, columns))
        hidden = [
            layer(state, position_ids=position_ids, position_embeddings=position_embeddings)[0] for state in hidden
        ]


SIMULATED = ("nf4sim", "nf4sim_mlp", "rtn4", "rtn3")


def decoder_linears(model):
    return {name: module for name, module in model.model.layers.named_modules() if isinstance(module, torch.nn.Linear)}


@torch.no_grad()
def simulate(weight, weight_format):
    if weight_format.startswith("nf4sim"):
        packed, state = bnb.quantize_4bit(weight, blocksize=64, quant_type="nf4")
        return bnb.dequantize_4bit(packed, state, blocksize=64, quant_type="nf4").to(weight.dtype)
    bits = int(weight_format[-1])
    groups = weight.float().view(weight.shape[0], -1, 128)
    low, high = groups.amin(dim=2, keepdim=True), groups.amax(dim=2, keepdim=True)
    scale = (high - low).clamp_min(1e-8) / (2**bits - 1)
    return ((groups - low) / scale).round().mul(scale).add(low).view_as(weight).to(weight.dtype)


def load_model(name, weight_format="bf16", wanda="none", calibration=None):
    quantization = {
        "bf16": None,
        "int8": BitsAndBytesConfig(load_in_8bit=True),
        "nf4": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        ),
    }.get(weight_format)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda",
        quantization_config=quantization,
    ).eval()
    model.requires_grad_(False)
    if wanda != "none":
        tokenizer = AutoTokenizer.from_pretrained(name)
        prune_wanda(model, text_windows(tokenizer, calibration, CALIBRATION_WINDOWS), wanda)
    if weight_format in SIMULATED:
        for name, module in decoder_linears(model).items():
            if weight_format != "nf4sim_mlp" or ".mlp." in name:
                module.weight.copy_(simulate(module.weight, weight_format))
    return model


@torch.no_grad()
def perplexity(model, tokenizer, path):
    losses = []
    for window in text_windows(tokenizer, path, PERPLEXITY_WINDOWS, skip=CALIBRATION_WINDOWS):
        window = window.cuda()
        losses.append(model(input_ids=window, labels=window).loss.item())
    return math.exp(sum(losses) / len(losses))
