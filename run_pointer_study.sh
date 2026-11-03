#!/usr/bin/env bash
# Discriminating experiments for the sparse copying failure; see POINTER_STUDY.md.
# Usage: bash run_pointer_study.sh {check|prepare|smoke|alpha|length|patch|factors|holdout|summary|all}
# Follow-up stages: followup_data, margin, quant, small, followup_summary.
# Batch 2 stages: batch2_data, temperature, competitors, weights, batch2_summary.
# Batch 3 stages: batch3_data, batch3, batch3_summary.
# Batch 4 stages: mechanism, batch4_data, batch4, batch4_summary, mechanism_phase, repair.
# Batch 5 stages: predecode, rtn4_mechanism, gap, holdout_gap, noise, repair_free, noise_sources, budget.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

ROOT="${POINTER_ROOT:-results/pointer}"
INPUTS="$ROOT/inputs_32k/inputs.pt"
mkdir -p "$ROOT"

check() {
  python -u check_math.py --out "$ROOT/check_math.json" 2>&1 | tee "$ROOT/check_math.log"
}

prepare() {
  python -u evaluate.py --split calibration --tasks synthetic_kv_retrieval --lengths 32768 \
    --synthetic-samples 4 --candidate dense --max-new-tokens 128 --prepare-only --out "$ROOT/inputs_32k"
  python - "$INPUTS" <<'PY'
import json, sys, torch
expected = json.load(open("pointer_calibration_hashes.json"))
actual = {s["sample_id"]: s["input_sha256"] for s in torch.load(sys.argv[1], weights_only=False)}
assert actual == expected, "regenerated prompts differ from the archived 32K sweep"
print("8 prompts match the archived 32K sweep")
PY
  python -u prepare_pointer_factors.py --inputs "$INPUTS" --out "$ROOT/factors_inputs/inputs.pt"
}

smoke() {
  python -u evaluate.py --split quick --tasks synthetic_kv_retrieval --lengths 4096 \
    --synthetic-samples 1 --candidate dense --max-new-tokens 128 --prepare-only --out "$ROOT/smoke_inputs"
  python -u pointer_patch_study.py --inputs "$ROOT/smoke_inputs/inputs.pt" --limit 1 --out "$ROOT/smoke_patch" \
    2>&1 | tee "$ROOT/smoke_patch.log"
  python -u evaluate.py --inputs "$ROOT/smoke_inputs/inputs.pt" --split quick --tasks synthetic_kv_retrieval \
    --candidate dense --candidate fp_row_norm:0.08 --max-new-tokens 128 --repeats 1 --profile --out "$ROOT/smoke_eval" \
    2>&1 | tee "$ROOT/smoke_eval.log"
}

alpha() {
  python -u evaluate.py --inputs "$INPUTS" --split calibration --tasks synthetic_kv_retrieval \
    --config pointer_candidates.json --max-new-tokens 128 --repeats 1 --profile --out "$ROOT/alpha" \
    2>&1 | tee "$ROOT/alpha.log"
}

length() {
  python -u evaluate.py --split calibration --tasks synthetic_kv_retrieval --lengths 8192 16384 24576 32768 \
    --synthetic-samples 4 --candidate dense --candidate fp_v1:0.08 --candidate fp_row_norm:0.08 \
    --candidate mean_balanced:0.08 --max-new-tokens 128 --repeats 1 --out "$ROOT/length" \
    2>&1 | tee "$ROOT/length.log"
}

patch() {
  python -u pointer_patch_study.py --inputs "$INPUTS" --out "$ROOT/patch" 2>&1 | tee "$ROOT/patch.log"
}

factors() {
  python -u evaluate.py --inputs "$ROOT/factors_inputs/inputs.pt" --split calibration \
    --tasks synthetic_kv_retrieval --candidate dense --candidate fp_v1:0.08 --max-new-tokens 128 --repeats 1 \
    --out "$ROOT/factors" 2>&1 | tee "$ROOT/factors.log"
}

holdout() {
  python -u evaluate.py --split holdout --tasks synthetic_kv_retrieval --lengths 32768 --synthetic-samples 25 \
    --candidate dense --candidate fp_v1:0.08 --candidate mean_native:0.08 --candidate fp_row_norm:0.08 \
    --candidate mean_balanced:0.08 --max-new-tokens 128 --repeats 1 --profile --out "$ROOT/holdout" \
    2>&1 | tee "$ROOT/holdout.log"
}

followup_data() {
  local data="$ROOT/followup_inputs"
  local models="$EXP1_STORAGE_ROOT/huggingface/models"
  local modelscope="https://www.modelscope.cn/models/Qwen"
  mkdir -p "$data" datasets/longbench_v2 datasets/loghub
  curl -sSL -o datasets/loghub/HDFS_2k.log \
    https://raw.githubusercontent.com/logpai/loghub/dd61d0952749ee7963bde24220d1be5ede023033/HDFS/HDFS_2k.log
  python -u prepare_log_copy.py --log datasets/loghub/HDFS_2k.log --out "$data/logs.pt"
  HF_HUB_OFFLINE=0 python -u evaluate.py --split ruler --lengths 32768 --ruler-samples 5 \
    --tasks ruler_niah_mk_1 ruler_niah_mk_2 ruler_niah_mk_3 ruler_niah_mq --candidate dense \
    --max-new-tokens 128 --prepare-only --out "$data/ruler"
  aria2c -x 8 -s 8 -k 16M -c --file-allocation=none --console-log-level=warn -d datasets/longbench_v2 -o data.json \
    "https://hf-mirror.com/datasets/THUDM/LongBench-v2/resolve/2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9/data.json"
  python -u evaluate.py --split modern --tasks longbench_v2 --longbench-v2-samples 15 --longbench-v2-min-tokens 8192 \
    --candidate dense --max-new-tokens 128 --prepare-only --out "$data/longbench"
  for name in Qwen2.5-0.5B-Instruct Qwen2.5-1.5B-Instruct; do
    mkdir -p "$models/$name"
    for file in config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json model.safetensors; do
      aria2c -x 8 -s 8 -k 64M -c --file-allocation=none --console-log-level=warn -d "$models/$name" -o "$file" \
        "$modelscope/$name/resolve/master/$file"
    done
    (cd "$models/$name" && sha256sum model.safetensors tokenizer.json) | tee "$data/$name.sha256"
  done
}

margin() {
  local data="$ROOT/followup_inputs"
  python -u pointer_followup_study.py margin --inputs "$data/ruler/inputs.pt" "$data/longbench/inputs.pt" "$data/logs.pt" \
    --out "$ROOT/margin" 2>&1 | tee "$ROOT/margin.log"
}

quant() {
  python -u pointer_followup_study.py quant --inputs "$ROOT/holdout/inputs.pt" --out "$ROOT/quant" 2>&1 | tee "$ROOT/quant.log"
}

small() {
  local models="$EXP1_STORAGE_ROOT/huggingface/models"
  for name in Qwen2.5-0.5B-Instruct Qwen2.5-1.5B-Instruct; do
    python -u evaluate.py --model "$models/$name" --inputs "$ROOT/holdout/inputs.pt" --split holdout \
      --tasks synthetic_kv_retrieval --candidate dense --max-new-tokens 128 --repeats 1 --out "$ROOT/small/$name" \
      2>&1 | tee "$ROOT/small_$name.log"
  done
}

followup_summary() {
  python -u summarize_followup.py "$ROOT"
  tar --exclude='*.pt' -czf "$ROOT.tar.gz" -C "$(dirname "$ROOT")" "$(basename "$ROOT")"
  ls -lh "$ROOT.tar.gz"
}

batch2_data() {
  local data="$ROOT/batch2_inputs"
  mkdir -p "$data" datasets/c4
  python -m pip install --index-url https://mirrors.aliyun.com/pypi/simple bitsandbytes==0.45.5
  aria2c -x 8 -s 8 -c --file-allocation=none --console-log-level=warn -d datasets/c4 -o c4-validation.00000-of-00008.json.gz \
    "https://hf-mirror.com/datasets/allenai/c4/resolve/main/en/c4-validation.00000-of-00008.json.gz"
  sha256sum datasets/c4/c4-validation.00000-of-00008.json.gz | tee "$data/c4.sha256"
  python - "$ROOT" <<'PY'
import sys, torch
root = sys.argv[1]
combined = []
for path in ("holdout/inputs.pt", "followup_inputs/logs.pt", "followup_inputs/ruler/inputs.pt"):
    for sample in torch.load(f"{root}/{path}", weights_only=False):
        combined.append({**sample, "split": "holdout"})
torch.save(combined, f"{root}/batch2_inputs/combined.pt")
print("combined prompts", len(combined))
PY
  python -u prepare_competitor_sweep.py --inputs "$ROOT/holdout/inputs.pt" --out "$data/competitors.pt"
}

temperature() {
  python -u evaluate.py --inputs "$ROOT/batch2_inputs/combined.pt" --split holdout --tasks synthetic_kv_retrieval \
    --candidate dense --candidate dense_temp_decode:0.05 --candidate dense_temp_decode:0.1 \
    --candidate dense_temp_all:0.05 --candidate dense_temp_all:0.1 \
    --max-new-tokens 128 --quality-only --out "$ROOT/temperature" 2>&1 | tee "$ROOT/temperature.log"
}

competitors() {
  python -u evaluate.py --inputs "$ROOT/batch2_inputs/competitors.pt" --split holdout --tasks synthetic_kv_retrieval \
    --candidate dense --candidate fp_v1:0.08 --max-new-tokens 128 --quality-only --out "$ROOT/competitors_7b" \
    2>&1 | tee "$ROOT/competitors_7b.log"
  python -u evaluate.py --model "$EXP1_STORAGE_ROOT/huggingface/models/Qwen2.5-0.5B-Instruct" \
    --inputs "$ROOT/batch2_inputs/competitors.pt" --split holdout --tasks synthetic_kv_retrieval --candidate dense \
    --max-new-tokens 128 --quality-only --out "$ROOT/competitors_0p5b" 2>&1 | tee "$ROOT/competitors_0p5b.log"
}

weights() {
  local calibration=datasets/c4/c4-validation.00000-of-00008.json.gz
  mkdir -p "$ROOT/ppl"
  python -u measure_ppl.py --calibration "$calibration" --out "$ROOT/ppl/bf16_none.json"
  for spec in ${WEIGHT_SPECS:-int8:none nf4:none bf16:unstructured bf16:2:4}; do
    local format="${spec%%:*}" wanda="${spec#*:}" name="${spec//:/_}"
    python -u measure_ppl.py --weight-format "$format" --wanda "$wanda" --calibration "$calibration" \
      --out "$ROOT/ppl/$name.json"
    python -u evaluate.py --inputs "$ROOT/batch2_inputs/combined.pt" --split holdout --tasks synthetic_kv_retrieval \
      --weight-format "$format" --wanda "$wanda" --calibration "$calibration" --candidate dense \
      --max-new-tokens 128 --quality-only --out "$ROOT/weights/$name" 2>&1 | tee "$ROOT/weights_$name.log"
  done
}

batch2_summary() {
  python -u summarize_batch2.py "$ROOT"
  tar --exclude='*.pt' -czf "$ROOT.tar.gz" -C "$(dirname "$ROOT")" "$(basename "$ROOT")"
  ls -lh "$ROOT.tar.gz"
}

batch3_data() {
  local data="$ROOT/batch3_inputs"
  local loghub=https://raw.githubusercontent.com/logpai/loghub/dd61d0952749ee7963bde24220d1be5ede023033
  mkdir -p "$data"
  [ -f datasets/loghub/OpenStack_2k.log ] || curl -sSL -o datasets/loghub/OpenStack_2k.log "$loghub/OpenStack/OpenStack_2k.log"
  sha256sum datasets/loghub/*.log | tee "$data/logs.sha256"
  python -u prepare_copy_tasks.py --source hdfs --path datasets/loghub/HDFS_2k.log --out "$data/hdfs.pt"
  python -u prepare_copy_tasks.py --source openstack --path datasets/loghub/OpenStack_2k.log --out "$data/openstack.pt"
  python -u prepare_copy_tasks.py --source manifest --samples 50 \
    --path "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" --out "$data/manifest.pt"
  python - "$data" <<'PY'
import sys, torch
data = sys.argv[1]
combined = [s for name in ("hdfs", "openstack", "manifest") for s in torch.load(f"{data}/{name}.pt", weights_only=False)]
torch.save(combined, f"{data}/copy.pt")
print("copy prompts", len(combined))
PY
}

batch3() {
  local inputs="$ROOT/batch3_inputs/copy.pt" calibration=datasets/c4/c4-validation.00000-of-00008.json.gz
  python -u evaluate.py --inputs "$inputs" --split holdout --tasks synthetic_kv_retrieval --candidate dense \
    --candidate fp_v1:0.08 --quality-only --out "$ROOT/batch3/bf16" 2>&1 | tee "$ROOT/batch3_bf16.log"
  for spec in nf4:none bf16:unstructured; do
    local format="${spec%%:*}" wanda="${spec#*:}" name="${spec//:/_}"
    python -u evaluate.py --inputs "$inputs" --split holdout --tasks synthetic_kv_retrieval \
      --weight-format "$format" --wanda "$wanda" --calibration "$calibration" --candidate dense \
      --quality-only --out "$ROOT/batch3/$name" 2>&1 | tee "$ROOT/batch3_$name.log"
  done
}

batch3_summary() {
  python -u summarize_batch3.py "$ROOT"
  tar --exclude='*.pt' -czf "$ROOT.tar.gz" -C "$(dirname "$ROOT")" "$(basename "$ROOT")"
  ls -lh "$ROOT.tar.gz"
}

mechanism() {
  python -u copy_mechanism.py --inputs "$ROOT/batch3_inputs/copy.pt" \
    --samples 'copy-hdfs-[0-9]+|copy-manifest-([0-9]|1[0-9]|2[0-4])' --out "$ROOT/mechanism" 2>&1 | tee "$ROOT/mechanism.log"
  python -u summarize_mechanism.py "$ROOT/mechanism"
}

mechanism_phase() {
  python -u copy_mechanism.py --inputs "$ROOT/batch3_inputs/copy.pt"     --samples 'copy-hdfs-[0-9]+|copy-manifest-([0-9]|1[0-9]|2[0-4])' --out "$ROOT/mechanism_phase"     --conditions bf16 nf4sim:all@prefill nf4sim:all@decode wanda:all@prefill wanda:all@decode 2>&1 | tee "$ROOT/mechanism_phase.log"
  python -u summarize_mechanism.py "$ROOT/mechanism_phase"
  python -u attention_check.py "$ROOT/mechanism_phase/targets.jsonl" | tee "$ROOT/mechanism_phase/attention_check.txt"
  python -u attention_check.py "$ROOT/mechanism/targets.jsonl" | tee "$ROOT/mechanism/attention_check.txt"
}

repair() {
  python -u copy_mechanism.py --inputs "$ROOT/batch3_inputs/copy.pt"     --samples 'copy-hdfs-[0-9]+|copy-manifest-([0-9]|1[0-9]|2[0-4])' --out "$ROOT/repair"     --conditions bf16 nf4sim:!attn nf4sim:!attnL0-18 nf4sim:!L0-9 wanda:!attn wanda:!attnL0-18 wanda:!L10-18     2>&1 | tee "$ROOT/repair.log"
  python -u summarize_mechanism.py "$ROOT/repair"
  python -u attention_check.py "$ROOT/repair/targets.jsonl" | tee "$ROOT/repair/attention_check.txt"
  python -u measure_ppl.py --weight-format nf4sim_mlp --calibration datasets/c4/c4-validation.00000-of-00008.json.gz     --out "$ROOT/ppl/nf4sim_mlp_none.json"
  copy_eval "$ROOT/batch4_inputs/formats.pt" "$ROOT/batch4_formats" nf4sim_mlp:none
  python -u summarize_batch3.py "$ROOT" batch4_formats batch4_inputs/formats.pt BATCH4_FORMATS.md
  tar --exclude='*.pt' -czf "$ROOT.tar.gz" -C "$(dirname "$ROOT")" "$(basename "$ROOT")"
}

batch4_data() {
  local data="$ROOT/batch4_inputs" purelib
  purelib="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  mkdir -p "$data"
  python -u prepare_copy_tasks.py --source hdfs --path datasets/loghub/HDFS_2k.log --prompt-budget 7936 \
    --id-prefix len8k --out "$data/len8k.pt"
  python -u prepare_copy_tasks.py --source hdfs --path datasets/loghub/HDFS_2k.log --prompt-budget 16128 \
    --id-prefix len16k --out "$data/len16k.pt"
  python -u prepare_copy_tasks.py --source manifest_pf --samples 50 --path "$purelib" --out "$data/path_first.pt"
  python - "$data" "$ROOT/batch3_inputs/copy.pt" <<'PY'
import re, sys, torch
data, copy = sys.argv[1:3]
torch.save([s for name in ("len8k", "len16k") for s in torch.load(f"{data}/{name}.pt", weights_only=False)], f"{data}/length.pt")
keep = r"copy-hdfs-[0-9]+|copy-manifest-([0-9]|1[0-9]|2[0-4])"
torch.save([s for s in torch.load(copy, weights_only=False) if re.fullmatch(keep, s["sample_id"])], f"{data}/formats.pt")
print("length", 50, "formats", 50)
PY
}

copy_eval() {
  local inputs="$1" out="$2" spec="$3" format wanda name
  format="${spec%%:*}" wanda="${spec#*:}" name="${spec//:/_}"
  python -u evaluate.py --inputs "$inputs" --split holdout --tasks synthetic_kv_retrieval --weight-format "$format" \
    --wanda "$wanda" --candidate dense --quality-only --out "$out/$name" 2>&1 | tee "${out}_$name.log"
}

batch4() {
  local data="$ROOT/batch4_inputs" spec
  for spec in bf16:none nf4:none bf16:unstructured; do
    copy_eval "$data/length.pt" "$ROOT/batch4_length" "$spec"
    copy_eval "$data/path_first.pt" "$ROOT/batch4_path_first" "$spec"
  done
  mkdir -p "$ROOT/ppl"
  for spec in nf4sim rtn4 rtn3; do
    python -u measure_ppl.py --weight-format "$spec" --calibration datasets/c4/c4-validation.00000-of-00008.json.gz --out "$ROOT/ppl/${spec}_none.json"
    copy_eval "$data/formats.pt" "$ROOT/batch4_formats" "$spec:none"
  done
  ln -sfn ../batch3/bf16 "$ROOT/batch4_formats/bf16"
  ln -sfn ../batch3/nf4_none "$ROOT/batch4_formats/nf4_none"
}

batch4_summary() {
  python -u summarize_batch3.py "$ROOT" batch4_length batch4_inputs/length.pt BATCH4_LENGTH.md
  python -u summarize_batch3.py "$ROOT" batch4_path_first batch4_inputs/path_first.pt BATCH4_PATH_FIRST.md
  python -u summarize_batch3.py "$ROOT" batch4_formats batch4_inputs/formats.pt BATCH4_FORMATS.md
  cat "$ROOT"/ppl/*.json
  tar --exclude='*.pt' -czf "$ROOT.tar.gz" -C "$(dirname "$ROOT")" "$(basename "$ROOT")"
  ls -lh "$ROOT.tar.gz"
}

predecode() {
  python -u predecode_probe.py --inputs "$ROOT/batch3_inputs/copy.pt"     --samples 'copy-hdfs-[0-9]+|copy-manifest-([0-9]|1[0-9]|2[0-4])' --out "$ROOT/predecode" 2>&1 | tee "$ROOT/predecode.log"
  python -u summarize_predecode.py "$ROOT"
}

rtn4_mechanism() {
  python -u copy_mechanism.py --inputs "$ROOT/batch3_inputs/copy.pt"     --samples 'copy-hdfs-[0-9]+|copy-manifest-([0-9]|1[0-9]|2[0-4])' --out "$ROOT/mechanism_rtn4"     --conditions bf16 rtn4:all rtn4:all@prefill rtn4:all@decode rtn4:L0-9 rtn4:L10-18 rtn4:L19-27 rtn4:attn rtn4:mlp     2>&1 | tee "$ROOT/mechanism_rtn4.log"
  python -u summarize_mechanism.py "$ROOT/mechanism_rtn4"
  python -u attention_check.py "$ROOT/mechanism_rtn4/targets.jsonl" | tee "$ROOT/mechanism_rtn4/attention_check.txt"
  python -u summarize_predecode.py "$ROOT"
}

gap() {
  python -u copy_mechanism.py --inputs "$ROOT/batch4_inputs/length.pt" "$ROOT/batch3_inputs/copy.pt"     --samples 'len8k-hdfs-([0-9]|1[0-6])|len16k-hdfs-([0-9]|1[0-6])|copy-hdfs-([0-9]|1[0-5])' --out "$ROOT/gap"     --conditions bf16 nf4sim:all rtn4:all wanda:all 2>&1 | tee "$ROOT/gap.log"
  python -u summarize_gap.py "$ROOT/gap"
}

holdout_gap() {
  python -u copy_mechanism.py --inputs "$ROOT/batch4_inputs/length.pt" "$ROOT/batch3_inputs/copy.pt"     "$ROOT/batch4_inputs/path_first.pt"     --samples 'len8k-hdfs-(1[7-9]|2[0-4])|len16k-hdfs-(1[7-9]|2[0-4])|copy-hdfs-(1[6-9]|2[0-4])|copy-manifest_pf-([0-9]|1[0-9]|2[0-4])'     --out "$ROOT/holdout_gap" --conditions bf16 nf4sim:all rtn4:all wanda:all fp_v1:all 2>&1 | tee "$ROOT/holdout_gap.log"
  python -u summarize_gap.py "$ROOT/holdout_gap" L23H11,L22H4,L22H7,L22H3,L22H1,L23H12,L22H25,L21H25
}

noise() {
  local heads=L23H11,L22H4,L22H7,L22H3,L22H1,L23H12,L22H25,L21H25
  python -u copy_mechanism.py --inputs "$ROOT/batch4_inputs/length.pt" "$ROOT/batch3_inputs/copy.pt"     --samples 'len8k-hdfs-([0-9]|1[0-6])|len16k-hdfs-([0-9]|1[0-6])|copy-hdfs-([0-9]|1[0-5])' --out "$ROOT/noise"     --conditions bf16 nf4sim:all nf4sim:!attn nf4sim:!qk rtn4:all wanda:all fp_v1:all --noise-heads "$heads"     2>&1 | tee "$ROOT/noise.log"
  python -u summarize_noise.py "$ROOT/noise" "$heads"
}

repair_free() {
  local spec
  for spec in nf4sim:none nf4sim_mlp:none nf4sim_noqk:none rtn4:none rtn4_mlp:none; do
    copy_eval "$ROOT/batch4_inputs/length.pt" "$ROOT/batch4_length" "$spec"
  done
  for spec in nf4sim_noqk:none rtn4_mlp:none; do
    copy_eval "$ROOT/batch4_inputs/formats.pt" "$ROOT/batch4_formats" "$spec"
  done
  for spec in nf4sim_noqk rtn4_mlp; do
    python -u measure_ppl.py --weight-format "$spec" --calibration datasets/c4/c4-validation.00000-of-00008.json.gz       --out "$ROOT/ppl/${spec}_none.json"
  done
  python -u summarize_repair.py "$ROOT"
}

noise_sources() {
  local heads=L23H11,L22H4,L22H7,L22H3,L22H1,L23H12,L22H25,L21H25
  python -u copy_mechanism.py --inputs "$ROOT/batch4_inputs/length.pt" "$ROOT/batch3_inputs/copy.pt"     --samples 'len8k-hdfs-([0-9]|1[0-6])|len16k-hdfs-([0-9]|1[0-6])|copy-hdfs-([0-9]|1[0-5])' --out "$ROOT/noise_sources"     --conditions bf16 rtn4:all rtn4:mlp rtn4:attn rtn4:!qk rtn4:L0-9 rtn4:L10-18 rtn4:L19-27 nf4sim:mlp nf4sim:attn     --noise-heads "$heads" 2>&1 | tee "$ROOT/noise_sources.log"
  python -u summarize_noise.py "$ROOT/noise_sources" "$heads"
}

budget() {
  local heads=L23H11,L22H4,L22H7,L22H3,L22H1,L23H12,L22H25,L21H25 spec
  for spec in nf4sim_mlp03 nf4sim_rand nf4sim_err rtn5 rtn6; do
    python -u measure_ppl.py --weight-format "$spec" --calibration datasets/c4/c4-validation.00000-of-00008.json.gz       --out "$ROOT/ppl/${spec}_none.json"
    copy_eval "$ROOT/batch4_inputs/length.pt" "$ROOT/batch4_length" "$spec:none"
    copy_eval "$ROOT/batch4_inputs/formats.pt" "$ROOT/batch4_formats" "$spec:none"
  done
  python -u copy_mechanism.py --inputs "$ROOT/batch4_inputs/length.pt" "$ROOT/batch3_inputs/copy.pt"     --samples 'len8k-hdfs-([0-9]|1[0-6])|len16k-hdfs-([0-9]|1[0-6])|copy-hdfs-([0-9]|1[0-5])' --out "$ROOT/noise_budget"     --conditions bf16 fmt=nf4sim_mlp03:all fmt=nf4sim_rand:all fmt=nf4sim_err:all fmt=rtn5:all fmt=rtn6:all     --noise-heads "$heads" 2>&1 | tee "$ROOT/noise_budget.log"
  python -u summarize_noise.py "$ROOT/noise_budget" "$heads"
  python -u summarize_repair.py "$ROOT"
}

summary() {
  python -u summarize_pointer_study.py "$ROOT"
  tar --exclude='*.pt' -czf "$ROOT.tar.gz" -C "$(dirname "$ROOT")" "$(basename "$ROOT")"
  ls -lh "$ROOT.tar.gz"
}

case "${1:-}" in
  check|prepare|smoke|alpha|length|patch|factors|holdout|summary|followup_data|margin|quant|small|followup_summary|batch2_data|temperature|competitors|weights|batch2_summary|batch3_data|batch3|batch3_summary|mechanism|batch4_data|batch4|batch4_summary|mechanism_phase|repair|predecode|rtn4_mechanism|gap|holdout_gap|noise|repair_free|noise_sources|budget) "$1" ;;
  all) check; prepare; smoke; alpha; patch; length; factors; holdout; summary ;;
  *) echo "usage: bash run_pointer_study.sh {check|prepare|smoke|alpha|length|patch|factors|holdout|summary|all}" >&2; exit 2 ;;
esac
