# Mean-key block-size study

This study uses the existing Qwen2.5-7B / FlashPrefill V1 capture path. One
forward pass saves the actual post-RoPE Q/K and V at selected layers. The offline
script then reuses those tensors to vary query and key block sizes without
rerunning the model. It measures block-mass ranking, within-block Jensen gaps,
same-token-budget oracles, and two-/four-subblock mean proxies. It is a mechanism
study; its diagnostic runtime is not prefill latency.

The scripts use the existing /root/autodl-tmp/conda/envs/exp1 environment from
env.sh. Data directories below are new, so existing runs remain intact.

## First 4K run

In the remote Jupyter Terminal, after the updated exp3 code is present:

~~~bash
bash -e <<'SH'
cd /root/autodl-tmp/exp3
source ./env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
run_tag=$(date +%Y%m%d_%H%M%S)
capture_dir="results/block_proxy_capture_4k_${run_tag}"
study_dir="results/block_proxy_pilot_4k_${run_tag}"
python -u v1_block_diagnostics.py \
  --inputs results/quick/inputs.pt \
  --sample-id synthetic-quick-4096-0-multi_key \
  --out "$capture_dir" \
  --layers 0 \
  --capture-only \
  --save-q-layers 0 \
  --save-pre-rope-q-layers 0 \
  --save-layer-input-layers 0 \
  --save-v-layers 0
python -u block_proxy_study.py \
  --capture "$capture_dir" \
  --out "$study_dir" \
  --budget-tokens 512
python plot_block_proxy_study.py "$study_dir"
SH
~~~

The capture-only mode still runs the actual V1 prefill and saves its route, but
skips the expensive full-sequence dense oracle. It saves query_pre_rope.pt,
query_post_rope.pt, key_pre_rope.pt, key_post_rope.pt, value.pt, layer_input.pt,
original v1_route.pt, token IDs, token mapping, model/input metadata, and
per-file hashes. The three legacy summary CSV files
are headers only in this mode. Full Q/K/V remain on the remote disk for later
output-error and alternate-proxy analysis.
The Qwen2.5 model and tokenizer are already cached by the earlier remote runs;
offline mode avoids a Hugging Face metadata request. The timestamped directories
preserve previous attempts, and `bash -e` stops before analysis or plotting if
capture fails.

## 32K study on existing calibration prompts

~~~bash
bash -e <<'SH'
cd /root/autodl-tmp/exp3
source ./env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
run_tag=$(date +%Y%m%d_%H%M%S)
capture_dir="results/block_proxy_capture_32k_${run_tag}"
study_dir="results/block_proxy_mechanism_32k_${run_tag}"
python -u v1_block_diagnostics.py \
  --inputs results/calibration_sweep_32k/inputs.pt \
  --sample-id synthetic-calibration-32768-0-multi_key \
  --sample-id synthetic-calibration-32768-4-multi_query \
  --out "$capture_dir" \
  --layers 0 7 14 21 27 \
  --capture-only \
  --save-q-layers 0 7 14 21 27 \
  --save-pre-rope-q-layers 0 7 14 21 27 \
  --save-layer-input-layers 0 7 14 21 27 \
  --save-v-layers 0 7 14 21 27
python -u block_proxy_study.py \
  --capture "$capture_dir" \
  --out "$study_dir" \
  --q-block-sizes 64 128 256 \
  --k-block-sizes 32 64 128 256 \
  --budget-tokens 2048 \
  --rows-per-tile 16
python plot_block_proxy_study.py "$study_dir"
SH
~~~

Default sampled Q tiles are near 25%, 50%, 75%, and 90% of each sequence; 16
rows are sampled evenly within each tile. Default Q heads 0 7 14 21 represent
the four KV-head groups. Run a denser follow-up with --rows-per-tile 128 and
additional heads only for the layers, samples, and mechanisms that merit it.
A new output directory is needed for each analysis configuration.

The Q/K/V and layer-input files preserve every token and every head at the
listed layers. The offline study only samples four Q heads and 16 query rows
per selected tile, but that sampling does not discard the captured tensors.
The extra pre-RoPE Q and layer input add about 4.4 GiB for the two 32K samples
across five layers on Qwen2.5-7B. Those files allow later comparison of
pre/post-RoPE QK structure and probing of the representation entering each
layer. MLP intermediate activations and unlisted layers are not captured.

For a previously saved rich capture, the capture command can be skipped if
each chosen layer contains query_post_rope.pt and key_post_rope.pt. Pass its
existing root to --capture; no new model forward is required.

## Saved results

| File | Content |
| --- | --- |
| summary.csv | Four oracles and proxy variants at fixed exact-token budget, retained mass, fifth-percentile row mass, regret, and margin. |
| v1_reference.csv | Retained mass and actual variable routed budget of the saved V1 mask on sampled rows when Q block size is 128. |
| decomposition.csv | Token-to-block granularity, Q-tile sharing, and mean-proxy loss in separate columns. |
| block_stats.csv.gz | Every eligible block's true mass, raw-proxy/true ranks, K-block Jensen gap, Q-tile Jensen gap, subblock-recovered gap, logit spread, effective support. |
| row_block_stats.csv.gz | Each sampled query row × eligible K block: true probability, mean logit, gap, subblock gain, spread, effective support. |
| miss_profiles.jsonl.gz | Highest-mass blocks missed by raw mean: positions and centered token logits for the query row that values the block most. Join token_start with the capture's tokens.jsonl to inspect text. |
| metadata.json | Exact CLI configuration, source prompt hashes, output sizes and hashes. |
| figures/*.png | First-pass curves, Q/K size regret map, gap/rank scatter, missed-block logit profiles. |
| block_proxy_report.zip | Downloadable bundle of the analysis files and figures; full Q/K/V stays in the capture directory. |

The four retained-mass oracles in summary.csv are: per-row token top-k,
per-row full-block top-k, shared full-block top-k, and the tested proxy. Their
differences separate execution granularity, Q-tile sharing, and proxy-ranking
loss. All variants share a fixed protected token complement. The remote
candidate interval is aligned to the largest K block size, so its tokens and
the exact-token budget are identical across K-size comparisons for a given
query tile. The query tile position and protection complement can change when
Q size changes; interpret the two-dimensional Q/K plot with its tile samples.

mean_raw corresponds to aggregating exponentiated mean logits across rows,
as in V1's score form, then using a fixed top-k budget. exact_raw replaces
the K proxy with true block mass but retains raw query aggregation.
mean_oracle_z divides by the true dense row normalizer and is diagnostic.
mean_balanced_remote normalizes within remote candidates only, so it is a
mechanism control, not the existing mean_balanced implementation.
sub2_raw/sub4_raw replace one mean with two/four subblock means. These
scores are offline proposals; they are not yet runtime selector kernels.

The first figure averages retained mass across sampled configurations. Check
summary.csv by sample/layer/head before interpreting an average. The second
figure isolates Q/K size effects; its values are mass percentage points. The
scatter only illustrates distributions: a true-gap-based rank relation cannot
serve as independent validation of a cheap predictor.

All remote commands above remain to be run on the user's GPU. Locally, only
Python syntax was checked.
