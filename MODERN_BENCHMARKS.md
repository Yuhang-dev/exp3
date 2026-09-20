# Modern Full-vs-V1 benchmark suite

This stage replaces the retrieval-heavy RULER diagnostic with two newer and
complementary evaluations while keeping the model, decoding, and V1 routing
fixed.

## Why these two benchmarks

- [LongBench v2](https://github.com/THUDM/LongBench) evaluates 503 realistic
  multiple-choice questions across six domains. Contexts range from 8K to 2M
  words and require comprehension/reasoning rather than only locating a planted
  string. Exp3 uses the official zero-shot prompt and exact A/B/C/D extractor.
- [BFCL V4](https://gorilla.cs.berkeley.edu/leaderboard.html) evaluates function
  calling. Exp3 uses the official 200-row `multi_turn_long_context` data,
  simulator backends, possible answers, and `multi_turn_checker` from
  `bfcl-eval==2025.12.17`.

The two results answer different questions. LongBench v2 is a fixed-prompt
quality comparison. BFCL is an end-to-end agent comparison: a changed tool call
can change backend state, tool output, and every later prompt.

## Pinned download

The preparation script downloads but does not install BFCL into the Torch
environment. It verifies the LongBench data file and BFCL wheel SHA256 before
extracting the wheel.
The installed Transformers build requires Hugging Face Hub below 1.0. Keep the
experiment environment on the pinned compatible client:

```bash
python -m pip install --no-deps --force-reinstall \
  --index-url https://pypi.org/simple \
  -r requirements-modern.txt
```

The explicit index is intentional: the AutoDL Aliyun mirror did not expose
the pinned wheel when this suite was first run.

For the AutoDL host it defaults only the Hugging Face transport endpoint to
`https://hf-mirror.com`; an explicit `HF_ENDPOINT` overrides this, and the
repository revision plus SHA256 remain unchanged.

```bash
cd /root/autodl-tmp/exp3
source ./env.sh
bash prepare_modern_benchmarks.sh
```

Pinned artifacts:

| Artifact | Pin | Expected SHA256 |
| --- | --- | --- |
| `THUDM/LongBench-v2/data.json` | HF revision `2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9` | `15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2` |
| `bfcl-eval` wheel | PyPI `2025.12.17` | `8555bc9407a56682ceb7d969e87eb724f6b679deb0ef05114d9c6e786406b103` |

## LongBench v2 native-32K pilot

Qwen2.5-7B-Instruct has a native 32K context in this project. Exp3 first
pre-filters the official `Short` category, then tokenizes the complete official
zero-shot prompt with the Qwen chat template. It keeps only prompts in
`[16,384, 32,640]` tokens, performs no middle truncation, and deterministically
round-robins across the six domains. The default pilot has 24 rows. It is
therefore labeled **LongBench v2 native-32K subset**, not the 503-row official
overall score.
The prompt and answer extractor are official, while generation uses Exp3's
deterministic greedy decoding for a paired causal comparison (the upstream
script requests temperature `0.1`).

Prepare and inspect the fixed input IDs without loading the model:

```bash
bash run_modern_benchmarks.sh prepare
```

The same command also writes the fixed 20-case BFCL selection and ground-truth
manifest without loading the model or touching CUDA.
Both GPU wrappers then reuse those exact LongBench token IDs and BFCL case IDs;
they do not silently resample.

Run Full attention and V1 at `alpha=0.08`:

```bash
bash run_modern_benchmarks.sh longbench
```

The run saves exact input IDs and hashes, source row/file hashes, raw generation
before scoring, official parsed choice, per-domain scores, three synchronized
prefill repeats, and an independent first-sample profile per configuration.

## BFCL V4 multi-turn long-context pilot

Run the deterministic 20-row, involved-class-stratified pilot:

```bash
bash run_modern_benchmarks.sh agent
```

If an Agent run is interrupted after complete episode records have already been
flushed, resume the same output directory with:

```bash
bash run_modern_benchmarks.sh agent-resume
```

Resume is accepted only when `generations.jsonl` is an exact prefix of the
fixed episode/configuration schedule. Completed episodes are not regenerated.
The failed attempt's metadata and timing file are archived under
`resume_attempts/`; timing rows from the incomplete episode are removed before
that episode is restarted. The wrapper uses the already populated local model
cache in Hub offline mode so a resume cannot stall on a metadata HEAD request.
A syntactically valid tool-call JSON object that is
missing `name` or `arguments` remains a model decode error: its raw text is
preserved in chat history, but no arguments are invented and no tool is run.

Run all 200 official rows by calling the entry point directly:

```bash
python -u bfcl_v4_agent.py \
  --bfcl-root third_party/bfcl_eval_2025_12_17 \
  --bfcl-wheel third_party/downloads/bfcl_eval-2025.12.17-py3-none-any.whl \
  --samples 200 \
  --methods dense fp_v1 \
  --alpha 0.08 \
  --repeats 3 \
  --max-new-tokens 512 \
  --out results/bfcl_v4_long_context_full200
```

Qwen2.5-7B-Instruct is not an exact built-in model entry in the pinned BFCL
package. The project adapter mirrors BFCL's Qwen `<tool_call>` and
`<tool_response>` formatting and uses the official simulator and state checker,
but the result is described as an Exp3 paired experiment rather than an
official leaderboard submission.

Each method starts from the same source episode and initial state. Raw responses
are flushed before scoring. The artifacts retain parsed calls, simulator
returns, official checker details, generated token IDs, and every dynamic
prompt's token count and SHA256. Prompts that cannot reserve the requested
generation tokens inside native 32K are not truncated; the episode is recorded
as a context-overflow failure.

BFCL latency has two reported meanings:

1. Episode workload: sum of the per-step median prefill times along that
   method's own trajectory.
2. Pure Full/V1 speedup: only steps whose complete prompt-token SHA256 is
   identical between Full and V1.

The second rule prevents a shorter failed trajectory from being presented as a
kernel speedup.

Re-run only the pinned official checker from saved trajectories, without model
loading or generation:

```bash
python -u bfcl_v4_rescore.py \
  results/bfcl_v4_long_context_pilot20 \
  --tag checker-audit
```

This creates an immutable derived snapshot under `rescoring/checker-audit/` and
records hashes of the original raw trajectories, stored scores, source package,
and metadata. It reparses saved raw model responses before invoking the state
checker, so both parser and checker changes can be audited without GPU work.

## Final 100+ paired evaluation

The 24/20 runs above are diagnostic pilots. The final comparison uses two
larger fixed panels:

- all 116 LongBench v2 `Short` rows whose complete chat-templated prompt fits
  `[8,192, 32,640]` tokens; no truncation or duplicate sampling;
- 100 deterministic BFCL V4 `multi_turn_base` rows, selected with the same
  involved-class-signature round-robin rule and evaluated with the official
  simulator, ground truth, and state checker.

`multi_turn_base` replaces `multi_turn_long_context` in the final Agent panel
because the latter gave Full only 2/20 successes. Increasing that floor-effect
pilot alone would estimate a failure-heavy regime more precisely without making
the Full/V1 capability comparison discriminating. The 20-row long-context run
remains a stress-test result.

Prepare both fixed manifests and run the panels separately:

```bash
bash run_final_benchmarks.sh prepare
bash run_final_benchmarks.sh longbench
bash run_final_benchmarks.sh agent
```

The final LongBench panel keeps 128 generation tokens. The BFCL base panel uses
1,024 tokens per step, three synchronized prefill repeats, a 20-step limit, and
native 32K overflow accounting. Interrupted runs resume only after validating
their saved generations and timings as an exact schedule prefix:

```bash
bash run_final_benchmarks.sh longbench-resume
bash run_final_benchmarks.sh agent-resume
```

## Output files

LongBench uses the normal Exp3 output contract and adds `domain_summary.csv`.
BFCL writes:

| File | Contents |
| --- | --- |
| `generations.jsonl` | scorer-independent trajectories, generated IDs, decoded calls, tool returns, prompt hashes, and timing repeats |
| `predictions.jsonl` | immutable copy plus official checker result/details |
| `quality.csv` | one official end-to-end success value per episode/config |
| `episodes.csv` | turns, steps, tool calls, token workload, summed prefill, and generation time |
| `timings.csv` | every synchronized step-level full-model prefill repeat |
| `official_results/` | BFCL-style raw result JSONL separated by configuration |
| `REPORT.md` | capability gap, trajectory-aware latency, regressions, and recoveries |
| `rescoring/<tag>/` | derived official-checker snapshot; no model or trajectory rerun |
| `resume_attempts/<n>/` | immutable failed-attempt metadata/timing snapshot and resume manifest |

Implementation checks are local and GPU measurements are run only on the remote
RTX 4090. Current execution status is recorded in `CHECKLIST.md`; the completed
pilot audit is in `MODERN_BENCHMARK_AUDIT.md`.
