# Final 100+ Full-vs-V1 benchmark audit

Date: 2026-09-21
Model: `Qwen/Qwen2.5-7B-Instruct`
Sparse configuration: FlashPrefill V1, `alpha=0.08`, block size 128

两个 benchmark 分别报告，不把 LongBench accuracy 与 BFCL episode success
混成总分。质量按相同样本配对；LongBench 延迟按相同固定 prompt 配对；
BFCL 的纯 prefill 速度只比较完整 prompt hash 相同的动态 step。

## Artifact integrity and independent rescoring

- 两个运行的 `metadata.status` 都是 `complete`。
- 原始归档保存在 `results/archives/final_100plus_20260921.tar.gz`，SHA256 为
  `e8ce375f1b43dc5f6f7bda753c411a75a3aa5b900ea16f92519ae3e710f25718`。
- LongBench 保存 116 个固定输入、232 个生成结果和 696 行同步计时。
  从原始生成文本独立重跑 scorer 后，0/232 个判定发生变化。
- BFCL 保存 100 个固定 case 和 200 条完整轨迹。用归档中固定的
  `bfcl-eval==2025.12.17` 官方 `multi_turn_checker` 重放后，0/200 个
  success 判定和 0/200 个 error type 发生变化。
- LongBench v2 `data.json` SHA256 为
  `15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2`；
  BFCL wheel SHA256 为
  `8555bc9407a56682ceb7d969e87eb724f6b679deb0ef05114d9c6e786406b103`。
- 每条 LongBench generation 的 input hash 都与固定输入一致；每条 BFCL
  source-row hash 和 ground truth 都与固定 wheel 一致。
- LongBench 中断恢复保留原来 103 个完整样本对的精确前缀：206 条生成和
  618 条完整计时进入最终结果，4 条未完成样本的计时单独隔离，未重复计入。

审计可由 `analyze_final_benchmarks.py` 从归档解压目录重新生成。完整本地结果在
`results/audits/final_100plus_20260921/results/analysis/`，包括独立 rescore、
paired outcomes、子组和长度分桶 CSV。

## Primary paired results

| Benchmark | n | Full | V1 | Δ V1−Full | Full-only / V1-only | Prefill Full | Prefill V1 | Paired speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LongBench v2 native 8–32K | 116 | 42/116 (36.21) | 45/116 (38.79) | +2.59 pp | 4 / 7 | 2557.1 ms | 2222.4 ms | 1.153× |
| BFCL V4 `multi_turn_base` | 100 | 22/100 | 21/100 | −1.00 pp | 2 / 1 | 5230.3 ms* | 4870.6 ms* | 1.011×** |

\* BFCL 此处是每个 episode 的 prefill 累加值中位数；Full/V1 轨迹会分叉，
所以它只能描述实际 workload，不能作为纯 kernel 配对比较。
\** 纯速度仅使用 V1 933 个生成 step 中 266 个完整 prompt 相同的配对 step，
覆盖率 28.5%。

配对不确定性：

- LongBench quality Δ 为 +2.59 pp，paired bootstrap 95% CI
  `[-2.59, +7.76]` pp，exact McNemar `p=0.549`。
- LongBench prefill 中位 paired speedup 为 1.153×，按样本 cluster bootstrap
  95% CI `[1.134, 1.188]`×。
- BFCL quality Δ 为 −1.00 pp，paired bootstrap 95% CI
  `[-5.00, +2.00]` pp，exact McNemar `p=1.000`。
- BFCL identical-prompt prefill 中位 speedup 为 1.011×，按 episode cluster
  bootstrap 95% CI `[1.005, 1.014]`×。

这些区间是固定面板上的描述性 paired bootstrap，不是预注册的 non-inferiority
检验。LongBench 的 +2.59 pp 只是净多答对 3 题，不能解释为稀疏注意力提高质量；
BFCL 也只有 3 个 discordant case，不能据此证明严格等价。

## LongBench sub-evaluations

| Dimension | Group | n | Full | V1 | Δ pp | Full-only / V1-only | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| domain | Code Repository Understanding | 3 | 33.33 | 66.67 | +33.33 | 0 / 1 | 1.27× |
| domain | Long In-context Learning | 7 | 14.29 | 42.86 | +28.57 | 0 / 2 | 1.20× |
| domain | Long Structured Data Understanding | 1 | 0.00 | 0.00 | +0.00 | 0 / 0 | 1.13× |
| domain | Long-dialogue History Understanding | 12 | 16.67 | 8.33 | −8.33 | 1 / 0 | 1.21× |
| domain | Multi-Document QA | 32 | 25.00 | 34.38 | +9.38 | 0 / 3 | 1.14× |
| domain | Single-Document QA | 61 | 49.18 | 45.90 | −3.28 | 3 / 1 | 1.14× |
| difficulty | easy | 42 | 45.24 | 42.86 | −2.38 | 3 / 2 | 1.16× |
| difficulty | hard | 74 | 31.08 | 36.49 | +5.41 | 1 / 5 | 1.15× |
| length | 8–16K | 27 | 40.74 | 40.74 | +0.00 | 0 / 0 | 1.085× |
| length | 16–24K | 46 | 34.78 | 39.13 | +4.35 | 1 / 3 | 1.139× |
| length | 24–32K | 43 | 34.88 | 37.21 | +2.33 | 3 / 4 | 1.222× |

最稳定的结构信号是速度随长度增加：从 8–16K 的 1.085× 上升到 24–32K
的 1.222×。质量子组中的大正负差多数由很少的 flip 构成；尤其 n=1/3/7 的
domain 只用于定位样本，不能形成方法结论。完整 sub-domain 表保存在
`longbench_subgroups.csv`。

独立 profile 仅运行了一个样本：V1 的 exact-pair ratio 为 18.4%，因此它只能
说明该 profile 样本的执行密度，不能当作 116 个样本的平均密度。两种方法记录的
peak allocated memory 都约 20.14 GiB。

## BFCL sub-evaluations

| Dimension | Group | n | Full | V1 | Δ pp | Full-only / V1-only | Matched steps | Speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| turns | 1–2 | 20 | 25.00 | 30.00 | +5.00 | 0 / 1 | 44 | 1.013× |
| turns | 3–4 | 47 | 19.15 | 17.02 | −2.13 | 1 / 0 | 137 | 1.012× |
| turns | 5–7 | 33 | 24.24 | 21.21 | −3.03 | 1 / 0 | 85 | 1.010× |
| APIs | cross API | 79 | 18.99 | 20.25 | +1.27 | 0 / 1 | 206 | 1.014× |
| APIs | single API | 21 | 33.33 | 23.81 | −9.52 | 2 / 0 | 60 | 0.997× |

`single API` 的 −9.52 pp 实际只有 2 个 Full-only case；20 个 class-signature
子组都很小，仍只作为错例索引。完整 signature/class-presence 表保存在
`bfcl_subgroups.csv`。

轨迹诊断：

| Method | Generated steps | Hit generation cap | Decode errors | Context overflow | Force-terminated episodes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full | 920 | 7 | 1 | 0 | 1 |
| V1 | 933 | 7 | 0 | 0 | 2 |

相同 prompt 的配对 step 全部短于 8K：`<4K` 为 36 个、speedup 0.994×；
`4–8K` 为 230 个、speedup 1.013×。因此 Agent panel 的结论不是“V1 kernel
没有长上下文收益”，而是当前 BFCL base 轨迹主要处于 V1 固定路由开销尚未摊薄的
长度区间；它测到了 Agent 质量，却不是 16–32K prefill 加速的有力测试。

## Interpretation relative to the earlier small tests

早期 32K 自定义 calibration 只有 8 个输入，而且专门强调 multi-key/multi-query
绑定失败；它揭示了真实 failure mode，但不能代表一般长文本质量。现在的 116 个
LongBench 固定问题把抽样噪声明显压低，并显示 V1 在更广任务上的总体差异是净 3 题，
不是早期自定义任务中的大幅下降。两者并不矛盾：当前结果支持“绑定任务是 V1 的
局部弱点”，不支持把该弱点外推成一般长上下文崩溃。

另一方面，116 个问题仍没有证明质量提升或严格 non-inferiority；BFCL 即使有
100 个 episode，也因只有 3 个 paired flip 而对小差异缺乏统计分辨率。后续 block
结构研究应直接解释和找回 routing miss，再用保存的同一批 raw outputs/scorer 口径
做验证，而不是继续根据小子组分数调参。
