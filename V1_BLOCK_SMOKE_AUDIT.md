# V1 block diagnostics 4K smoke 审计

审计日期：2026-09-20。该运行只验证 V1 诊断采集链路，不用于评价方法优劣。

## 阶段结论

**PASS。** 运行和样本状态均为 `complete`；所有张量形状、GQA 头映射、因果概率、V1 选块计数、保护块和同预算 oracle 均通过。14/14 个 manifest 文件的大小与 SHA256 一致，运行所用三个源码 hash 与提交 `296028d` 后的本地源码一致。

这是一条 3962-token、仅 layer 0 的 quick multi-key 输入。它证明采集实现可进入 32K 阶段，但不证明任何 block 结构、选择规律或质量结论可以泛化。

审计同时发现一个展示层命名歧义：旧 CSV 的 `mean_pool_jensen_gap_max=28.7128` 实际是“block/query-tile 行均值的最大值”，真正的单 query-row 最大 Jensen gap 是 `43.2190`，底层 `.pt` 两个张量一直分别正确保存。32K 前已将 CSV 拆成 `mean_pool_jensen_gap_tile_mean_max` 和 `mean_pool_jensen_gap_row_max`。

## 归档与环境

- 原始归档：`C:/Users/Yuhang/Downloads/v1_blocks_smoke_4k.tar.gz`
- SHA256：`058319745E8ED28CE3A69AF3092F0CE82E026A2848C759FAE20727CC7E446107`
- 大小：6,034,931 bytes
- 项目内原样副本：`results/archives/v1_blocks_smoke_4k_20260920_sha05831974.tar.gz`
- 解包审计目录：`results/audits/v1_blocks_smoke_4k_20260920_1344/results/v1_blocks_smoke_4k/`
- 模型：`Qwen/Qwen2.5-7B-Instruct`，revision `a09a35458c702b33eeacc393d103063234e8bc28`
- 环境：RTX 4090；Torch 2.6.0+cu124；Transformers 4.51.3；Triton 3.2.0
- V1：`alpha=0.08`，block size 128；完整保存的 prompt 边界
- 输入：3962 tokens，31 blocks，最后一块 122 tokens；28 Q heads / 4 KV heads
- 捕获：layer 0；未启用可选 Q、V、row-by-block mass 或 output-vector 归档

`results/` 被 Git 忽略，原始归档和张量不会上传到 GitHub。压缩包中额外存在一个未列入 manifest 的 Jupyter `.ipynb_checkpoints/metadata-checkpoint.json`；审计只认根目录 `metadata.json` 和 manifest 中的文件。

## 完整性与张量契约

| 检查 | 结果 |
| --- | --- |
| 根 / 样本 `metadata.status` | `complete` / `complete` |
| manifest | 14 条；14/14 文件存在、大小一致、SHA256 一致 |
| source / capture input hash | 相同：`2b98eb65...34bb66` |
| token / decoded block 行数 | 3962 / 31 |
| `block_summary.csv` | 124 = 31 blocks × 4 KV heads |
| `tile_summary.csv` | 868 = 31 query blocks × 28 Q heads |
| `layer_summary.csv` | 1 行，仅 layer 0 |
| raw pre-/post-RoPE K | `[3962, 4, 128]`, BF16 |
| V1 mean K | `[31, 4, 128]`, BF16 |
| V1 score / block masks | `[31, 31, 28]` |
| exact block probability及 Mean Pool 误差张量 | `[31, 31, 28]`, FP32 |
| retained mass / output error | `[3962, 28]`, FP32 |

## Dense oracle sanity

以下数值直接读取 `dense_block_oracle.pt/sanity`：

| 检查 | 数值 |
| --- | ---: |
| block probability sum 最大绝对误差 | `1.7881e-7` |
| future block mass 最大绝对值 | `0` |
| retained mass 范围 | `[0.0517670, 1.00000036]` |
| selected count mismatch | `0` |
| missing protected entries | `0` |
| same-budget routed-count mismatch | `0` |
| non-finite values | `0` |
| 用保存的 Mean K 重建 V1 full-block 分布的最大误差 | `1.7822e-4` |
| exact − V1 proxy 聚合 log-mass 最小值 | `0.200630` |

最后一项为正，与 `logmeanexp(qk) >= mean(qk)` 一致；V1 score 重建误差很小，说明 mean、query 聚合和 head 映射没有明显实现错位。

## Layer 0 汇总

| 指标 | 数值 |
| --- | ---: |
| selected causal block ratio | `65.65%` |
| non-protected routed ratio | `38.26%` |
| dense attention retained mass mean / p05 / min | `96.32% / 84.18% / 5.18%` |
| same-budget dense oracle retained mass mean | `96.43%` |
| selected-vs-dense output relative L2 mean / p95 / max | `0.88% / 3.81% / 47.50%` |
| output cosine mean | `0.999795` |
| proxy-vs-exact cosine，全 causal / 仅 remote | `0.3896 / 0.9345` |
| exact-mass-weighted Mean Pool Jensen gap | `2.0076` nats |
| tile-mean Jensen gap 最大值 | `28.7128` nats |
| 单 query-row Jensen gap 最大值 | `43.2190` nats |
| V1 BF16 mean vs FP32 block mean relative L2 最大值 | `0.2448%` |

同预算 oracle 只比 V1 多保留约 `0.12` 个百分点的平均 mass，说明在这个 4K/layer-0 样本上，给定相同 routed budget 后，V1 的整体选块已经接近 mass oracle；这不能外推到 32K 或后层。

## 初步结构信号（只作下一阶段检查项）

1. **实现误差与方法误差必须分开。** V1 Mean Pool kernel 与 FP32 块均值的最大向量误差只有 `0.2448%`；但 exact-mass-weighted Jensen gap 为 `2.0076` nats，且单行最高 `43.2190`。后者来自块内 token logits 经指数后的质量差异，不是 Mean Pool kernel 算错。
2. **layer 0 的 K 具有极强共同方向。** pre-/post-RoPE mean-reconstruction MSE 均值分别约 `0.099%` / `0.207%`，directional concentration 均值约 `0.99901` / `0.99793`。同时仍出现很大的 query-conditioned Jensen gap，说明仅看 K 空间欧氏重构误差可能严重低估 softmax-mass 误差。
3. **remote 指标比全 causal 指标更有解释力。** 全 causal proxy/exact cosine 只有 `0.3896`，但去掉必保留区域后为 `0.9345`；后续必须把 protected local/sink 与真正 routed remote block 分开分析。
4. **attention mass 与 Value 输出误差不是单调对应。** 最低 retained mass 出现在 token 3005 / block 23 / head 15：仅 `5.18%`，但 relative L2 为 `9.36%`、cosine 为 `0.9956`。最大 relative L2 出现在 token 1948 / evidence block 15 / head 22：retained mass 仍有 `70.58%`，但 relative L2 为 `47.50%`、cosine 为 `0.8804`。这正是 32K rich capture 需要保留 V 的理由。
5. **均值会隐藏头部/位置尾部。** layer 均值输出误差很小，但 head 0 的 query blocks 20–23 只保留 6 个 protected blocks、没有 routed remote block，tile mean retained mass 约 `49.8%–52.4%`，relative L2 mean 约 `13.0%–14.1%`。不能只看层均值。

这些现象来自一条 quick 样本和第一层，只定义 32K 要验证的统计量，不作为新方法依据。

## 下一阶段门槛

32K rich capture 应保存两个任务各一条样本的全部 Q/V，并在 layers `0,7,14,21,27` 保存 row-by-block mass。进入新方法设计前至少完成：

- 分层比较 pre-/post-RoPE K 的结构；
- 分开 protected 与 routed remote block；
- 检查 Jensen gap、exact mass miss、Value output error 三者是否在 multi-key 和 multi-query 都稳定相关；
- 再用全部八条 calibration 输入做轻量确认，避免把两条 rich 样本当作统计证据。
