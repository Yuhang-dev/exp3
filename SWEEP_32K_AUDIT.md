# 32K calibration alpha sweep 审计

审计日期：2026-09-18。远端运行完成于 `2026-09-17T14:34:08+0800`。

## 阶段结论

- `cgf_mean:0.04` 和 `dispersion_mean:0.04` 在 multi-key 上均达到 100，精算比例分别约 25.7% 和 23.9%；同一 alpha 的两种 mean 基线为 83.33，精算比例约 30%。这是值得保留的选择器信号。
- multi-query 的最好稀疏得分仍是 91.67，来自 `mean_native:0.04`、`mean_balanced:0.04/0.08`。CGF 最好为 75；dispersion 最好也为 75，但出现在 alpha=0.08，不是 0.04。
- 没有一个稀疏配置在两类任务上同时追平 dense。不能仅凭某个 alpha 或当前未融合实现的速度淘汰整个 CGF/dispersion 分支。
- 离线重评 112 条原始生成，分数变化为 0；独立重算 28 行得分、prefill 中位数与配对加速比，结果全部一致。原有六个配置的 48 条输出也逐字重现。

## 数据范围与保存位置

本轮是自定义 `synthetic_kv_retrieval` 的 32K calibration：multi-key 和 multi-query 各 4 条输入，每条有 3 个目标。共 8 条不同输入、14 个配置、112 条生成，**不是前一轮的 80 条官方数据 RULER，也没有扩充质量样本数**。

每个任务/配置实际只有 4 个独立样本、12 个目标；一个目标对应 8.33 分，三个目标来自同一 prompt，不能当作三个独立样本。未把两个任务平均成总分。RULER pilot 只验证了 dense/V1，不能把其稳定性直接转移给本轮新方法。

- 模型：`Qwen/Qwen2.5-7B-Instruct`，revision `a09a35458c702b33eeacc393d103063234e8bc28`。
- 环境：RTX 4090、BF16、batch=1；Torch 2.6.0+cu124、Transformers 4.51.3、Triton 3.2.0。
- 总预算 32768，生成预留 128；实际 prompt 为 32636–32639 tokens。
- 原始归档：`C:/Users/Yuhang/Downloads/calibration_sweep_32k.tar.gz`，481334 bytes。
- 项目内原样副本：`results/archives/calibration_sweep_32k_20260918_shaB5C2DC01.tar.gz`。
- 解包审计目录：`results/audits/calibration_sweep_32k_20260918/results/calibration_sweep_32k/`，以下称 RUN。
- 原始归档和项目副本 SHA256 一致：`B5C2DC0105D0B0ACF42D46D7BBD2A06CAF4AB0551B55069E52C1FD7409CE8FAB`。

`results/` 被 Git 忽略；原始归档、token IDs、生成文本和派生重评保存在本地，不随本报告自动上传。原有 `REPORT.md`、预测和计时没有被重写。

## 完整性与 scorer 核验

| 检查 | 结果 |
| --- | --- |
| `metadata.status` | `complete` |
| 输入 / 原始生成 / predictions / quality 行数 | 8 / 112 / 112 / 112 |
| 不重复的 sample/config 生成组合 | 112，无重复或缺失 |
| 原始 timing 行数 | 336；每个 sample/config 恰好有 repeat 0/1/2 |
| 独立 profile | 784 行 = 14 配置 × 2 任务 × 28 层；每组仅首条样本 |
| 生成记录输入 hash 与 inputs 对照 | 112/112 一致 |
| 原始生成文本与 scored prediction 文本 | 112/112 一致 |
| scorer 实现 hash 与远端 manifest | 一致 |
| 原始生成离线复评分 | 112/112 原分重现，变化 0 |
| 从原始 repeats 和复评分独立重算 summary | 28/28 一致；最大得分误差 1.42e-14，时间误差 0 ms，加速比误差 2.22e-16 |
| 生成结束 | 112/112 正常 EOS，无 max-token 结束 |
| 与旧 calibration 的共同输入 | 8/8 输入 hash 和答案一致 |
| 与旧 calibration 的共同配置输出 | 48/48 文本逐字一致、分数一致 |

旧校准对照来自 `results/calibration_debug_hotpot_audit_20260917/`。共同配置为 dense 与五种 alpha=0.08 稀疏方法。复现的是相同输入上的结果，不是新增 48 个独立样本。

离线重评版本为 `exp3-scorers-v3-2026-09-17`，快照在 `RUN/rescoring/audit-20260918/`，包含独立 `scores.jsonl`、`quality.csv`、`SUMMARY.md` 和 `manifest.json`。关键 hash：

| 文件 | SHA256 |
| --- | --- |
| `scoring.py` | `b0b5ae49297866984c9a18f16c3e2ae20e5f4c1d9b298e0bd3b6e12b9ae453b3` |
| `inputs.jsonl` | `b076c7db570cb4bf77d8df1d8edf6ca9074ba6bf6548c305a199c25a403959ae` |
| `inputs.pt` | `5ac9e611dd187f72c312cccd8e3500d79f609aa4b1ca901d48a7818fe6ebaff8` |
| `generations.jsonl` | `90e94d6f87abd457a8742773fdf6be0c5ab271118ca3d5b77803802b1214da92` |
| `predictions.jsonl` | `3d88499fdba6b815ab2764ca7eb49a53287584ae3ef2d9d52b31a6812a88708f` |

无需重新调用模型即可再评：

```bash
python -u rescore.py results/calibration_sweep_32k --tag another-scorer-audit
python -u diagnose_quality.py results/calibration_sweep_32k
```

换 scorer 时使用新的 tag，保留已有快照。此次本地仅分析保存结果，没有运行 GPU 实验或修改模型、选择器、评分规则。

## 全部质量结果

主分数是 key 对应 value 的逐目标准确率（0–100）。最后一列为「所有三个目标都正确」的样本数，顺序固定为 multi-key / multi-query。

| 配置 | alpha | Multi-key | Multi-query | 全对样本数 MK / MQ |
| --- | ---: | ---: | ---: | ---: |
| dense | — | 100.00 | 100.00 | 4/4 / 4/4 |
| fp_v1 | 0.08 | 58.33 | 33.33 | 1/4 / 0/4 |
| mean_native | 0.04 | 83.33 | 91.67 | 3/4 / 3/4 |
| mean_native | 0.08 | 58.33 | 41.67 | 1/4 / 0/4 |
| mean_native | 0.16 | 58.33 | 50.00 | 1/4 / 1/4 |
| mean_balanced | 0.04 | 83.33 | 91.67 | 3/4 / 3/4 |
| mean_balanced | 0.08 | 75.00 | 91.67 | 2/4 / 3/4 |
| mean_balanced | 0.16 | 75.00 | 66.67 | 2/4 / 2/4 |
| cgf_mean | 0.04 | 100.00 | 75.00 | 4/4 / 2/4 |
| cgf_mean | 0.08 | 75.00 | 58.33 | 2/4 / 0/4 |
| cgf_mean | 0.16 | 75.00 | 58.33 | 2/4 / 1/4 |
| dispersion_mean | 0.04 | 100.00 | 66.67 | 4/4 / 2/4 |
| dispersion_mean | 0.08 | 75.00 | 75.00 | 2/4 / 2/4 |
| dispersion_mean | 0.16 | 83.33 | 58.33 | 2/4 / 1/4 |

### 哪些结构有信号

1. **均值补偿**：alpha=0.08 的 mean_native 相对 V1，在 multi-query 从 33.33 到 41.67（多对 1 个目标），multi-key 同为 58.33。mean_native:0.04 的明显恢复同时改变了选择阈值；本轮没有 V1:0.04，不能把这一全部增量单独归因于补偿。
2. **逐 query 归一化再聚合**：同为 alpha=0.08，mean_balanced 相对 mean_native 为 75/91.67 对 58.33/41.67，且记录到的精算比例略低。这是本轮最清楚的跨两类任务的选择规则改善；alpha=0.04 时两者分数持平，说明收益依赖阈值。
3. **CGF 二阶选择**：alpha=0.04 时相对 mean_balanced:0.04，multi-key 提高 16.67 分，multi-query 下降 16.67 分；精算比例从约 29.6% 降到 25.7%。不能称为普遍更准确，但其 multi-key 结果值得在留出集验证。
4. **dispersion 的 Value-aware 加权**：alpha=0.08 时相对 CGF，multi-key 持平，multi-query 提高 16.67 分，精算比例更低；alpha=0.16 时 multi-key 比 CGF 高 8.33 分，multi-query 持平。alpha=0.04 时反而在 multi-query 比 CGF 低 8.33 分。因此不能整体放弃，也不能称它稳定优于 CGF。

这些是端到端配置对照，不是固定激活、固定相同密度的消融。相同 alpha 不是相同预算。CGF 二阶项只用于选择；所有 mean 系列的补偿仍是同一个一阶 block-mean 形式。

### 增益集中在哪些样本

- multi-key 的样本 0：mean_native:0.04 和 mean_balanced:0.04 都只对 1/3，CGF:0.04 和 dispersion:0.04 为 3/3；另外三个样本均为 3/3。相对这两个 mean 基线的 +16.67 全部来自这一条输入的两个目标。
- CGF 和 dispersion 从 alpha=0.08 到 0.04，multi-key 都修复样本 2 的一个目标和样本 3 的两个目标，因此从 75 到 100。
- multi-query 的 dispersion:0.08 相对 CGF:0.08，在样本 5 和 6 各多对一个目标，其他两条分数不变。
- dispersion 的 multi-query 从 alpha=0.08 到 0.04 并不单调改善：样本 4 从 1/3 到 3/3，样本 5 从 3/3 到 1/3，样本 6 保持 3/3，样本 7 从 2/3 到 1/3，合计由 75 降到 66.67。更密的路由没有保证最终生成更好。

样本号取自 `synthetic-calibration-32768-{id}-{variant}`。保留逐样本变化，不将这些小样本差异解释成统计显著性。

## 错误是否来自评分

112 条生成共形成 336 个「配置 × 目标」判断，其中 89 个失败判断：86 个是在正确 key 后输出错误 value，3 个是真值出现在其他 key 后的绑定错误。这些是重复配置上的判断数，不是 336 个独立样本。

未发现正确 value 已绑定到正确 key 却因格式被漏判的案例；没有大小写单独造成的失分，没有 max-token 结束。三个 misbound 案例会在 value-only recall 下多得分，但该指标忽略了题目要求的 key–value 对应关系，不能据此改写主得分。

失败值的诊断分类为：35 个同命名空间错误值、33 个正确随机后缀配错索引、18 个复制另一目标值、2 个值字符串自身截短、1 个其他。这里的两个截短值出现在正常 EOS 输出内，不是碰到了生成长度上限。

逐目标证据已写入 `RUN/target_diagnosis.csv`，详细对照为 `RUN/QUALITY_DIAGNOSIS.md`。当前证据定位到答案选择/复制/绑定差异；复评分一致不等于证明所有算法实现细节正确，但本轮没有 scorer-only 的解释。

## 实际延迟与精算比例

下表所有斜线左右均分别表示 **multi-key / multi-query**，没有跨任务平均。prefill 先在每条样本内取三次计时中位数，再对四条样本取中位数；加速比先逐样本计算 dense/method，再取中位数。

精算比例来自每任务/配置首条输入的独立 28 层 profile，不是四条样本的平均，也不包含 mean/selector 的计算成本。

| 配置 | alpha | 精算 token-pair 比例 % MK / MQ | Prefill ms MK / MQ | 配对加速比 MK / MQ |
| --- | ---: | ---: | ---: | ---: |
| dense | — | 100.00 / 100.00 | 4833.5 / 4833.0 | 1.00 / 1.00 |
| fp_v1 | 0.08 | 20.38 / 20.32 | 3826.5 / 3827.2 | 1.26 / 1.26 |
| mean_native | 0.04 | 30.40 / 30.22 | 4706.4 / 4702.2 | 1.03 / 1.03 |
| mean_native | 0.08 | 19.66 / 19.64 | 4520.0 / 4520.7 | 1.07 / 1.07 |
| mean_native | 0.16 | 12.28 / 12.20 | 4387.5 / 4388.4 | 1.10 / 1.10 |
| mean_balanced | 0.04 | 29.56 / 29.62 | 5096.4 / 5093.6 | 0.95 / 0.95 |
| mean_balanced | 0.08 | 18.44 / 18.50 | 4909.7 / 4909.8 | 0.98 / 0.98 |
| mean_balanced | 0.16 | 11.35 / 11.52 | 4788.8 / 4789.4 | 1.01 / 1.01 |
| cgf_mean | 0.04 | 25.70 / 25.77 | 5225.2 / 5235.1 | 0.92 / 0.92 |
| cgf_mean | 0.08 | 16.49 / 16.72 | 5081.7 / 5088.6 | 0.95 / 0.95 |
| cgf_mean | 0.16 | 10.56 / 10.64 | 4976.0 / 4983.5 | 0.97 / 0.97 |
| dispersion_mean | 0.04 | 23.90 / 23.97 | 5355.3 / 5365.9 | 0.90 / 0.90 |
| dispersion_mean | 0.08 | 15.63 / 15.73 | 5224.8 / 5233.2 | 0.92 / 0.92 |
| dispersion_mean | 0.16 | 10.15 / 10.22 | 5135.7 / 5140.1 | 0.94 / 0.94 |

峰值 allocated 约 20.283 GiB，reserved 约 21.467 GiB，各配置基本相同；本实现仍保留完整 KV cache。

alpha=0.04 的 CGF 和 dispersion 比 dense 分别约慢 8% 和 11%。下面以 multi-query 的首条 profile 为例，记录 28 层时间之和，单位 ms：

| 配置 | Selector | Exact | Mean | Merge |
| --- | ---: | ---: | ---: | ---: |
| fp_v1:0.08 | 13.6 | 432.5 | 0 | 0 |
| mean_native:0.04 | 15.8 | 595.5 | 526.4 | 142.6 |
| mean_balanced:0.08 | 445.2 | 382.3 | 521.3 | 142.6 |
| cgf_mean:0.04 | 661.1 | 498.2 | 520.0 | 142.6 |
| dispersion_mean:0.04 | 823.4 | 461.0 | 517.7 | 142.6 |
| dispersion_mean:0.08 | 827.1 | 324.6 | 516.0 | 142.6 |
| dispersion_mean:0.16 | 828.0 | 237.7 | 520.2 | 142.6 |

还存在 descriptor、indices 及模型非 attention 部分；此表不是完整 prefill 的加法分解。profile 来自额外前向，不进入主计时。

Exact 路径已复用 Triton kernel，dense/V1 也已有优化；当前主要未融合的是新增 selector、mean tail、merge 等路径。dispersion 从 0.04 到 0.16 时 exact 大幅减少，但约 0.83 秒 selector、0.52 秒 mean、0.14 秒 merge 几乎不降，因此完整 prefill 只从约 5.37 秒到 5.14 秒。

这说明后续优化应关注新增路径的代理扫描与融合开销；不能根据精算比例直接承诺优化后的速度。就当前实际完整时间而言，CGF/dispersion 尚未胜过 dense 或最快 mean 基线。

## 下一阶段建议（尚未冻结、尚未运行）

先在未见过的 holdout 上验证本轮信号，不继续针对这 8 条输入调 alpha。优先考虑以下七个配置：

- 对照：dense、fp_v1:0.08。
- mean 基线：mean_native:0.04（当前质量与实际时间折中）、mean_balanced:0.08（更低精算比例的 balanced 对照）。
- 新候选：cgf_mean:0.04、dispersion_mean:0.04、dispersion_mean:0.08。

dispersion_mean:0.16 的约 10% 精算比例和 multi-key 83.33 也保留为低密度信号；若下一阶段重点是优化后的低密度潜力，可在运行前将它加入冻结名单，而不是删除其结果。mean_balanced:0.04 本轮分数与 mean_native:0.04 完全相同，时间更慢，优先级较低。

七配置是建议，不是已发布的冻结配置文件。现有 holdout 入口默认每种合成任务/长度 8 条（16K 和 32K 共 32 条），另有 16 条 HotpotQA，共 48 条不同输入；七配置共 336 条生成。它仍是自定义合成任务加 QA，不是新的 RULER。尚未读取任何 holdout 得分，也尚未开展 kernel 优化。
