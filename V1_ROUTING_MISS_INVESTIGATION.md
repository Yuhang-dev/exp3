# V1：从 Mean Pool 误差到实际漏选块

日期：2026-09-20。状态：已有张量的离线调查；没有实现新选择器，没有重跑模型、质量测评或延迟测试。

## 本轮结论

1. V1 确实逐 query 与 Mean Key Block 计算，然后跨 query 聚合并按阈值选块。“仅替换 Mean Pool”应准确称为“固定原路由预算，仅替换块质量估计的离线消融”，不是另一种现成的 V1 路径。
2. 相对同预算 dense-mass oracle，原 V1 漏选 213 条；块质量改用精确值、保留 V1 query 聚合后，找回 59 条，但又丢掉 41 条，仍漏 195 条。不能把原来的 213 条全部归因于 Mean Pool。
3. 213 条中有 158 条位于原选中边界之后的前两名。每个有 routed budget 的 query-tile/head 多保留 6 个候选，候选外仅剩 9 条。这个结果只是“候选覆盖能力”，不是已实现的修复效果。
4. 在上述 +6 候选中，使用精确块质量、仍按 V1 的未逐行归一化规则重排，实际还漏 194 条。因此扩大候选与有效复核是两个独立问题。
5. 数学上，给所有 post-RoPE Key 加同一个向量不改变 full attention，却可能改变 V1 的跨 query 聚合排序。这提供了一个与块内残差互补的诊断轴：共同方向是否通过 query 权重影响选块。

本轮暂将“降到个位数”解释为：这个冻结样本、相同 routed budget 下，漏选条目不超过 9。后续跨样本按漏选率和 mass 损失报告，不能沿用绝对条数，也不把 +6 冻结为新方法参数。

## 数据与计数口径

- 源档案 SHA256：`058319745E8ED28CE3A69AF3092F0CE82E026A2848C759FAE20727CC7E446107`。
- 一条 quick multi-key 输入，3962 tokens，仅 layer 0；31 key blocks，28 Q heads，4 KV heads，block size 128，V1 alpha 0.08。
- 一条“漏选”指一个 `(query tile, key block, Q head)` 条目；不是一个独立样本，也不是一个唯一文本块。
- 保持 V1 的全部 protected blocks。每个 query-tile/head 的 remote budget `k` 固定为其原 V1 routed count。
- 同预算 oracle 在 remote 候选中，按逐 query softmax 后的平均真实 block mass 选 top-k。
- 共 7728 个 eligible remote 条目，最终 routed slots 为 2957；596 组有非零 routed budget，另有 48 组有 remote 候选但 budget 为零。
- 原漏选率：`213 / 2957 = 7.2032%`。这不是所有 dense 有用信息的遗漏率。
- 用保存的 Mean Pool 聚合分数重建 top-k，与原 V1 mask 的差异为 0；用 dense mean block probability 重建 oracle，与保存的 oracle mask 的差异也为 0。

零 routed budget 时，同预算 oracle 也不能选 remote，因此其漏选条数恒为零。预算不足引起的质量损失必须另算；本轮不通过这个口径宣称“所有有用块都已找回”。原 V1 的保护区、当前对角块与 alpha 阈值问题也没有被同预算消融覆盖。

完整派生数字、输入张量 hash、候选扫描与统计口径保存在 [JSON 记录](D:/long-context/exp3/results/audits/v1_routing_miss_4k_20260920.json)。原始归档和张量保持不变。所有统计只在 CPU 上读取已有张量；FP32 保存值提升为 FP64 后汇总。

## V1 与“仅替换 Mean Pool”的准确含义

以下省略 head，讨论完全可见、等长的 remote blocks；公式忽略 BF16 舍入，实际数值对照使用保存的 V1 BF16 mean。设 query tile 为 `T`，每块 `B=128` 个 token：

\[
z_{qi}=q^T k_i/\sqrt d,\qquad
\bar k_b=\frac1B\sum_{i\in b}k_i,\qquad
a_{qb}=q^T\bar k_b/\sqrt d.
\]

原 V1 块分数在共同归一化常数之外是：

\[
S_b^{\rm mean}=\sum_{q\in T}e^{a_{qb}}.
\]

它不先池化 Q，也不是对 query 取最大值。kernel 中对 query 的最大值用于指数计算的数值稳定，随后实际求和；跨 key-block 的 `alpha * max_b score_b` 是后续选择阈值。

“仅替换”的离线分数为：

\[
S_b^{\rm exact,raw}
=\sum_{q\in T}\frac1B\sum_{i\in b}e^{z_{qi}}.
\]

即将每个 query 对 mean key 的一个指数，换为该 query 对块内全部 key 的指数均值；保留跨 query 的直接求和。在原 budget 下重排 remote blocks。选中后的 V1 exact attention / Value 路径没有在本轮重新执行，也没有加入 mean compensation。

而本轮 dense oracle 的目标为：

\[
P_b=\frac1{|T|}\sum_{q\in T}
\frac{\sum_{i\in b}e^{z_{qi}}}{Z_q},\qquad
Z_q=\sum_{i\le q}e^{z_{qi}}.
\]

所以即使块内质量完全准确，直接跨 query 求和也不等于平均真实 attention mass。令 `p_qb` 为真实 block probability，则 `S_exact,raw` 与 `sum_q Z_q p_qb` 成比例；它给不同 query 加上了与 `Z_q` 成比例的权重。

实现证据：[V1 分数 kernel](D:/long-context/exp3/upstream/flashprefill_native_forward.py:94)、[阈值与保护规则](D:/long-context/exp3/upstream/flashprefill_native_forward.py:345)、[跨块统一归一化](D:/long-context/exp3/upstream/flashprefill_native_forward.py:398)、[dense oracle 的保存与同预算选择](D:/long-context/exp3/v1_block_diagnostics.py:572)。

## 已完成的离线消融

| 路由分数，全部保持原最终预算 | 相对 dense oracle 漏选条目 | 平均保留 dense attention mass |
| --- | ---: | ---: |
| 原 V1 Mean Pool 分数 | 213 | 96.315717% |
| 精确块质量 + 原 query 聚合 | 195 | 96.316611% |
| dense oracle，本轮参照目标 | 0（定义如此） | 96.433051% |

精确块质量的替换一共引入 102 个新条目，其中 59 个属于原先漏掉的 oracle 条目；另外 43 个不属于 oracle。同时丢掉 41 个原先选对的 oracle 条目：`213 - 59 + 41 = 195`。

这支持“单修块内质量估计不足以解决当前同预算漏选”，而不是证明所有剩余问题都只能由一种新聚合方法解决。尚缺 mean-estimate + row-normalized aggregation 这个正交对照。

## 边界候选能覆盖多少漏块

对每个非零 budget 的 query-tile/head，按原 Mean Pool 分数保留前 `min(k+r, eligible_count)` 个候选；最终仍只允许选 `k` 个。零预算组不扩张。

| 每组额外候选 r | 总 remote 候选条目 | 相对原路由额外候选 | 候选外 oracle 漏块 | 候选内精确块质量 + 原聚合重排后漏块 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 2957 | 0 | 213 | 213 |
| 1 | 3447 | 490 | 113 | 199 |
| 2 | 3898 | 941 | 55 | 194 |
| 3 | 4308 | 1351 | 32 | 195 |
| 4 | 4686 | 1729 | 22 | 194 |
| 6 | 5345 | 2388 | 9 | 194 |
| 8 | 5901 | 2944 | 3 | 194 |

原漏块相对第 k 名的排名差分布：第 1 名之后 100 条，第 2 名之后 58 条，第 3 名之后 23 条，第 4 名之后 10 条；最远为第 k+15 名。

+6 覆盖 204/213 个原漏块，但需要额外 2388 个候选条目，相当于原 routed slots 的 80.76%。这是候选检查范围，不是运行时涨幅，也不是额外最终精算预算。只有用 dense oracle 在候选内重排这个不可直接廉价实现的参照，才会只剩 9 个漏块。

+6 后仍在候选外的 9 条 `(query block, key block, Q head)` 为：`(18,11,3)`、`(21,12,20)`、`(24,5,10)`、`(26,2,3)`、`(26,21,15)`、`(27,16,0)`、`(28,4,3)`、`(28,16,0)`、`(28,21,17)`。它们不全是极低 mass：如 `(26,21,15)` 的 tile mean mass 为 4.3227%，单 query 最大 mass 为 87.6432%。固定扩张边界仍可能错过少数重要尖峰。

## 数学角度一：块内残差什么时候真的引起翻转

写作 `k_i = mean_k_b + r_i`，令 `delta_qi = q^T r_i / sqrt(d)`。则：

\[
M_{qb}=\sum_{i\in b}e^{z_{qi}}
=B e^{a_{qb}}e^{J_{qb}},\qquad
J_{qb}=\log\left(\frac1B\sum_{i\in b}e^{\delta_{qi}}\right).
\]

理想均值下 `J >= 0`。但大 J 本身不代表漏选：别的块可能有同样大的 J，或者该块本就受保护/排名很高。

对同一种跨 query 聚合，定义 `Delta_b = log S_exact,raw_b - log S_mean_b`。一个当前未选块 u 能反超已选块 s，当且仅当：

\[
\Delta_u-\Delta_s
>\log S_s^{\rm mean}-\log S_u^{\rm mean}.
\]

应该研究的是“块残差引起的相对修正是否跨过选中边界”，不是平均重构误差或者孤立的方差大小。`q^T Sigma_b q / d`、沿 query 的少数极端 token、多模态子簇都可以是解释这个相对修正的候选结构；只存 K 无法验证它们与实际 query 的对应关系。

Mean Pool 信息不足还有一个简单反例：一维 query 为 1，块 A 的 keys 为 `[0,0]`，块 B 为 `[-c,c]`。两块 mean 都是 0，但真实质量分别为 2 与 `2 cosh(c)`。只看 mean 的规则无法区分；想可靠补回这一类块，必须增加残差信息或读取部分原始 K。

可研究的逐级精化方向：把块划为少量子组，保存每组计数 `n_j`、均值 `mu_j`、残差半径 `R_j=max_i ||k_i-mu_j||`。对一个 query，令 `a_j=q^T mu_j/sqrt(d)`、`rho_j=||q||R_j/sqrt(d)`，则有：

\[
\underbrace{\sum_j n_j e^{a_j}}_{L_{qb}}
\le M_{qb}\le
\underbrace{\sum_j n_j e^{a_j}\cosh(\rho_j)}_{U_{qb}}.
\]

下界来自 Jensen，上界来自指数函数在 `[-rho,rho]` 上的割线和组内零均值。这个界可能很松，是否有用必须测；它不是已验证的新方法。

如果每个可见块都有有效 mass 区间，则真实 row probability 也有区间：

\[
\frac{L_b}{L_b+\sum_{c\ne b}U_c}
\le p_b\le
\frac{U_b}{U_b+\sum_{c\ne b}L_c}.
\]

跨 query 平均区间后，只精化有机会越过第 k 名边界的块。该思路同时考虑块内残差和逐行归一化；当前部分可见块需要单独正确处理。理论上精化到逐 token 就会精确，但低成本阶段能收紧多少尚未验证，不能承诺单个位数与净加速同时实现。

## 数学角度二：共同方向与选块的平移不变性

对所有 post-RoPE keys 加同一个向量 c，`z'_qi = z_qi + q^T c / sqrt(d)`。对同一 query，这个偏移对所有可见 tokens 相同，所以 row softmax 和 full-attention output 完全不变。

但 V1 聚合变成：

\[
S_b'=\sum_q e^{q^Tc/\sqrt d} e^{a_{qb}}.
\]

这是对 query 重新加权，不一定保留 block 排名。该现象即使每个块内 K 完全相同、Mean Pool 没有任何近似误差，也可以发生。

例如两个等长块、两个 query，每块内 keys 完全相同，logits 为 `[[0,2],[1,0]]`。在固定 top-1 的抽象对照中，raw exp 聚合选块 B；给第二行加 10 后变为 `[[0,2],[11,10]]`，raw 聚合改选 A。但真实 row-normalized 平均概率始终是 A=0.42513、B=0.57487，应该选 B。这里只证明排序性质，不是重演实际 alpha/保护区配置。

现有 K 的共同方向能量占比 `N||mean_all_K||^2 / sum_i||K_i||^2`：

| KV head | pre-RoPE | post-RoPE |
| ---: | ---: | ---: |
| 0 | 99.9747% | 99.8968% |
| 1 | 99.9072% | 99.7056% |
| 2 | 99.9229% | 99.7001% |
| 3 | 99.7970% | 99.0758% |

这使“共同方向与真正区分块的残差分离”成为值得检查的结构假设；它不能证明这些共同方向已造成实际漏选，因而还需要 Q 检查 `q^T c` 的幅度及与漏块的对应关系。离线平移对照可验证路由不变性；未来若使用数据估计中心，必须保持因果可见性。

RoPE 在这里的用途是区分内容结构和位置旋转引入的结构，不是据此删除 RoPE 或认定它有害。应对相同的 block/query 错误条目比较 pre-/post-RoPE 残差，而不是把整体低秩或均值缩小直接当成漏选原因。

## 后续调查，不直接实现方法

1. 用现有 32K rich-capture 命令补齐 Q/V：multi-key 与 multi-query 各一条，覆盖深度；Q/K 允许离线重算任意代理与 row normalizer，V 用于衡量补块后的输出误差。当前 smoke 没有 Q/V，不能完成这些对照。
2. 做估计与聚合的 2×2 对照：Mean / exact block mass × raw query sum / row-normalized query mean；全部固定 protected mask 和最终 routed budget。分别报告找回、又丢、净漏选、mass 改善，不混用原 alpha 的预算变化。
3. 对原漏块、找回块、又丢块做同 query/head 下的边界配对。研究 query 投影残差、稀有 token/子簇、共同方向和 pre-/post-RoPE 差异；同时保留随机打散、位置/距离匹配的对照，排除重复 token 与位置结构造成的假象。
4. 比较两种后续补块模式。第一种在候选里复核再换块，固定最终预算；第二种先算原选中块，利用其 exact softmax normalizer 再筛未选块并追加。第二种不能冒充固定预算：追加块、额外读 K/V、额外扫描与合并都单独记账。若用 selected normalizer 近似 full normalizer，偏差因子为 `1 / retained_mass_q`，不能把它称为真实逐行归一化。
5. 两条 rich 样本只用于发现机制；先在 8 条 calibration 输入分任务确认，再检查保存的 RULER 输入。layer 0 的局部 dense 对照共享最初 Q/K；后层捕获的是 V1 前向轨迹上的 Q/K，需要区分块本身的结构与之前稀疏层已经改变的激活。
6. 只有实际廉价复核规则在未用于设计的样本上有效，才能说漏选降到个位数水平或有稳定低漏选率。当前未运行这些实验，未宣称输出质量改善或净加速。

现有采集命令与存储开销见 [V1_BLOCK_DIAGNOSTICS.md](D:/long-context/exp3/V1_BLOCK_DIAGNOSTICS.md)。
