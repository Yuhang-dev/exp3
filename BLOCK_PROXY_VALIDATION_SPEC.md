# Block proxy 验证实验说明（交给 Codex 执行）

## 0. 背景与要回答的问题

已有审计（`BLOCK_PROXY_RESULTS_20260923.md`，目录 `results/audits/block_proxy_reports_20260923/`）的结论：

- L14、Q=K=128 时，MeanPool 相对共享块 oracle 丢失 7.53（RULER）/ 9.82（LongBench）个百分点的注意力质量。
- 把 MeanPool 换成真实块指数和后，损失降到 0.86 / 1.31；四等分子块均值、换行分母几乎无效。
- 高质量漏选块由单 token 尖峰主导：N_eff 约 3（命中块约 9），峰值 token 占块内条件质量 >99.9%。
- 但 FlashPrefill 在 RULER / BFCL 上的任务精度没有下降。

本轮要回答的问题：

- **Q1** COBS 的二阶项（块内协方差）能否修复这类漏选？
- **Q2** 漏选的机制是“稀释”（尖峰被平均掉）还是“抵消”（块内正负 logit 相消）？RoPE 高频是否参与？
- **Q3** 尖峰 token 是否是 sink 类、与 query 无关？
- **Q4** 漏选对 attention **输出**（而不是注意力质量）的影响有多大？（value 侧）
- **Q5** RBS 式半径补救能否捞回这些块，额外代价是多少？
- **Q6** 真实证据块（RULER needle）在 MeanPool 下会不会被漏？

## 1. 约束

- **Phase 1 只用已有 capture 离线计算**（CPU 即可），不跑新的模型前向。Phase 2 的 GPU 部分只写脚本和运行命令，不编造结果。
- 输出到新目录 `results/audits/block_proxy_validation_<YYYYMMDD>/`，不覆盖旧结果；记录全部输入文件的 SHA256。
- **复用现有代码的口径**：`analyze_block_proxy_reports.py`、`inspect_cases.py` 中的加载、分块、候选区域、causal 处理、oracle 与 `mean_raw` 选择逻辑。不要另写一套口径不同的实现；需要改动时在报告中说明。
- 注意归档文件名与内容相反的问题，以归档内 metadata 为准。
- 单位严格区分：注意力质量百分点 / 输出相对误差 / 任务准确率，不能混用。
- 各层分开报告，不跨覆盖范围不一致的层求平均。RULER 三条输入共享同一背景文本，不算独立重复；LongBench 目前只有 1 条（其余 7 条补跑完成后重跑本分析）。
- 无法忠实复现的方法（RBS 的 β、FlashPrefill V2 补偿）必须在报告里标注“近似实现”。
- 报告中不写新颖性或 SOTA 结论。

## 2. 共同定义（先读现有代码核实，冲突时以现有代码为准并说明）

- **模型**：从 capture metadata / config 读取。预期为 Qwen2.5-7B：28 层、28 个 Q 头、4 个 KV 头、head_dim d=128。GQA 映射 Q 头 h → KV 头 h // (n_q / n_kv)，从 config 核实。
- **logit**：`s_ij = q_i · k_j / sqrt(d)`，使用 RoPE 之后的 q、k，float32 计算，logsumexp 用 float64。
- **主配置**：Q block = K block = 128；远端预算 2048 token；sink/local = 256/384；每个 Q tile 采样 16 行；tile 位于上下文 25/50/75/90% 处。若 capture 含全部 28 个头的 Q/K/V，则所有头都计算，并标明覆盖范围。
- **行级量**（行 i，块 b，块大小 B）：
  - `mu_ib` = 块内 logit 均值
  - `Z_ib = Σ_j exp(s_ij)`
  - `g_ib = log(Z_ib / B) - mu_ib`（Jensen gap，≥ 0）
  - `P_ib = Z_ib / Z_i`，Z_i 为该行 dense 全分母
  - 块内条件分布 `r_ij = exp(s_ij) / Z_ib`；`N_eff = 1 / Σ_j r_ij^2`；`top1_share = max_j r_ij`
- **tile 级集合**（每个 layer、head、tile）：
  - `C*` = 共享块 oracle 的选中集合；`M` = MeanPool（现有 `mean_raw`）的选中集合
  - `hit = C* ∩ M`，`missed = C* \ M`，`false_pos = M \ C*`
- **行统计聚合到 (tile, block)**：按行的 `P_ib` 加权，与现有 `block_distributions.csv` 口径一致。
- **regret**：相对 `C*` 的保留质量差，单位百分点，与现有 `method_layer.csv` 一致。

## 3. Phase 1 实验（离线）

### E0 复现检查（先做；不通过就停止并报告）

L14、Q=K=128：

- MeanPool regret：RULER 7.53 / LongBench 9.82
- 真实块指数和 regret：0.86 / 1.31

允许误差 ±0.05 pp。

### E1 COBS 二阶上界与累积量发散

新增评分 `mean_var2`：每行每块的 log-mass 估计为

`log B + mu_ib + 0.5 * Var_j(s_ij)`

这里用的是精确的块内 logit 方差（等于 `q^T Σ_b q / d`，满秩、逐行），是任何 COBS 式压缩的上界。

- 把 `mean_raw` 里的逐行块分数替换成这个估计，**跨行聚合和预算选块步骤保持不变**，计算 regret。
- 以 `method_layer.csv` 相同格式输出所有层（Q=K=128）的结果；L14 另外对 K ∈ {32, 64, 128, 256} 跑一遍。

累积量诊断（按 hit / missed 分组）：

- 块内 logit 经验累积量 κ1..κ4；部分和 `K_n = Σ_{m≤n} κ_m / m!`（K_1 = μ，K_2 = μ + Var/2）。
- 与真值 `log(Z_ib / B)` 比较：报告每组 n=1..4 的 `|K_n - 真值|` 中位数，以及“n=4 的误差大于 n=2 的误差”的块比例（发散指示）。
- `Δ_eff = log(top1_share / (1 - top1_share)) + log(B - 1)`。
- 二阶高估的比例：`0.5 * Var > g` 的块占比。

**预测**：`mean_var2` 在 L14 的 regret 远高于真实指数和（>5 pp）；missed 块的累积量部分和不收敛。

### E2 稀释 vs 抵消

每行取参考水平 `c_i`，两种都报告：

- (a) 选择边界：`c_i = min_{b∈M} mu_ib`
- (b) 该行远端候选块 `mu_ib` 的中位数

对每个 (行, 块) 计算：

- `A_plus = mean_j max(s_ij - c_i, 0)`，`A_minus = mean_j max(c_i - s_ij, 0)`。校验恒等式 `mu_ib - c_i = A_plus - A_minus`。
- `mu_minus1` = 去掉块内 logit 最大的 token 后的均值
- `top1_share`、`N_eff`、`A_plus` 中来自 top-1 token 的比例

分组：hit / missed / false_pos / 边界竞争者（M 中排名在 cutoff ±2 以内的块）。报告中位数、均值、按 tile 重采样的 bootstrap 95% CI 和效应量。

**判据**：

- 稀释：missed 的 `A_minus` 与 false_pos / 边界竞争者相当（差异 CI 含 0），且 `A_plus` 主要来自 top-1。
- 抵消：missed 的 `A_minus` 显著大于竞争者。

### E3 RoPE 频带分解

- 从 config / metadata 读取 `rope_theta` 和 `rope_scaling`（如 YaRN），重建 `inv_freq`。确认 Qwen 的 rotate_half 约定：维度 j 与 j + d/2 配对，频率为 `inv_freq[j]`，j = 0..d/2-1。
- 若只保存了 post-RoPE 的 q/k，用绝对位置反旋得到 pre-RoPE。校验：再旋转回去与原值的 max abs err < 1e-3（bf16 容差）。
- 频带 logit（用 post-RoPE 向量）：`s_ij_f = (q_i[f] k_j[f] + q_i[f+d/2] k_j[f+d/2]) / sqrt(d)`，校验 `Σ_f s_ij_f = s_ij`（误差 < 1e-4）。
- 理论衰减：`a_f(B) = |sin(B θ_f / 2) / (B sin(θ_f / 2))|`。B=128 时的分组：high `a_f < 0.5`，mid `0.5–0.9`，low `a_f > 0.9`；其他 B 另算。（参考：θ=1e6、d=128、B=128 时，约 17/64 个频率对 a<0.5，43/64 个 a>0.9。）
- 对每个 (行, 块)：
  - 各组对块内 logit 方差的贡献份额，用协方差分解：`share_f = Cov(s_f, s) / Var(s)`（可正可负，总和为 1）。
  - 尖峰优势 `Δ = s_peak - mu` 中各组的贡献份额：`(s_peak_f - mean_j s_j_f) / Δ`。
- 经验衰减：`rho_bf = ||mean_j k_post_j[f 对]|| / mean_j ||k_post_j[f 对]||`，按 f 画均值曲线，与 `a_f` 对比。
- 按 hit / missed 分组报告。

### E4 峰值 token 的去 RoPE 反事实

对象：每层、每个数据集 missed 块中按块质量排前 50 的块，加上三个已解码案例：

- RULER MQ:59，L14 H7，K 块起点 9984（峰值 ` won`，位置 10053）
- LongBench，L14 H7，K 块起点 11008（峰值 `/h`，位置 11110）
- LongBench，L14 H7，K 块起点 13184（峰值 `1`，位置 13207）

计算：

- `s_nope_ij = q_pre_i · k_pre_j / sqrt(d)`
- RoPE 下的峰值 token 在 NoPE 下的块内排名；`Δ_rope` 与 `Δ_nope`；峰值 token 到 query 的相对距离；E3 中高频/低频对 Δ 的贡献。

分类：

- 内容驱动：NoPE 下仍是块内 top-1，且 `Δ_nope ≥ 0.5 Δ_rope`
- 位置驱动：NoPE 下不再突出
- 混合

### E5 尖峰 token 是否 sink 类

对 E4 中的峰值 token j*（按位置去重）：

- **query 无关性**：在所有可用的行中（所有 tile、所有采样行，满足 causal 且 j* 位于远端候选区），j* 是其所在块 argmax 的比例；j* 在整行 logit 中的百分位。同一 KV 头下的各个 Q 头分别统计。
- **范数**：`||k_j*||`、`||v_j*||` 在本层本 KV 头全部 token 中的百分位，以及相对块内中位数的倍数。
- **复现性**：同一位置在多少个 (layer, head) 中是尖峰。
- 用 `inspect_cases.py` 的方式解码 token 文本（需要本地原始 input ids 和 tokenizer）。

**预测**：query 无关比例 > 0.8 的判为 sink 类。

### E6 输出误差（value 侧）

对每条采样行、每个 (layer, head)、每种选择方案，计算稀疏输出 O_s（只在 sink + local + 选中远端块上做 softmax 并重新归一化）与 dense 输出 O_d。

方案：

- dense
- C*（共享块 oracle）
- M（MeanPool）
- `mean_var2`（E1）
- 真实块指数和选择
- RBS-like（E7）
- M + FlashPrefill V2 均值补偿（实现方式见下）

逐行指标：

- `rel_err = ||O_d - O_s|| / ||O_d||`
- 沿 O_d 方向的分量：`e_par = (O_d - O_s) · Ô_d / ||O_d||`（只改变大小）
- 垂直分量：`e_perp = ||(O_d - O_s) - (e_par · ||O_d||) Ô_d|| / ||O_d||`（改变内容）
- `cos(O_d, O_s)`

missed 块逐块指标：`v̄_b` 为块内按注意力加权的 value 均值。

- `P_b`、`||v̄_b - O_d|| / ||O_d||`、`cos(v̄_b, O_d)`、`||v̄_b|| / ||O_d||`
- 单块删除误差：`P_b / (1 - P_b) * ||v̄_b - O_d|| / ||O_d||`。用公式算一遍，再直接重算校验。

相关性：跨 (行, head) 计算 mass regret 与 `rel_err` 的 Spearman ρ，分层报告。

可选：

- 若能加载该层 `o_proj` 权重（只需这几层的这一个矩阵），把所有头的输出拼接后乘 `o_proj`，报告 `||W_o (O_d - O_s)|| / ||W_o O_d||`。
- 若 capture 中有该层残差流 hidden state，再报告相对残差范数的大小。
- 无法加载就跳过并说明。

**V2 均值补偿**：阅读 FlashPrefillv2 commit `75b58f2` 的两个位置：

- `flashprefill_ops/flashprefill/prefill.py`（约 L60，默认开关）
- `flashprefill_ops/mainloop_fwd_sm90_tma_gmma_ws.hpp`（约 L1531，补偿实现）

用 numpy 复刻：对未选中的远端块加入替代项，预期形式为分子加 `B · exp(q·k̄_b/√d) · v̄_b`、分母加 `B · exp(q·k̄_b/√d)`，**以代码为准**。报告有补偿和无补偿时的 `rel_err`、`e_par`、`e_perp`。

**预测**：

- M 的 `rel_err` 远小于其质量 regret 所暗示的量级。
- 若漏选块的 `v̄_b ≈ 0`，误差主要在 `e_par`（输出被缩放）。
- V2 补偿对尖峰块补不回来，因为它把这些块的质量低估了约 `e^g` 倍。

### E7 RBS 式半径补救的离线评估

阅读 arXiv 2609.20971 的 Eq. 7–13，尽量忠实实现：

- centroid `c_b` = 块内 key 均值（论文未说明时用 post-RoPE，与 FlashPrefill 口径一致）
- `r_b = max_j ||k_j - c_b||`
- 补救分数 = `q·c_b/√d + β_b · ||q|| · r_b / √d`
- `β_b` 由本 prompt、本层、本头的半径分布（p50–p90）决定
- Q 块聚合按论文 Eq. 10–11
- 基础分支和补救分支分别按相对 max 的阈值选块，取并集，再加上 sink / window

若无法获取论文细节，改用简化版 `β_b = β0 · clip((r_b - r_p50) / (r_p90 - r_p50), 0, 1)`，β0 ∈ {0.25, 0.5, 1.0}，并标注“RBS-like 近似”。

两种比较方式：

- (a) 同总预算（远端 16 块）：`k_base + k_rescue = 16`，扫 `k_rescue ∈ {0, 2, 4, 8}`
- (b) 阈值模式：扫 `α_rescue`，记录实际密度

指标：

- missed 块的召回率（按质量加权）
- 三个已解码案例是否被捞回
- 误报：补救分支选中但不在 C* 中的块数
- mass regret
- E6 的 `rel_err`

**关键问题**：捞回尖峰块能不能降低输出误差，还是只降低了质量 regret。

### E8 真实证据块（RULER）

- 从样本元数据或 input ids 中定位 needle 的 key span 和 value span（MK1:56、MK1:71、MQ:59）。
- 相关的查询行是 prompt 末尾的问题 token。先检查 capture 是否保存了末尾 tile 的 Q（或全部位置的 Q）：
  - **有**：对末尾 tile，计算包含 needle 的 K 块在 MeanPool 下的排名、真实质量排名、g、N_eff、top1_share，并与 E4 的尖峰块对比。
  - **没有**：跳过计算，写一个最小补跑脚本，只捕获末尾 tile 的 Q（K/V 已有则不重复捕获），放入 Phase 2。

**预测**：needle 块的 g 小、MeanPool 排名靠前，不会被漏。

## 4. Phase 2（需要远端 GPU：只写脚本和命令，不生成结论）

- **G1 末尾 tile Q 的补捕获**（为 E8 服务）。
- **G2 任务压力测试**：比较 FlashPrefill（实际使用的版本和阈值）、V2（补偿开/关）与 dense。
  - 任务：RULER 各类 NIAH、VT、CWE、FWE、QA，长度 32K / 64K / 128K；可加 many-shot ICL 和 BFCL multi-turn。
  - 记录准确率和实际密度。
  - 预注册预测：掉点集中在 VT 和需要精确标识符的任务；聚合类任务（CWE/FWE）因为 α·max 阈值自适应，相对稳定。
- **G3 误差传播**：同一 prompt 分别做 dense prefill 和稀疏 prefill，保存若干层的 K/V 和最后位置的 hidden state。
  - 报告逐层 KV 相对偏移、答案首 token 分布的 KL。
  - BFCL multi-turn 中，每轮都用稀疏 prefill，看偏移是否随轮数累积。

## 5. 交付物

- **CSV**：`e1_second_order.csv`、`e1_cumulants.csv`、`e2_dilution_cancellation.csv`、`e3_bands.csv`、`e3_attenuation.csv`、`e4_nope_peaks.csv`、`e5_sink.csv`、`e6_output_error.csv`、`e6_block_drop.csv`、`e7_rbs.csv`、`e8_needle_blocks.csv`
- **图**：
  - E1 各方法 regret 条形图（包含 `mean_var2`）
  - E2 各组 `A_plus` / `A_minus` 的分布
  - E3 `rho_f` 与 `a_f` 的对比曲线；频带贡献堆叠图
  - E6 mass regret 与 `rel_err` 的散点图（按层着色）
  - E7 召回率与额外块数的曲线
- **报告** `VALIDATION_REPORT.md`：
  - 对 Q1–Q6 逐条给出“支持 / 不支持 / 不确定”，附关键数字、样本量和限制
  - 与各条预测逐一对照
  - 列出所有近似实现和所有未通过的校验
- 复现命令和依赖列表。

## 6. 必须通过的校验

- E0 复现数字。
- 频带 logit 求和等于总 logit（max abs err < 1e-4）。
- RoPE 反旋与再旋一致。
- 按块拼出的 dense 输出等于直接计算的 dense 输出（rel err < 1e-5）。
- 单块删除公式等于直接重算（rel err < 1e-4）。
- 恒等式 `mu - c = A_plus - A_minus` 成立。
- **任何校验失败**：停止相关实验并在报告中说明，不要通过调参绕过。
