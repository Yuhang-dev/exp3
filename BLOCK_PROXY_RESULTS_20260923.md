# 两份 block proxy 报告：实测结果与后续实验

分析日期：2026-09-23。下列结果来自用户返回的真实实验文件，CPU 分析已完成；没有运行新 GPU 实验。

## 1. 实际覆盖范围

文件名顺序与用户描述相反，以归档内 metadata 为准：

| 返回文件 | 实际实验 | 样本与层 |
| --- | --- | --- |
| `block_proxy_report.zip` | LongBench v2 | 1 条 Single-Document QA / Academic；层 0、14、27 |
| `block_proxy_report (1).zip` | RULER NIAH | MK1:56、MK1:71 各层 0、7、14、21、27；MQ:59 仅层 0、7、14 |

LongBench 样本为 `longbench-v2-66ebc0c95a08c7b9b35de7f3`，问题是 LLaMA 预训练使用的代码数据比例。其既有完整评测中 Dense 和 V1 均答对。不能把本次局部注意力质量损失直接解释为答题失败。

两份报告均统计全部 28 个 Query heads；详细块分布和 token profiles 只覆盖 heads 0、7、14、21。Q block 为 64/128/256，K block 为 32/64/128/256；每个 Q tile 采样 16 行，在上下文 25%、50%、75%、90% 附近取 tile。固定远端预算 2048 tokens，sink/local 参数为 256/384。

五项主数据文件的 SHA256 均匹配归档 metadata。summary 无重复配置、无数值缺失，oracle 次序成立；约 1e-7 的负 regret 是浮点舍入。源 input token ID 的哈希也与 capture 对齐。证据见本地 `results/audits/block_proxy_reports_20260923/coverage.json`、`integrity.json`。

**捕获命令遗漏：** commit `59b175d` 中的 LongBench 命令缺少 `--all-samples`，因此默认只捕获子集第 0 条。这是交付命令的问题，不是用户操作问题。已修正文档，并提供只补跑其余 7 条的脚本，见末节。

## 2. 先把三种损失拆开

在相同远端区域、相同 token 预算和相同保护区域下，定义：

- T：每个 Query 独立选 token 的 oracle 保留质量。
- R：每个 Query 独立选整块的 oracle 保留质量。
- C：同一 Query tile 共享块集合的 oracle 保留质量。
- M：MeanPool 代理选块后的保留质量。

于是 `T - M = (T - R) + (R - C) + (C - M)`，依次是整块粒度损失、Query 共享集合损失、代理排序损失。这里的质量均指 dense attention probability mass。

以下为 Q=K=128、全部 28 heads 的均值，单位为注意力质量百分点，**不是任务准确率百分点**：

| 数据 | 层 | 整块粒度 T−R | Query 共享 R−C | 代理排序 C−M |
| --- | ---: | ---: | ---: | ---: |
| RULER | 0 | 9.00 | 1.20 | 1.93 |
| RULER | 14 | 5.68 | 2.36 | 7.53 |
| RULER | 27 | 12.00 | 2.47 | 0.78 |
| LongBench | 0 | 8.78 | 1.47 | 2.48 |
| LongBench | 14 | 4.47 | 2.11 | 9.82 |
| LongBench | 27 | 8.73 | 3.28 | 1.03 |

**这批样本的第 14 层主要暴露代理不足，第 27 层主要暴露整块执行的粒度代价。** 第 27 层即便把 selector 换成最优共享块 selector，可收回的质量也有限。不能把所有层统一归因于 MeanPool 失效。

RULER 第 27 层只有两条样本，其他上述层有三条；LongBench 只有一条，尚不能宣称跨任务普遍规律。各层分开报告，不用不匹配的层覆盖计算任务总体优劣。

![不同 block size 的损失分解](results/audits/block_proxy_reports_20260923/01_loss_decomposition.png)

证据：`layer_blocksize.csv`、`sample_layer.csv`、`head_layer.csv`。

## 3. 块内分布：漏选块存在少数 token 占主导的情况

对块内条件概率 r，定义有效支撑 `N_eff = 1 / sum(r_j^2)`。均匀分布时等于块大小，单 token 独占时接近 1。

在第 14 层、Q=K=128、详细保存的四个 heads 上，按真实块质量加权得到：

| 数据 | 共享块 oracle 选中且 MeanPool 命中 | oracle 选中但 MeanPool 漏选 |
| --- | ---: | ---: |
| RULER：有效支撑 | 8.94 | 3.20 |
| LongBench：有效支撑 | 9.98 | 2.87 |
| RULER：Jensen gap | 3.45 | 6.48 |
| LongBench：Jensen gap | 4.57 | 8.29 |

这支持一个具体机制：**部分重要块的质量来自极少数高 logit token；均值描述符稀释了它们。** 这不是说所有块都尖锐，也不是说四个详细 heads 代表全部 heads。

进一步解码高质量漏选块，找到以下实例；排名从 1 开始，块质量为 sampled Query rows 的均值：

| 样本 / L14 head | K block 起点 | 块质量 | 真实质量排名 → 代理排名 | 峰值 token |
| --- | ---: | ---: | ---: | --- |
| RULER MQ:59 / H7 | 9984 | 13.7% | 1 → 167 | ` won`，位置 10053 |
| LongBench / H7 | 11008 | 11.8% | 2 → 107 | `/h`，位置 11110 |
| LongBench / H7 | 13184 | 11.7% | 1 → 150 | `1`，位置 13207 |

所保存代表 Query 行中，峰值 token 分别占**块内条件质量**的 99.92%、99.99%、99.98%，不是整行注意力的这些比例。RULER 的 `won` 来自普通背景文章中的 “won't”；LongBench 的 `/h` 来自代词串，`1` 来自参考文献年份。

RULER 三条输入在相近位置出现同一背景文本的尖峰，不能算三种独立语料的重复验证。当前可以称为“普通文本上的高注意力集中位置”；是否属于 attention sink、是否由 K norm 导致、是否影响输出，需要额外检验。不能仅凭注意力概率称其为语义关键 token。

![漏选块的 token 分布实例](results/audits/block_proxy_reports_20260923/03_missed_block_spikes.png)

证据：`block_distributions.csv`、`largest_missed_blocks.csv`、`decoded_missed_blocks.json`。解码使用对应 Qwen tokenizer，并核验原始 input IDs 与 capture 哈希。

## 4. 数学解释与诊断干预

对实际 RoPE 后的 Q/K，令 `s_ij = q_i^T k_j / sqrt(d)`，块大小为 B。令 `mu_ib` 为块内平均 logit，`Z_ib = sum_j exp(s_ij)`，则：

`log Z_ib = log B + mu_ib + g_ib`

`g_ib = log(mean_j exp(s_ij)) - mean_j s_ij = KL(u || r_ib) >= 0`。

MeanPool 给出的是 `mu_ib`。对于相同大小的两个块，若代理认为 a 优于 b，margin 为 `m = mu_ia - mu_ib > 0`，真实质量排序正确的精确条件是 `m > g_ib - g_ia`。所以要研究的是**竞争块之间的 gap 差与 margin**，而不是要求每个块近似均匀。

这是逐 Query 行的恒等式。对多个 Query 行共享块集合时，还要处理各行全局分母及跨行聚合；不能直接拿平均 g 代入逐行判据宣布 tile 排序正确。

第 14 层的诊断结果，仍为相对共享块 oracle 的质量损失，单位百分点：

| 评分方法 | RULER | LongBench |
| --- | ---: | ---: |
| MeanPool 原始跨行聚合 | 7.53 | 9.82 |
| 四个等长子块分别 MeanPool | 7.40 | 9.71 |
| MeanPool + 真实全局逐行分母 | 7.19 | 9.88 |
| 真实块指数和 + 原始跨行聚合 | 0.86 | 1.31 |

在这些配置里，补回块内指数和结构的诊断收益远大于仅替换行分母。四个子均值也不能充分恢复单 token 尖峰：128-token block 切成四段后，每段仍有 32 tokens。

“真实块指数和”要读取完整 logits，是定位问题的昂贵诊断，不是可部署算法。各项干预改变不同组成，不能把这些差值当成独立、可加的因果贡献。

证据：`method_layer.csv`、`paired_method_gain.csv`。

## 5. Block size 给故事线增加了什么

LongBench 第 14 层、固定 Q=128 和远端 2048-token 预算：

| K block | 平均 gap | 整块粒度损失 | 代理排序损失 |
| --- | ---: | ---: | ---: |
| 32 | 2.27 | 2.64 | 10.59 |
| 64 | 2.60 | 3.46 | 10.60 |
| 128 | 2.94 | 4.47 | 9.82 |
| 256 | 3.21 | 5.45 | 9.55 |

块变大时，块内非均匀性增加，但相对块 oracle 的代理损失没有随之增加。原因分析必须同时考虑 oracle 上限降低、选中块数减少，以及竞争块的 gap 差和 margin。**不能由较大的 g 直接推出排序更差，也不能由较小的相对 regret 推出大块更好。**

当前可以组织成这样的研究主线：先建立 token→独立 block→共享 block→廉价 proxy 的分层解释，再用 gap 与 margin 定位代理的有效范围，最后针对少量尖峰驱动的漏选进行修复。粒度损失与代理损失分别测量，使算法改进对准真实瓶颈。

## 6. 下一步的具体顺序

1. 补齐其余七条自然文档输入，覆盖多文档、长对话、代码等任务，再判断上述规律的覆盖面。
2. 利用已保存 Q/K，针对全头统计中表现突出的 heads 增加详细分析：如 L14 的 H4/H7/H9/H24，以及自然样本 L0 的 H3/H15。无需重新做模型前向捕获。沿用实际 RoPE 后 logits，距离作为分析变量保留。
3. 利用已保存 V，测量高质量漏选块的局部输出影响。固定该层 Q/K/V，删除块 b 并重新归一化时：`O - O_without_b = sum_{j in b} p_j (v_j - O) / (1 - P_b)`。质量高但 V 与当前输出相近的块，删除后的影响可能很小。
4. 再检验“均值描述背景 + 少量 token 代表尖峰”的廉价描述符。首先测能否找到这些峰值、改善排序及输出，再测 selector 开销。K norm 或跨 Query 重复出现频率可作为待验证候选，当前没有证据证明任一候选足够有效，也未作新颖性或 SOTA 结论。

本报告的漏选实例来自固定预算 `mean_raw` 诊断，它只用每 tile 的 16 条 sampled Query rows。真实 V1 使用完整 Query block 和阈值选块；归档 `v1_reference.csv` 提供汇总统计，未提供完整选择 mask。因此不能把这里的具体漏选块直接说成真实 V1 必然漏选。

## 7. 补跑命令与复现

在远端执行，读取原来的八条输入子集，仅捕获索引 1–7，写入新的时间戳目录，不覆盖已有实验：

```bash
cd /root/autodl-tmp/exp3
git pull --ff-only
bash run_longbench_block_proxy_remaining.sh
```

全部七条 capture 和离线分析仍待远端运行。

本地汇总分析脚本：

```powershell
python analyze_block_proxy_reports.py `
  'C:/Users/Yuhang/Downloads/block_proxy_report.zip' `
  'C:/Users/Yuhang/Downloads/block_proxy_report (1).zip' `
  --out results/audits/block_proxy_reports_20260923
```

该脚本输出 coverage、CSV 和前两张对比图。token 解码案例及第三张图由本次审计目录中的 `inspect_cases.py` 生成，另需本地保存的原始输入和对应 tokenizer；报告 ZIP 本身未包含原始 input IDs。
