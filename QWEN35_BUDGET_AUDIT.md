# Qwen3.5-4B CL-bench：生成预算与显存校准

日期：2026-09-21 至 2026-09-22。设备：RTX 4090 24GB。

## 已确认的问题

原正式运行使用 thinking + greedy + 2,048-token 输出上限。进度检查时，
前 14 条 Full/V1 输出全部触及上限，思考未闭合，最终答案为空。
这批输出保留在原目录，不作为新协议的续跑前缀。

将最长输入（固定样本索引 55，60,525 tokens）的 Full 输出上限放宽至
32,768 后，greedy 仍触顶。原始文本中的同一组表格检查重复约 75 次，
没有 `</think>`，也没有最终答案。这证明仅增加预算不能解决该样本的循环。
该轨迹耗时 1,011.1 秒，CUDA 峰值分配 17.748 GiB，峰值保留 20.313 GiB；
未发生 OOM。原始轨迹保存在远端
`results/clbench_qwen35_budget32k_probe/generations.jsonl`。

## 新的生成协议

采用本地固定模型快照 README 中的 general thinking 参数：
temperature=1.0、top_p=0.95、top_k=20、presence_penalty=1.5、
repetition_penalty=1.0；保留 thinking，输出上限 32,768，遇 EOS 提前停止。
Presence penalty 只作用于已生成 token。两种方法逐题重置 CUDA 采样随机数
生成器，种子均为 `42 + sample_index`。相同种子不意味着不同分布会生成相同文本。

这是对此前 greedy 协议的明确变更，两批质量结果应分开报告。
Full/V1 的模型、输入、输出预算和采样设置相同。
更换采样协议的直接依据是完整轨迹中的重复和模型作者的通用设置；
目前没有据此声称质量得分提高。

参考：固定快照 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` 的
`models/Qwen3.5-4B/README.md`，Sampling Parameters 与 Adequate Output Length。

## 校准与正式运行

采样校准使用固定样本 55（最长输入）、0（金融题）和 3（游戏规则题），
每题分别运行 Full 和 V1；原始生成和逐条显存/耗时记录保存在
`results/clbench_qwen35_budget32k_sampling_probe/`，已下载到本地同名目录。

| 样本 | 输入 tokens | 方法 | 输出 tokens | 完整生成秒数 | 峰值分配 GiB | 峰值保留 GiB |
|---|---:|---|---:|---:|---:|---:|
| 55，最长输入 | 60,525 | Full | 5,474 | 180.9 | 17.748 | 20.313 |
| 55，最长输入 | 60,525 | V1 | 6,986 | 645.1 | 17.746 | 20.061 |
| 0，金融题 | 18,442 | Full | 6,474 | 194.0 | 11.334 | 12.434 |
| 0，金融题 | 18,442 | V1 | 6,496 | 189.4 | 11.334 | 12.322 |
| 3，游戏规则题 | 28,153 | Full | 5,010 | 152.6 | 12.808 | 14.195 |
| 3，游戏规则题 | 28,153 | V1 | 5,278 | 155.4 | 12.809 | 14.158 |

6/6 均为 EOS 结束、thinking closed、非空 final answer，没有触顶或 OOM。
最长输入 V1 的 645.1 秒包含该形状的首次 Torch 编译，不能直接拿这一行
与 Full 计算 attention 加速比。该表用于生成完整性和显存校准；正式 prefill
测速仍先 warmup，再测每种方法的三次重复。

选择 32,768 作为正式上限，既符合模型说明，也给实测约 5K–7K 的轨迹留出余量。
小规模校准无法保证全部 100 题都在此上限内终止；正式输出保留 end_reason，
后续应报告触顶率和空答案率。当前结论是 RTX 4090 24GB 已通过这组校准。

正式入口为 `bash run_clbench_qwen35_long.sh run`；中断后使用同一入口的
`resume`。该入口派生新的输入快照，仅把每条的 `max_new_tokens` 改为 32,768；
题号、题序、messages、rubrics、input_ids 和所有对应 hash 保持不变。
总上下文上限为 98,304，容纳最长原始输入与输出预算，不重新筛选题目。

正式输出目录为 `results/clbench_qwen35_full100_thinking32k_sampling`。
100 题 × Full/V1 共 200 条生成，prefill 每种方法每题重复 3 次。
正式运行与校准分开；数学 gate、原始输入和旧结果均保留。

2026-09-22 00:42（UTC+8），正式任务通过 `screen` 会话 `exp35_long` 启动。
远端核验新旧 100 条输入在排除 `max_new_tokens` 字段后逐项相等。
日志为 `results/clbench_qwen35_full100_thinking32k_sampling.log`。
本次只启动生成和性能记录，质量评测仍需之后运行官方 judge。
