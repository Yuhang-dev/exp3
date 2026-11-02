# 指针跳转判别实验

状态：脚本已写好；除语法和离线造数检查外，**全部实验等待远端运行**，下面没有任何结果。

## 问题

32K 合成检索上 `fp_v1` 目标准确率 45.8%，dense 100%（8 个 prompt）。离线重算显示错误基本是抄错来源，而不是数值漂移。本实验回答：错误是不是实现造成的？错在哪个 token？答案读到的哪部分 K/V 造成了错误？任务的哪个结构触发了错误？

## 阶段

| 阶段 | 内容 | 判读 |
| --- | --- | --- |
| `check` | 现有 kernel 数学检查 | 环境门槛 |
| `prepare` | 重建 8 个 32K prompt，SHA256 必须与归档一致；生成反事实 prompt | 不一致即停止 |
| `smoke` | 4K 单 prompt 跑通 patch、margin 和 `fp_row_norm` | 流程门槛 |
| `alpha` | V1 的 α=0–0.16、`fp_row_norm`、`mean_native`、`mean_balanced` | α=0 未达 100% 或 K/V 明显偏离 dense → 实现问题优先；随 α 单调上升 → 路由问题 |
| `patch` | 18 个 K/V 组合 + 4 个方法基线，每个都做自由生成和 teacher-forced margin | 见下表 |
| `length` | 8K/16K/24K/32K | 定位 16K 到 32K 的断崖 |
| `factors` | 12 个 level × 8 prompt，dense 与 V1 | 见下表 |
| `holdout` | 32 个新 prompt，配置已冻结为 α=0.08 | 校准集上的结论能否推广；只跑一次 |

`fp_row_norm` 是新方法：`mean_balanced` 的逐行归一化路由，加 V1 的纯精确输出，与现有方法构成路由 × 尾项 2×2 消融。

## K/V patching 的判读

所有方法都只 prefill `input[:-1]`，之后走 dense 解码，所以条件之间只有答案读到的 K/V 不同。

| 结果 | 含义 |
| --- | --- |
| `v1_fix_all` 不等于 `dense_base` | patch 流程有误，其余结果作废 |
| `v1_fix_needle` 能救回 | 错误集中在 needle 自身的表示 |
| `v1_fix_needle_k` 救回、`_v` 救不回 | 寻址问题：答案找不到正确记录 |
| `v1_fix_needle_v` 救回、`_k` 救不回 | 内容问题：找到了，但内容已损坏 |
| `v1_fix_filler` 能救回、needle 救不回 | 干扰项表示被改变，成为竞争者 |
| `v1_fix_question` 能救回 | 问题行的表示被改变（解码读到的 query 上下文） |
| 某一层段单独能救回 | 错误由该层段的 K/V 决定 |
| `dense_break_*` 掉分 | 该区域的 V1 K/V 足以造成错误 |

## 反事实任务的判读

| 因子 | 改动 | 判读 |
| --- | --- | --- |
| `filler_index_7` | 干扰项 index 首位 0→7 | 错 index 变成 7 → 被干扰项多数模式带偏；仍是 0 → 另有原因 |
| `order_cyclic` | 输出顺序改为 1、2、0 | 错 index 变成 1 → 抄第一行；仍是 0 → 不是抄第一行 |
| `filler_other_ns` | 干扰项换 namespace | 恢复 → 需要共享前缀才会失败 |
| `position_swap` | key0 放 86%，key2 放 8% | 区分 index 效应和位置效应 |
| `shift_16…112` | 同一内容相对块网格平移 | 同一 needle 靠近边界才失败 → 块边界线索成立 |

块边界的阈值（偏移 ≤21 或 ≥120）是在归档数据上事后发现的，这里用新平移数据做检验。`filler_other_ns` 的 tokenization 会变化，个别 prompt 最长到 34417 token（约 +5%）；变长只会让任务更难，所以恢复仍可解读，不恢复时要看长度。

## 运行

```bash
mkdir -p /root/autodl-tmp && cd /root/autodl-tmp && git clone -b pointer-study https://github.com/Yuhang-dev/exp3.git && cd exp3
bash setup_instance.sh all
bash run_pointer_study.sh all
```

粗估总时长约 3 小时（按归档 sweep 每次 prefill 加生成约 7 秒推算，未实测），`smoke` 和 `alpha` 跑完后可据日志修正。RTX 4090（24 GB）即可，32K 峰值显存约 20 GB；磁盘留 60 GB。

可分阶段运行，`run_pointer_study.sh` 的每个阶段名都可单独作为参数。结果打包为 `results/pointer.tar.gz`（不含 `.pt`），汇总在 `results/pointer/SUMMARY.md`。

## 依赖的数据

本实验只需要 Qwen2.5-7B-Instruct 权重（commit `a09a354`），合成 prompt 由种子确定性重建，不需要下载数据集。

RULER 13 任务和 LongBench 的后续实验才需要数据：`bash setup_instance.sh ruler`、`bash setup_instance.sh longbench`。`run_rescue_benchmarks.sh longbench` 还依赖 `results/longbench_v2_native32k_inputs116/inputs.pt`，需要重新生成或上传。
