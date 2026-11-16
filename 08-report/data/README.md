# 统计表

报告中的表格和图表都从这些 CSV 取数。人工评分每条视频先取四名评审的中位数；`member_slot` A、B、C、D 依次对应 Wan 2.2、HunyuanVideo 1.5、LTX-Video、LongCat-Video；`target_frames` 为 129（短片）、513（长片）或 `all`。

| 文件 | 内容 | 报告中的位置 |
| --- | --- | --- |
| `human_summary.csv` | 各模型、各维度的人评均值与中位数，按全部／短片／长片分组 | 表 2、图 2 |
| `human_agreement.csv` | 四维的序数 Krippendorff α、两两同分率和相差不超过 1 分的比例 | 表 2 |
| `human_video_medians.csv` | 96 条视频各维度的四人中位数、最低分和最高分 | 附录 B |
| `human_excluding_owner.csv`、`human_excluding_owner_cases.csv` | 剔除生成该模型的成员后的均值对照，以及中位数发生变化的案例 | — |
| `vbench_summary.csv` | VBench-I2V 四个指标按模型和长短分组的统计 | 表 3 |
| `vbench_paired.csv` | 模型两两之间的 VBench 配对差值与自助法区间 | — |
| `vbench_human_correlations.csv` | VBench 指标与对应人评维度的 Spearman 相关 | — |
| `qwen_diagnostic_ratings.csv` | Qwen3-VL 对 96 条视频的四维评分（匿名编号） | — |
| `qwen_summary.csv` | Qwen3-VL 评分按模型和长短分组的统计 | 表 4 |
| `qwen_human_pairs.csv` | 逐视频、逐维度的 Qwen 分数与人评中位数配对 | 图 5 |
| `qwen_human_comparison.csv` | Qwen 与人评的 Spearman 相关和平均绝对分差 | 表 4、图 5 |
| `qwen_sampling_coverage.csv` | Qwen 每条视频实际采样的帧数占比 | — |
| `qwen_common_cases.json` | 各维度下 Qwen 与人评都有有效分数、可以配对的案例编号 | — |

数据来源：人评来自 [`05-human-evaluation/human-evaluation-result/`](../../05-human-evaluation/human-evaluation-result/) 的四份评分表，VBench 和 Qwen 的逐视频结果分别由 [`ai6132_benchmark/vbench.py`](../../ai6132_benchmark/vbench.py) 和 [`ai6132_benchmark/qwen_fast.py`](../../ai6132_benchmark/qwen_fast.py) 产生。
