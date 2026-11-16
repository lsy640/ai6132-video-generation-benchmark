# AI6132 Video Generation Benchmark

[English](README.md) | 简体中文

AI6132 Project 1 · Technical Review

在相同首图、英文提示词和种子下，比较四个开源图生视频模型的首图保持、动作自然度、提示词遵循、背景细节保持与推理成本。评估结合四人人工评分、VBench-I2V 自动指标和 Qwen3-VL 评审，并分析三者之间的一致性。

最终报告：[`group_11_report.pdf`](08-report/Review_of_Image_to_Video_Generation/group_11_report.pdf)（源文件 [`main.tex`](08-report/Review_of_Image_to_Video_Generation/main.tex)），成员贡献说明：[`group_11_contribution.pdf`](08-report/group_11_contribution.pdf)

## 实验设计

24 个固定案例（VBench-I2V 16:9 子集，11 类；16 个主体案例、8 个背景案例），种子均为 42，每案例每模型生成一次，共 96 条视频。每模型 18 条短片（目标 129 帧）和 6 条长片（目标 513 帧，单次连续生成）。输入清单见 [`01-dataset/cases.canonical.csv`](01-dataset/cases.canonical.csv)。

| 成员 | 模型 | 步数 | 原生分辨率／fps | 短／长帧数 |
| --- | --- | --- | --- | --- |
| A | Wan2.2-TI2V-5B | 10 | 1248×704／24 | 125／509 |
| B | HunyuanVideo-1.5 480p I2V step-distilled | 8 | 848×480／24 | 129／513 |
| C | LTX-Video 13B v0.9.8 distilled | 7+3 | 768×432／24 | 129／513 |
| D | LongCat-Video 480p + CFG-step 蒸馏 LoRA | 16 | 832×480／15 | 129／513 |

已知协议偏差：A 以 10 步运行（默认 50 步）且每条少 4 帧；D 为 15 fps，相同帧数下物理时长更长。各模型步数、分辨率不同，结果比较的是这些具体配置。

## 主要结果

人工评分（每视频取四人中位数，再对 24 条求均值，1–5 分）：

| 模型 | 首图保持 | 动作自然度 | 提示词遵循 | 背景保持 | 短片耗时 (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wan 2.2 | 3.79 | 2.96 | 4.08 | 3.42 | 464 |
| HunyuanVideo 1.5 | 4.25 | 3.71 | 4.48 | 4.02 | 173 |
| LTX-Video | **4.67** | **3.92** | **4.71** | **4.75** | **25** |
| LongCat-Video | 4.00 | 3.69 | 4.15 | 3.88 | 447 |

- 短片上四个模型接近；长片上只有 LTX-Video 保持稳定，Wan 2.2 和 LongCat-Video 的首图保持与背景保持均降到 2 分以下。
- 四名评审的序数 Krippendorff α 为 0.54–0.66，属中等一致。
- Qwen3-VL 在首图保持（ρ=0.60）和提示词遵循（ρ=0.51）上与人评相关，动作自然度与人评无相关（ρ=−0.06），动作分只作方法诊断。
- VBench 相似度指标集中在 0.96–0.99，不能复现人评排序。

样本为单种子、24 个固定案例，结论限于本次配置。完整分析与局限见报告。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| [`01-dataset/`](01-dataset/README.md) | 24 个案例的清单和参考图 |
| [`02-pilot/tc2/`](02-pilot/tc2/README.md) | TC2 环境准备脚本、SLURM 作业入口和生成记录 |
| [`03-generation/`](03-generation/) | 四个模型的生成代码、配置和交接工具 |
| [`ai6132_benchmark/`](ai6132_benchmark/) | 共享数据校验、视频处理、VBench 与 Qwen 评估实现 |
| [`04-vbench-evaluation/`](04-vbench-evaluation/README.md) | 评分副本制作与 VBench-I2V 评估 |
| [`05-human-evaluation/`](05-human-evaluation/README.md) | 人工评分网页／Excel 工具、收表校验与四份评分表 |
| [`06-vlm-evaluation/`](06-vlm-evaluation/README.md) | Qwen3-VL 评审协议 |
| [`07-analysis/`](07-analysis/README.md) | 报告图表脚本 |
| [`08-report/`](08-report/README.md) | LaTeX 报告、成员贡献说明和统计表 |
| `video-output/` | 各成员交付的原片与运行记录（不纳入版本控制） |

原始图片、模型权重、生成视频和匿名映射不纳入版本控制。实验协议与评分量表见 [PLAN.md](PLAN.md)。

## 运行

在 NTU TC2 上，Head Node 只用于编辑、Git 和提交作业；Python 和 GPU 工作全部通过 `sbatch` 在计算节点执行。

```bash
export AI6132_CODE="$PWD"
export AI6132_CPU_ENV=/absolute/path/to/conda/environment
sbatch 02-pilot/tc2/run_cpu.sh dataset --verify 01-dataset/cases.canonical.csv   # 校验输入
sbatch 07-analysis/make_report_figures.sh                                        # 重绘报告图
sbatch 08-report/build_report.sh                                                 # 编译报告和贡献说明
```

各模型的生成命令见 [Wan2.2](03-generation/Wan2.2/BENCHMARK.md)、[Hunyuan](03-generation/hunyuan_video/README.md)、[LTX](03-generation/ltx_video/README.md)、[LongCat](03-generation/longcat_video/README.md)；评估流程见 04–06 各目录说明。
