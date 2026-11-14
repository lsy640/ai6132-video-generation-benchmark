# AI6132 图生视频评测

AI6132 Project 1 · Technical Review

在同一批 VBench-I2V 首图与提示词上比较四个图生视频模型。四人各负责一个模型的生成，并独立评价四个模型的全部正式输出。结果包括 VBench-I2V 自动指标、人工盲评、Qwen3-VL 独立评审，以及推理成本和生成成功率。

[完整实验计划、评分量表和四人分工](PLAN.md)

## 阶段目录

| 目录 | 用途 |
| --- | --- |
| [`01-dataset/`](01-dataset/) | 样本抽取、共享清单和输入预处理 |
| [`02-pilot/`](02-pilot/) | 单卡试跑、解码校验和评测验证 |
| [`03-generation/`](03-generation/) | 四个模型的推理入口和运行配置 |
| [`04-vbench-evaluation/`](04-vbench-evaluation/) | VBench-I2V 自动评分和统一视频预处理 |
| [`05-human-evaluation/`](05-human-evaluation/) | 人工盲评量表、匿名流程和评分表结构 |
| [`06-vlm-evaluation/`](06-vlm-evaluation/) | Qwen3-VL 评审提示词、推理和输出校验 |
| [`07-analysis/`](07-analysis/) | 一致性计算、结果合并和图表 |
| [`08-report/`](08-report/) | 技术报告与组员贡献说明 |

## C 的运行入口

- [TC2 资源检查与基础环境](02-pilot/tc2/README.md)
- [CPU 清单检查、人评工具与测试入口](02-pilot/tc2/run_cpu.sh)
- [LTX-Video 2B 两阶段生成](03-generation/ltx_video/README.md)
- [人工盲评准备与收表检查](05-human-evaluation/README.md)

共享输入由 A 统一冻结。仓库保存代码、配置和说明；权重、视频、评分及匿名映射另行存储。

## A 实际清单与 B 的运行入口

当前基准采用A提供的11类24案例，按原分配生成18条129帧短片与6条513帧长片，四模型共96条。人工、VBench、Qwen均覆盖完整时长，Qwen另做长片四段诊断。

2026-11-14：47项CPU回归及真实513帧VAE解码显存回归通过，固定版本资产核验通过。首轮54400在VAE解码阶段OOM，记录保留；修复后的正式作业54414已在TC2N08的L40S上运行，按用户指示跳过生成试跑。代码在生成期间先行同步main，完整24例结果以后续全片验收为准。见 [验证记录](02-pilot/tc2/VALIDATION.md)。

- [原始输入与规范清单](01-dataset/README.md)
- [B 的 Hunyuan 生成与试跑门槛](03-generation/hunyuan_video/README.md)
- [全片预处理、匿名交接与 VBench](04-vbench-evaluation/README.md)
- [Qwen 整体及分段评审](06-vlm-evaluation/README.md)

CPU通过 `run_cpu.sh`，生成和GPU评估通过 `run_b_gpu.sh` 提交。真实模型的共同试跑与正式实验状态须以验收记录为准，不以代码或模拟测试完成代替。
