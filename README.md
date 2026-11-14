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
- [LTX-Video 2B 两阶段生成](03-generation/ltx_video/README.md)
- [人工盲评准备与收表检查](05-human-evaluation/README.md)

共享输入由 A 统一冻结。仓库保存代码、配置和说明；权重、视频、评分及匿名映射另行存储。
