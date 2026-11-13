# 图生视频评测计划

AI6132 Project 1 · Technical Review

## 目标

给定相同首图与原始英文提示词，比较四个图生视频模型的首图保持、动作自然度、提示词遵循和推理成本。每名成员负责一套生成管线，并独立评价四个模型的全部正式视频。

## 模型与分工

| 成员 | 生成模型与主配置 | 协作职责 |
| --- | --- | --- |
| A | **Wan2.2-TI2V-5B**。官方 I2V，50 步，CPU offload。主批次完成后，可用六类各 1 组做 25 步对照。 [官方推理说明](https://github.com/Wan-Video/Wan2.2#run-text-image-to-video-generation) | 维护公共样本清单，冻结首图、提示词、样本 ID 和种子。 |
| B | **HunyuanVideo-1.5-480P-I2V-step-distill**。480p、8 步，开启 offload，关闭提示词改写和超分。 [官方仓库](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) | 汇总生成日志，检查视频解码，制作统一评分副本，运行 VBench-I2V 和 Qwen3-VL。 |
| C | **LTX-Video 2B v0.9.8 Distilled**。官方两阶段配置，首图作为第 0 帧条件，关闭提示词扩写。 [官方配置](https://github.com/Lightricks/LTX-Video/blob/main/configs/ltxv-2b-0.9.8-distilled.yaml) | 编制统一评分表，随机化每人的观看顺序，收齐四份原始评分。 |
| D | **LongCat-Video**。48 GB 卡优先试跑 480p、16 步蒸馏。24 GB 卡或 LongCat 试跑未达标时，试跑 **CogVideoX1.5-5B-I2V** 的官方 offload 配置。 [LongCat 官方示例](https://github.com/meituan-longcat/LongCat-Video/blob/main/run_demo_image_to_video.py) · [CogVideoX 模型卡](https://huggingface.co/zai-org/CogVideoX1.5-5B-I2V) | 合并人评、模型评审与自动指标，计算一致性并整合报告。 |

## 数据、试跑与统一生成标准

1. **冻结样本。** 从 VBench-I2V 的单人、多人、动物、建筑、风景、室内六类各用固定随机种子抽 4 组。正式评估前冻结 `cases.csv`，包含 `case_id`、类别、首图路径、原始英文 `prompt` 和 `seed`。四个模型共用同一份横版首图裁剪与原始提示词，关闭提示词扩写。 [VBench-I2V 官方协议](https://github.com/Vchitect/VBench/blob/master/vbench2_beta_i2v/README.md)
2. **试跑。** 每套候选配置用相同的 3 个独立样本检查显存、生成、解码和评分。根据单卡串行实测耗时冻结第四模型。试跑样本用于验证流程和校准评分。
3. **正式生成。** 每个 `case_id` 和模型按清单种子生成 1 条，按预先冻结的批次顺序运行。先完成每类前 2 组，共 **12 组 × 4 模型 = 48 条**。预算允许时统一扩到 **24 组 × 4 模型 = 96 条**。
4. **统一评分副本。** 保留原片。评分副本取前 5 秒，统一为 **768 × 432、12 fps、H.264**，保持画面比例并按需补边。
5. **交接。** 每位成员交付视频和 `runs.csv`。记录 `case_id`、`model_id`、权重版本、配置、`seed`、原生规格、GPU 型号、耗时、峰值显存、状态及失败原因。

## VBench-I2V 自动评估

主体类报告 `i2v_subject`、`motion_smoothness` 和 `dynamic_degree`，背景类报告 `i2v_background`。保存逐视频结果并按 `case_id`、`model_id` 合并。报告清楚标注所用 VBench-I2V 子集及每组 1 条的采样设置。 [评分维度与代码](https://github.com/Vchitect/VBench/blob/master/vbench2_beta_i2v/README.md#5-evaluation)

## 四人独立人工盲评

四人先用 3 条试跑视频校准量表。正式视频用匿名 `clip_id`，移除文件名和界面中的模型标识。每位成员按单独随机顺序查看全部正式视频、对应首图和提示词。四份独立评分全部提交后再解盲和讨论。 [人评协议参考](https://artificialanalysis.ai/methodology/video) · [动作与状态变化参考](https://arxiv.org/abs/2406.08656)

| 维度 | 1 分 | 3 分 | 5 分 |
| --- | --- | --- | --- |
| 首图保持 | 主要主体或场景明显改变，难以辨认为同一对象或场景。 | 关键特征大体保留，有局部变形或明显漂移。 | 主体身份与场景主要特征始终可辨，变化与动作或视角相容。 |
| 动作自然度 | 明显卡顿、跳变、肢体变形，或应有动作基本缺失。 | 动作可辨，有短暂顿挫、重复或局部不自然。 | 动作连贯，速度和变化符合情境，未见明显跳变。 |
| 提示词遵循 | 主要指令缺失或明显违背。 | 完成部分要求，关键细节或动作不完整。 | 主要主体、动作与场景要求均能在视频中确认。 |

2 和 4 分表示相邻锚点的中间状态。同分允许。提示词未指定动作时，按动作与画面情境是否相容判断。每人提交一份 `ratings.csv`，每条视频一行，含 `rater_id`、`clip_id`、`reference_fidelity`、`motion_naturalness`、`prompt_adherence` 和 `comment`。生成或解码失败的评分留空并记录原因。C 检查记录完整性、重复 ID 和分数范围，在解盲前请原评审补齐漏填项。

最低批次每人评 48 条，四人共 192 条记录。完整批次每人评 96 条，四人共 384 条记录。每条记录含三个维度。解盲时用 `clip_map.csv` 将 `clip_id` 关联到 `case_id` 与 `model_id`。

## Qwen3-VL 独立评审

B 在生成释放 GPU 后，用 **Qwen3-VL-8B-Instruct** 本地逐条评估同一批匿名视频。每条视频开启独立请求，输入为对应首图、原始英文 prompt、评分副本与同一套量表。先用试跑视频验证解码、显存和输出格式。正式评审设定为 5 秒视频均匀采样 24 帧、保留时间信息、固定权重版本与视觉尺寸预算，使用 `do_sample=False`。交卷前，人评与模型评审各自看不到对方分数。 [官方模型卡](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) · [官方视频输入说明](https://github.com/QwenLM/Qwen3-VL#new-qwen-vl-utils-usage)

> 根据首图、原始提示词和视频，按附上的 1、3、5 分锚点评分，2 和 4 分用于中间状态。只依据视频中可见的内容，为每项给一句简短证据。仅返回 JSON，字段为 reference_fidelity、motion_naturalness、prompt_adherence 和 evidence。前三项为 1 至 5 的整数，evidence 含对应三项的说明。

保存逐条原始回复与标准化 JSON，按 `clip_id` 关联。如果回复不可解析或分数越界，要求按同一量表重输 JSON 一次，保留两次回复，仍失败记为缺失。记录实际帧数、配置与异常。

## 一致性与结果整合

- **人工评审一致性。** 用四份独立正式评分，按维度计算序数 **Krippendorff’s α**，报告有效视频数、缺失数、两两完全同分率和相差不超过 1 分的比例。缺乏变异导致系数无法计算时标为 NA 并注明原因。复看高分歧样本。 [Krippendorff 一致性计算说明](https://www.asc.upenn.edu/sites/default/files/2021-03/Computing%20Krippendorff%27s%20Alpha-Reliability.pdf)
- **人工与模型评审对照。** 按视频连接 Qwen3-VL 评分与四位评审中位分，逐维度报告 Spearman 等级相关、平均绝对分差和有效配对数。等级无变异时将相关值记为 NA。结合完整视频复核分歧，注意抽帧对瞬时瑕疵判断的影响。
- **主表。** 人评分先取每条视频四位评审的中位数，再按模型汇总各维度的中位数、四分位区间与有效样本数。横向质量对比用四模型均成功的共同样本。主表单列 VBench-I2V 指标、模型评审分、人评分布、成功率、失败数、耗时和峰值显存。不同 GPU 上的耗时注明硬件与配置。另做一版剔除所属跑模成员评分的汇总，检查结论变化。
- **图表。** 给出各维度一致性表、3 组代表性的并排帧序列和失败案例。若执行 Wan 25 步对照，四人用同一量表独立评分并单列结果。 [Wan 默认采样配置](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_ti2v_5B.py)

## 排期与课程交付

- **2026 年 11 月 14 日中午**冻结模型配置、数据清单和评分表。四人各跑自己的模型，按 GPU 队列协调。
- **11 月 15 日中午**冻结生成输出，下午完成人工盲评、模型评审与汇总。
- **11 月 16 日**由 D 代表小组在 NTULearn 提交至少 4 页的 `group_XX_report.pdf`、`group_XX_contribution.pdf` 和 `group_XX_code.zip`。代码包包含源代码与运行配置。成员贡献说明逐人记录负责的模型、完成的评审与承担的汇总工作。

相关评测阅读：[VBench-2.0](https://arxiv.org/abs/2503.21755) · [Video-Bench](https://github.com/Video-Bench/Video-Bench) · [StreamEval](https://github.com/StreamEval/StreamEval) · [Avatar-Forever](https://arxiv.org/abs/2608.12107) · [AlayaWorld](https://github.com/AlayaLab/AlayaWorld/blob/main/inference/README.md)
