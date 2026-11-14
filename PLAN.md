# 图生视频评测计划

AI6132 Project 1 · Technical Review

## 目标

给定相同首图与原始英文提示词，比较四个图生视频模型的首图保持、动作自然度、提示词遵循和推理成本。每名成员负责一套生成管线，并独立评价四个模型的全部正式视频。

## 模型与分工

| 成员 | 生成模型与主配置 | 协作职责 |
| --- | --- | --- |
| A | **Wan2.2-TI2V-5B**。官方 I2V，50 步，CPU offload。主批次完成后，可用预先冻结的代表子集做 25 步对照，单列结果且不计入正式96条。 [官方推理说明](https://github.com/Wan-Video/Wan2.2#run-text-image-to-video-generation) | 维护公共样本清单，冻结首图、提示词、样本 ID 和种子。 |
| B | **HunyuanVideo-1.5-480P-I2V-step-distill**。480p、8 步，开启 offload，关闭提示词改写和超分。 [官方仓库](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) | 汇总生成日志，检查视频解码，制作统一评分副本，运行 VBench-I2V 和 Qwen3-VL。 |
| C | **LTX-Video 2B v0.9.8 Distilled**。官方两阶段配置，首图作为第 0 帧条件，关闭提示词扩写。 [官方配置](https://github.com/Lightricks/LTX-Video/blob/main/configs/ltxv-2b-0.9.8-distilled.yaml) | 编制统一评分表，随机化每人的观看顺序，收齐四份原始评分。 |
| D | **LongCat-Video**。48 GB 卡优先试跑 480p、16 步蒸馏。24 GB 卡或 LongCat 试跑未达标时，试跑 **CogVideoX1.5-5B-I2V** 的官方 offload 配置。 [LongCat 官方示例](https://github.com/meituan-longcat/LongCat-Video/blob/main/run_demo_image_to_video.py) · [CogVideoX 模型卡](https://huggingface.co/zai-org/CogVideoX1.5-5B-I2V) | 合并人评、模型评审与自动指标，计算一致性并整合报告。 |

## 数据、试跑与统一生成标准

1. **冻结样本。** 采用 A 已交付的 `01-dataset/cases.csv` 和 `01-dataset/dataset/`，共 **24 个案例、11 类**。单人、多人各 3 例，动物、交通工具、食物、植物、其他主体、建筑、风景、室内、抽象各 2 例；主体 16 例、背景 8 例。保留原始顺序、提示词、全部 `seed=42` 及长短分配，不重新抽样。抽样过程未提供，报告称固定子集，不声称已验证随机抽样。[VBench-I2V 分类](https://github.com/Vchitect/VBench/blob/master/vbench2_beta_i2v/README.md#25-image-content)
2. **清单适配。** 原始 CSV 与图片不覆盖。派生 `cases.canonical.csv`：`case_id,category,image_path,prompt,seed,source_frame_num,target_frames`。图片数字前缀 01—24 对应 `case_0001`—`case_0024`，`image_type` 映射 `category`，原 `frame_num` 保留为 `source_frame_num`。原 128/512 分别映射为合法原生 **129/513 帧**；生成器按逐条目标执行，不裁回 128/512。所有生成入口只接受该规范，正式批次默认读取全部24例，帧数由每行目标决定。保存原清单、规范清单及图片哈希。
3. **共同试跑。** 每套候选配置使用相同的 3 个独立样本，至少覆盖一个 129 帧短片和一个 513 帧长片，验证生成、完整解码、主机内存、显存及评估格式。试跑与正式输入分离，试跑分数不计入正式结果。513 帧须实测，合法帧数不代表资源可行；第四模型按长短共同试跑冻结。

   本次B生成按2026-11-14用户明确指示跳过生成试跑，直接生成正式24例；在冻结批次中记录 `pilot_validation.status=skipped_by_user` 与原始指示。继续保留输入、资产、参数、完整解码和失败记录校验，未将环境检查声称为模型试跑通过。
4. **正式生成。** 每案例每模型按清单种子生成 1 条：每模型 **18 条短片＋6 条长片＝24 条**；四模型共 **72 条短片＋24 条长片＝96 条**。使用模型原生帧率并记录实际秒数；24 fps 下约 5.38/21.38 秒，15 fps 下约 8.60/34.20 秒。长片主协议为单次连续生成，失败不得静默改为续写、拼接、循环或插帧。冻结顺序，保留成功、失败和中断尝试，禁止自动重抽。
5. **全片评分副本。** 保留原片；评分副本覆盖完整视频，取消只取前 5 秒的限制。统一为 **768×432、12 fps、H.264、yuv420p**，保持比例并补边；采用固定 FFmpeg、CRF 18、medium preset，去除音轨与来源 metadata。转换后完整解码，时长误差至多 1/12 秒。原片有效而转换工具失败属于流程故障，不计为模型生成失败。
6. **交接。** 每位成员交付原片、完整 `runs.csv`、选定案例记录和冻结配置。记录 case/model、种子、来源/目标/实际帧数、fps、时长、版本、GPU、耗时、显存、状态及失败原因。未启动或缺失项保留，不能只用已有视频计成功率。B 自动混排并冻结匿名 ID；C 只接收六列匿名清单、视频和首图，映射与详细日志独立封存。

## VBench-I2V 自动评估

主体七类报告 `i2v_subject`、`motion_smoothness` 和 `dynamic_degree`，背景四类报告 `i2v_background`。使用与人评及 Qwen 相同的完整评分副本、固定上游版本和缓存权重；通过 `custom_input` 关联对应首图，保留逐视频原始结果并按 case/model 合并。无失败时共有 224 个适用的视频—指标项（64×3＋32）。失败和评估异常保留空值及原因，不填零。明确标注 24 案例固定子集、每组每模型 1 条及统一预处理，不作为完整官方排行榜总分。[评分维度与代码](https://github.com/Vchitect/VBench/blob/master/vbench2_beta_i2v/README.md#5-evaluation)

## 四人独立人工盲评

四人先用 3 条试跑视频校准量表。正式视频用匿名 `clip_id`，移除文件名和界面中的模型标识。每位成员按单独随机顺序查看全部正式视频、对应首图和提示词。四份独立评分全部提交后再解盲和讨论。 [人评协议参考](https://artificialanalysis.ai/methodology/video) · [动作与状态变化参考](https://arxiv.org/abs/2406.08656)

| 维度 | 1 分 | 3 分 | 5 分 |
| --- | --- | --- | --- |
| 首图保持 | 主要主体或场景明显改变，难以辨认为同一对象或场景。 | 关键特征大体保留，有局部变形或明显漂移。 | 主体身份与场景主要特征始终可辨，变化与动作或视角相容。 |
| 动作自然度 | 明显卡顿、跳变、肢体变形，或应有动作基本缺失。 | 动作可辨，有短暂顿挫、重复或局部不自然。 | 动作连贯，速度和变化符合情境，未见明显跳变。 |
| 提示词遵循 | 主要指令缺失或明显违背。 | 完成部分要求，关键细节或动作不完整。 | 主要主体、动作与场景要求均能在视频中确认。 |

2 和 4 分表示相邻锚点的中间状态。同分允许。提示词未指定动作时，按动作与画面情境是否相容判断。每人提交一份 `ratings.csv`，每条视频一行，含 `rater_id`、`clip_id`、`reference_fidelity`、`motion_naturalness`、`prompt_adherence` 和 `comment`。生成或解码失败的评分留空并记录原因。C 检查记录完整性、重复 ID 和分数范围，在解盲前请原评审补齐漏填项。

每人观看完整视频并覆盖 **96 条正式记录**，四人共 **384 条记录**，每条包含三个维度；技术失败保留行并留空分数。可分场完成，但保持分配顺序和独立评分。B 先提交自己的评分并冻结哈希，再执行 Qwen；交卷前不打开匿名映射或模型评审结果。全部四份人评收齐后，才释放映射及 Qwen 输出给 D。这是流程隔离，不是对同一账户的强制保密；保留可能识别自生成视频的局限。

## Qwen3-VL 独立评审

B 在生成释放 GPU 且个人评分交卷后，用固定版本 **Qwen3-VL-8B-Instruct** 本地单卡 BF16、batch size 1 评估同一批完整匿名评分视频。每个请求独立对话，只输入首图、原始英文 prompt、视频和量表，不输入生成来源、人评分数或其他自动结果。采用 `do_sample=False`、`max_new_tokens=512`，不默认量化。[官方模型卡](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) · [视频 metadata 接口](https://github.com/QwenLM/Qwen3-VL#new-qwen-vl-utils-usage)

- **整体主评分：**短片全片均匀抽 32 帧，长片全片抽 128 帧，均包含首尾。三项评分保持原量表；要求依据全片而非开头或最佳片段，关注后程漂移、重复、停滞和跳变，不要求一次性动作在各段反复完成。
- **长片辅助诊断：**按完整实际时长分四个连续等时窗口，每窗独立抽 32 帧，输入原始首图与提示词，保留绝对时间。仅评价首图保持和动作自然度，不单独判断提示词是否在别段完成；分段不读取整体评分，不平均或替代主分。
- **视觉预算：**视频每帧最多 384×224 像素，首图最多 768×432，长短使用同一每帧预算。保存实际尺寸、输入帧数/索引/时间戳和视觉网格，禁止 processor 二次抽帧。最长请求须在试跑中验证显存。
- **记录数量：**无失败时为 96 次整体请求＋96 次分段请求＝192 次请求。整体表始终保留 96 行，辅助表保留 96 行；视频技术失败或模型回复失败留空并写原因。

> 根据首图、原始提示词和视频，按附上的 1、3、5 分锚点评分，2 和 4 分用于中间状态。只依据视频中可见的内容，为每项给一句简短证据。仅返回 JSON，字段为 reference_fidelity、motion_naturalness、prompt_adherence 和 evidence。前三项为 1 至 5 的整数，evidence 含对应三项的说明。

保存逐请求原始回复、规范 JSON、采样记录、配置、耗时及显存，按匿名 clip/request ID 关联。仅整体请求包含 prompt_adherence；分段只返回两个诊断维度。每个请求最多修复一次非法 JSON、字段或分数范围，两次回复均保留；仍失败记缺失。某个分段失败不丢弃已完成整体评分。流程/GPU 错误保留已完成项并停止，不批量伪造模型评分失败。Qwen 输出封存至四份人评收齐。

## 一致性与结果整合

- **人工评审一致性。** 用四份独立正式评分，按维度计算序数 **Krippendorff’s α**，报告有效视频数、缺失数、两两完全同分率和相差不超过 1 分的比例。缺乏变异导致系数无法计算时标为 NA 并注明原因。复看高分歧样本。 [Krippendorff 一致性计算说明](https://www.asc.upenn.edu/sites/default/files/2021-03/Computing%20Krippendorff%27s%20Alpha-Reliability.pdf)
- **人工与模型评审对照。** 按视频连接 Qwen3-VL 评分与四位评审中位分，逐维度报告 Spearman 等级相关、平均绝对分差和有效配对数。等级无变异时将相关值记为 NA。结合完整视频复核分歧，注意抽帧对瞬时瑕疵判断的影响。
- **主表。** 人评分先取每条视频四位评审的中位数，再按模型汇总各维度的中位数、四分位区间与有效样本数。横向质量对比用四模型均成功的共同样本。主表单列 VBench-I2V 指标、模型评审分、人评分布、成功率、失败数、耗时和峰值显存。不同 GPU 上的耗时注明硬件与配置。另做一版剔除所属跑模成员评分的汇总，检查结论变化。
- **长度与类别分层。**整体、短片和长片分别报告分布、有效数量、失败数和时长，相关性只使用对应全片主评分，不能将四段当成独立案例。类别不均衡且每类只有 2—3 例，不做稳定类别级长片排名。短长组采用不同输入，组间均分差不能直接解释为长度的因果效应；长片内部前后变化仅作辅助证据。
- **图表。** 给出各维度一致性表、3 组代表性的并排帧序列和失败案例。若执行 Wan 25 步对照，四人用同一量表独立评分并单列结果。 [Wan 默认采样配置](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_ti2v_5B.py)

## 排期与课程交付

- **2026 年 11 月 14 日中午**冻结模型配置、数据清单和评分表。四人各跑自己的模型，按 GPU 队列协调。
- **11 月 15 日中午**冻结生成输出，下午完成人工盲评、模型评审与汇总。
- **11 月 16 日**由 D 代表小组在 NTULearn 提交至少 4 页的 `group_XX_report.pdf`、`group_XX_contribution.pdf` 和 `group_XX_code.zip`。代码包包含源代码与运行配置。成员贡献说明逐人记录负责的模型、完成的评审与承担的汇总工作。

相关评测阅读：[VBench-2.0](https://arxiv.org/abs/2503.21755) · [Video-Bench](https://github.com/Video-Bench/Video-Bench) · [StreamEval](https://github.com/StreamEval/StreamEval) · [Avatar-Forever](https://arxiv.org/abs/2608.12107) · [AlayaWorld](https://github.com/AlayaLab/AlayaWorld/blob/main/inference/README.md)
