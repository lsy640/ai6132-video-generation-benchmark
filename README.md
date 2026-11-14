# AI6132 Video Generation Benchmark

AI6132 Project 1 · Technical Review

在相同首图、英文提示词和种子下，比较四个图生视频模型的首图保持、动作自然度、提示词遵循与推理成本。项目结合 VBench-I2V 自动指标、四人独立人工盲评和 Qwen3-VL 评审，记录生成成功率、耗时及显存，并分析评审之间的一致性。

[实验计划与评分量表](PLAN.md) · [数据说明](01-dataset/README.md) · [环境与运行说明](02-pilot/tc2/README.md) · [验收记录](02-pilot/tc2/VALIDATION.md)

## 实验设计

共享输入为 A 提供的 **24 个固定案例，覆盖 11 类**，包含 16 个主体案例与 8 个背景案例。四个模型使用相同输入，每案例每模型生成一次，共 **96 条正式视频记录**；每名评审独立评价全部记录，四人共提交 **384 条评分记录**。

| 长度组 | 原始帧数标注 | 原生目标帧数 | 每模型视频数 | 四模型合计 |
| --- | ---: | ---: | ---: | ---: |
| 短片 | 128 | 129 | 18 | 72 |
| 长片 | 512 | 513 | 6 | 24 |
| 合计 | — | — | **24** | **96** |

129/513 帧为当前统一的原生生成目标。各模型记录实际帧率和完整时长；以 24 fps 为例，两档视频约为 5.38 秒和 21.38 秒。长片采用单次连续生成。

生成器统一读取 [`cases.canonical.csv`](01-dataset/cases.canonical.csv)：

```text
case_id,category,image_path,prompt,seed,source_frame_num,target_frames
```

案例编号为 `case_0001` 至 `case_0024`，种子均为 42。原始清单、图片、提示词、顺序和长短分配保留，规范清单及图片由 SHA256 凭据绑定。该数据是固定子集，抽样过程未提供，类别级结果仅作描述性分析。

## 模型与成员分工

| 成员 | 生成模型 | 公共职责 |
| --- | --- | --- |
| A | Wan2.2-TI2V-5B | 准备并冻结公共数据与输入清单 |
| B | HunyuanVideo-1.5-480P-I2V-step-distill | 汇总生成记录、制作匿名评分副本、运行 VBench-I2V 与 Qwen3-VL |
| C | LTX-Video 2B v0.9.8 Distilled | 准备评分材料、随机观看顺序与收表校验 |
| D | LongCat-Video；备选 CogVideoX1.5-5B-I2V | 整合结果、计算一致性并完成报告 |

第四模型依据共同试跑结果冻结。各模型的步数、精度、卸载配置和备选规则见 [PLAN.md](PLAN.md)；所有成员均参与四模型的人工评审。

## 评估方法

- **统一评分素材：**保留原片，另制作覆盖完整时长的 768×432、12 fps、H.264 评分副本，保持比例并补边，完整解码验证后匿名分发。
- **VBench-I2V：**主体案例评估 `i2v_subject`、`motion_smoothness`、`dynamic_degree`；背景案例评估 `i2v_background`，保留逐视频结果。
- **人工盲评：**四位评审按各自随机顺序观看全片，独立给出首图保持、动作自然度、提示词遵循三项 1–5 分评分。
- **Qwen3-VL：**使用 Qwen3-VL-8B-Instruct；短片全片抽取 32 帧，长片全片抽取 128 帧进行整体评分。长片另分四个等时窗口，各抽取 32 帧作辅助诊断，分段分数独立保存。
- **结果分析：**比较生成成功率、质量分布和推理成本，报告人工评审一致性及人评与 Qwen 的相关性；整体、短片、长片分别统计。

失败、缺失和中断均保留在正式记录中。B 先提交个人评分，再运行正式 Qwen；四份人评收齐后解盲并释放模型评审结果。详细方法见 [完整协议](PLAN.md)。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| [`01-dataset/`](01-dataset/README.md) | 原始清单、规范清单与输入校验凭据 |
| [`02-pilot/tc2/`](02-pilot/tc2/README.md) | 环境准备、SLURM 入口与验收记录 |
| [`03-generation/`](03-generation/) | 模型生成配置及 LTX 生成、交接工具 |
| [`ai6132_benchmark/`](ai6132_benchmark/) | 共享数据校验、Hunyuan 生成、视频处理及评估实现 |
| [`04-vbench-evaluation/`](04-vbench-evaluation/README.md) | 全片预处理、匿名交接与 VBench 使用说明 |
| [`05-human-evaluation/`](05-human-evaluation/README.md) | 人工盲评材料、评分模板与收表工具 |
| [`06-vlm-evaluation/`](06-vlm-evaluation/README.md) | Qwen 整体评分与长片诊断使用说明 |
| [`07-analysis/`](07-analysis/) | 结果整合与图表目录，待填充 |
| [`08-report/`](08-report/) | 课程报告与成员贡献说明目录，待填充 |
| [`tests/`](tests/) | 共享工具与评估流程回归测试 |

## 快速开始

### 1. 获取代码与输入

```bash
git clone https://github.com/lsy640/ai6132-video-generation-benchmark.git
cd ai6132-video-generation-benchmark
```

图片与模型权重另行交接或下载。将 A 冻结的原始图片放入 `01-dataset/dataset/`，保留清单中的文件名；输入清单及校验凭据已随仓库提供。

### 2. 配置环境并校验数据

CPU 工具使用 Python 3.10+、Pillow，以及支持 libx264 的 FFmpeg/ffprobe。不同模型与评估阶段使用各自的个人 Conda 环境，安装步骤见 [TC2 说明](02-pilot/tc2/README.md) 和各模型文档。

在 NTU TC2 上，Head Node 用于编辑、Git 与提交/查询作业；所有 Python、测试、安装构建和 GPU 工作通过 `sbatch` 在计算节点执行。每次提交前查询当前配额与队列，按实际资源调整作业参数。

```bash
export AI6132_CODE="$PWD"
export AI6132_CPU_ENV=/absolute/path/to/personal/conda/environment
mytcinfo
squeue -u "$USER"
sbatch 02-pilot/tc2/run_cpu.sh dataset --verify 01-dataset/cases.canonical.csv
```

保留返回的作业 ID，用 `sacct -j <JOB_ID> --format=JobID,State,ExitCode` 确认成功后再进行下一步。默认 CPU 日志写入提交目录的 `ai6132_cpu_<JOB_ID>.out` 与 `.err`。各入口的资源、环境变量及完整参数见对应文档。

### 3. 生成与评估

按下列顺序执行，各阶段使用独立输出目录并冻结输入、配置和版本：

1. [Hunyuan 生成](03-generation/hunyuan_video/README.md)或 [LTX 生成与已完成批次交接](03-generation/ltx_video/README.md)。其余模型由相应成员交付同一规范的运行记录。
2. 汇总四模型输出，进行[全片验收、匿名副本制作与 VBench 评估](04-vbench-evaluation/README.md)。
3. 按[人工盲评流程](05-human-evaluation/README.md)分发材料并收表；B 提交个人评分后开展 [Qwen 评估](06-vlm-evaluation/README.md)。
4. 四份人评收齐后解盲，按 [PLAN.md](PLAN.md)整合结果与报告。

常规共同试跑使用独立的三个长短案例。本次 B 正式生成按明确指示跳过生成试跑，该例外已写入冻结记录；后续阶段仍执行各自的验收要求。

### 4. 运行测试

在 CPU 环境与配额满足要求、前一作业完成后提交：

```bash
sbatch 02-pilot/tc2/run_cpu.sh test
```

测试覆盖 LTX 生成协议与交接、人工评审工具、共享数据和评估流程，包含真实 FFmpeg 视频转换及合成模型接口回归。

## 当前进度

截至 **2026-11-14**：

| 阶段 | 状态与证据范围 |
| --- | --- |
| 公共输入 | 24 例、11 类规范清单及图片校验已完成 |
| LTX 生成 | C 报告 24/24 完成；代码已合入 main，实际视频与原始凭据待交接后独立验收，见 [C 批次报告](02-pilot/tc2/C_FORMAL_REPORT.md) |
| Hunyuan 生成 | 正式批次进行中，已有 129 帧与 513 帧成功记录；整批 24 例独立全片验收尚未完成 |
| 工具验证 | 最新 52 项回归测试全部通过，见 [验收记录](02-pilot/tc2/VALIDATION.md) |
| 正式评估与报告 | VBench、Qwen 和人工盲评工具已实现；全组正式评分、统计分析和最终报告待完成 |

当前尚无四模型完整质量比较结果。成员报告、工具回归与实际视频验收分别保留证据，生成及评估的详细状态以各阶段验收记录为准。

## 复现与产物管理

仓库保存源代码、配置、共享清单与文档。原始图片、模型权重、生成视频、原始评分和匿名映射不纳入版本控制，按需单独存储和交接。各阶段记录固定上游版本、依赖、输入及文件哈希、生成参数和真实运行状态；成功、失败与中断原件均保留。

本项目使用 VBench-I2V 固定子集及统一预处理，结果用于课程实验比较，不作为完整官方排行榜总分。完整实验约定、评分锚点、统计方法及课程交付要求见 [PLAN.md](PLAN.md)。
