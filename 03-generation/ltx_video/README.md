# C：LTX-Video 2B v0.9.8 Distilled

使用官方两阶段 I2V，首图条件为第 0 帧，关闭提示词扩写。

2026-11-14 已在 TC2 L40S 上完成一条合成环境测试：`768×432 / 121 帧 / 24 fps`，H.264 原片完整解码通过，时长约 5.04 秒。详见 [验证记录](../../02-pilot/tc2/VALIDATION.md)。共同试跑和正式生成待 A 提供数据后进行，参数由全组试跑后冻结。

## 环境与权重

使用独立 Python 3.10 环境，在 TC2 的 GPU 作业中安装和推理。所有 Python（包括 CPU 清单检查和测试）均通过 `sbatch` 在计算节点执行。每次提交前查询自己的 QoS；完整环境、资源和日志说明见 [TC2 说明](../../02-pilot/tc2/README.md)。从仓库根目录提交：

```bash
export AI6132_ROOT="$HOME/ai6132-c"
export AI6132_CODE="$PWD"
mytcinfo
sbatch --partition=MGPU-TC2 --qos=normal 02-pilot/tc2/bootstrap_env.sh
# 确认上一作业成功后，再提交安装及固定版本资产下载作业。
sbatch --partition=MGPU-TC2 --qos=normal 02-pilot/tc2/install_ltx.sh
```

`prepare_environment.sh` 固定官方 commit `50f0aef1297574a4b29acbdb874e3a363a6d5f2e`，安装 torch 2.5.1 / torchvision 0.20.1 的 cu121 构建。`constraints.txt` 中的依赖组合已在 Python 3.10.22、L40S / 驱动 570.148.08 环境通过依赖、导入、CUDA/BF16 和生成检查。每批输出同时保存实际安装版本。

下载固定 Hugging Face revision，并保存每个文件的 SHA256：

| 内容 | 仓库及 revision |
|---|---|
| `ltxv-2b-0.9.8-distilled.safetensors`、`ltxv-spatial-upscaler-0.9.8.safetensors` | `Lightricks/LTX-Video@8984fa25007f376c1a299016d0957a37a2f797bb` |
| `text_encoder/`、`tokenizer/` | `PixArt-alpha/PixArt-XL-2-1024-MS@b89adadeccd9ead2adcb9fa2825d3fabec48d404` |

推理使用本地权重，Hugging Face 设为离线模式。保留官方两阶段 7+3 个 timesteps、BF16、CFG=1、STG=0，设置 `prompt_enhancement_words_threshold=0` 关闭扩写。

## 共享样本清单

由 A 冻结 `cases.csv` 和横版首图。清单格式示例：

```csv
case_id,category,image_path,prompt,seed
single_001,single_person,images/single_001.png,"Original English prompt from the shared manifest",42
```

`case_id` 由英文字母、数字、点、下划线或短横线组成，整批唯一。`category` 保持 A 的原始标签，正式清单包含六类。`seed` 为 0 至 2³²−1 的整数，prompt 保留原文。

首图路径字段也接受 `image`、`first_frame`、`first_frame_path`、`reference_image`；多个字段同时存在时非空值必须相同。相对路径以 CSV 所在目录为基准，也可使用 `--image-root` 指定。首图由 A 统一裁为 16:9，与输出比例一致；官方推理会补边，再裁回请求尺寸。

```bash
# 检查清单、种子、图片路径和样本顺序。
export AI6132_CPU_ENV="$AI6132_ROOT/env"
sbatch 02-pilot/tc2/run_cpu.sh validate-cases \
  --cases /absolute/path/cases.csv --phase formal --per-category 2
```

## Smoke、共同试跑、正式批次

三个阶段各用独立输出目录：

- `--phase smoke`：1 条合成样本，仅用于环境检查。
- `--phase pilot`：A 提供的 3 条共同试跑样本，用于检查生成、解码和评分。
- `--phase formal --per-category 2`：CSV 中每类前 2 组，按 CSV 原顺序运行，共 12 条；小组批准统一扩展后用 `--per-category 4` 共 24 条。

```bash
# 共同试跑。
sbatch --partition=MGPU-TC2 --qos=normal 02-pilot/tc2/run_ltx.sh \
  pilot /absolute/shared/pilot_cases.csv "$AI6132_ROOT/outputs/ltx-pilot"

# 全组冻结正式数据和试跑配置后：
sbatch --partition=MGPU-TC2 --qos=normal 02-pilot/tc2/run_ltx.sh \
  formal /absolute/shared/cases.csv "$AI6132_ROOT/outputs/ltx-formal" --per-category 2
```

同一命令再次运行时，只继续尚未启动的 case。已成功、失败或中断的记录均保留，技术失败由小组确认后处理。正式批次始终使用同一输出目录和冻结的配置、输入及权重，每个 case 按清单种子生成一次。扩至 24 条时，需从一开始就使用完整 24 条清单，续跑时只改 `--per-category 4`。

`selection.json` 单独记录当前正式选中的 12 或 24 条及顺序，允许 12→24 扩批，拒绝缩回 12 条。`runs.csv` 在首个 worker 启动前就覆盖全部选中样本；未启动样本为 `not_started`，无耗时和 GPU 实测值，也不会创建 case 目录，后续可正常启动。中断后的既有尝试仍保留为 `incomplete`，不会自动重抽。B 应以 `selection.json` 核对交付行数，并保留 `not_started`/`incomplete` 等非成功状态；不能只统计已有视频。

代码哈希也在 `batch.json` 中冻结。升级 runner 后不能修改旧记录来绕过校验；保留旧输出，使用原代码续跑，或经小组确认后为新代码建立独立批次。只有旧版 attempt log 而没有 `selection.json` 时，单独读取记录不会猜测批次大小。

试跑阶段可通过 `--width`、`--height`、`--num-frames`、`--frame-rate` 调整配置，正式运行时锁定参数。官方代码只在 GPU 总显存小于 30 GiB 时启用 CPU offload，因此 48GB L40S 上记录为 requested=true、effective=false。

## 输出与计时口径

每个 case 的 `native/` 保存原始 MP4。全部帧可解码、尺寸/帧数/fps 符合请求且时长不少于 5 秒时记为 `success`；画面质量由后续评审判断。

`runs.csv` 与 `selection.json` 一起交给 B，含 case/model ID、种子、版本、配置路径及哈希、原生规格、真实状态与失败原因、GPU/驱动、耗时和显存。`batch.json` 固定完整输入清单和图片哈希；`pipeline.json` 是实际配置；`environment.txt` 保存真实安装版本；每个已启动 case 有 `request.json`、`result.json`（中断时可能缺失）和 `inference.log`。

每个样本在独立进程调用官方 `infer`。`elapsed_seconds` 包含本地权重加载、文本编码、两阶段生成和视频写盘，不含下载和后续解码检查。`worker_wall_seconds` 另含导入与解码验证。显存记录为 PyTorch 本进程的 peak allocated / peak reserved（MiB）。

退出码 0 表示选中样本全部成功；1 表示存在技术失败、中断或未完成；2 表示清单、环境或冻结校验失败。中断且没有终态的记录标为 `incomplete`，需检查原日志及进程。

## 测试

```bash
sbatch 02-pilot/tc2/run_cpu.sh test
```

测试使用合成数据，覆盖清单冻结、完整交付记录、扩批/续跑，以及人评素材与评分校验；不替代 GPU 共同试跑。历史 20 项测试结果见环境验证记录，修复后的验证结果另行记录。

官方依据：[仓库与安装](https://github.com/Lightricks/LTX-Video/tree/50f0aef1297574a4b29acbdb874e3a363a6d5f2e)、[2B 两阶段配置](https://github.com/Lightricks/LTX-Video/blob/50f0aef1297574a4b29acbdb874e3a363a6d5f2e/configs/ltxv-2b-0.9.8-distilled.yaml)、[推理入口](https://github.com/Lightricks/LTX-Video/blob/50f0aef1297574a4b29acbdb874e3a363a6d5f2e/ltx_video/inference.py)、[PyTorch 2.5.1 安装](https://pytorch.org/get-started/previous-versions/#v251)。
