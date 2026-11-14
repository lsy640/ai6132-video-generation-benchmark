# C：LTX-Video 2B v0.9.8 Distilled

使用官方两阶段 I2V，首图条件为第 0 帧，关闭提示词扩写。

2026-11-14 已在 TC2 L40S 上完成一条合成环境测试：`768×432 / 121 帧 / 24 fps`，H.264 原片完整解码通过，时长约 5.04 秒。详见 [验证记录](../../02-pilot/tc2/VALIDATION.md)。共同试跑和正式生成待 A 提供数据后进行，参数由全组试跑后冻结。

## 环境与权重

使用独立 Python 3.10 环境，在 TC2 的 GPU 作业中安装和推理。Slurm 提交方式见 [TC2 说明](../../02-pilot/tc2/README.md)。

```bash
# 替换为实际存储路径。
C_STORAGE=/absolute/persistent/project/path
conda create -p "$C_STORAGE/envs/ltx-c" python=3.10 pip -y
conda activate "$C_STORAGE/envs/ltx-c"

# 在仓库根目录安装环境。
bash 03-generation/ltx_video/prepare_environment.sh "$C_STORAGE/LTX-Video"

# 下载模型、空间 upscaler、T5 文本编码器和 tokenizer。
python 03-generation/ltx_video/download_assets.py --assets "$C_STORAGE/ltx-assets"
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
python 03-generation/ltx_video/run_cases.py \
  --cases /absolute/path/cases.csv --phase formal --per-category 2 --validate-only
```

## Smoke、共同试跑、正式批次

三个阶段各用独立输出目录：

- `--phase smoke`：1 条合成样本，仅用于环境检查。
- `--phase pilot`：A 提供的 3 条共同试跑样本，用于检查生成、解码和评分。
- `--phase formal --per-category 2`：CSV 中每类前 2 组，按 CSV 原顺序运行，共 12 条；小组批准统一扩展后用 `--per-category 4` 共 24 条。

```bash
# 共同试跑。
python 03-generation/ltx_video/run_cases.py \
  --cases /absolute/shared/pilot_cases.csv --phase pilot \
  --upstream "$C_STORAGE/LTX-Video" --assets "$C_STORAGE/ltx-assets" \
  --output "$C_STORAGE/outputs/ltx-pilot" --offload-to-cpu

# 全组冻结正式数据和试跑配置后：
python 03-generation/ltx_video/run_cases.py \
  --cases /absolute/shared/cases.csv --phase formal --per-category 2 \
  --upstream "$C_STORAGE/LTX-Video" --assets "$C_STORAGE/ltx-assets" \
  --output "$C_STORAGE/outputs/ltx-formal" --offload-to-cpu
```

同一命令再次运行时，只继续尚未启动的 case。已成功、失败或中断的记录均保留，技术失败由小组确认后处理。正式批次始终使用同一输出目录和冻结的配置、输入及权重，每个 case 按清单种子生成一次。扩至 24 条时，需从一开始就使用完整 24 条清单，续跑时只改 `--per-category 4`。

试跑阶段可通过 `--width`、`--height`、`--num-frames`、`--frame-rate` 调整配置，正式运行时锁定参数。官方代码只在 GPU 总显存小于 30 GiB 时启用 CPU offload，因此 48GB L40S 上记录为 requested=true、effective=false。

## 输出与计时口径

每个 case 的 `native/` 保存原始 MP4。全部帧可解码、尺寸/帧数/fps 符合请求且时长不少于 5 秒时记为 `success`；画面质量由后续评审判断。

`runs.csv` 可直接交给 B，含 case/model ID、种子、版本、配置路径及哈希、原生规格、真实状态与失败原因、GPU/驱动、耗时和显存。`batch.json` 固定清单和图片哈希；`pipeline.json` 是实际配置；`environment.txt` 保存真实安装版本；每个 case 有 `request.json`、`result.json` 和完整 `inference.log`。

每个样本在独立进程调用官方 `infer`。`elapsed_seconds` 包含本地权重加载、文本编码、两阶段生成和视频写盘，不含下载和后续解码检查。`worker_wall_seconds` 另含导入与解码验证。显存记录为 PyTorch 本进程的 peak allocated / peak reserved（MiB）。

退出码 0 表示选中样本全部成功；1 表示存在技术失败、中断或未完成；2 表示清单、环境或冻结校验失败。中断且没有终态的记录标为 `incomplete`，需检查原日志及进程。

## 测试

```bash
cd 03-generation/ltx_video
python -m unittest -v test_runner.py
```

2026-11-14 本目录 6 项生成协议测试、人评工具 14 项测试通过，共 20 项。

官方依据：[仓库与安装](https://github.com/Lightricks/LTX-Video/tree/50f0aef1297574a4b29acbdb874e3a363a6d5f2e)、[2B 两阶段配置](https://github.com/Lightricks/LTX-Video/blob/50f0aef1297574a4b29acbdb874e3a363a6d5f2e/configs/ltxv-2b-0.9.8-distilled.yaml)、[推理入口](https://github.com/Lightricks/LTX-Video/blob/50f0aef1297574a4b29acbdb874e3a363a6d5f2e/ltx_video/inference.py)、[PyTorch 2.5.1 安装](https://pytorch.org/get-started/previous-versions/#v251)。
