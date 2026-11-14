# C：LTX-Video 2B v0.9.8 Distilled

使用官方两阶段 I2V，首图条件为第 0 帧，关闭提示词扩写。这里不训练模型，不产生或替代人工评分。

当前 `768×432 / 121 帧 / 24 fps` 已于 **2026-11-14 在 TC2 的 L40S 上通过 1 条合成样本 smoke**，生成的 H.264 原片可完整解码，时长约 5.04 秒。这次环境检查不计入共同试跑、评分或正式结果；预设仍待 A 的 3 个共同样本验证和全组冻结。完整指标及计时口径见 [验证记录](../../02-pilot/tc2/VALIDATION.md)。官方内部会按模型要求补边，最终输出裁回请求尺寸。不得根据单条视频观感更换种子或重抽。

## 环境与权重

在 TC2 已分配的 GPU 作业中准备，Mac 用于连接、编辑和查看结果。使用独立 Python 3.10 环境；不要在登录节点进行推理。持久存储位置和磁盘额度应先确认。

```bash
# 先使用学校提供的 conda 创建并激活独立 Python 3.10 环境。
# 以下路径自行填为项目实际持久存储位置；C_STORAGE 不是系统变量。
C_STORAGE=/absolute/persistent/project/path
conda create -p "$C_STORAGE/envs/ltx-c" python=3.10 pip -y
conda activate "$C_STORAGE/envs/ltx-c"

# 在本小组仓库根目录运行；仅安装环境，不下载权重。
bash 03-generation/ltx_video/prepare_environment.sh "$C_STORAGE/LTX-Video"

# 明确执行下载：只取固定2B模型、空间upscaler、T5文本编码器与tokenizer。
python 03-generation/ltx_video/download_assets.py --assets "$C_STORAGE/ltx-assets"
```

`prepare_environment.sh` 固定官方 commit `50f0aef1297574a4b29acbdb874e3a363a6d5f2e`，安装 torch 2.5.1 / torchvision 0.20.1 的 cu121 构建。`constraints.txt` 的依赖组合已在本次 Python 3.10.22、L40S / 驱动 570.148.08 环境通过 `pip check`、官方入口导入、CUDA/BF16 检查和单条 smoke。它不是上游提供的锁文件；实际安装版本仍须随批次保存。换环境后应重新检查。

下载固定 Hugging Face revision，并保存每个文件的 SHA256：

| 内容 | 仓库及 revision |
|---|---|
| `ltxv-2b-0.9.8-distilled.safetensors`、`ltxv-spatial-upscaler-0.9.8.safetensors` | `Lightricks/LTX-Video@8984fa25007f376c1a299016d0957a37a2f797bb` |
| `text_encoder/`、`tokenizer/` | `PixArt-alpha/PixArt-XL-2-1024-MS@b89adadeccd9ead2adcb9fa2825d3fabec48d404` |

推理阶段强制 Hugging Face 离线，使用已下载的绝对路径，不自动拉取其他版本。保留官方两阶段 7+3 个 timesteps、BF16、CFG=1、STG=0；唯一模型配置修改是 `prompt_enhancement_words_threshold=0`。没有下载扩写使用的 Florence/Llama。不得把模型换成 13B、0.9.6、FP8 或 LTX-2。

## 共享样本清单

由 A 冻结 `cases.csv` 和相同横版首图。规范字段：

```csv
case_id,category,image_path,prompt,seed
single_001,single_person,images/single_001.png,"Original English prompt from the shared manifest",42
```

这只是格式示意，不能当作正式样本。`case_id` 只能用英文字母、数字、点、下划线和短横线，且不能重复。`category` 保持 A 的原始标签；正式清单须有六类。`seed` 必须是清单中的显式整数，范围 0 至 2³²−1。保留 prompt 原文，包括空格与标点。

首图路径字段也接受 `image`、`first_frame`、`first_frame_path`、`reference_image`；多个字段同时存在时非空值必须相同。相对路径默认相对于 CSV 所在目录，必要时显式使用 `--image-root`。工具不另作裁图，实际推理前要求首图与输出同为 16:9，避免各模型输入内容不一致。

```bash
# Mac/CPU 可先检查 CSV、种子、图片存在性和样本顺序，无须推理依赖或权重。
python 03-generation/ltx_video/run_cases.py \
  --cases /absolute/path/cases.csv --phase formal --per-category 2 --validate-only
```

## Smoke、共同试跑、正式批次

三种阶段分别用独立输出目录，不能混用：

- `--phase smoke`：恰好 1 条明确标识的环境检查样本。用于确认环境，**不计入共同试跑、评分或正式结果**。
- `--phase pilot`：恰好 3 条与队友相同的独立试跑样本，用于检查生成/解码/评分。未拿到 A 的共同样本前不要启动。
- `--phase formal --per-category 2`：CSV 中每类前 2 组，按 CSV 原顺序运行，共 12 条；小组批准统一扩展后用 `--per-category 4` 共 24 条。

```bash
# 共同试跑示例；首次连接只需完成环境步骤，不要用临时样本冒充共同试跑。
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

原样重跑同一命令会继续尚未启动的 case，已成功、失败或中断的 case 均不自动重跑。扩至 24 条时应使用同一份已包含完整 24 条的清单和同一输出目录，只改 `--per-category 4`。配置、文件内容、prompt、seed 或权重变化会被拒绝。正式输出目录应始终唯一；不要换目录绕过防重复保护。技术失败需和小组确认处置并保留原始记录，不允许用新结果覆盖失败或低质量原片。

`--width`、`--height`、`--num-frames`、`--frame-rate` 只用于试跑阶段显式讨论配置，不会自动降配；正式运行时保存并锁定参数。官方代码只在 GPU 总显存小于 30 GiB 时实际启用 CPU offload，因此 48GB L40S 上 requested=true、effective=false 属于官方行为。

## 输出、成本口径与限制

每个 case 的 `native/` 保留官方原始 MP4；不重编码或覆盖。全部帧可解码、尺寸/帧数/fps符合请求、时长不少于 5 秒才记为 `success`。质量差但可正常解码的片段仍算生成成功，交给人评正常评分。

`runs.csv` 可直接交给 B，含 case/model ID、种子、版本、配置路径及哈希、原生规格、真实状态与失败原因、GPU/驱动、耗时和显存。`batch.json` 固定清单和图片哈希；`pipeline.json` 是实际配置；`environment.txt` 保存真实安装版本；每个 case 有 `request.json`、`result.json` 和完整 `inference.log`。

每个样本在独立进程调用官方 `infer`。`elapsed_seconds` 从模型构建前开始，包含本地权重加载、文本编码、两阶段生成和官方视频写盘，不包含下载和后续完整解码检查；不是纯去噪时长。`worker_wall_seconds` 另外包含导入与解码验证。两项显存分别为 PyTorch 本进程的 peak allocated / peak reserved（MiB），不等于整卡 NVML 总占用。

退出码 0 表示选中样本全部成功；1 表示存在技术失败/中断/未完成；2 表示清单、环境或冻结校验失败。中断且没有终态的记录标为 `incomplete`，先查原日志及进程，不重新抽样。不要删除原目录以强行续跑。

本次单条合成 smoke 的 `elapsed_seconds` 为 19.791758 秒，PyTorch peak allocated 为 26623.0693 MiB、peak reserved 为 37226 MiB。它们仅描述这一条输入，不能作为性能平均值、显存下限或所有输入都能成功的保证。官方 T5 初始化先加载再转换 BF16，主机 RAM 也可能限制试跑。遇到 OOM 保留错误，先检查已分配 GPU/RAM；工具不私自量化、降低分辨率或减少帧数。

本地协议测试（不安装推理依赖、不运行 GPU）：

```bash
cd 03-generation/ltx_video
python -m unittest -v test_runner.py
```

2026-11-14 本目录 6 项生成协议测试、人评工具 14 项 CPU 测试均通过，共 20 项。CPU 测试检查工具逻辑，不替代共同样本的生成和评分验收。

官方依据：[仓库与安装](https://github.com/Lightricks/LTX-Video/tree/50f0aef1297574a4b29acbdb874e3a363a6d5f2e)、[2B 两阶段配置](https://github.com/Lightricks/LTX-Video/blob/50f0aef1297574a4b29acbdb874e3a363a6d5f2e/configs/ltxv-2b-0.9.8-distilled.yaml)、[推理入口](https://github.com/Lightricks/LTX-Video/blob/50f0aef1297574a4b29acbdb874e3a363a6d5f2e/ltx_video/inference.py)、[PyTorch 2.5.1 安装](https://pytorch.org/get-started/previous-versions/#v251)。
