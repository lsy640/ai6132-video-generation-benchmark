# C：LTX-Video 13B v0.9.8 Distilled

使用 BF16、官方两阶段 7+3 timesteps，第二阶段 `tone_map_compression_ratio=0.6`，首图条件为第 0 帧，关闭提示词扩写。模型 ID 为 `ltxv-13b-0.9.8-distilled`。

2026-11-15 全组同意从 2B 改为 13B，使用同一规范清单重跑全部 24 条：18 条 129 帧、6 条 513 帧，均为 768×432 / 24 fps；原图、prompt、seed 和顺序不变。13B 已通过 case_0001 的 129/513 帧技术试跑，正式作业 54620 进行中。旧 2B 作业 54359 的 24 条结果和 [PR #2](https://github.com/lsy640/ai6132-video-generation-benchmark/pull/2) 来源保留为历史，详见 [C 批次报告](../../02-pilot/tc2/C_FORMAL_REPORT.md)。

## 环境与权重

使用独立 Python 3.10 环境，在 TC2 的 GPU 作业中安装和推理。所有 Python（包括 CPU 清单检查和测试）均通过 `sbatch` 在计算节点执行。每次提交前查询自己的 QoS；完整环境、资源和日志说明见 [TC2 说明](../../02-pilot/tc2/README.md)。从仓库根目录提交：

```bash
export AI6132_ROOT="$HOME/ai6132-c"
export AI6132_CODE="$PWD"
export AI6132_LTX_ASSETS="$AI6132_ROOT/ltx_assets_13b"
mytcinfo
sbatch --partition=MGPU-TC2 --qos=normal 02-pilot/tc2/bootstrap_env.sh
# 确认上一作业成功后，再提交安装及固定版本资产下载作业。
sbatch --partition=MGPU-TC2 --qos=normal 02-pilot/tc2/install_ltx.sh
```

`prepare_environment.sh` 固定官方 commit `50f0aef1297574a4b29acbdb874e3a363a6d5f2e`，建立 torch 2.5.1 / torchvision 0.20.1 的 cu121 基础环境。该旧环境在513帧VAE解码失败；长短正式生成使用独立的 `env-cu126`（torch2.6.0+cu126、torchvision0.21.0、cuDNN9.5.1）。基础环境安装完成后，通过 `bootstrap_cu126.sh` 克隆升级；安装会检查100GB Home预算，须在GPU批处理作业执行。已有cu126环境时直接复用。不要在新环境重跑旧 constraints 安装。每批输出冻结完整实际版本，并逐案例记录cuDNN版本。

```bash
mytcinfo
sbatch 02-pilot/tc2/bootstrap_cu126.sh
# 确认安装作业成功后；run_ltx.sh 也默认选择这个环境。
export AI6132_LTX_ENV="$AI6132_ROOT/env-cu126"
export AI6132_CPU_ENV="$AI6132_LTX_ENV"
```

下载固定 Hugging Face revision，并保存每个文件的 SHA256：

| 内容 | 仓库及 revision |
|---|---|
| `ltxv-13b-0.9.8-distilled.safetensors`、`ltxv-spatial-upscaler-0.9.8.safetensors` | `Lightricks/LTX-Video@8984fa25007f376c1a299016d0957a37a2f797bb` |
| `text_encoder/`、`tokenizer/` | `PixArt-alpha/PixArt-XL-2-1024-MS@b89adadeccd9ead2adcb9fa2825d3fabec48d404` |

安装和生成入口默认使用独立的 `$AI6132_ROOT/ltx_assets_13b`，可通过 `AI6132_LTX_ASSETS` 指定已校验的 13B 资产目录。旧 2B 权重及输出继续保留；下载前核对 Home 剩余空间。

推理使用本地权重，Hugging Face 设为离线模式。`ltxv-13b-0.9.8-no-enhance.json` 保留官方两阶段 7+3 个 timesteps、BF16、CFG=1、STG=0 及第二阶段 tone map compression ratio 0.6，设置 `prompt_enhancement_words_threshold=0` 关闭扩写。正式条件固定为 `conditioning_strength=1.0`、`image_cond_noise_scale=0.15`，使用官方首图预处理。

`memory_adapter.py` 按组件顺序加载：先缓存 T5 正负提示词编码并释放 T5，再加载 BF16 Transformer；两阶段完成后释放 Transformer，再沿官方路径解码。Transformer 不在两阶段之间卸载到 CPU。策略记录为 `sequential_bf16_prompt_cache_final_transformer_release`；相同适配在 2B 129 帧回归中与原基线视频 SHA256 一致。13B 长短技术试跑和证据范围见 [验收记录](../../02-pilot/tc2/VALIDATION.md)。

## 共享样本清单

由 A 冻结原始 `cases.csv` 和横版首图，生成器只读取七列规范清单 `cases.canonical.csv`：

```csv
case_id,category,image_path,prompt,seed,source_frame_num,target_frames
case_0001,single-human,dataset/01.jpg,"Original English prompt from the shared manifest",42,512,513
```

`case_id` 格式为 `case_` 加四位数字且整批唯一，`category` 保持 A 原始标签，seed为0至2³²−1整数，prompt保留原文。正式清单固定为11类24案例、18条129帧和6条513帧，逐条校验source_frame_num/target_frames的128→129、512→513映射。详见[清单适配](../../01-dataset/README.md)。

首图路径字段统一为 `image_path`。相对路径以 CSV 所在目录为基准，也可使用 `--image-root` 指定。首图由 A 统一裁为 16:9，与输出比例一致；官方推理会补边，再裁回请求尺寸。

```bash
# 检查清单、种子、图片路径和样本顺序。
export AI6132_CPU_ENV="$AI6132_ROOT/env-cu126"
sbatch 02-pilot/tc2/run_cpu.sh validate-cases \
  --cases 01-dataset/cases.canonical.csv --phase formal
```

## Smoke、共同试跑、正式批次

三个阶段各用独立输出目录：

- `--phase smoke`：1 条采用同一规范字段的合成样本，目标129帧，仅用于环境检查。
- `--phase pilot`：A 提供的 3 条独立共同试跑样本，必须同时覆盖129和513帧，用于检查生成、解码和评分。
- `--phase formal`：默认完整使用A的11类24案例，18条129帧＋6条513帧，保持原顺序。

```bash
# 共同试跑。
sbatch --partition=MGPU-TC2 --qos=normal 02-pilot/tc2/run_ltx.sh \
  pilot /absolute/shared/pilot_cases.csv "$AI6132_ROOT/outputs/ltx-pilot"

# 当前 13B 修订的完整正式批次：
sbatch --partition=MGPU-TC2 --qos=normal 02-pilot/tc2/run_ltx.sh \
  formal "$AI6132_CODE/01-dataset/cases.canonical.csv" "$AI6132_ROOT/outputs/ltx13b-formal-20261115"
```

本次 13B 修订按全组决定，沿用 case_0001 的 129/513 帧技术试跑后直接重跑正式 24 条，未另行完成三个独立共同试跑样本。该例外与试跑结果记录在 [PLAN.md](../../PLAN.md) 和批次报告。

同一命令再次运行时，只继续尚未启动的 case；成功、失败或中断均保留，不自动重抽。13B 使用新输出目录，不续写 2B 目录或混选两模型的视频。失败保留原长度与原因，不改为续写或降低长度。

`selection.json` 冻结协议、阶段、规范清单哈希及全部案例顺序，必须与 `batch.json` 中的完整清单一致。`runs.csv` 从首个worker前覆盖所有案例，未启动为not_started、无实测GPU/耗时，中断尝试为incomplete；不为未启动项创建目录。B必须核对全部行和状态。

代码哈希也在 `batch.json` 中冻结；schema 3 另保存 `source_receipt.json` 及实际源码快照，逐 case 绑定同一凭据。升级 runner 后不能修改旧记录来绕过校验；保留旧输出，使用原代码续跑，或经小组确认后为新代码建立独立批次。读取记录也必须提供匹配的 `selection.json`。

试跑可调整 width/height/frame-rate，正式运行固定 768×432 / 24 fps；生成帧数始终取自每案例 target_frames。`run_ltx.sh` 保留 offload 请求记录，当前内存适配实际不做 CPU offload，记录 requested=true、effective=false。

## B 接收 13B 与历史 2B 批次

13B 正式批次完成后，交付完整目录：`batch.json`、`selection.json`、`pipeline.json`、`runs.csv`、环境与源码凭据，以及 `cases/` 下的 request/result/native。B 在收到的副本上提交 CPU 验收，保留原记录和视频：

```bash
export AI6132_CODE="$PWD"
export AI6132_CPU_ENV=/absolute/personal/environment
mytcinfo
sbatch 02-pilot/tc2/run_cpu.sh import-ltx \
  --cases 01-dataset/cases.canonical.csv \
  --bundle /absolute/received/ltx13b-formal-20261115 \
  --output /absolute/new/ltx13b-canonical-intake
```

该工具逐条核对冻结 prompt、seed、图片哈希、帧数、请求/结果/CSV 和视频哈希，并用 FFmpeg 完整解码。新 13B 批次保留 `case_0001` 至 `case_0024`；历史 2B Shared24 的 01 至 24 编号映射为规范编号，模型 ID 仍为 2B。导出的独立 `runs.csv` 和接收凭据可交给 B 的 media 入口。工具不从原记录的远端绝对路径读取视频，环境须有 Pillow 及 FFmpeg/ffprobe；不完整或含失败的批次会拒绝作为完整成功交付验收，原记录继续保留。

B 用 `ltxv-13b-0.9.8-distilled` 建立新的全片评分副本、匿名映射、VBench 与 Qwen 批次。如果旧 2B 已转码或评分，保留旧文件与结果，不改标签继承分数。人评工具和量表不用修改，按新的匿名 96 条材料重新准备评分包，替换的视频重新独立评分。评分副本继续覆盖完整时长、768×432 / 12 fps；映射和模型评审结果按人评流程封存。

## 输出与计时口径

每个case的native/保留完整原片。完整解码且尺寸、逐条帧数与fps符合请求时记success。画面质量由评审判断，不能用质量低替代技术失败。

`runs.csv` 与 `selection.json` 一起交给 B，含 case/model ID、种子、版本、配置路径及哈希、原生规格、真实状态与失败原因、GPU/驱动、耗时和显存。`batch.json` 固定完整输入清单和图片哈希；`pipeline.json` 是实际配置；`environment.txt` 保存真实安装版本；每个已启动 case 有 `request.json`、`result.json`（中断时可能缺失）和 `inference.log`。

每个样本在独立进程调用官方 `infer`。`elapsed_seconds` 包含本地权重加载、文本编码、两阶段生成和视频写盘，不含下载和后续解码检查。`worker_wall_seconds` 另含导入与解码验证。显存记录为 PyTorch 本进程的 peak allocated / peak reserved（MiB）。

退出码 0 表示选中样本全部成功；1 表示存在技术失败、中断或未完成；2 表示清单、环境或冻结校验失败。中断且没有终态的记录标为 `incomplete`，需检查原日志及进程。

## 测试

```bash
sbatch 02-pilot/tc2/run_cpu.sh test
```

测试使用合成数据，覆盖规范字段、全部24例与逐案例帧数、清单冻结、完整交付记录、续跑，以及人评素材与评分校验；不替代 GPU 共同试跑。验证结果见 [验证记录](../../02-pilot/tc2/VALIDATION.md)。

官方依据：[仓库与安装](https://github.com/Lightricks/LTX-Video/tree/50f0aef1297574a4b29acbdb874e3a363a6d5f2e)、[13B 两阶段配置](https://github.com/Lightricks/LTX-Video/blob/50f0aef1297574a4b29acbdb874e3a363a6d5f2e/configs/ltxv-13b-0.9.8-distilled.yaml)、[2B 历史配置](https://github.com/Lightricks/LTX-Video/blob/50f0aef1297574a4b29acbdb874e3a363a6d5f2e/configs/ltxv-2b-0.9.8-distilled.yaml)、[推理入口](https://github.com/Lightricks/LTX-Video/blob/50f0aef1297574a4b29acbdb874e3a363a6d5f2e/ltx_video/inference.py)、[PyTorch 版本安装](https://pytorch.org/get-started/previous-versions/)。
