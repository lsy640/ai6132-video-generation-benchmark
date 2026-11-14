# C：LTX-Video 2B v0.9.8 Distilled

使用官方两阶段 I2V，首图条件为第 0 帧，关闭提示词扩写。

2026-11-14 已在 TC2 L40S 上完成一条合成环境测试：`768×432 / 121 帧 / 24 fps`，H.264 原片完整解码通过，时长约 5.04 秒。该记录仅为历史实测；当前129/513帧规范仍须用3个独立共同样本验证。详见 [验证记录](../../02-pilot/tc2/VALIDATION.md)。

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

由 A 冻结原始 `cases.csv` 和横版首图，生成器只读取七列规范清单 `cases.canonical.csv`：

```csv
case_id,category,image_path,prompt,seed,source_frame_num,target_frames
case_0001,single-human,dataset/01.jpg,"Original English prompt from the shared manifest",42,512,513
```

`case_id` 格式为 `case_` 加四位数字且整批唯一，`category` 保持 A 原始标签，seed为0至2³²−1整数，prompt保留原文。正式清单固定为11类24案例、18条129帧和6条513帧，逐条校验source_frame_num/target_frames的128→129、512→513映射。详见[清单适配](../../01-dataset/README.md)。

首图路径字段统一为 `image_path`。相对路径以 CSV 所在目录为基准，也可使用 `--image-root` 指定。首图由 A 统一裁为 16:9，与输出比例一致；官方推理会补边，再裁回请求尺寸。

```bash
# 检查清单、种子、图片路径和样本顺序。
export AI6132_CPU_ENV="$AI6132_ROOT/env"
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

# 全组冻结正式数据和试跑配置后：
sbatch --partition=MGPU-TC2 --qos=normal 02-pilot/tc2/run_ltx.sh \
  formal "$AI6132_CODE/01-dataset/cases.canonical.csv" "$AI6132_ROOT/outputs/ltx-formal-mixed"
```

同一命令再次运行时，只继续尚未启动的case；成功、失败或中断均保留，不自动重抽。新混合协议使用独立目录，不修改旧批次冻结配置。513帧尚需共同GPU试跑验收，合法帧数不代表已验证资源可行；不得因失败静默改成续写或降低长度。

`selection.json` 冻结协议、阶段、规范清单哈希及全部案例顺序，必须与 `batch.json` 中的完整清单一致。`runs.csv` 从首个worker前覆盖所有案例，未启动为not_started、无实测GPU/耗时，中断尝试为incomplete；不为未启动项创建目录。B必须核对全部行和状态。

代码哈希也在 `batch.json` 中冻结。升级 runner 后不能修改旧记录来绕过校验；保留旧输出，使用原代码续跑，或经小组确认后为新代码建立独立批次。读取记录也必须提供匹配的 `selection.json`。

试跑可调整width/height/frame-rate，正式运行时冻结；生成帧数始终取自每案例target_frames。官方只在GPU总显存小于30GiB时启用CPU offload；48GB L40S记录requested=true、effective=false，长片可行性需实测。

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

官方依据：[仓库与安装](https://github.com/Lightricks/LTX-Video/tree/50f0aef1297574a4b29acbdb874e3a363a6d5f2e)、[2B 两阶段配置](https://github.com/Lightricks/LTX-Video/blob/50f0aef1297574a4b29acbdb874e3a363a6d5f2e/configs/ltxv-2b-0.9.8-distilled.yaml)、[推理入口](https://github.com/Lightricks/LTX-Video/blob/50f0aef1297574a4b29acbdb874e3a363a6d5f2e/ltx_video/inference.py)、[PyTorch 2.5.1 安装](https://pytorch.org/get-started/previous-versions/#v251)。
