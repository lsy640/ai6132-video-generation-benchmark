# LTX-Video 13B（成员 C）

`run_cases.py` 逐案例调用官方的推理入口，官方源码固定在提交 `50f0aef1297574a4b29acbdb874e3a363a6d5f2e`。

## 实际配置

| 项目 | 取值 |
| --- | --- |
| 模型 | `ltxv-13b-0.9.8-distilled`，配置见 [`ltxv-13b-0.9.8-no-enhance.json`](ltxv-13b-0.9.8-no-enhance.json) |
| 采样 | 官方两阶段 7+3 步，CFG 1，STG 0，第二阶段 tone map compression ratio 0.6 |
| 条件 | 首图作为第 0 帧，`conditioning_strength=1.0`，`image_cond_noise_scale=0.15`；种子 42，关闭提示词扩写 |
| 精度 | BF16 |
| 分辨率与帧率 | 768×432，24 fps |
| 帧数 | 短片 129 帧、长片 513 帧，单次连续生成 |
| 内存策略 | 先缓存 T5 的提示词编码并释放 T5，再加载 Transformer；两阶段结束后释放 Transformer 再解码（`memory_adapter.py`） |
| 硬件 | 单张 NVIDIA L40S，8 CPU，30G 主机内存 |
| 耗时 | 短片平均 25.0 秒，长片平均 206.2 秒 |
| 显存峰值 | 短片 29.7 GiB，长片 36.9 GiB |

24 条全部生成成功，整批作业用时 30 分 27 秒。逐视频的哈希、耗时和显存见 [`ltx13b-formal-summary.json`](../../02-pilot/tc2/ltx13b-formal-summary.json)。小组最初用 2B 版本生成过一批，之后改为 13B 并重新生成了全部 24 条，报告只使用 13B 的结果。

权重按固定 revision 下载并记录每个文件的 SHA256：

| 内容 | 仓库与 revision |
| --- | --- |
| `ltxv-13b-0.9.8-distilled.safetensors`、`ltxv-spatial-upscaler-0.9.8.safetensors` | `Lightricks/LTX-Video@8984fa25007f376c1a299016d0957a37a2f797bb` |
| `text_encoder/`、`tokenizer/` | `PixArt-alpha/PixArt-XL-2-1024-MS@b89adadeccd9ead2adcb9fa2825d3fabec48d404` |

## 运行

所有步骤都通过 `sbatch` 在计算节点执行，从仓库根目录提交，确认上一个作业成功后再提交下一个。

```bash
export AI6132_CODE="$PWD"
export AI6132_ROOT="$HOME/ai6132-c"                   # 环境、权重和输出的个人目录
export AI6132_LTX_ASSETS="$AI6132_ROOT/ltx_assets_13b"

sbatch 02-pilot/tc2/bootstrap_env.sh                  # 基础环境（PyTorch 2.5.1+cu121）
sbatch 02-pilot/tc2/install_ltx.sh                    # 安装官方代码，下载 13B、upscaler 和 T5
sbatch 02-pilot/tc2/bootstrap_cu126.sh                # 生成用的环境（PyTorch 2.6.0+cu126）

export AI6132_LTX_ENV="$AI6132_ROOT/env-cu126"
export AI6132_CPU_ENV="$AI6132_LTX_ENV"
sbatch 02-pilot/tc2/run_cpu.sh validate-cases \
  --cases 01-dataset/cases.canonical.csv --phase formal        # 检查清单、图片和种子
sbatch 02-pilot/tc2/run_ltx.sh formal \
  "$AI6132_CODE/01-dataset/cases.canonical.csv" "$AI6132_ROOT/outputs/ltx13b-formal"
```

基础的 cu121 环境在 513 帧的 VAE 解码时失败，所以生成使用单独的 cu126 环境。推理只读取本地权重，Hugging Face 设为离线。

## 输出

输出目录中，`batch.json` 记录冻结的输入清单、图片哈希和代码哈希，`pipeline.json` 是实际使用的配置，`runs.csv` 覆盖全部 24 个案例，每个案例保存请求、结果、原片和推理日志。

- 同一命令重复运行时只处理尚未开始的案例，成功、失败和中断的记录都保留，不自动重新抽样。
- 耗时包含加载权重、文本编码、两阶段生成和视频写盘，不含下载和之后的解码检查。显存为 PyTorch 进程的峰值分配量。
- 退出码 0 表示全部成功，1 表示有失败或未完成，2 表示清单、环境或冻结校验没有通过。

## 交接

`import_completed.py` 用于接收成员 C 交付的批次：逐条核对提示词、种子、图片哈希、帧数和视频哈希，用 FFmpeg 完整解码，再导出下游使用的 `runs.csv`。

```bash
sbatch 02-pilot/tc2/run_cpu.sh import-ltx \
  --cases 01-dataset/cases.canonical.csv \
  --bundle /absolute/received/raw/formal-13b-20261115 \
  --output /absolute/new/ltx13b-intake
```

制作评分副本时，把 `ltxv-13b-0.9.8-distilled=/absolute/received/delivery/runs.csv` 作为 `media` 的一项输入，见 [04-vbench-evaluation](../../04-vbench-evaluation/README.md)。
