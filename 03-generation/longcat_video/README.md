# LongCat-Video（成员 D）

## 实际配置

| 项目 | 取值 |
| --- | --- |
| 模型 | `meituan-longcat/LongCat-Video`，加 CFG-step 蒸馏 LoRA |
| 采样 | 16 步，CFG 1，种子 42，不做提示词扩写、续写、拼接或插帧 |
| 精度 | BF16（权重按官方推理精度逐分片转换，不是额外的低位量化） |
| 分辨率与帧率 | 832×480，15 fps |
| 帧数 | 短片 129 帧（8.6 秒）、长片 513 帧（34.2 秒），单次连续生成 |
| 内存策略 | 短片全部留在 GPU；长片把 48 个 Transformer 块中的 40 个卸载到 CPU，LoRA 留在 GPU |
| 硬件 | 单张 NVIDIA L40S，4 CPU，28G 主机内存 |
| 耗时 | 短片平均 447 秒，长片平均 3747 秒 |
| 显存峰值 | 短片 34.7 GiB，长片 37.3 GiB |

24 条全部生成成功。帧率为 15 fps，相同帧数下的物理时长比其他三个模型长，报告中已说明。逐案例的配置、日志和计时在 [`02-pilot/tc2/longcat24-records/`](../../02-pilot/tc2/longcat24-records/README.md)。

| 内容 | 固定版本 |
| --- | --- |
| 模型权重 | `meituan-longcat/LongCat-Video@03b55529b1d1d4045f5fbe14d65c8c6e8116b278` |
| 官方源码 | `meituan-longcat/LongCat-Video@58eb336edc5c8c63389838727e40b8c18434e54e` |
| PyTorch | 2.6.0+cu124，torchvision 0.21.0+cu124 |
| FlashAttention | 2.7.4.post1 |
| 其他 | diffusers 0.35.1、transformers 4.41.0，完整版本见 `frozen/environment-freeze.txt` |

## 文件

| 路径 | 用途 |
| --- | --- |
| `frozen/` | 生成这 24 条视频时实际使用的代码和配置：引擎 `engine.py`、调度 `worker.py`、作业脚本 `wave.slurm`、批次配置 `batch.json`，以及权重和环境清单。内容与生成记录中的哈希绑定，不要修改 |
| `prepare_runtime.slurm` | 在新路径下安装运行环境 |
| `run_case.py`、`run_case.slurm` | 用 `frozen/engine.py` 生成单个案例 |
| `rebuild_batch.py`、`worker.py`、`submit_batch.sh`、`tools_cpu.slurm` | 在新路径下按同样的配置重新生成全部 24 条 |
| `import_completed.py`、`intake_cpu.sh` | 接收并校验成员 D 交付的批次，导出下游使用的 `runs.csv` |

## 运行

所有步骤都通过 `sbatch` 在计算节点执行，从仓库根目录提交。

```bash
export AI6132_CODE="$PWD"
export LONGCAT_ROOT="$HOME/ai6132-longcat"
export LONGCAT_BOOTSTRAP_PYTHON=/absolute/path/to/python3.11
export LONGCAT_UV=/absolute/path/to/uv
sbatch 03-generation/longcat_video/prepare_runtime.slurm          # 安装环境并准备权重

sbatch 03-generation/longcat_video/run_case.slurm \
  --cases "$AI6132_CODE/01-dataset/cases.canonical.csv" \
  --case-id case_0001 --out "$LONGCAT_ROOT/replay/case_0001"      # 生成单个案例
```

帧数、提示词和种子取自规范清单，只允许写入新的输出目录。

6 条长片放不进一个 6 小时的作业，所以整批分两段运行：先单独生成 `case_0001`（长片），再用一段跑 18 条短片，另一段跑其余 5 条长片。

```bash
export AI6132_CPU_PYTHON=/absolute/path/to/cpu-env/bin/python     # Python 3.11，需要 Pillow
sbatch 03-generation/longcat_video/tools_cpu.slurm rebuild-batch \
  --cases "$AI6132_CODE/01-dataset/cases.canonical.csv" \
  --runtime "$LONGCAT_ROOT" \
  --reuse-case1 "$LONGCAT_ROOT/replay/case_0001" \
  --out "$LONGCAT_ROOT/batches/shared24"

bash 03-generation/longcat_video/submit_batch.sh "$LONGCAT_ROOT/batches/shared24"   # 提交两段作业
```

已经成功的案例不会重跑。遇到显存不足时，调度程序允许用同一种子、开启 CPU 卸载重试一次，每次尝试都保留记录；本次交付的 24 条都是一次完成的。

## 交接

`import_completed.py` 核对交付批次的源码、权重清单、图片、逐条请求与结果、16 步的执行记录和视频字节，再用 FFmpeg 完整解码。交付包里的 `runs.csv` 必须与 `02-pilot/tc2/longcat24-records/verification.json` 中记录的摘要逐字节一致，所以这个记录目录需要完整保留。

```bash
sbatch 03-generation/longcat_video/intake_cpu.sh export \
  --raw /received/longcat16_full24_20261115 \
  --cases "$AI6132_CODE/01-dataset/cases.canonical.csv" \
  --out /received/longcat-delivery
```

制作评分副本时，把 `longcat-video-480p-16step-distill=/received/longcat-delivery/runs.csv` 作为 `media` 的一项输入，见 [04-vbench-evaluation](../../04-vbench-evaluation/README.md)。PyAV 解码模式另需安装 `intake-requirements.txt`。
