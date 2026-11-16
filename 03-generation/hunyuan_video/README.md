# HunyuanVideo-1.5（成员 B）

生成入口是 [`ai6132_benchmark/hunyuan.py`](../../ai6132_benchmark/hunyuan.py)，调用固定在提交 `60783e704160023913bee78f0b47036d393d4dfa` 的官方源码。环境和作业脚本在 [`02-pilot/tc2/`](../../02-pilot/tc2/README.md)，依赖版本约束见 [`constraints.txt`](constraints.txt)。

## 实际配置

| 项目 | 取值 |
| --- | --- |
| 模型 | `HunyuanVideo-1.5-480P-I2V-step-distill` |
| 采样 | 8 步（步数蒸馏），种子 42，关闭提示词改写和超分 |
| 精度 | Transformer 为 BF16，文本编码器按官方的 FP16 加载 |
| 分辨率与帧率 | 848×480（由官方分桶选择），24 fps |
| 帧数 | 短片 129 帧、长片 513 帧，单次连续生成 |
| 内存策略 | Transformer 分组卸载到 CPU，编码器和 VAE 留在 GPU；VAE 解码按 64 像素分块、重叠 0.25 |
| 硬件 | 单张 NVIDIA L40S，8 CPU，30G 主机内存 |
| 耗时 | 短片平均 173 秒，长片平均 1145 秒 |
| 显存峰值 | 短片 32.7 GiB，长片 33.2 GiB |

24 条全部生成成功。第一次运行时，513 帧的长片在 VAE 解码阶段显存不足；改用上表的分块解码后重新生成了全部 24 条，步数、种子、分辨率和帧数没有改变。本模型没有先做独立的试跑，直接生成了正式批次，批次记录中标明了这一点。

## 运行

所有步骤都通过 `sbatch` 在计算节点执行，从仓库根目录提交。

```bash
export AI6132_CODE="$PWD"
export AI6132_B_ROOT="$HOME/ai6132-b"                 # 环境、权重和输出的个人目录

sbatch 02-pilot/tc2/bootstrap_hunyuan.sh              # 建立环境（PyTorch 2.6、CUDA 12.4）
sbatch 02-pilot/tc2/check_hunyuan_env.sh              # 在 GPU 上检查环境，不加载权重
sbatch 02-pilot/tc2/preflight_hunyuan.sh              # 校验输入清单和权重的下载权限
sbatch 02-pilot/tc2/download_hunyuan.sh               # 按固定 revision 下载权重
sbatch 02-pilot/tc2/finalize_hunyuan_assets.sh        # 核对权重并写入哈希清单
sbatch 02-pilot/tc2/generate_hunyuan_formal.sh        # 生成 24 条视频
sbatch 02-pilot/tc2/verify_hunyuan_formal.sh          # 完整解码并核对全部输出
```

每一步确认上一个作业成功后再提交。各脚本需要的其他环境变量和默认资源写在脚本开头。视觉编码器的权重需要 Hugging Face 访问权限，令牌通过权限为 600 的私有文件提供，不写入命令行或日志。

生成脚本内部调用的是：

```bash
sbatch 02-pilot/tc2/run_b_gpu.sh hunyuan \
  --cases 01-dataset/cases.canonical.csv --phase formal \
  --upstream /absolute/HunyuanVideo-1.5 --commit 60783e704160023913bee78f0b47036d393d4dfa \
  --assets /absolute/hunyuan-assets --out /absolute/outputs/hunyuan-formal \
  --skip-pilot-reason "不做试跑，直接生成正式批次"
```

## 输出

输出目录中，`batch.json` 记录冻结的输入和配置，`runs.csv` 覆盖全部 24 个案例，每个案例保存请求、结果、原片和日志。

- 同一命令重复运行时只处理尚未开始的案例，成功、失败和中断的记录都保留，不自动重新抽样。
- 耗时从加载模型算到视频写盘，不含下载和之后的解码检查。
- 作业时间不够时在两个案例之间停止，下次提交接着运行。
