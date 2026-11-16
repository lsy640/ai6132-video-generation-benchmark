# Wan2.2-TI2V-5B（成员 A）

本目录是 Wan2.2 的官方源码，固定在提交 `1ea34ff48f87168174e12956e200b1d908b1c5ff`。报告中的 24 条 Wan 2.2 视频由其中的 `generate.py` 生成。

## 实际配置

| 项目 | 取值 |
| --- | --- |
| 权重 | `Wan-AI/Wan2.2-TI2V-5B@921dbaf3f1674a56f47e83fb80a34bac8a8f203e` |
| 任务 | `ti2v-5B`，输入首图和原始英文提示词，种子 42，不启用提示词扩写 |
| 采样 | UniPC，10 步，shift 5.0，CFG 5.0（官方默认为 50 步） |
| 分辨率与帧率 | 请求 `1280*704`，16:9 输入实际输出 1248×704，24 fps |
| 帧数 | 短片 125 帧、长片 509 帧（清单中的 128／512 取不超过它的 4n+1） |
| 内存策略 | `offload_model=True`、`t5_cpu=True`、`convert_model_dtype=True` |
| 硬件 | 单张 NVIDIA A40（46 GB） |
| 耗时 | 短片平均 464 秒，长片平均 1823 秒；未记录显存峰值 |

与其他三个模型相比有两处偏差，报告中已说明：步数为 10 而非默认的 50，每条视频比目标少 4 帧。

## 运行

按日志中记录的参数，单条视频的命令相当于：

```bash
python generate.py --task ti2v-5B --size "1280*704" \
  --ckpt_dir /absolute/path/to/Wan2.2-TI2V-5B \
  --offload_model True --convert_model_dtype --t5_cpu \
  --image "01-dataset/dataset/01_a man is running on a dirt road.jpg" \
  --prompt "a man is running on a dirt road" \
  --base_seed 42 --frame_num 509 --sample_steps 10 \
  --save_file /absolute/path/to/output/01_seed42_f509.mp4
```

在 TC2 上需要把它放进 `sbatch` 脚本，在计算节点执行。成员 A 的批处理脚本没有随交付提供，仓库中没有收录；原片和逐作业日志在 `video-output/wan22_10steps_output/`（日志在其中的 `run_metadata/`）。依赖安装见官方的 [INSTALL.md](INSTALL.md) 和 [README.md](README.md)。

## 源码说明

- 来源和逐文件的 Git blob 校验值记录在 [`UPSTREAM.json`](UPSTREAM.json)。
- 唯一的本地改动是 `wan/__init__.py`：改为按需导入，TI2V 不再提前加载语音和动画管线的依赖。
- 上游的 `examples/`、`assets/`、`tests/`（演示用的音视频、README 配图和多卡测试脚本）与本项目无关，已移除。
