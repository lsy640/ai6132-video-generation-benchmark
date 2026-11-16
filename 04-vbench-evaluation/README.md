# 评分副本与 VBench-I2V 评估

本阶段把四个模型的原片转换成统一的评分副本，并运行 VBench-I2V 的四个指标。

## 结果

96 条视频的 224 个适用指标项全部成功（作业 54818，独立复核 54853）。

| 模型 | I2V 主体 (n=16) | I2V 背景 (n=8) | 动作平滑 (n=16) | 显著运动 (n=16) |
| --- | ---: | ---: | ---: | ---: |
| Wan 2.2 | 0.9787 | 0.9760 | 0.9857 | 43.75% |
| HunyuanVideo 1.5 | 0.9730 | 0.9747 | 0.9904 | 68.75% |
| LTX-Video | 0.9612 | 0.9811 | 0.9917 | 43.75% |
| LongCat-Video | 0.9654 | 0.9637 | 0.9839 | 75.00% |

逐模型、按长短分组的统计见 [`vbench_summary.csv`](../08-report/data/vbench_summary.csv)，与人工评分的对照见[报告](../08-report/Review_of_Image_to_Video_Generation/group_11_report.pdf)第 4.4 节。

## 指标

| 指标 | 适用案例 | 含义 |
| --- | --- | --- |
| `i2v_subject` | 16 个主体案例 | 首图与各帧、相邻帧之间的 DINO 特征相似度 |
| `i2v_background` | 8 个背景案例 | 同上，使用 DreamSim 特征 |
| `motion_smoothness` | 16 个主体案例 | AMT 插帧模型重建被抽掉帧的误差 |
| `dynamic_degree` | 16 个主体案例 | RAFT 光流超过阈值的视频比例，表示运动量 |

主体和背景分数的计算方式是 0.4 × 首图相似度最大值 + 0.3 × 邻帧相似度均值 + 0.3 × 邻帧相似度最小值。它们衡量全局特征是否一致，不衡量人物身份或背景清晰度；`dynamic_degree` 衡量运动多少，不衡量运动是否自然。本项目使用固定的 24 例子集，结果不等同于官方排行榜分数。

## 流程

所有命令从仓库根目录经 `sbatch` 提交。CPU 环境需要 Python 3.10+、Pillow 和带 libx264 的 FFmpeg/ffprobe。

**1. 制作评分副本。** 把每条原片转成完整时长的 768×432、12 fps、H.264/yuv420p 副本（CRF 18，保持比例并补边，移除音轨和来源元数据），不截取片段。输入是四个模型各自的 `runs.csv`。

```bash
export AI6132_CPU_ENV=/absolute/path/to/cpu/env
sbatch 02-pilot/tc2/run_cpu.sh media --cases 01-dataset/cases.canonical.csv \
  --runs MODEL_A=/absolute/A/runs.csv MODEL_B=/absolute/B/runs.csv \
         MODEL_C=/absolute/C/runs.csv MODEL_D=/absolute/D/runs.csv \
  --outputs-frozen --out /absolute/private/handoff
```

输出目录中，`handoff/` 是匿名的视频、首图和六列清单，`private/` 是随机映射、原始运行记录和转换审计。`COMPLETE.json` 存在才表示整批准备成功。`private/` 不随评分包分发。

**2. 冻结缓存并评估。** 上游固定为提交 `fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`。DINO、DreamSim、AMT、RAFT 权重预先下载到缓存目录并写入哈希凭据，推理阶段禁止联网，缺少缓存时直接失败。

```bash
sbatch 02-pilot/tc2/run_cpu.sh vbench-cache --cache /absolute/vbench-cache \
  --receipt-cache /absolute/private/vbench-cache.json

export AI6132_GPU_ENV=/absolute/path/to/vbench/env
sbatch 02-pilot/tc2/run_b_gpu.sh vbench \
  --manifest /absolute/private/handoff/handoff/anonymous_clips.csv \
  --clip-map /absolute/private/handoff/private/clip_map.csv \
  --upstream /absolute/VBench --commit fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490 \
  --cache /absolute/vbench-cache --cache-receipt /absolute/private/vbench-cache.json \
  --out /absolute/private/vbench
```

评估使用官方的 `custom_input` 接口，不修改指标公式。每个指标保留官方原始 JSON 和逐视频结果；失败或缺失的项留空并记录原因，已完成的请求不重跑。

只有部分模型交付时，可给 `media` 和 `vbench` 加 `--phase diagnostic`，接受 1 至 4 个模型，每个模型仍需覆盖全部 24 例。

## 说明

- VBench 的输入是 12 fps 副本，人工评分使用原生帧率副本，Qwen 从原片抽帧，三者不是同一份像素。
- Wan 2.2 的 10 步、125/509 帧和 LongCat-Video 的 15 fps 按实际交付评估，没有补帧或重采样到相同时长。
- 最终结果由两批合并而成：A、B、C 的 168 项（54672）和 D 的 56 项（54818），两批使用相同的上游提交、缓存、环境和指标定义。
