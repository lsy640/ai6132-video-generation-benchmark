# 分析与图表

`make_report_figures.py` 读取 [`08-report/data/`](../08-report/data/) 中的统计表和四个模型的原片，生成 LaTeX 报告使用的全部图片，输出到 `08-report/Review_of_Image_to_Video_Generation/figures/`。

```bash
sbatch 07-analysis/make_report_figures.sh    # 在仓库根目录提交
```

| 输出 | 内容 | 数据来源 |
| --- | --- | --- |
| `dataset_overview.jpg` | 24 张参考图及类别、长短分组 | `01-dataset/` |
| `human_scores.pdf` | 四维人评，按短片／长片分组 | `human_summary.csv` |
| `qwen_vs_human.pdf` | Qwen3-VL 与人评中位数逐视频对照 | `qwen_human_pairs.csv`、`qwen_human_comparison.csv` |
| `qualitative_*.jpg` | 10 个案例的四模型抽帧对比：全部 6 个长片（01、04、07、09、19、23）和 4 个短片（05、08、12、22） | 各模型原片 |

脚本需要 matplotlib 和 Pillow（默认使用集群的 Anaconda），抽帧使用 FFmpeg。Hunyuan 原片目录、Python 和 FFmpeg 路径可用 `AI6132_HUNYUAN_DIR`、`AI6132_PLOT_PYTHON`、`AI6132_FFMPEG` 覆盖。

人评每视频取四人评分的中位数。不同指标的适用案例数和含义不同，不合成总分。
