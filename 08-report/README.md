# 报告

| 路径 | 内容 |
| --- | --- |
| [`Review_of_Image_to_Video_Generation/`](Review_of_Image_to_Video_Generation/) | 课程最终报告：`main.tex`、`references.bib`、`figures/` 和编译好的 `group_11_report.pdf` |
| [`data/`](data/README.md) | 报告中表格和图表所用的统计表 |
| `group_11_contribution.tex`、`group_11_contribution.pdf` | 成员贡献说明，按 [`PLAN.md`](../PLAN.md) 的分工逐人记录负责的模型、完成的评审和承担的汇总工作 |
| `build_report.sh` | 用 Tectonic 编译报告和贡献说明的 SLURM 脚本 |

## 编译

```bash
sbatch 08-report/build_report.sh      # 在仓库根目录提交，输出 group_11_report.pdf 和 group_11_contribution.pdf
```

脚本默认使用 `~/tools/tectonic/tectonic`，可用 `AI6132_TECTONIC` 指定其他路径；首次运行会把所需的 TeX 宏包下载到 `~/.cache/Tectonic`。也可以把 `Review_of_Image_to_Video_Generation/` 整个上传到 Overleaf 编译，或在本地运行 `pdflatex main && bibtex main && pdflatex main && pdflatex main`。

提交前在 `main.tex` 开头填写组号、四名成员的姓名和学号（`\groupid`、`\memberA`–`\memberD`、`\matricA`–`\matricD`），再重新编译。

`figures/` 中的图片由 [`07-analysis/make_report_figures.py`](../07-analysis/README.md) 生成。Qwen 的动作分为诊断性结果，报告中不用于模型排序。
