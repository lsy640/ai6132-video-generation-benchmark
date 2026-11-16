# 人工评分

四名成员各自对四个模型的 96 条视频独立评分。本目录包含评分网页、Excel 工具、收表校验和四份提交的评分表。

## 结果

四份表各 96 条全部有效，共 384 条记录、1536 项分数，无缺失、无 NA（收表作业 54957）。提交的原始表在 [`human-evaluation-result/`](human-evaluation-result/)。

| 模型 | 首图／主体外观保持 | 动作自然度 | 提示词遵循 | 背景细节保持 |
| --- | ---: | ---: | ---: | ---: |
| Wan 2.2 | 3.79 | 2.96 | 4.08 | 3.42 |
| HunyuanVideo 1.5 | 4.25 | 3.71 | 4.48 | 4.02 |
| LTX-Video | 4.67 | 3.92 | 4.71 | 4.75 |
| LongCat-Video | 4.00 | 3.69 | 4.15 | 3.88 |
| 序数 Krippendorff α | 0.656 | 0.573 | 0.582 | 0.538 |

每格是 24 个"视频四人中位数"的均值。按长短分组的统计、评审一致性和剔除模型负责人后的对照见 [`data/`](../08-report/data/) 中的 `human_summary.csv`、`human_agreement.csv`、`human_excluding_owner.csv`。

## 评分量表

四个维度分别打 1–5 的整数分，互不影响，不规定分数比例。完整的五级锚点定义在 `human_workbooks.py` 的 `RUBRICS` 中，并写入每份 Excel 的说明页。

| 维度 | 1 分 | 5 分 |
| --- | --- | --- |
| 首图／主体外观保持 | 主体或场景严重改变，人物被替换或难以辨认 | 全程主要外观与场景特征稳定，变化与动作或视角相容 |
| 动作自然度 | 严重卡顿、跳变或结构变形，或提示词要求的动作基本缺失 | 动作持续连贯，局部细节可辨，速度、轨迹和接触符合情境 |
| 提示词遵循 | 主要要求缺失或与视频明显矛盾 | 主体、动作、场景的主要要求都能在视频中确认 |
| 背景细节保持 | 可辨认的背景结构或纹理几乎全部丢失，或持续严重模糊 | 主要背景结构、边缘和纹理在可比较处基本保持 |

评分要点：

- 先以 1 倍速看完整片，再按需暂停或慢放。
- 动作检查手脚结构、接触、滑步、持续模糊、重复和停滞；合理的瞬时运动模糊不扣分。
- 背景要拿中后段和首图比较，主体清晰不能抵消背景退化。
- 背景本来就无法比较时才填 `NA`；原先清晰、后来持续虚化应按退化程度打分。

## 评分包

每名成员收到一个只含本人材料的 ZIP，解压后双击 `viewer.html` 即可离线使用，不需要 Python、服务器或网络。

```text
评分包_A_HTML.zip
  viewer.html            评分网页：首图、视频和提示词同页展示
  rating_app.js  workbook_io.js  rating_config.js
  vendor/                本地 fflate 0.8.2（读写 xlsx 用）
  ratings_A.xlsx         空白表单，也可以直接在 Excel 中填写
  order.js               本人的随机观看顺序
  anonymous_manifest.csv 匿名编号、首图、提示词
  images/  videos/       24 张首图和 96 条视频
```

- 视频使用匿名编号，每人的顺序各自随机。模型与编号的映射只由组织者保存。
- 素材由原片转成 768×432、H.264 副本，保留原生帧率、全部帧和完整时长（A、B、C 为 24 fps，D 为 15 fps）。
- 网页把草稿保存在浏览器本地。每次结束时下载 JSON 备份；换浏览器或清缓存后可用"导入进度/Excel"恢复。
- 96 条全部评完后"导出最终 Excel"，提交这份 xlsx。JSON 只用于恢复进度。

## 文件

| 文件 | 作用 |
| --- | --- |
| `human_workbooks.py` | 四维评分包的制作、核验、升级和收表校验（`human-xlsx` 子命令） |
| `rating_viewer.html`、`rating_app.js`、`workbook_io.js` | 评分网页及 Excel 读写 |
| `vendor/` | fflate 0.8.2、MIT 许可和来源校验 |
| `complete_d_tc2.sh` | 把后交付的模型补入已分发的评分包 |
| `human_eval.py` | 早期的三维 CSV 评分工具；VBench 入口复用其中读取匿名清单的函数 |

## 命令

所有命令经 `sbatch` 提交。环境需要 Python 3.10+、openpyxl 3.1.5、Pillow 和 FFmpeg/ffprobe。

```bash
export AI6132_CODE="$PWD"
export AI6132_CPU_ENV=/absolute/path/to/human-tools-env

# 校验一份提交的评分表并导出 CSV；最终交卷加 --final
sbatch 02-pilot/tc2/run_cpu.sh human-xlsx validate-workbook \
  --packet-root /path/to/packet-root --rater A \
  --workbook /path/to/ratings_A.xlsx \
  --report /path/to/checks/A.json --export-csv /path/to/exports/A.csv --final
```

`human-xlsx` 的其他子命令：`prepare`（从原片制作四份评分包）、`verify`（核验评分包）、`upgrade-web`（更新已有评分包的网页文件）、`complete-d`（补入后交付的模型）。`prepare` 和 `complete-d` 要处理整批视频，需要把默认的 10 分钟时限调大。

## 局限

- 四名评审各自生成了其中一个模型，可能认出自己的视频；LongCat-Video 的视频晚于其他三个模型补入。评分不是完全双盲。剔除模型负责人后，各项均值的变化不超过 0.25 分。
- 计划中的 3 条共同校准样本没有完成。
- Qwen 和 VBench 评估在人工评分提交之前运行，顺序与原计划相反；自动评分没有分发给评审。
- 评分素材统一为 768×432，不能据此判断各模型原生分辨率下的绝对清晰度。
