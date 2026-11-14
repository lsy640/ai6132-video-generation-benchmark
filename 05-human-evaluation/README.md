# C：准备人工盲评与检查收表

本目录实现 `PLAN.md` 中 C 的评分组织工作。只依赖 Python 3.10+ 标准库，可在 Mac 上运行，无需 GPU。它生成**空评分表**，不生成、补写或调整任何人工分数。

## 与 B 的输入约定

B 完成视频解码校验和统一评分副本后，提供 `anonymous_clips.csv`。表头必须与 [`anonymous_clips.template.csv`](anonymous_clips.template.csv) 一致且按以下顺序；模板仅有表头，没有实验数据。

| 字段 | 约定 |
| --- | --- |
| `clip_id` | 匿名中性编号，例如 `clip_0001`；`clip_` 后至少四位数字，整批唯一，不含模型名 |
| `video_path` | 相对于该 CSV 的 MP4 评分副本路径；成功条目必须是 `videos/<clip_id>.mp4`，技术失败留空 |
| `image_path` | 相对于该 CSV 的首图路径；必须是 `images/<clip_id>.png`（也支持小写 jpg/jpeg/webp 扩展名） |
| `prompt` | 冻结的原始英文提示词，原样保存；有逗号或换行时按 CSV 规范加引号 |
| `status` | 仅 `success`、`generation_failed`、`decode_failed` |
| `failure_reason` | 成功留空；技术失败只填下述中性原因代码，不传入 traceback、原始目录或模型名 |

不要给此清单加 `case_id`、`model_id`、模型评分或人工评分列；工具拒绝额外列。`clip_map.csv` 由负责匿名映射的成员另行保管，待全部交卷后交给 D。输入文件、映射和原始技术日志不要放进公开评分包或提交到公开仓库。

C 本人也参与盲评，因此 **B 交给 C 的输入就必须已经匿名**，不能等 C 制作分发包时才隐藏模型名。B 应混排来源后分配匿名编号，避免按模型分段编号；首图也按每条 clip 复制命名。生成失败原因代码：`out_of_memory`、`timeout`、`generation_error`、`missing_output`、`other`；解码失败：`decode_error`、`empty_video`、`other`。原始详细原因由 B 另行记录，解盲后交给 D。工具拒绝其他失败原因字符串和非约定素材路径。

`success` 表示可正常评审，**不代表画面质量好**。可播放但质量差的视频必须正常打分。只有生成或解码失败才能标成技术失败，并保留对应的匿名条目，使四人仍覆盖相同的 48 或 96 个计划输出。

准备工具检查文件存在、扩展名和清单结构，但不会重新验证视频时长、分辨率、编码、是否真的可解码。B 应先完成：前 5 秒、768×432、12 fps、H.264，以及去除画面、界面、容器 metadata 中的模型标识。工具只重新命名复制素材；无法自动识别画面水印或原始 prompt 中意外出现的模型名称。

## 生成四人评分材料

从仓库根目录执行，下面的输入路径请替换成 B 实际交接位置。`--out` 须使用仓库外的全新私有目录，不会覆盖已有材料：

```bash
python3 05-human-evaluation/human_eval.py prepare \
  --manifest /path/to/private_handoff/anonymous_clips.csv \
  --out /path/to/private_review/formal_48 \
  --raters A B C D \
  --seed 6132 \
  --expected-clips 48
```

扩展批次改成 `--expected-clips 96`，且四人必须共同使用同一批次；不能各评各的子集。校准使用独立的三条试跑清单、独立输出目录和 `--phase calibration --expected-clips 3`。校准评分不并入正式评分。

输出结构：

```text
formal_48/
  assets/                         # 匿名重命名的视频和首图
  preparation.json               # 批次、种子、清单与观看顺序的 SHA-256
  raters/
    A/order.csv                   # 单独随机观看顺序、首图和原始 prompt
    A/ratings.csv                 # 六列表头，成功视频全部空分
    A/评分说明.txt                 # 完整量表及独立评分要求
    B/…
    C/…
    D/…
```

每位评审的顺序由总种子和 `rater_id` 通过 SHA-256 派生随机种子，并在规范化 `clip_id` 列表上洗牌，可复现且互相独立。保留 Python 版本和输出 `preparation.json`；不要在正式评审中重新选择种子。`order.csv` 中素材路径相对于该文件所在目录。

分别给每位评审完整 `assets/` 与**仅属于其本人的** `raters/<rater_id>/`，保留上述相对目录结构。不要把收集中的真实评分放在四人共享的目录或公开 GitHub 仓库里。各人交卷前不看他人评分；C 本人也应先交自己的独立原始评分，再检查他人的提交。

失败条目的三个分数字段留空；`comment` 自动填入“未能生成视频”或“视频无法解码”。具体 traceback、模型名称、原始素材目录均不进入分发清单。其余条目由评审本人填写整数 1–5，备注可选。表格软件导入时把 ID、路径、prompt、comment 列设为文本，避免修改文字内容。

## 检查四份原始评分

将四份提交保存为不同的原始文件；校验工具只读这些文件，不合并分数、不修改评分：

```bash
python3 05-human-evaluation/human_eval.py validate \
  --manifest /path/to/private_handoff/anonymous_clips.csv \
  --preparation /path/to/private_review/formal_48/preparation.json \
  --raters A B C D \
  --ratings A=/path/to/submitted/A.csv B=/path/to/submitted/B.csv \
            C=/path/to/submitted/C.csv D=/path/to/submitted/D.csv \
  --report /path/to/private_review/check_01.json
```

检查内容：清单是否仍为准备阶段冻结的同一份文件；四人是否各交独立文件；表头与 rater_id 是否一致；是否缺行、重复或出现未知 clip_id；成功视频三维度是否全部是整数 1–5；技术失败是否三分全空且备注非空。

退出码：`0` 为通过，`1` 为评分内容有问题，`2` 为输入或文件错误。JSON 报告列出需修正的评审、clip_id、字段和问题类型，不列出具体分数。报告文件也不覆盖已有文件。将对应问题私下交回原评审，由其补齐；保留首次提交和补齐版本，不替评审填分。

四份评分全部通过后，将独立原始评分、必要的补齐记录、冻结匿名清单及 `preparation.json` 交给 D。D 再结合独立保管的 `clip_map.csv` 做解盲与统计。工具仅检查格式、覆盖和失败规则，无法证明评审独立性或评分内容是否合理。

## 本地验证

```bash
python3 -m unittest discover -s 05-human-evaluation -p 'test_*.py' -v
```

测试使用临时合成文件和明确的测试分数，覆盖可复现随机顺序、匿名路径、空模板、批次隔离、重复/未知/缺失 ID、错误评审、非整数与越界分数、技术失败规则和不改动原始提交。这些测试数据不属于实验结果。
