# 人工盲评

使用 Python 3.10+ 标准库。`human_eval.py` 用于生成四人的评分材料和检查收回的评分表。

TC2 的 Head Node 不运行 Python；准备、核验、收表和测试均使用下列 CPU 批处理入口。提交前运行 `mytcinfo` 核对当前 QoS；资源及环境设置见 [CPU 作业说明](../02-pilot/tc2/README.md#cpu-工具与测试)。从仓库根目录提交，先将 `AI6132_CPU_ENV` 设为个人 Python 3.10+ Conda 环境的路径（默认 `$AI6132_ROOT/env`）。

## B 的交接材料

B 提供已匿名的 `anonymous_clips.csv`、视频和首图。CSV 字段及顺序须与 [`anonymous_clips.template.csv`](anonymous_clips.template.csv) 一致：

| 字段 | 约定 |
| --- | --- |
| `clip_id` | 整批唯一的匿名编号，如 `clip_0001`；`clip_` 后至少四位数字 |
| `video_path` | 相对于 CSV 的路径：`videos/<clip_id>.mp4`；技术失败留空 |
| `image_path` | 相对于 CSV 的路径：`images/<clip_id>.png`，也支持小写 jpg/jpeg/webp 扩展名 |
| `prompt` | 原始英文提示词；含逗号或换行时按 CSV 规范加引号 |
| `status` | 仅 `success`、`generation_failed`、`decode_failed` |
| `failure_reason` | 成功留空，技术失败填写下述原因代码 |

失败原因代码：

- `generation_failed`：`out_of_memory`、`timeout`、`generation_error`、`missing_output`、`other`。
- `decode_failed`：`decode_error`、`empty_video`、`other`。

B 混排模型来源后分配 `clip_id`，首图按每条 clip 复制命名。C 也参与评分，因此交接清单中只保留上述六列；模型映射 `clip_map.csv` 和详细错误日志单独保存，收齐四份评分后交给 D。

B 负责确认评分副本完整解码、保留完整时长、768×432、12 fps、H.264，并清除文件名和 metadata 中的模型标识。若画面本身含明确模型标识，应暂停分发并核查，不能声称转码自动消除了画面标识。正式批次固定为11类混合长短基准的96条，四人均观看全片；可播放但质量差的视频仍标为success，技术失败保留行。准备与核验评分包均执行该数量约束。

## 准备评分材料

从仓库根目录提交，替换为实际交接路径。素材和评分存放在仓库外，`--out` 使用新目录：

```bash
sbatch 02-pilot/tc2/run_cpu.sh human-eval prepare \
  --manifest /path/to/private_handoff/anonymous_clips.csv \
  --out /path/to/private_review/formal_mixed_96 \
  --raters A B C D \
  --seed 6132 \
  --expected-clips 96
```

当前正式批次固定 `--expected-clips 96`，B先提交自己的完整评分，再运行Qwen；其他人交卷前封存映射和Qwen结果。校准使用独立的三条试跑清单及输出目录，参数为 `--phase calibration --expected-clips 3`；校准评分与正式评分分开保存。

输出结构：

```text
formal_mixed_96/
  assets/                         # 视频和首图
  preparation.json               # 批次、种子和文件校验信息
  raters/
    A/order.csv                   # 观看顺序、素材路径和 prompt
    A/ratings.csv                 # 待填写的评分表
    A/评分说明.txt                 # 量表和评分说明
    B/…
    C/…
    D/…
```

每位评审有独立、可复现的随机顺序。v2 `preparation.json` 冻结匿名清单以及复制后的视频、首图、四份 `order.csv` 和评分说明的 SHA256；可编辑的 `ratings.csv` 不纳入冻结。收表校验会核对整个保留包。

旧 v1 记录缺少素材哈希，不能证明评分包完整性。保留旧包和原始评分，在新目录重新准备并确认分发材料；不能通过手改版本号升级旧记录。

给每位评审分发完整 `assets/` 和其本人的 `raters/<rater_id>/`，保留上述目录结构。`order.csv` 中素材路径相对于该文件所在目录。

交接时也应核验实际收到的副本。使用准备方独立保留的可信 `preparation.json`，而非接受接收方重写的记录；`--rater` 只检查该评审的目录和共享素材，无需分发其他人的评分表：

```bash
sbatch 02-pilot/tc2/run_cpu.sh human-eval verify-packet \
  --preparation /path/to/private_review/formal_mixed_96/preparation.json \
  --packet-root /path/to/received/A_packet --rater A
```

`A_packet` 仍包含 `assets/` 和 `raters/A/` 两层路径。核验失败时暂停评分，重新交接正确材料，不覆盖已收到的原件。此检查应在开始评分前完成。

四人按各自顺序独立评分，全部交卷前不查看他人分数或模型映射。C 先提交自己的评分，再检查其他人的提交；收回的表单单独保存。

成功视频的三个维度填写整数 1–5，备注可选。技术失败的三个分数留空，保留自动填写的失败备注。表格软件导入时将 ID、路径、prompt 和 comment 列设为文本。

## 收表检查

四位评审各提交一份原始 CSV：

```bash
sbatch 02-pilot/tc2/run_cpu.sh human-eval validate \
  --manifest /path/to/private_handoff/anonymous_clips.csv \
  --preparation /path/to/private_review/formal_mixed_96/preparation.json \
  --packet-root /path/to/private_review/formal_mixed_96 \
  --raters A B C D \
  --ratings A=/path/to/submitted/A.csv B=/path/to/submitted/B.csv \
            C=/path/to/submitted/C.csv D=/path/to/submitted/D.csv \
  --report /path/to/private_review/check_01.json
```

校验覆盖冻结清单、实际评分素材、观看顺序及说明、表头、评审 ID、缺行、重复或未知 clip_id、整数 1–5 分，以及技术失败的空分和备注。`--packet-root` 默认是 `preparation.json` 所在目录；移动整个评分包后可显式指定新位置。

退出码：`0` 通过，`1` 评分有缺漏或格式问题，`2` 输入或文件错误。报告列出评审、clip_id、字段和问题类型；每次检查使用新的报告文件名。

漏填和错误由原评审修正，首次提交与修正版分别保留。四份表通过后，将原始评分、修正记录、匿名清单及 `preparation.json` 交给 D，结合 `clip_map.csv` 解盲并统计。

## 测试

```bash
sbatch 02-pilot/tc2/run_cpu.sh test
```
