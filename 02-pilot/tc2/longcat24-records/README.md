# D：LongCat 24组生成记录

本目录只提供生成记录：24/24技术成功，18条129帧短片、6条513帧长片。原片为832×480 / 15fps，时长分别8.6秒和34.2秒。D已在本地用PyAV12完整解码5400帧，并核对原片SHA256；B的独立接收验收尚待进行，未开展正式质量评分。

| 文件/目录 | 内容 |
| --- | --- |
| runs.csv | 对齐共享media入口的24行交接表，含路径、模型、种子、帧数、fps、耗时、显存及哈希 |
| seed_prompt.csv | 逐条case、category、原始prompt、seed=42、长短目标和图片SHA256 |
| configs/ | 24份原始请求参数，runs.csv中的config以本目录为基准 |
| model_version.json | 权重/源码revision、CFG/step LoRA、BF16、16步及内存策略 |
| cases/ | 逐条request/run/result、16步耗时、GPU规格与推理日志 |
| source/ | 未改写的原始runs.csv、冻结批次配置、状态与完成记录 |
| provenance/ | 原始及BF16权重SHA256、源码凭据和真实环境版本 |
| slurm/ | 两个Slurm作业的日志与COMPLETED/0退出码记录 |
| verification.json | D本地逐视频完整解码和冻结输入/配置/来源核验记录 |
| summary.json | 计数、耗时及证据范围摘要 |

生成采用通用`meituan-longcat/LongCat-Video`，官方480p/16步CFG-step蒸馏配置，CFG=1、BF16。原图和英文prompt沿用共享规范清单，seed均为42，单次连续I2V生成，没有续写、拼接或插帧。短片GPU全驻留，长片卸载40/48个DiT块；相同RoPE频率共享一份缓存。

## 交接字段

`runs.csv`是派生交接表：`succeeded`映射为`success`，统一模型ID为`longcat-video-480p-16step-distill`，补齐`weight_version`、相对配置路径、输入哈希、GPU及MiB显存字段。`source/runs.csv`保留原始字节及原始状态；没有改写旧记录。未实测的peak reserved显存留空。

原片单独提供。接收时把24个原字节视频放到本目录的`native/case_0001.mp4`至`native/case_0024.mp4`，然后将`longcat-video-480p-16step-distill=/absolute/path/to/longcat24-records/runs.csv`作为B四模型media入口的一项输入。其余三模型由相应成员提供。

正式评分副本由B按仓库media流程从native重新制作：全长、保持比例补边、768×432 / 12fps。此前本地调试evaluation副本不作为本记录交接的正式评分输入。

## 计时与范围

本次23条新生成从2026-11-15 14:59运行到22:26，含最终校验打包约7小时27分钟；case_0001此前已完成并复用。完整24条逐条elapsed_seconds之和约8小时29分钟，包含加载、编码、采样、解码、写盘和D的解码检查，不含首次下载和安装。短片均值约7分27秒，长片均值约1小时2分27秒，配置与不同模型的步数/分辨率需分别报告。

未完成计划中的3个独立共同试跑样本。case_0001复用同参数技术试跑，明确标记`imported_validated_trial`；源记录`formal_evaluation=false`和`phase=full24`原样保留，待小组/B确认该协议例外后纳入正式96条。试跑前的依赖缺失、显存不足和内存调试记录未被当作完成视频；本批次24行的完成尝试均为1次，没有根据画质重抽。Cog的20条生成结果属于暂停的备选批次，不混入LongCat记录。

目录内原始记录按MANIFEST.sha256逐文件绑定；.gitattributes关闭文本换行转换，以保留日志和CSV原始字节。
