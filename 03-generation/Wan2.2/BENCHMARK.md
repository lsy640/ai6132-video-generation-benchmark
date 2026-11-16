# Wan2.2 TI2V-5B 项目接入

本目录为用户上传的官方源码，审查时87个文件的Git blob SHA1均与官方固定提交`1ea34ff48f87168174e12956e200b1d908b1c5ff`一致。许可证保留在`LICENSE.txt`，来源及本地补丁见`UPSTREAM.json`。官方README与安装说明保留原文；其中Python、安装、编译和多卡示例在TC2必须改由sbatch运行，不能在Head直接执行。

该上传不包含A的原始批处理脚本、生成时源码快照或权重，因此不建立与已交付A视频的历史执行绑定。历史A为10步、125/509帧，保持原始记录和评估；新增入口不会改写或自动替换它们。

## 审查与修复

| 问题 | 接入方式 |
| --- | --- |
| 官方CLI默认121帧、默认随机seed，无法直接承接项目七列清单 | `benchmark.py`读取完整规范清单，固定原prompt／seed42及129/513目标帧数 |
| CLI只在帮助中说明4n+1，未拒绝128/512等非法请求帧数 | 项目入口严格读取已校验的target_frames，不直接传原始source_frame_num；实际解码帧数必须与目标相同 |
| 官方`save_video`捕获编码异常后仅记录日志，CLI可能正常退出 | 新worker使用返回码受检查的FFmpeg写入，父进程再次全片解码、核对帧数／尺寸／fps／时长及视频SHA256 |
| `wan/__init__.py`提前加载全部音频／动画管线 | 本地补丁按请求惰性加载，仅使用TI2V所需依赖；FlashAttention仍为真实5B模型的必需项 |
| 缺少批次冻结、失败保留及共享交接表 | 输入／源码／模型receipt绑定到新batch，独占写锁，逐案例结果原子写入，24行runs.csv保留未启动和失败 |
| 上游测试含多卡、扩写及其他模型任务 | 项目CPU测试用`wan_cpu.sh test`；不执行上游`tests/test.sh`作为基准回归 |

原始上传已在个人审查证据中备份；推理数学、采样器和VAE代码保持原字节。仅包级惰性导入及.gitignore有本地补丁。官方示例音视频／图片不纳入源码合并，仍保留在用户原始目录。

## 配置与限制

默认`formal50`：TI2V-5B、50步UniPC、shift5、CFG5、BF16、CPU offload、T5 CPU、关闭扩写，完整24例、18×129＋6×513帧。输入是16:9，官方1280×704面积预算及32像素网格实际选择1248×704，输出24fps。显示评分副本仍由共享media流程制作。

`diagnostic10`为独立的新10步诊断模型ID，仍严格保留129/513帧，不重现历史向下取帧错误，不与已交付A的125/509帧混称同一批次。`formal50`名称仅表示计划参数，不能证明共同试跑、GPU资源或完整生成已验收。

模型、T5、VAE、tokenizer必须提前以完整revision下载到个人获授权位置。`assets.json`绑定实际文件集合及逐文件SHA256；prepare与每次生成均核对它及源图、清单、程序，新增或变更权重均拒绝。运行时离线，不下载模型、不启用prompt扩写。上游requirements为依赖范围，不是本项目GPU环境锁；还须通过计算节点验证Torch／Transformers／FlashAttention兼容性与资源。

assets入口只为已经下载的文件记账，不向官方仓库核验来源；下载阶段的固定revision与官方LFS／Git凭据须另外保留，不能把本地receipt一致当成官方权重真实性认证。

每个GPU作业只处理一个显式指定案例，3小时为可调整资源默认值，不是运行耗时保证。当前未验证真实50步513帧在TC2的内存／显存峰值；未满足共同试跑和资源条件前不能声称正式生成可行。该入口不执行训练，不生成训练checkpoint，也不自动重提。成功案例校验后直接返回，已有失败或中断的attempt拒绝自动重抽；明确重试应使用新批次并保留旧批次。只有24条均成功才创建DONE.json。

## TC2 操作

以下命令从项目根目录提交。所有Python、安装与GPU工作均在计算节点；每次提交前查询mytcinfo并确认个人存储配额。CPU入口默认MGPU-TC2/normal、1CPU、2G、0GPU、30分钟；GPU入口默认4CPU、28G、1GPU、3小时，均1节点且不固定nodelist。

```bash
export AI6132_CODE="$PWD"
export AI6132_CPU_ENV=/absolute/personal/cpu-conda-environment
export AI6132_WAN_ENV=/absolute/personal/wan-conda-environment
export AI6132_WAN_ASSETS=/absolute/personal/Wan2.2-TI2V-5B
export AI6132_WAN_BATCH=/absolute/personal/outputs/wan-formal50-new
export AI6132_WAN_REVISION=FULL_40_CHARACTER_MODEL_REVISION
mytcinfo
sbatch 02-pilot/tc2/wan_cpu.sh assets --root "$AI6132_WAN_ASSETS" \
  --revision "Wan-AI/Wan2.2-TI2V-5B=$AI6132_WAN_REVISION"
```

确认receipt作业COMPLETED/0:0后，重新核对资源并准备新批次；已有有效receipt不重复创建：

```bash
mytcinfo
sbatch 02-pilot/tc2/wan_cpu.sh prepare \
  --cases "$AI6132_CODE/01-dataset/cases.canonical.csv" \
  --assets "$AI6132_WAN_ASSETS" --revision "$AI6132_WAN_REVISION" \
  --profile formal50 --out "$AI6132_WAN_BATCH"
```

prepare验证通过后，再在满足试跑／资源前提时逐案例提交。GPU脚本显式激活AI6132_WAN_ENV和cuda/12.8.0；实际环境另行准备和验收，不修改其他生成或评测环境。

```bash
mytcinfo
sbatch 02-pilot/tc2/run_wan.sh --batch "$AI6132_WAN_BATCH" --case-id case_0001
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
```

后续案例使用同一batch、不同case-id，每次在前一个案例结束后提交；文件锁拒绝并发写同一批次。日志为提交目录`ai6132_wan_cpu_JOBID.out/.err`、`ai6132_wan_JOBID.out/.err`，案例日志在batch下`cases/case_ID/worker.log`。

交接时以`wan2.2-ti2v-5b-formal50=/absolute/batch/runs.csv`作为四模型media的一项输入；10步新诊断ID为`wan2.2-ti2v-5b-diagnostic10`。表保留24个位置，成功项使用相对视频／配置路径并含版本、输入／视频／配置哈希及资源字段；整个batch搬移后仍可用于media交接。重新生成需要原冻结源码、权重与输入路径，不能把迁移交接误称为已验证的异地恢复。

## 验证证据

2026-11-16作业55027执行12项Wan合约及真实FFmpeg129/513帧视频解码回归、25项LongCat、30项LTX、19项人评、15项共享测试，共101项通过；60个Python文件AST语法检查及官方纯尺寸函数1248×704核对通过。源码前后哈希一致。该作业仅验证编排、文件和编码路径，合成权重／视频不计实验结果；没有下载权重、加载真实Wan模型或运行GPU生成。

另以当前本地shared media／Qwen扩展构建整合快照，55031在TC2N01以COMPLETED/0:0、29秒完成Wan12项及共享43项，共55项相关回归；源码前后哈希一致。与101项集合有重叠，不合称156项不同测试。此验证不把本地未提交扩展纳入Wan提交，也不增加GPU验收范围。

个人审查证据目录为`ai6132-review-evidence/wan-integration-20261116/`，包含原始上传备份、官方87文件比对、测试日志与验证脚本。后续GPU验证须另行记录环境、完整权重来源、内存／显存、原生长短片和独立完整解码。
