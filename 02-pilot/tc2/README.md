# TC2 环境、批处理入口与验证

先按学校指南登录集群，执行 `mytcinfo` 与 `sinfo -N -l`，核对自己的 QoS、分区、资源上限与节点状态。SSH 主节点只用于文件管理与提交作业；本目录的脚本通过 `sbatch` 在计算节点执行。

截至2026-11-16 15:29（新加坡时间），LongCat审查修复已以`d9987f3`推送main；55016完成89项候选及68项关联整合回归和310项历史记录校验。四模型VBench224项与人评384条收表完成；Qwen54992只通过诊断技术准入，原严格门槛仍失败，54993／55005已处理24条、55013运行中。版本、缺失评分及验收边界见[验证记录](VALIDATION.md#当前范围与状态快照)。

发布边界：`human-xlsx`、新增评估环境脚本、诊断扩展和32B加速提交链是个人工作区入口，其代码尚未随本文发布。下文相应段落记录本地操作；已发布的LTX、LongCat及历史共享入口按各自版本使用，不能从公开仓库的旧`run_cpu.sh`直接调用本地新增子命令。

## 检查 GPU

从保存日志的目录提交（用自己的实际分区与 QoS）：

```bash
sbatch --partition=MGPU-TC2 --qos=normal /path/to/inspect_gpu.sh
```

查看 GPU 型号、显存、驱动和 `CUDA_VISIBLE_DEVICES`，沿用调度器分配的 GPU 编号。`--mem` 指系统内存。

## Wan2.2 项目入口

Wan TI2V-5B使用[`wan_cpu.sh`](wan_cpu.sh)的assets／prepare／test子命令，以及[`run_wan.sh`](run_wan.sh)的单案例GPU生成入口，不调用上游多卡测试脚本。CPU默认1CPU／2G／0GPU／30分钟，GPU默认4CPU／28G／1GPU／3小时，1节点、无固定nodelist；每次sbatch前仍须重新查询mytcinfo。个人环境由AI6132_CPU_ENV和AI6132_WAN_ENV显式指定。

默认新formal50批次为50步、129/513帧；每个GPU作业只处理一个case-id，24行runs.csv保留未启动项，成功不重抽，失败／中断不自动重提。GPU资源和独立共同试跑仍需另行验收，CPU测试不能解锁真实生成可行性。历史A的10步125/509帧与新入口分开。完整路径、receipt与续跑边界见[Wan接入说明](../../03-generation/Wan2.2/BENCHMARK.md)。

## 建立基础环境

`bootstrap_env.sh` 使用 `miniconda3/25.5.1` 模块，在 `AI6132_ROOT` 下建立 Python 3.10 环境，安装 FFmpeg、PyTorch 2.5.1 和 torchvision 0.20.1（CUDA 12.1 wheel），并检查 BF16 矩阵运算。已验证环境为 L40S / 驱动 570.148.08，详见 [验证记录](VALIDATION.md)。

```bash
export AI6132_ROOT="$HOME/ai6132-c"
sbatch --partition=MGPU-TC2 --qos=normal /path/to/bootstrap_env.sh
```

默认申请 1 GPU、8 CPU、28GB 系统内存、2 小时。提交前应确认符合自己的配额；本次 C 账号的配额为 1 GPU、10 CPU、30GB 系统内存、最长 6 小时。可使用 `--nodelist` 指定已确认可用的节点。

环境创建完后，将本仓库放在 `$AI6132_ROOT/code`（或通过 `AI6132_CODE` 指定路径），再提交安装作业：

```bash
sbatch --partition=MGPU-TC2 --qos=normal /path/to/install_ltx.sh
```

该作业安装固定上游代码及依赖，下载固定版本的 13B、upscaler 和 T5 资产，并保存完整环境导出。当前资产目录默认为 `$AI6132_ROOT/ltx_assets_13b`，也可通过 `AI6132_LTX_ASSETS` 指定；旧 2B 目录保留。下载前核对 Home 空间。详见 [LTX-Video 安装与生成步骤](../../03-generation/ltx_video/README.md)。

## 完整流程检查与后续批次

2026-11-15 全组同意将 C 改为 13B v0.9.8 Distilled，并在新目录重跑同一规范清单的全部 24 例。BF16、官方 7+3 timesteps、第二阶段 tone map compression ratio 0.6、关闭提示词扩写，18×129＋6×513 帧及 768×432 / 24 fps 固定。正式作业 **54620 已完成 24/24，COMPLETED / 0:0 / 30 分 27 秒**；CPU 作业 54622 的 64 项回归及 24 条 FFmpeg 全片验收通过，本地 PyAV 再次完整解码 5400 帧。B后续独立接收54638已通过，新13B已纳入四模型评估，本轮四份人工表已收齐。详见 [汇总 JSON](ltx13b-formal-summary.json)与[C 批次报告](C_FORMAL_REPORT.md)，其中保留旧 2B 作业 54359 及 PR #2 历史。

旧 cu121 环境在 2B 513 帧 VAE 解码失败；当前 `run_ltx.sh` 默认激活 `AI6132_LTX_ENV=${AI6132_ROOT}/env-cu126`。在基础环境和 LTX 安装成功后，查询配额并提交 `sbatch 02-pilot/tc2/bootstrap_cu126.sh` 创建独立 cu126 环境。脚本默认 MGPU-TC2/normal、1节点、8CPU、28G、1GPU、1小时，无节点绑定；安装前检查 Home 用量、克隆体积和 12GB 升级缓存预留。已有环境时直接复用，不在其中重跑旧 constraints。Conda hook 与 activate 均保护 nounset，启用 cuDNN v8 API 路径。

13B 沿用 torch2.6.0+cu126 / cuDNN9.5.1。内存适配先缓存提示词编码后释放 T5，两阶段完成后释放 Transformer 再解码，实际 CPU offload 为 false。2B129 适配回归与旧基线 SHA256 一致；13B 正式短片/长片生成均值为 25.0237/206.1915 秒。此前 case_0001 的长短技术试跑分别为 40.5888/208.5975 秒，单列保存，不计入正式统计；本次未另行完成三个独立共同试跑样本。详细证据见 [验证记录](VALIDATION.md)。

从仓库根目录提交 CPU 作业，建立合成环境检查输入（个人环境中需有 Pillow）：

```bash
export AI6132_CPU_ENV="$AI6132_ROOT/env"
sbatch 02-pilot/tc2/run_cpu.sh make-smoke --out /absolute/private/smoke-input
```

确认 CPU 作业成功、输入文件已写完后，提交环境检查作业。`run_ltx.sh` 默认申请 30GB 系统内存：

```bash
sbatch --partition=MGPU-TC2 --qos=normal --time=00:30:00 /path/to/run_ltx.sh \
  smoke /absolute/smoke-input/cases.csv /absolute/artifacts/smoke-001
```

合成测试仅用于检查环境。A 提供三个共同试跑样本后，用独立目录提交：

```bash
sbatch --partition=MGPU-TC2 --qos=normal /path/to/run_ltx.sh \
  pilot /absolute/shared/pilot_cases.csv /absolute/artifacts/pilot
```

当前 A 的规范清单已冻结，正式入口默认使用全部 24 个案例和逐案例帧数。本次 13B 修订沿用上述 case_0001 长短技术试跑，按全组决定直接提交完整正式批次：

```bash
export AI6132_CODE="$PWD"
export AI6132_LTX_ENV="$AI6132_ROOT/env-cu126"
export AI6132_LTX_ASSETS="$AI6132_ROOT/ltx_assets_13b"
sbatch --partition=MGPU-TC2 --qos=normal 02-pilot/tc2/run_ltx.sh \
  formal "$AI6132_CODE/01-dataset/cases.canonical.csv" \
  "$AI6132_ROOT/outputs/ltx13b-formal-20261115"
```

新模型使用独立输出目录；旧 2B 冻结记录不覆盖、不混入 13B 统计。示例路径需替换为实际路径。

## 作业状态

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
```

`PENDING` 表示作业尚未开始，排队原因见 `squeue`；完成状态和退出码见 `sacct`。

## CPU 工具与测试

`run_cpu.sh` 覆盖人评准备/素材核验/收表、合成首图创建、生成清单检查和三套单元测试。所有 Python 均在 `sbatch` 分配的计算节点执行；不要在 Head Node 用 `bash run_cpu.sh` 或直接运行 Python。无需 GPU，默认 `MGPU-TC2 / normal / 1 node / 1 CPU / 2G / 0 GPU / 10 分钟`，无节点绑定；这些是脚本默认值，每次提交前需用 `mytcinfo` 和必要的 `sacctmgr` 查询核对，按实际权限用 sbatch 参数覆盖。

从仓库根目录提交；若在别处提交，先设置 `AI6132_CODE` 为仓库绝对路径。脚本加载 `anaconda` 模块，以非交互方式激活 `AI6132_CPU_ENV`；默认环境为 `${AI6132_ROOT:-$HOME/ai6132-c}/env`。可改用已有个人Python3.10+环境。创建图片需要Pillow，`human-xlsx`另需openpyxl3.1.5和可用FFmpeg/ffprobe；本次隔离环境在`/home/msai/lius0131/XJIANG026/liu/ai6132-evaluation/human-tools-env`，节点TC2N08可访问。不要将其他Conda环境的bin目录放在激活环境之前，FFmpeg可用`AI6132_FFMPEG`及`AI6132_FFPROBE`显式指定。

```bash
export AI6132_CODE="$PWD"
export AI6132_CPU_ENV=/absolute/path/to/personal/conda/env
mytcinfo
sbatch 02-pilot/tc2/run_cpu.sh validate-cases \
  --cases 01-dataset/cases.canonical.csv --phase formal
sbatch 02-pilot/tc2/run_cpu.sh test
```

人评参数示例见[人评说明](../../05-human-evaluation/README.md)。新版human-xlsx提供prepare、verify、upgrade-web、complete-d、validate-workbook；网页支持直接评分、导入已有Excel和导出Excel。当前仅分发四份96条完整包，upgrade-web不再生成网页升级补丁，已有进度可导入新完整包继续。prepare/complete-d处理整批视频，需覆盖默认10分钟时限；D已接收，无需再次补入。浏览器集成测试另外使用个人human-web-test-env和Chromium，通过sbatch运行，不在Head Node启动浏览器；成员使用评分包无需这些依赖。日志为ai6132_cpu_JOBID.out/.err。上述CPU任务无需训练checkpoint、signal或自动重提。

新增CPU子命令：dataset（适配/验证原始清单）、assets（已下载资产的revision/SHA256凭据）、media（完整时长匿名交接）、vbench-cache（缓存凭据）、import-ltx（完整解码接收 C 的完整 24 条，支持新 13B 规范批次与历史 2B Shared24）。新13B已接收并重建独立评分材料；旧2B结果保留。test覆盖LTX、人评及共享工具，除Pillow外需要FFmpeg/ffprobe，本地Excel扩展另需openpyxl。LongCat25项单独经`tools_cpu.slurm test`提交，环境还需PyAV，详见[LongCat说明](../../03-generation/longcat_video/README.md#验证范围)。

B的生成与评估统一通过 `run_b_gpu.sh hunyuan|vbench|qwen`，显式设置个人 `AI6132_GPU_ENV`。默认MGPU-TC2/normal、1node、4CPU、28G、1GPU、2小时，无节点绑定；提交前重新查询资源与配额。stdout/stderr为提交目录的ai6132_b_JOBID.out/.err。Hunyuan、VBench、Qwen按单GPU配额串行；CPU任务只有在总CPU/内存配额允许时并行。这里是推理批处理，不使用训练checkpoint或无限自动重提；逐案例/请求日志支持继续未启动项。

正式生成前可提交 `preflight_hunyuan.sh` 检查输入和下载权限；`bootstrap_hunyuan.sh` 用于独立Hunyuan环境安装与CPU导入检查，均无GPU需求。参数、路径和验收边界见 [B生成说明](../../03-generation/hunyuan_video/README.md)。

参考：[PyTorch 对应版本安装命令](https://pytorch.org/get-started/previous-versions/)、项目 `PLAN.md`、学校 TC2 用户指南。

## 隔离评估环境与诊断批次

已交付部分模型时可用独立diagnostic批次预评估，仍保持每模型完整24例及统一评分副本；默认正式四模型约束保持不变。环境、缓存和权重使用明确授权且有足够空间的存储，提交前查询配额和挂载可写性。Head Node挂载只读时，在获授权的计算节点批处理中检查和写入。

`bootstrap_evaluation.sh` 克隆现有B生成环境到独立评估目录，优先尝试Conda固定依赖，所需固定组合不可用时安装对应PyPI版本。生成环境保持不变。默认2CPU/8G/0GPU/1小时；`download_evaluation.sh` 默认2CPU/4G/0GPU/1小时，预留40GB空间并校验固定Qwen revision、官方Git/LFS哈希及VBench权重来源。两脚本均要求设置 `AI6132_CODE` 与 `AI6132_EVAL_ROOT`；默认克隆目标为该目录下env，可通过 `AI6132_EVAL_ENV`指定新环境。

Decord 0.6.0的公开py3 wheel内部带有cp36标签，较新pip check会报告不支持平台。脚本仅允许该精确元数据告警，原文保存在receipts；其他依赖错误仍失败。实际导入与全部已准备评分副本的Decord完整解码另行验证，未修改上游wheel或解码二进制。环境版本和校验凭据另存，不将元数据告警说成pip check全部通过。

评分副本准备完成后，`evaluate_diagnostic.sh` 需要以下变量：

- `AI6132_CODE`：本仓库；`AI6132_EVAL_ROOT`：已授权评估存储；`AI6132_GPU_ENV`：已验证的独立Conda环境。
- `AI6132_DIAGNOSTIC_DIR`：含COMPLETE.json、handoff及private的诊断副本目录。
- `AI6132_DIAGNOSTIC_REASON`：允许在人评前运行诊断Qwen的真实用户指示；缺失时拒绝执行。
- `AI6132_EVAL_ATTEMPT`：独立输出后缀，默认v1；配置变更或故障恢复应保留旧目录并显式选择新后缀。

默认MGPU-TC2/normal、1节点、4CPU、28G、1GPU、5小时50分钟；脚本不固定节点，必要时在提交命令指定已核验的节点。受单GPU配额限制，VBench与Qwen依次执行；下载、环境和CPU转换可在总配额允许时并行。每个指标/请求立即保存结果，中断和失败保留，不无限重提。stdout/stderr为ai6132_diagnostic_JOBID.out/.err，预检与源码快照在诊断目录preflight_JOBID，最终分析在analysis-<attempt>。

当前Qwen v3.5使用外部证据目录中独立冻结的`evaluate_fast.sh`及Head后继提交器，采用8CPU／28G／1GPU／3小时、每次12视频；不等同于上面的历史v1入口。提交器在每次新提交前查询mytcinfo，并以锁、父作业标记及SLURM备注防重复。原始回复和无效视图保留，普通作业错误停止。通过SLURM与日志查看状态，当前输入、门槛及后继边界见[Qwen协议](../../06-vlm-evaluation/QWEN_V3.md#当前状态快照)。
