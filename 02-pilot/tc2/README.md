# C 的 TC2 环境检查

先按学校指南登录集群，执行 `mytcinfo` 与 `sinfo -N -l`，核对自己的 QoS、分区、资源上限与节点状态。SSH 主节点只用于文件管理与提交作业；本目录的脚本通过 `sbatch` 在计算节点执行。

## 检查 GPU

从保存日志的目录提交（用自己的实际分区与 QoS）：

```bash
sbatch --partition=MGPU-TC2 --qos=normal /path/to/inspect_gpu.sh
```

查看 GPU 型号、显存、驱动和 `CUDA_VISIBLE_DEVICES`，沿用调度器分配的 GPU 编号。`--mem` 指系统内存。

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

该作业安装固定上游代码及依赖，下载固定版本的 2B、upscaler 和 T5 资产，并保存完整环境导出。详见 [LTX-Video 安装与生成步骤](../../03-generation/ltx_video/README.md)。正式推理前须冻结模型版本与生成配置。

## 完整流程检查与后续批次

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

当前A实际11类混合长度清单先经CPU适配，再提交 `formal /absolute/cases.canonical.csv /absolute/artifacts/formal-mixed`，默认使用全部24个案例和逐案例帧数。129/513帧须共同试跑验收。示例路径均需替换为实际路径。

## 作业状态

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
```

`PENDING` 表示作业尚未开始，排队原因见 `squeue`；完成状态和退出码见 `sacct`。

## CPU 工具与测试

`run_cpu.sh` 覆盖人评准备/素材核验/收表、合成首图创建、生成清单检查和三套单元测试。所有 Python 均在 `sbatch` 分配的计算节点执行；不要在 Head Node 用 `bash run_cpu.sh` 或直接运行 Python。无需 GPU，默认 `MGPU-TC2 / normal / 1 node / 1 CPU / 2G / 0 GPU / 10 分钟`，无节点绑定；这些是脚本默认值，每次提交前需用 `mytcinfo` 和必要的 `sacctmgr` 查询核对，按实际权限用 sbatch 参数覆盖。

从仓库根目录提交；若在别处提交，先设置 `AI6132_CODE` 为仓库绝对路径。脚本加载 `anaconda` 模块，以非交互方式激活 `AI6132_CPU_ENV`；默认环境为 `${AI6132_ROOT:-$HOME/ai6132-c}/env`。可改用已有个人 Python 3.10+ 环境；只有创建合成图片需要 Pillow，其他 CPU 入口仅使用标准库。

```bash
export AI6132_CODE="$PWD"
export AI6132_CPU_ENV=/absolute/path/to/personal/conda/env
mytcinfo
sbatch 02-pilot/tc2/run_cpu.sh validate-cases \
  --cases 01-dataset/cases.canonical.csv --phase formal
sbatch 02-pilot/tc2/run_cpu.sh test
```

人评参数示例见 [人评说明](../../05-human-evaluation/README.md)。日志写入提交目录的 `ai6132_cpu_<jobid>.out` 和 `.err`。这些 CPU 任务无需 checkpoint、训练恢复、signal 或后继作业；测试中出现的模拟中断日志不代表真实推理失败。

新增CPU子命令：dataset（适配/验证原始清单）、assets（已下载资产的revision/SHA256凭据）、media（完整时长匿名交接）、vbench-cache（缓存凭据）。详见各阶段README。test也覆盖共享工具回归；完整CPU套件除Pillow外需要可用FFmpeg/ffprobe。

B的生成与评估统一通过 `run_b_gpu.sh hunyuan|vbench|qwen`，显式设置个人 `AI6132_GPU_ENV`。默认MGPU-TC2/normal、1node、4CPU、28G、1GPU、2小时，无节点绑定；提交前重新查询资源与配额。stdout/stderr为提交目录的ai6132_b_JOBID.out/.err。Hunyuan、VBench、Qwen按单GPU配额串行；CPU任务只有在总CPU/内存配额允许时并行。这里是推理批处理，不使用训练checkpoint或无限自动重提；逐案例/请求日志支持继续未启动项。

正式生成前可提交 `preflight_hunyuan.sh` 检查输入和下载权限；`bootstrap_hunyuan.sh` 用于独立Hunyuan环境安装与CPU导入检查，均无GPU需求。参数、路径和验收边界见 [B生成说明](../../03-generation/hunyuan_video/README.md)。

参考：[PyTorch 对应版本安装命令](https://pytorch.org/get-started/previous-versions/)、项目 `PLAN.md`、学校 TC2 用户指南。
