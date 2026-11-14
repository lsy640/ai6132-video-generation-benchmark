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

## 长视频运行环境：已通过预检

513 帧预检在旧环境（PyTorch 2.5.1+cu121 / cuDNN 9.1）完成去噪后，于 VAE 解码的大 Conv3D 处回退并发生显存不足。独立 `env-cu126` 已在 L40S 上通过 513 帧和 129 帧各一条预检，使用 PyTorch 2.6.0+cu126、torchvision 0.21.0 和 cuDNN 9.5.1。两次生成计时分别约 166.22 秒和 13.16 秒，属于单条观测，详见 [验证记录](VALIDATION.md)。

正式 24 条作业 `54359` 已提交到 `formal-cu126-20261114`，尚未完成整批验收。

`bootstrap_cu126.sh` 克隆已安装的基础环境，保留其余依赖，再升级 PyTorch 相关包，旧 `env` 保持不变。模型权重、BF16、两阶段配置、768×432、24 fps 和请求帧数不变。`prepare_environment.sh` 与原 `constraints.txt` 仍用于 2.5.1 基础环境，不在新环境中重跑，以免降级。

以下为新部署示例；已有 `env-cu126` 时直接选择该环境，不重复创建。第二项等待安装成功，再用一条目标为 512 帧的原始数据行检查 513 帧输出；子清单保留 A 的字段、prompt 和 seed。

```bash
export AI6132_CODE="$PWD"
export AI6132_LTX_ENV="$AI6132_ROOT/env-cu126"
export TORCH_CUDNN_V8_API_DISABLED=0
C_SETUP_JOB=$(sbatch --parsable --partition=MGPU-TC2 --qos=normal \
  02-pilot/tc2/bootstrap_cu126.sh)
sbatch --dependency="afterok:$C_SETUP_JOB" --partition=MGPU-TC2 --qos=normal \
  02-pilot/tc2/run_ltx.sh smoke /absolute/shared/preflight-long-01.csv \
  "$AI6132_ROOT/outputs/preflight-long-cu126-01" --protocol shared24
```

`run_ltx.sh` 通过 `AI6132_LTX_ENV` 选择环境，未设置时仍使用旧 `env`。新环境检查使用独立输出目录，保留此前失败的日志和记录。

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

全组冻结配置及正式样本后，提交 `formal /absolute/shared/cases.csv /absolute/artifacts/formal --per-category 2`。扩批与续跑规则见生成工具说明。示例路径均需替换为实际路径。

## 作业状态

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
```

`PENDING` 表示作业尚未开始，排队原因见 `squeue`；完成状态和退出码见 `sacct`。

## CPU 工具与测试

`run_cpu.sh` 覆盖人评准备/素材核验/收表、合成首图创建、生成清单检查和两套单元测试。所有 Python 均在 `sbatch` 分配的计算节点执行；不要在 Head Node 用 `bash run_cpu.sh` 或直接运行 Python。无需 GPU，默认 `MGPU-TC2 / normal / 1 node / 1 CPU / 2G / 0 GPU / 10 分钟`，无节点绑定；这些是脚本默认值，每次提交前需用 `mytcinfo` 和必要的 `sacctmgr` 查询核对，按实际权限用 sbatch 参数覆盖。

从仓库根目录提交；若在别处提交，先设置 `AI6132_CODE` 为仓库绝对路径。脚本加载 `anaconda` 模块，以非交互方式激活 `AI6132_CPU_ENV`；默认环境为 `${AI6132_ROOT:-$HOME/ai6132-c}/env`。可改用已有个人 Python 3.10+ 环境；只有创建合成图片需要 Pillow，其他 CPU 入口仅使用标准库。

```bash
export AI6132_CODE="$PWD"
export AI6132_CPU_ENV=/absolute/path/to/personal/conda/env
mytcinfo
sbatch 02-pilot/tc2/run_cpu.sh validate-cases \
  --cases /absolute/shared/cases.csv --phase formal --per-category 2
sbatch 02-pilot/tc2/run_cpu.sh test
```

人评参数示例见 [人评说明](../../05-human-evaluation/README.md)。日志写入提交目录的 `ai6132_cpu_<jobid>.out` 和 `.err`。这些 CPU 任务无需 checkpoint、训练恢复、signal 或后继作业；测试中出现的模拟中断日志不代表真实推理失败。

参考：[PyTorch 对应版本安装命令](https://pytorch.org/get-started/previous-versions/)、项目 `PLAN.md`、学校 TC2 用户指南。
