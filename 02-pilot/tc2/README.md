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

在本目录运行以下命令，建立合成环境检查输入（需要 Pillow）：

```bash
python make_smoke_case.py --out /absolute/private/smoke-input
```

上传输入文件夹后提交环境检查作业。`run_ltx.sh` 默认申请 30GB 系统内存：

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

参考：[PyTorch 对应版本安装命令](https://pytorch.org/get-started/previous-versions/)、项目 `PLAN.md`、学校 TC2 用户指南。
