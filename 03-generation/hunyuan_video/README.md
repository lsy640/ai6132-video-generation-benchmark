# B：Hunyuan 原生混合长度生成

入口为 `ai6132_benchmark.hunyuan`，统一通过 `02-pilot/tc2/run_b_gpu.sh hunyuan` 提交。固定480p step-distilled、8步、Transformer BF16、24fps，Transformer分组CPU卸载，编码器及VAE驻留GPU；关闭overlap offload、缓存、改写和超分。原生分辨率由官方分桶选择并记录；同一案例不自动重抽。

TC2当前主机内存配额30G，无法同时常驻全部卸载组件。适配器在GPU初始化组件、按官方最终fp16精度直接加载文本编码器、保留Transformer分组卸载钩子并关闭全组件CPU迁移；这些选择写入冻结设置。输出逐帧编码，避免为513帧视频同时分配多份全片浮点数组。生成和长片可行性以实际运行结果为准。

分别准备 Hunyuan、VBench/CPU 和 Qwen 的个人环境，不复用 C 的 LTX 依赖版本。Hunyuan 使用固定官方 commit 的 requirements（当前官方组合需要 torch>=2.6、diffusers0.35、transformers4.57.1，须将最终兼容版本锁定）。安装、编译、导入和模型检查也只能经 sbatch。此仓库不会在运行时安装依赖或自动下载权重。

### 正式作业前置检查

`02-pilot/tc2/preflight_hunyuan.sh` 使用CPU作业完整校验正式输入、检查独立试跑清单和本地运行目录，并查询官方模型的固定revision、所需文件大小及权重访问状态；不下载权重。设置 `AI6132_CODE`、含Pillow的 `AI6132_CPU_ENV` 和仓库外新文件路径 `AI6132_PREFLIGHT_REPORT`，查询 `mytcinfo` 后用 `sbatch` 提交。若已有独立试跑清单，可追加 `--pilot-cases /absolute/pilot.canonical.csv`。退出2表示有未满足的前置条件，不能视为正式生成失败或成功。

官方源码固定在 `60783e704160023913bee78f0b47036d393d4dfa`。`02-pilot/tc2/bootstrap_hunyuan.sh` 在计算节点新建 `$HOME/ai6132-b/env`，以Conda安装基础环境、以匹配CUDA12.4的PyTorch2.6/vision0.21/audio2.6和官方requirements安装推理依赖；兼容约束见 `constraints.txt`。作业检查依赖及上游导入，保存完整环境导出与 `environment/READY.json`；该凭据不代表GPU、模型权重或试跑验收通过。提交前核对Home配额、剩余空间及当前QoS；默认2CPU/8G/0GPU/30分钟，无节点绑定。

2026-11-14初次预检54338验证正式24例及图片通过，未提供凭据时视觉编码器返回HTTP401。随后按用户指示安全复用既有私有凭据文件，认证预检54363确认四个HF仓库权重均可访问（HTTP200），其退出2仅表示资产尚未下载。凭据文件须由当前用户拥有、为普通文件且权限600，内容仅在计算节点内存使用，不写入代码、参数、日志或Git；重定向离开HF域名时移除认证头。使用 `--token-file /private/credential` 指定路径。状态见[验证记录](../../02-pilot/tc2/VALIDATION.md)。

环境安装54342已完成，依赖和官方导入检查通过，实际版本冻结于个人 `ai6132-b/environment/`。安装后可通过 `sbatch 02-pilot/tc2/check_hunyuan_env.sh` 检查CUDA/BF16及单卡NCCL和官方初始化（默认2CPU/4G/1GPU/5分钟，每次提交前查询QoS）。54349已在NVIDIA A40上通过此检查；`gpu_check_54349.json` 绑定环境凭据。该作业不加载模型权重，不能替代三例共同试跑或正式生成验收。

先只读确认个人配额和磁盘剩余量，再按固定revision下载必要资产到个人Home；不要下载整个多模型仓库。官方加载目录包括 `transformer/480p_i2v_step_distilled`、`vae`、`scheduler`、`text_encoder/llm`、`text_encoder/byt5-small`、`text_encoder/Glyph-SDXL-v2`、`vision_encoder/siglip`。视觉编码器按官方说明核实访问权限；密钥仅用已有安全环境注入，不写入脚本或命令参数。所有源仓库的固定revision及本地文件SHA256需写入 `assets.json`：

```bash
sbatch 02-pilot/tc2/run_cpu.sh assets --root /absolute/hunyuan-assets \
  --revision tencent/HunyuanVideo-1.5=FULL_REVISION_SHA Qwen/Qwen2.5-VL-7B-Instruct=FULL_REVISION_SHA OTHER_SOURCE=FULL_REVISION_SHA
```

占位符须替换为实际完整40位SHA；这一步只审计已下载文件，不能把浮动下载变成已验证来源。保留下载时的 revision/文件清单。

```bash
export AI6132_CODE="$PWD"
export AI6132_GPU_ENV=/absolute/personal/hunyuan/environment
mytcinfo
sbatch 02-pilot/tc2/run_b_gpu.sh hunyuan \
  --cases /absolute/independent/pilot.canonical.csv --phase pilot \
  --upstream /absolute/HunyuanVideo-1.5 --commit FULL_UPSTREAM_SHA \
  --assets /absolute/hunyuan-assets --out /absolute/artifacts/hunyuan-pilot
```

试跑3条全部成功并包含长短两档后产生 `pilot_pass.json`。常规正式入口用该凭据核对同一代码、环境、权重和设置，并拒绝与试跑图片重合：

```bash
sbatch 02-pilot/tc2/run_b_gpu.sh hunyuan \
  --cases 01-dataset/cases.canonical.csv --phase formal \
  --upstream /absolute/HunyuanVideo-1.5 --commit FULL_UPSTREAM_SHA \
  --assets /absolute/hunyuan-assets --out /absolute/artifacts/hunyuan-formal \
  --pilot-receipt /absolute/artifacts/hunyuan-pilot/pilot_pass.json --max-new-cases 24
```

本次用户明确要求“不用试跑，直接开始正式生成”，使用 `--skip-pilot-reason` 记录原始指示，冻结为 `pilot_validation.status=skipped_by_user`，不声称试跑通过。具体入口为 `02-pilot/tc2/generate_hunyuan_formal.sh`，MGPU-TC2/normal、1节点、8CPU、30G、1GPU、5小时50分钟；按用户要求固定TC2N08，每次提交前查询配额和节点状态。输入为同一正式24例，输出在个人 `ai6132-b/outputs/formal-20261114/`。

可根据已完成案例的实测时间调整每作业 `--max-new-cases`，留出时间余量。CLI读取SLURM结束时间，用对应长度已完成案例的耗时加余量判断是否继续启动；首次估计至少给短片15分钟、长片60分钟，并额外保留10分钟。预算不足时只在案例之间停止，保存 `continuation_JOBID.json`，已完成输出保持不变，后继批处理继续not_started项。此时退出0表示本段无技术失败，仍须等到 `DONE.json` 和独立24例验收通过才算正式批次完成。

重复同一批次命令只启动未尝试项，成功/失败/中断均保留；正式批次出现首个技术失败即停止，剩余项保留not_started，已有失败的批次不继续启动更多案例。退出1表示存在失败或显式限制条数后仍未完成。没有训练checkpoint或自动重提；中断原件和记录保留，先查明原因再安排后续工作。

`batch.json` 冻结输入和协议，`selection.json`/`runs.csv` 从首个worker前就覆盖所有案例；每例保存request、result、原片、完整日志。耗时包含模型加载到官方输出写盘，不含下载及后续解码；另记worker时间和两种PyTorch显存峰值。

本次按用户指示直接使用正式案例验证实际生成，尤其记录513帧的主机内存和显存；CPU模拟与环境检查不作为长片成功证据。模型或资源不支持时保留真实失败记录，不静默改变长度、步数或改用续写。
