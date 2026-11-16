# TC2 环境与作业入口

本目录包含在 NTU TC2 集群上准备环境、提交生成和评估作业的脚本。

## 使用约定

- Head Node 只用于编辑文件、Git 和提交作业。Python、安装和 GPU 工作全部通过 `sbatch` 在计算节点执行。
- 每次提交前运行 `mytcinfo` 和 `squeue -u "$USER"` 核对配额。本项目使用的 `normal` QoS 上限为 10 CPU、30G 内存、1 GPU、2 个作业、6 小时。
- 环境、权重和输出放在个人目录，通过环境变量传给脚本。脚本开头的资源申请是默认值，可用 `sbatch` 参数覆盖。
- 作业结束后用 `sacct -j <JOB_ID> --format=JobID,State,Elapsed,ExitCode` 确认状态，再进行下一步。

## 脚本

| 脚本 | 用途 |
| --- | --- |
| `run_cpu.sh` | CPU 工具的统一入口，子命令见下表 |
| `run_b_gpu.sh` | GPU 入口：`hunyuan`（生成）、`vbench`、`qwen`（评估） |
| `bootstrap_env.sh`、`install_ltx.sh`、`bootstrap_cu126.sh` | LTX-Video 的基础环境、代码与权重、生成用的 cu126 环境 |
| `run_ltx.sh` | LTX-Video 生成 |
| `bootstrap_hunyuan.sh`、`check_hunyuan_env.sh` | 建立并检查 HunyuanVideo 环境 |
| `preflight_hunyuan.sh`、`download_hunyuan.sh`、`finalize_hunyuan_assets.sh` | HunyuanVideo 的输入与下载权限检查、权重下载、权重核对 |
| `generate_hunyuan_formal.sh`、`verify_hunyuan_formal.sh` | HunyuanVideo 生成 24 条视频，完整解码并核对输出 |
| `bootstrap_evaluation.sh`、`download_evaluation.sh` | 建立评估环境，下载并核对 VBench 和 Qwen 的权重 |

`run_cpu.sh` 的子命令：

| 子命令 | 作用 |
| --- | --- |
| `dataset` | 生成并校验规范案例清单 |
| `validate-cases` | 按生成入口的规则检查清单 |
| `import-ltx` | 接收并完整解码 LTX-Video 的交付 |
| `media` | 制作完整时长的评分副本和匿名清单 |
| `assets`、`vbench-cache` | 为已下载的权重或缓存记录 revision 和 SHA256 |
| `human-xlsx` | 人工评分包的制作、核验与收表 |
| `human-eval` | 早期的三维 CSV 评分工具；VBench 入口复用其中读取匿名清单的函数 |

各模型的生成步骤见 [Wan 2.2](../../03-generation/Wan2.2/BENCHMARK.md)、[HunyuanVideo](../../03-generation/hunyuan_video/README.md)、[LTX-Video](../../03-generation/ltx_video/README.md)、[LongCat-Video](../../03-generation/longcat_video/README.md)，评估步骤见 [04](../../04-vbench-evaluation/README.md)、[05](../../05-human-evaluation/README.md)、[06](../../06-vlm-evaluation/README.md)。Wan 2.2 和 LongCat-Video 由成员在各自的环境中生成，不经过本目录的脚本。

## 生成记录

| 文件 | 内容 |
| --- | --- |
| `ltx13b-formal-summary.json` | LTX-Video 13B 逐视频的哈希、耗时和显存 |
| `longcat24-records/` | LongCat-Video 24 条视频的配置、日志和计时；接收校验依赖其中的 `verification.json`，目录需完整保留 |

## 注意事项

- Decord 0.6.0 的 wheel 带有旧的平台标签，`pip check` 会给出一条元数据告警，评估环境只放行这一条。
- 默认的 ASCII 环境读取含中文的 JSON 会失败，CPU 和 GPU 入口都已设置 `PYTHONUTF8=1`。
- `--mem` 是主机内存，与 GPU 显存无关。HunyuanVideo 和 LongCat-Video 的长片依赖 CPU 卸载，主机内存需要申请到 28–30G。
