# AI6132 Video Generation Benchmark

AI6132 Project 1 · Technical Review

在相同首图、英文提示词和种子下，比较四个图生视频模型的首图保持、动作自然度、提示词遵循、背景细节保持与推理成本。项目结合 VBench-I2V 自动指标、四人独立人工评价和 Qwen3-VL 评审，记录生成成功率、耗时及显存，并分析评审之间的一致性。

[实验计划与评分量表](PLAN.md) · [数据说明](01-dataset/README.md) · [环境与运行说明](02-pilot/tc2/README.md) · [验收记录](02-pilot/tc2/VALIDATION.md)

发布范围：LongCat修复代码已随`d9987f3`发布；本文及阶段说明也记录个人工作区的进展。新版人评网页／Excel、Qwen v2/v3.5和诊断报告实现及其新增作业入口仍未发布，对应命令需使用个人工作区的冻结版本；模型、视频、原始评分及私有映射另行保存。

截至 **2026-11-16 15:29（新加坡时间）**，四模型96条原片已接收，VBench全部224个适用项完成；四份人工表已通过只读校验，共384条记录、1536项数值评分。32B Qwen加速版已完成70请求开发检查，技术准入通过，原严格质量门槛仍失败；诊断全量前两个分片已处理24条，第三分片运行中，尚无完整96条Qwen比较结果。LongCat三项审查问题已修复并以`d9987f3`推送main。A的步数／帧数偏差、D试跑及来源例外继续保留，详见[验证记录](02-pilot/tc2/VALIDATION.md)。

## 实验设计

共享输入为 A 提供的 **24 个固定案例，覆盖 11 类**，包含 16 个主体案例与 8 个背景案例。四个模型使用相同输入，每案例每模型生成一次，共 **96 条正式视频记录**；每名评审独立评价全部记录，四人共提交 **384 条评分记录**。

| 长度组 | 原始帧数标注 | 原生目标帧数 | 每模型视频数 | 四模型合计 |
| --- | ---: | ---: | ---: | ---: |
| 短片 | 128 | 129 | 18 | 72 |
| 长片 | 512 | 513 | 6 | 24 |
| 合计 | — | — | **24** | **96** |

129/513 帧为当前统一的原生生成目标。各模型记录实际帧率和完整时长；以 24 fps 为例，两档视频约为 5.38 秒和 21.38 秒。长片采用单次连续生成。

生成器统一读取 [`cases.canonical.csv`](01-dataset/cases.canonical.csv)：

```text
case_id,category,image_path,prompt,seed,source_frame_num,target_frames
```

案例编号为 `case_0001` 至 `case_0024`，种子均为 42。原始清单、图片、提示词、顺序和长短分配保留，规范清单及图片由 SHA256 凭据绑定。该数据是固定子集，抽样过程未提供，类别级结果仅作描述性分析。

## 模型与成员分工

| 成员 | 生成模型 | 公共职责 |
| --- | --- | --- |
| A | Wan2.2-TI2V-5B | 准备并冻结公共数据与输入清单 |
| B | HunyuanVideo-1.5-480P-I2V-step-distill | 汇总生成记录、制作匿名评分副本、运行 VBench-I2V 与 Qwen3-VL |
| C | LTX-Video 13B v0.9.8 Distilled | 准备评分材料、随机观看顺序与收表校验 |
| D | LongCat-Video 480p、16步CFG-step蒸馏；CogVideoX仅保留为历史备选 | 整合结果、计算一致性并完成报告 |

实际第四模型为LongCat，832×480／15fps，18条129帧与6条513帧；三个独立共同试跑未完成。A实际交付为10步、125/509帧，与计划50步、129/513帧存在偏差。各模型配置及例外见 [PLAN.md](PLAN.md)；所有成员均参与四模型的人工评审。

2026-11-15 全组同意将 C 的 LTX 从 2B 改为 13B，使用 BF16、官方两阶段 7+3 timesteps、第二阶段 tone map compression ratio 0.6，关闭提示词扩写。全部 24 例已在新批次重跑，公共输入、长短分配及 768×432 / 24 fps 不变。旧 2B 批次与 PR #2 结果保留为历史；新 13B 使用独立评分材料与评估批次。

## 评估方法

VBench已汇总四模型224项并通过独立复核。人工四维评分已收齐并计算一致性。Qwen当前采用[32B v3.5加速诊断协议](06-vlm-evaluation/QWEN_V3.md)，原片全片观察及固定连续片段共5个视图；历史v1的8B整体／长片四窗输出和v2失败试验独立保留。

Qwen开发检查54992已完成70/70有效视图，但仍漏检4个乱序对照及手部缺陷样例，只按用户授权进入诊断全量。54993、55005各处理12条；第二片的`clip_0020`保留`missing_view`，不能把作业正常退出等同于全部评分有效。55013处理第25—36条，后续分片与最终报告按现有有限串行流程生成；新旧分数不混算，暂不据此排名。

- **评分素材：**保留原片，制作完整时长的768×432、H.264副本并完整解码验收。VBench及历史Qwen v1使用12fps；人工v3保留原生帧，ABC为24fps、D为15fps。新版Qwen从原片采样，明确区分输入版本。
- **VBench-I2V：**主体案例评估 `i2v_subject`、`motion_smoothness`、`dynamic_degree`；背景案例评估 `i2v_background`，保留逐视频结果。
- **人工评价：**每人有独立随机顺序的离线网页与Excel，四项评分为首图／主体外观保持、动作自然度、提示词遵循、背景细节保持。当前ABCD共96条可评，网页仅要求四项评分，问题类型和背景NA原因均为选填，已移除问题时间段与评审声明。支持导入旧版Excel／进度、草稿恢复、JSON备份与导出Excel；全部评分完成即可最终导出。报告继续披露已见分数、讨论缺陷及分阶段交付的匿名性局限。
- **Qwen3-VL：**当前为Qwen3-VL-32B-Instruct NF4诊断批次，保留原采样像素、四维锚点及逐视图回复；96×5＝480个全量视图终结并复核后才形成全量报告。技术准入不代表严格质量门槛或独立人评校准通过。
- **结果分析：**比较生成成功率、质量分布和推理成本，报告人工评审一致性及人评与 Qwen 的相关性；整体、短片、长片分别统计。

失败、缺失和中断均保留。原正式流程要求B先提交个人评分；本轮在人评前运行诊断Qwen的用户授权和已见分数影响继续披露。四份人工表现已收齐，Qwen全量及最终报告仍待完成。详细方法见 [完整协议](PLAN.md)。

2026-11-15 按明确指示，对 Hunyuan 与历史 LTX 2B 提前运行了诊断性 Qwen，并记录了人评顺序例外。该批次共 48 条视频，独立保存；后续人评须披露已见模型评分的影响。正式四模型流程仍按上述顺序执行。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| [`01-dataset/`](01-dataset/README.md) | 原始清单、规范清单与输入校验凭据 |
| [`02-pilot/tc2/`](02-pilot/tc2/README.md) | 环境准备、SLURM 入口与验收记录 |
| [`03-generation/`](03-generation/) | Hunyuan、LTX和LongCat生成、回放与交接工具 |
| [`ai6132_benchmark/`](ai6132_benchmark/) | 共享数据校验、Hunyuan 生成、视频处理及评估实现 |
| [`04-vbench-evaluation/`](04-vbench-evaluation/README.md) | 全片预处理、匿名交接与 VBench 使用说明 |
| [`05-human-evaluation/`](05-human-evaluation/README.md) | 人工盲评材料、评分模板与收表工具 |
| [`06-vlm-evaluation/`](06-vlm-evaluation/README.md) | 当前32B诊断协议、校准限制及历史Qwen流程 |
| [`07-analysis/`](07-analysis/) | 结果整合与图表目录，待填充 |
| [`08-report/`](08-report/) | 课程报告与成员贡献说明目录，待填充 |
| [`tests/`](tests/) | 共享工具与评估流程回归测试 |

## 快速开始

### 1. 获取代码与输入

```bash
git clone https://github.com/lsy640/ai6132-video-generation-benchmark.git
cd ai6132-video-generation-benchmark
```

图片与模型权重另行交接或下载。将 A 冻结的原始图片放入 `01-dataset/dataset/`，保留清单中的文件名；输入清单及校验凭据已随仓库提供。

### 2. 配置环境并校验数据

CPU 工具使用 Python 3.10+、Pillow，以及支持 libx264 的 FFmpeg/ffprobe；新版人工Excel工具另需openpyxl3.1.5。不同模型与评估阶段使用各自的个人 Conda 环境，安装步骤见 [TC2 说明](02-pilot/tc2/README.md) 和各模型文档。

在 NTU TC2 上，Head Node 用于编辑、Git 与提交/查询作业；所有 Python、测试、安装构建和 GPU 工作通过 `sbatch` 在计算节点执行。每次提交前查询当前配额与队列，按实际资源调整作业参数。

```bash
export AI6132_CODE="$PWD"
export AI6132_CPU_ENV=/absolute/path/to/personal/conda/environment
mytcinfo
squeue -u "$USER"
sbatch 02-pilot/tc2/run_cpu.sh dataset --verify 01-dataset/cases.canonical.csv
```

保留返回的作业 ID，用 `sacct -j <JOB_ID> --format=JobID,State,ExitCode` 确认成功后再进行下一步。默认 CPU 日志写入提交目录的 `ai6132_cpu_<JOB_ID>.out` 与 `.err`。各入口的资源、环境变量及完整参数见对应文档。

### 3. 生成与评估

按下列顺序执行，各阶段使用独立输出目录并冻结输入、配置和版本：

1. [Hunyuan 生成](03-generation/hunyuan_video/README.md)、[LTX 13B 生成与批次交接](03-generation/ltx_video/README.md)或 [LongCat 生成与交接](03-generation/longcat_video/README.md)。各成员交付同一规范的运行记录。
2. 汇总四模型输出，进行[全片验收、匿名副本制作与 VBench 评估](04-vbench-evaluation/README.md)。
3. 按[人工评价流程](05-human-evaluation/README.md)分发每人独立的四维网页／Excel和96条匿名素材，D已补齐。一般流程B先交卷再运行[Qwen评估](06-vlm-evaluation/README.md)；本轮已授权提前运行诊断Qwen，该顺序变更保留在记录和报告中。
4. 四份人评收齐后解盲，按 [PLAN.md](PLAN.md)整合结果与报告。

常规共同试跑使用独立的三个长短案例。本次 B 正式生成按明确指示跳过生成试跑，该例外已写入冻结记录；后续阶段仍执行各自的验收要求。

C 的13B修订使用case_0001的129/513帧技术试跑后，按全组决定完成24例重跑；未另行完成三个独立共同试跑样本。B已验收新原片并将`ltxv-13b-0.9.8-distilled`纳入ABC诊断批次。历史2B素材和分数单独保留；本轮使用四维人工v3重新独立评分。

### 4. 运行测试

在 CPU 环境与配额满足要求、前一作业完成后提交：

```bash
sbatch 02-pilot/tc2/run_cpu.sh test
```

该入口覆盖LTX、人评及共享工具；LongCat测试单独提交，环境另需PyAV，提交前再次核对配额：

```bash
export AI6132_CPU_PYTHON=/absolute/path/to/personal/conda/environment/bin/python
mytcinfo
sbatch 03-generation/longcat_video/tools_cpu.slurm test
```

已发布`d9987f3`候选的基础64项加LongCat25项，共89项通过；保留本地Qwen扩展的关联整合另通过68项，二者有重叠，不作为157项不同测试。详见[验证范围](02-pilot/tc2/VALIDATION.md#2026-11-16longcat审查修复与发布)。

## 当前进度

状态快照：**2026-11-16 15:29（新加坡时间）**。运行中作业的后续状态以SLURM和逐请求凭据为准。

| 阶段 | 状态与证据范围 |
| --- | --- |
| 公共输入 | 24 例、11 类规范清单及图片校验已完成 |
| Hunyuan 生成 | 作业 54414 完成 24/24；独立验收 54435 通过全片解码、冻结输入、配置与哈希核验，原片为 848×480 / 24 fps，见 [B 生成说明](03-generation/hunyuan_video/README.md#当前结果) |
| LTX 13B 生成 | 作业 54620 完成 24/24，独立接收 54638 通过完整解码及冻结记录核验；原片为 768×432 / 24 fps，见 [C 生成说明](03-generation/ltx_video/README.md)及[逐条统计](02-pilot/tc2/ltx13b-formal-summary.json) |
| LTX 2B 历史 | 24/24 原片通过独立验收；既有素材和评分保留在历史批次 |
| 历史诊断评估 | Hunyuan＋LTX 2B 共 48 条视频，112/112 个 VBench 指标项、48/48 次 Qwen 整体及 48/48 次窗口评分成功；CPU 恢复导出与独立统计复核均通过，见 [VBench 结果范围](04-vbench-evaluation/README.md#当前验证状态)及 [Qwen 结果范围](06-vlm-evaluation/README.md#历史v1验证状态) |
| ABC诊断评估 | 72条全片副本、168个VBench指标项、72次Qwen整体及72次长片窗口评分完成，作业54672及独立统计核验通过；Qwen动作分随后未通过敏感性对照，保留为历史输出 |
| 工具验证 | LongCat修复55016为COMPLETED/0:0，发布候选89项、关联整合68项通过，310项历史记录校验通过；95项人评／浏览器历史验证54791另保留，不混计版本 |
| A、D 来源 | A日志支持10步及125/509帧，生成时源码／权重哈希不足；D原片通过54790，补充记录通过54800：24/24原片／首图哈希、prompt、seed42、配置及16步日志一致；记录配置为LongCat 480p/16steps/CFG1/BF16，实际权重／执行源码未在接收端重算哈希，试跑例外及原formal_evaluation=false保留 |
| D LongCat 代码与记录 | PR #4、#5已同步；临时目录、已发布摘要绑定及重试终态／退出码修复已通过`d9987f3`推送。新重建批次使用修复worker，历史冻结证据不改写 |
| 四模型VBench | 54818已完成224/224项，独立复核54853通过；包含已验收ABC168项与D新增56项 |
| 人工评价 | 四份96行表全部有效，共384条、1536项数值；四维序数α为0.656、0.573、0.582、0.538。原始表及映射保留，见[收表状态](05-human-evaluation/README.md#当前收表与分析状态) |
| 当前Qwen与报告 | 54992技术准入通过、原严格门槛失败；54993／55005已处理24条，其中1条missing_view；55013运行中。尚未完成96条诊断全量或最终报告 |

当前尚无四模型完整质量比较结果。成员报告、工具回归与实际视频验收分别保留证据，生成及评估的详细状态以各阶段验收记录为准。

C 的交接文件名为 `C_LTX13B_shared24_handoff_20261115.zip`，包含可搬移的 `delivery/runs.csv` 与原片、完整原始记录、代码及验收证据。B已完成新13B原片接收核验；本轮四模型人工表已收齐，最终整合仍等待当前Qwen批次终结和复核。

LongCat审查修复已发布；新版Excel、Qwen及诊断报告扩展仍为本地未提交工作，命令与测试数量按版本区分。历史v1存在天花板和时序敏感性失败；当前32B加速版仍漏检乱序及手部缺陷，诊断全量准入不等于有效性通过。

## 复现与产物管理

仓库保存源代码、配置、共享清单与文档。原始图片、模型权重、生成视频、原始评分和匿名映射不纳入版本控制，按需单独存储和交接。各阶段记录固定上游版本、依赖、输入及文件哈希、生成参数和真实运行状态；成功、失败与中断原件均保留。

本项目使用 VBench-I2V 固定子集及统一预处理，结果用于课程实验比较，不作为完整官方排行榜总分。完整实验约定、评分锚点、统计方法及课程交付要求见 [PLAN.md](PLAN.md)。
