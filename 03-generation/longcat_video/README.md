# D：LongCat-Video 生成、环境与交接代码

测量配置为通用 `meituan-longcat/LongCat-Video`：480p、16步 CFG/step 蒸馏 LoRA、BF16、CFG=1、seed=42。原生832×480 / 15fps，18条129帧短片和6条513帧长片，分别8.6秒和34.2秒。没有提示词扩写、续写、拼接或插帧。

2026-11-15完成24/24技术生成。作业54702完成18短片，54703完成剩余5长片及收尾；case_0001复用此前同参数成功试跑54644。完整[生成记录](../../02-pilot/tc2/longcat24-records/README.md)已经合入main；D本地对24条原片完整解码5400帧、输入/配置/哈希核验通过。B独立验收及正式评分待进行；三样本共同试跑未完成、源`formal_evaluation=false`及复用记录均保留，见[证据范围](../../02-pilot/tc2/longcat24-records/README.md#计时与范围)。

## 文件与版本

| 路径 | 用途 |
| --- | --- |
| `frozen/engine.py` | 实际主批次生成引擎：T5、DiT、VAE分进程加载；完整native及历史调试evaluation导出 |
| `frozen/worker.py`、`wave.slurm`、`batch.json` | 实际两段式23条新生成、1条长片导入、状态汇总及打包程序与配置 |
| `worker.py` | 新重建批次的调度程序：绑定绝对图片路径、将耗尽重试的中断记录转为明确失败、通过退出码报告最终失败 |
| `frozen/trial_engine.py`、`trial_config.json` | case_0001成功试跑的原代码和配置，与原始代码SHA证明绑定 |
| `frozen/runtime.json`、`weights-manifest.json`、`environment-freeze.txt` | 官方源码、原始/BF16权重及真实依赖版本凭据 |
| `frozen/prepare_assets.py`、`prepare_assets.slurm` | 资产下载、逐分片BF16转换与安装归档；Slurm副本已包含setuptools依赖修复 |
| `prepare_runtime.slurm` | 新增可配置路径的CPU安装入口，不在CPU作业导入需要CUDA的pipeline |
| `run_case.py`、`run_case.slurm` | 新增单案例回放入口，使用原始engine字节和绝对图片路径 |
| `rebuild_batch.py`、`submit_batch.sh` | 新路径下冻结同一两段式配方并提交；原历史批次保持不变 |
| `import_completed.py`、`intake_cpu.sh` | 对已发布原始批次进行完整接收校验，另导出或验证B兼容的相对路径交接表 |
| `tools_cpu.slurm`、`test_*.py` | CPU测试、批次重建及故障/不可覆盖/交接格式回归 |

主批次engine、worker、wave与batch保留原始字节。历史路径用于审计；跨账号时使用新增入口绑定新路径并冻结新代码哈希，不修改旧证据。新入口已通过CPU结构、路径、哈希及回归测试，本次没有为验证这些入口重新下载权重或重跑GPU生成。

| 内容 | 固定版本 |
| --- | --- |
| 模型权重 | `meituan-longcat/LongCat-Video@03b55529b1d1d4045f5fbe14d65c8c6e8116b278` |
| 官方源码 | `meituan-longcat/LongCat-Video@58eb336edc5c8c63389838727e40b8c18434e54e` |
| PyTorch / CUDA wheel | 2.6.0+cu124 / torchvision0.21.0+cu124 |
| FlashAttention | 2.7.4.post1，Python3.11 / torch2.6 / CXX11 ABI FALSE wheel |
| 其他 | diffusers0.35.1、transformers4.41.0；完整版本见frozen环境清单 |

权重下载时先核对原始LFS SHA256，再按官方BF16推理精度逐分片转换；不是额外低位量化。原始和转换后文件哈希都保留。短片GPU全驻留，长片40/48块CPU卸载，LoRA留在GPU；48层相同RoPE频率共用CPU缓存。不开启torch.compile、稀疏注意力或720p细化。

## TC2安装和单案例回放

所有Python、安装、测试和GPU操作均通过sbatch在计算节点执行。Head只编辑文件、Git和提交/查询作业。以下为新运行示例；不覆盖已完成批次。提交前先用`mytcinfo`检查QoS、显存和home剩余容量。实际测量使用TC2N04的一块L40S、4CPU、28GB主机内存。

从仓库根目录设置个人路径，准备Linux Python3.11及uv；A的原图放在规范CSV所引用的位置：

```bash
export AI6132_CODE="$PWD"
export LONGCAT_ROOT="$HOME/ai6132-longcat"
export LONGCAT_BOOTSTRAP_PYTHON=/absolute/path/to/python3.11
export LONGCAT_UV=/absolute/path/to/uv
mytcinfo
sbatch 03-generation/longcat_video/prepare_runtime.slurm
```

用`sacct -j JOB_ID --format=JobID,State,ExitCode`确认成功后再提交GPU案例。环境和模型已验证完整时可直接复用，不必重复安装。

```bash
sbatch 03-generation/longcat_video/run_case.slurm \
  --cases "$AI6132_CODE/01-dataset/cases.canonical.csv" \
  --case-id case_0001 --out "$LONGCAT_ROOT/replay-new/case_0001"
```

帧数、原始prompt与seed来自规范清单；只允许新输出目录。wrapper把图片解析成绝对路径，因此不依赖历史engine中的Cog项目目录前缀。实际生成仍调用未改动的frozen engine。`run.json`、`steps.jsonl`、每阶段日志和原生全片校验均保留。

## 同配方全24组重建

原测量使用先完成的case_0001长片，再运行18短＋5长；新批次也要求提供一条同参已完成case_0001，防止把6条长片塞进一个6小时作业。首次准备可用上面的单案例入口生成该长片。其native、evaluation与配置哈希必须有效；不能用短片替代。

CPU环境需Python3.11及Pillow；PyAV模式和测试另安装`intake-requirements.txt`（TC2安装也须通过CPU作业）。以下的`AI6132_CPU_PYTHON`应指向已准备好的个人CPU环境：

```bash
export AI6132_CPU_PYTHON=/absolute/path/to/cpu-env/bin/python
sbatch 03-generation/longcat_video/tools_cpu.slurm rebuild-batch \
  --cases "$AI6132_CODE/01-dataset/cases.canonical.csv" \
  --runtime "$LONGCAT_ROOT" \
  --reuse-case1 "$LONGCAT_ROOT/replay-new/case_0001" \
  --out "$LONGCAT_ROOT/batches/replay-shared24-new"
```

CPU准备成功后，在Head用Shell一次提交两段依赖任务：

```bash
bash 03-generation/longcat_video/submit_batch.sh \
  "$LONGCAT_ROOT/batches/replay-shared24-new"
```

新批次复制当前`worker.py`并绑定Slurm运行根目录，重新记录完整代码哈希，历史engine字节不变。两作业先held提交，写入job IDs后release；finish依赖afterany，先接收未完成短片再运行5条剩余长片，最终输出24行状态、完整校验与tar.gz。最多2个submitted作业、1块GPU，每段最多6小时。关闭本机、VPN、Codex不影响已提交作业；进度查询和本地下载需重连。

成功样本不会重跑。历史worker允许同seed的资源错误切换CPU卸载后最多一次技术恢复，保留每次尝试；实际已交批次所有完成尝试均为1次。流程故障/超时仍可能造成失败，记录真实状态，只有24条均成功才标记all_succeeded。

新worker先恢复已验证成功的attempt，再检查剩余次数；两次尝试均未成功时，将遗留的`running/interrupted`记录转为`failed`，保留原状态、作业ID、错误和全部attempt。最终收尾将其他非终态也明确记为失败，照常保留24行记录并打包；存在失败时finish作业退出码为1，全成功为0。历史`frozen/worker.py`与已提交批次保持不变；要使用修复，应在新目录重新运行重建入口，不能替换旧批次中受哈希绑定的代码。

## B接收与字段适配

正常交接使用已合并的记录目录，加单独提供的native视频。完整原始结果包需要额外审计时，下面的适配器严格核对本次已发布冻结批次、源码、原始/转换权重凭据、canonical图片、逐条请求/结果/CSV、16步记录和视频字节；不会访问旧账号的远端绝对路径。

源`runs.csv`还必须逐字节匹配仓库`longcat24-records/verification.json`中的`source_runs_sha256`，然后才解码原片。该摘要来自已发布记录，不接受接收包自报的替代值；同时改写视频、run.json和包内CSV也会被拒绝。应保留完整仓库的记录目录；缺少已发布锚点时验收失败。

```bash
export AI6132_CPU_PYTHON=/absolute/path/to/cpu-env/bin/python
sbatch 03-generation/longcat_video/intake_cpu.sh export \
  --raw /received/longcat16_full24_20261115 \
  --cases "$AI6132_CODE/01-dataset/cases.canonical.csv" \
  --out /received/new-longcat-delivery
```

默认用共享FFmpeg/ffprobe完整解码；可用`--decoder pyav`切换已验证的PyAV12后端。`verify`替代`export`时核验现有delivery，并可用`--verification-out /new/receipt.json`保存新凭据。完整原始包与delivery一起搬移后仍可验证：`native/`和`configs/`均是相对路径，原片不重编码。

模型ID映射为`longcat-video-480p-16step-distill`，原`status=succeeded`另导出为共享`success`；保留source_status/origin/source_formal_evaluation，补齐weight_version/config/image_sha256/GPU/显存字段。预留显存没有实测，留空。该适配器绑定历史发布批次；新重建批次应建立自己的来源凭据后按组内流程验收，不能改写旧哈希冒充历史产物。

B的四模型`--runs`参数中，对应项为`longcat-video-480p-16step-distill=/received/new-longcat-delivery/runs.csv`。正式评分副本由B从native经共享media流程重新制作，保持比例补边、全时长、768×432/12fps并移除来源metadata；历史engine的直接scale evaluation仅是调试副本。B独立接收和正式质量评分的结果应另行记录。

## 验证范围

```bash
sbatch 03-generation/longcat_video/tools_cpu.slurm test
```

当前包含25项CPU回归，覆盖接收/来源/协同篡改/缺失锚点/真实PyAV解码及四模型media格式集成、临时目录独立于checkout深度、路径回放/版本/不可覆盖/冻结源码/批次重建，以及当前worker的成功不重抽、资源恢复、重试耗尽终态、末次成功恢复、最终退出码和24行归档。测试临时文件使用系统临时目录，可在提交前通过`TMPDIR`指定个人目录。新wrapper和worker的CPU验证不代表重跑GPU测量；历史24条生成证据保持独立。

原24条实测：短片均值447.359秒，长片均值3747.219秒；逐条elapsed总和30535.774秒，含加载、生成、导出与D解码检查。批量23条作业总墙钟约7小时27分钟，另有此前1条长片。详见已合并的[记录与计时说明](../../02-pilot/tc2/longcat24-records/README.md)。Cog20条暂停产物单独保留，不作为本模型完成数。

官方依据：[源码revision](https://github.com/meituan-longcat/LongCat-Video/tree/58eb336edc5c8c63389838727e40b8c18434e54e)、[模型revision](https://huggingface.co/meituan-longcat/LongCat-Video/tree/03b55529b1d1d4045f5fbe14d65c8c6e8116b278)。
