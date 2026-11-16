# B：完整时长预处理、匿名交接与 VBench

**四模型作业（2026-11-16）：**来源核验54812通过全部96条原片，确认ABC现有12fps素材与168项已验收VBench原件未变；GPU作业54818只新增D的56项，再按案例／模型／类别核对汇总224项，保存两批独立来源凭据，不伪造一个重新评分批次。D输入由已核验的`runs.accepted.csv`制作完整时长768×432／12fps副本。官方commit、缓存哈希、适配器、环境和指标定义须与ABC一致才能合并。结果入口为`/home/msai/lius0131/ai6132-b/evaluation/abcd-automatic-20261116/`的`vbench-combined-v1/`和`analysis-vbench/`；状态以SLURM及验收凭据为准。A的10步／125-509帧、D的15fps和来源例外保留，横向结果仍标记为诊断评价。

**本轮VBench已完成：**54818为COMPLETED/0:0、4分50秒，224/224项全部成功；逐项结果见`vbench-combined-v1/vbench_results.csv`，合并核验见`MERGED_PROVENANCE.json`，模型与长短片统计见`analysis-vbench/`。独立审计54853的`report-review/verified/VALIDATED.json`为passed。Qwen54820是已失败的历史v2校准；当前32B加速诊断批次的严格门槛仍失败，状态见[Qwen说明](../06-vlm-evaluation/README.md)。

所有命令从仓库根目录经 sbatch 执行。CPU环境需 Python3.10+、Pillow、含libx264的FFmpeg及ffprobe。可设置 `AI6132_FFMPEG`、`AI6132_FFPROBE` 为个人安装的绝对路径。固定实际FFmpeg版本；视频转换为全片768×432/12fps/H.264/yuv420p，CRF18、medium、保持比例补边、移除音轨/来源metadata，禁止只取前5秒。

全组冻结输出后，提供恰好四个不同 `MODEL_ID=RUNS.csv`，model ID必须与对应CSV一致。统一输入字段至少为 case_id/model_id/seed/status/weight_version/config，成功项还需native_video；原始CSV独立保存。其他成员使用不同字段时须显式转换，不自动猜测。缺失交付、未启动、中断均保留；原片有效而转换失败会暂停任务，不伪装成模型失败。

2026-11-15 全组同意 C 改用 `ltxv-13b-0.9.8-distilled`，同一24例重跑后通过C技术验收及B独立接收54638，见[逐条统计与哈希](../02-pilot/tc2/ltx13b-formal-summary.json)。随后54672完成Wan、Hunyuan和LTX13B共72视频、168个适用VBench指标项；旧2B副本与指标保留为历史。当前四模型共96条，沿用原图、prompt、seed和长短分配，D在新批次补齐。

## 当前验证状态

截至2026-11-16 15:29（新加坡时间），当前96条／四模型VBench224项及其独立统计复核均已完成，包含新LTX13B和LongCat；四份人工表已收齐。Qwen全量尚未完成，不影响已验收VBench结果的范围。以下两模型48条／112项属于历史批次，不能代替当前四模型结果。

已完成Hunyuan与历史LTX 2B的诊断预评估，共48条完整时长评分副本，**112/112个适用VBench指标项成功**。GPU作业54533保存全部原始结果后在末尾中文证据导出时失败；CPU恢复54549仅重建导出与统计，模型调用为0，独立复核54550通过原始结果、CSV及统计核对。

上述历史评估使用本地诊断扩展；当前四模型224项也沿用明确冻结的诊断协议。诊断扩展仍未随LongCat修复提交发布；已合并代码的正式流程命令见下文，不能把当前诊断结果改称完整官方排行榜总分。

## 正式流程

```bash
export AI6132_CPU_ENV=/absolute/personal/cpu/environment
sbatch 02-pilot/tc2/run_cpu.sh media --cases 01-dataset/cases.canonical.csv \
  --runs MODEL_A=/absolute/A/runs.csv MODEL_B=/absolute/B/runs.csv \
         MODEL_C=/absolute/C/runs.csv MODEL_D=/absolute/D/runs.csv \
  --outputs-frozen --out /absolute/private/formal-handoff
```

输出新目录：`handoff/`仅含匿名六列CSV、视频和转为无来源metadata PNG的首图；`private/`含随机映射、原始runs、逐条转换审计和统一运行记录。匿名ID包含全部96项，随机种子不打印。`COMPLETE.json`表示整个准备成功，失败目录不得分发。此入口制作自动评估的12fps副本；人工v3另由[四维Excel工具](../05-human-evaluation/README.md)从原片制作原生帧率素材、独立H3编号及D留位。两个输入版本不能混称为同一评分副本，private目录不得随成员包分发。

VBench使用同一全片副本和官方 `custom_input`，七个主体类运行i2v_subject/motion_smoothness/dynamic_degree，四背景类运行i2v_background。视频在私有暂存区使用 `clip_0001-0.mp4` 对应 `clip_0001.png`，避免官方文件名解析漏项。固定干净的上游commit，单独安装其依赖，并在安装/下载批处理中准备DINO、AMT、RAFT和DreamSim等缓存。冻结整个缓存树：

```bash
sbatch 02-pilot/tc2/run_cpu.sh vbench-cache --cache /absolute/vbench-cache \
  --receipt-cache /absolute/private/vbench-cache.json
export AI6132_GPU_ENV=/absolute/personal/vbench/environment
sbatch 02-pilot/tc2/run_b_gpu.sh vbench \
  --manifest /absolute/private/formal-handoff/handoff/anonymous_clips.csv \
  --clip-map /absolute/private/formal-handoff/private/clip_map.csv \
  --upstream /absolute/VBench --commit FULL_UPSTREAM_SHA \
  --cache /absolute/vbench-cache --cache-receipt /absolute/private/vbench-cache.json \
  --out /absolute/private/vbench-formal
```

DreamSim的默认相对 `.cache` 也位于给定缓存根目录内，TORCH_HOME/HF_HOME由入口显式指定；缓存必须预先准备。推理阶段阻止联网和下载子进程，缺失缓存时保留失败并停止。官方依赖中若有不兼容的本地操作，须在共同试跑定位，不能通过关闭离线检查静默下载浮动权重。

保存官方原始JSON及逐视频结果长表，无失败时224项（64主体视频×3＋32背景视频）。失败/缺失留空并保留原因；不计算完整排行榜总分。批次哈希变化拒绝续跑；已有请求不自动重抽。操作故障停止后，其余条目仍导出为not_started。首次使用新环境或缓存时仍须进行实际接口验证。

## 已交付模型的诊断预评估

`media --phase diagnostic` 接受1至4个已交付模型，每模型仍保留同一完整24例；正式入口默认四模型96条约束保持不变。2026-11-15先评价B的Hunyuan和C的历史LTX 2B，共48条、112个适用VBench指标项。以下示例复现该历史批次；新13B必须换用新模型ID及已验收13B清单，并选择独立输出目录重新评分。C原始交接先使用 `run_cpu.sh import-ltx` 完整解码并派生规范编号，保持原件。

```bash
sbatch 02-pilot/tc2/run_cpu.sh media --phase diagnostic --outputs-frozen \
  --cases 01-dataset/cases.canonical.csv \
  --runs HunyuanVideo-1.5-480P-I2V-step-distill=/absolute/B/runs.csv \
         ltxv-2b-0.9.8-distilled=/absolute/C/canonical-intake/runs.csv \
  --out /absolute/private/bc-diagnostic
```

评分副本仍采用完整时长768×432/12fps/H.264；诊断记录单独标记phase，不能作为四模型正式评分包分发。VBench可使用同一批输入并加 `--phase diagnostic`，要求每个模型覆盖全部24例。

评估环境与权重准备见 [TC2说明](../02-pilot/tc2/README.md#隔离评估环境与诊断批次)。使用固定官方VBench、DINO、DreamSim与AMT版本及已校验缓存。适配器将首图所在目录传给官方custom-input接口，并将VBench的DINO路径/分辨率元数据转换为固定本地源码加原始checkpoint加载；不修改指标公式或官方源码。每个指标保留官方原始结果及实际耗时。
