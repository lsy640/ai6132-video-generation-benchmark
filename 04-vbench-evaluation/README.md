# B：完整时长预处理、匿名交接与 VBench

所有命令从仓库根目录经 sbatch 执行。CPU环境需 Python3.10+、Pillow、含libx264的FFmpeg及ffprobe。可设置 `AI6132_FFMPEG`、`AI6132_FFPROBE` 为个人安装的绝对路径。固定实际FFmpeg版本；视频转换为全片768×432/12fps/H.264/yuv420p，CRF18、medium、保持比例补边、移除音轨/来源metadata，禁止只取前5秒。

全组冻结输出后，提供恰好四个不同 `MODEL_ID=RUNS.csv`，model ID必须与对应CSV一致。统一输入字段至少为 case_id/model_id/seed/status/weight_version/config，成功项还需native_video；原始CSV独立保存。其他成员使用不同字段时须显式转换，不自动猜测。缺失交付、未启动、中断均保留；原片有效而转换失败会暂停任务，不伪装成模型失败。

2026-11-15 全组同意 C 改用 `ltxv-13b-0.9.8-distilled`，同一 24 例已重跑并通过 C 的完整技术验收，见[逐条统计与哈希](../02-pilot/tc2/ltx13b-formal-summary.json)。B 尚待接收；复核后以交接包的 `delivery/runs.csv` 为 C 模型输入，在新目录建立全片副本、匿名映射和 VBench 批次。若已有 2B 副本或指标，保留为历史，不改模型标签或继承旧分数。其余三模型保持原输入，仍为四模型共 96 条，原图、prompt、seed、长短分配不变。

```bash
export AI6132_CPU_ENV=/absolute/personal/cpu/environment
sbatch 02-pilot/tc2/run_cpu.sh media --cases 01-dataset/cases.canonical.csv \
  --runs MODEL_A=/absolute/A/runs.csv MODEL_B=/absolute/B/runs.csv \
         MODEL_C=/absolute/C/runs.csv MODEL_D=/absolute/D/runs.csv \
  --outputs-frozen --out /absolute/private/formal-handoff
```

输出新目录：`handoff/`仅含匿名六列CSV、视频和转为无来源metadata PNG的首图；`private/`含随机映射、原始runs、逐条转换审计和统一运行记录。匿名ID包含全部96项，随机种子不打印。`COMPLETE.json` 表示整个准备成功；失败目录不得分发，用新目录保留失败证据后重做。只将 `handoff/` 交给C，按人评README生成并校验评分包；不要将private目录一起分发。

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

保存官方原始JSON及逐视频结果长表，无失败时224项（64主体视频×3＋32背景视频）。失败/缺失留空并保留原因；不计算完整排行榜总分。批次哈希变化拒绝续跑；已有请求不自动重抽。操作故障停止后，其余条目仍导出为not_started。GPU缓存和实际接口仍需共同试跑。
