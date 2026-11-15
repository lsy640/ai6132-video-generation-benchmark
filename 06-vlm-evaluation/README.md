# B：Qwen 全片主评分与长片辅助诊断

主协议见PLAN.md。短片整体32帧、长片整体128帧；长片另分四个实际等时窗各32帧。整体三维评分，分段仅首图保持/动作自然度，不把已在别段完成的动作判为提示词不遵循。96次整体＋96次窗口请求；窗口不当成独立案例或平均为主分。

2026-11-15 C 的正式模型改为 `ltxv-13b-0.9.8-distilled`，同一 24 例已重跑并通过 C 的技术验收，见[批次凭据](../02-pilot/tc2/ltx13b-formal-summary.json)。B 验收新原片、生成新的完整时长匿名材料并提交对应个人评分后，为新材料建立独立 Qwen 批次；这些评估尚待执行。已有 2B 的回复与分数保留为历史，不能改标签或复用为 13B 结果；新批次的整体与窗口评分均重新生成。

单独准备个人Qwen环境：支持Qwen3VLForConditionalGeneration的固定Transformers、Torch、qwen-vl-utils（官方当前示例为0.0.14）、Pillow、NumPy、FFmpeg/ffprobe。安装、依赖检查、权重下载和GPU检查均使用独立sbatch；不更改C环境。下载前检查个人配额，固定 Qwen/Qwen3-VL-8B-Instruct 的40位revision，运行时只加载本地safetensors，不自动下载、不默认量化。必须用assets CPU入口记录已固定来源的本地文件，生成根目录assets.json；推理会核对revision和全部文件哈希：

```bash
sbatch 02-pilot/tc2/run_cpu.sh assets --root /absolute/local/Qwen3-VL-8B-Instruct \
  --revision Qwen/Qwen3-VL-8B-Instruct=FULL_MODEL_REVISION_SHA
```

每次请求直接抽取指定帧，按32倍数及像素预算缩放；传入原始帧索引/fps/total_num_frames视频metadata，禁用processor再缩放/抽帧，并核对视觉网格的实际帧数。视频单帧预算384×224、首图768×432；BF16、SDPA、batch1、do_sample=False、max_new_tokens512。最长请求的真实显存必须经共同试跑验证。

B先按C顺序完成个人评分并提交，才运行正式Qwen。`--b-submission`必须是覆盖相同匿名项的完整B评分表；其哈希被封存，分数不会传给模型。映射只供runner确定帧数档位，不传给模型；控制台不输出生成来源。结果目录需继续封存，四份人评交齐后再交D。

```bash
export AI6132_GPU_ENV=/absolute/personal/qwen/environment
mytcinfo
sbatch 02-pilot/tc2/run_b_gpu.sh qwen \
  --manifest /absolute/private/formal-handoff/handoff/anonymous_clips.csv \
  --clip-map /absolute/private/formal-handoff/private/clip_map.csv \
  --model /absolute/local/Qwen3-VL-8B-Instruct --revision FULL_MODEL_REVISION_SHA \
  --b-submission /absolute/submitted/B.csv --out /absolute/private/qwen-formal
```

试跑采用独立输入和独立评分文件，加 `--phase pilot`，不得将试跑分数合并正式结果。正式模式强制24案例×4模型及A的类别、长短分布。

`batch.json`绑定全部输入、权重内容、revision、代码、实际依赖版本和B提交哈希。每个clip/request保存原始request、采样索引/绝对时间、最多两次raw回复及性能信息、最终JSON。非法JSON、缺字段/证据、非整数或分数越界只修复一次；失败为缺失。已尝试但没有result的请求保留incomplete，不自动重抽。输入、配置、代码或权重改变须新批次。

整体 `qwen_ratings.csv` 保留96行、辅助 `qwen_windows.csv` 保留96行；证据和输入网格保留在逐请求JSON，分段的prompt_adherence留空。GPU/流程错误停止并保留尚未启动记录，单个窗口失败不删除已有主分。此工具的CPU假后端测试不代表真实QwenGPU验收。
