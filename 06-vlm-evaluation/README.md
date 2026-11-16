# B：当前Qwen诊断评估与历史评分流程

**当前进展（2026-11-16 15:29，新加坡时间）：**VBench224项已完成。Qwen v3.5的54992已COMPLETED/0:0，70/70开发视图有效，技术准入通过；原严格门槛仍失败，保留4个乱序及手部缺陷漏检。54993、55005各处理12条，共23条聚合success、1条missing_view（clip_0020）；55013处理第25—36条，仍RUNNING。总目标96×5＝480个全量视图尚未终结，当前无完整诊断报告，详见[状态快照](QWEN_V3.md#当前状态快照)。

当前为Qwen3-VL-32B-Instruct NF4、GPU预处理／原帧缓存及简洁输出的独立批次；54989匹配请求快2.30倍，慢版54974／54981已取消且原件保留。技术准入允许在用户授权下采集诊断输出，不表示独立人评或质量门槛通过。四份人评共384条已收齐；开发样例已见人工结果，不能当作留出校准集。新旧自动分数分开，原Excel和失败回复保留，人工分数不输入模型。下文三项评分、长片四窗及512token上限均为历史v1规格。

## 2026-11-15：评分器有效性复核

三模型72条全片评分中，动作自然度65条为5分；长片窗口64/72为5分。后续时序对照作业54708对三模型的跑步、切鱼案例分别测试原片、逐帧冻结和完全乱序版本，18次动作评分全部为5，说明当前v1评分未通过该敏感性检查。已有原始回复、分数与统计保留，暂不据此声称动作质量排名。执行和统计验收通过，不等同于与人评一致。

用户报告LTX13B case5手部动作模糊、Hunyuan case4人物外观中段改变，以及Wan/Hunyuan后程背景虚化。原片与评分副本对照已导出，原片中可见的退化并非仅由转码造成。v1只有整体首图保持、动作自然度和提示词遵循，未单列背景细节保持。

历史v2试验先记录带实际时间的观察，分开检查主体外观、动作局部细节和背景纹理，增加原生连续帧及高分辨率输入，并试验background_detail_retention。原有效性门槛要求先通过时序对照及独立人工校准；该门槛未通过，原结论不改写。随后人工评价升级为[四维Excel v3](../05-human-evaluation/README.md)；当前v3.5全量仅基于后续明确的诊断技术准入授权，不作为正式已校准人机对照。

首轮试验54710已完成16请求，15条结构校验通过、1条无观察证据而保留invalid_reply。真实输入张量与视频token核验正常，结构化提问正确识别两个冻结对照，并指出Wan case1背景细节丢失；对乱序、Hunyuan case4外观改变和LTX case5手部细节仍识别不足，尚未达到全量新评分的有效性门槛。

主协议见PLAN.md。短片整体32帧、长片整体128帧；长片另分四个实际等时窗各32帧。整体三维评分，分段仅首图保持/动作自然度，不把已在别段完成的动作判为提示词不遵循。96次整体＋96次窗口请求；窗口不当成独立案例或平均为主分。

2026-11-15 C的正式模型改为`ltxv-13b-0.9.8-distilled`，同一24例重跑后通过C技术验收及B独立接收54638，见[批次凭据](../02-pilot/tc2/ltx13b-formal-summary.json)。随后已在独立ABC诊断批次完成新13B及Wan/Hunyuan共72条的整体与窗口评分，作业54672；结果保留上述有效性局限。旧2B回复、分数与匿名映射继续保留。

## 历史v1验证状态

本地诊断扩展已对Hunyuan与历史LTX 2B各24条视频完成 **48/48次整体评分、48/48次长片窗口评分**。按2026-11-15明确指示在人评前运行，原始指示和顺序例外随冻结批次保存；正式四模型流程仍要求B先提交个人评分。诊断扩展尚未随本次文档更新发布，下文命令使用已合并的正式流程。

GPU作业54533的96次请求全部成功，末尾导出因ASCII环境读取中文证据失败；CPU作业54549启用UTF-8恢复导出，模型调用为0。独立复核54550检查原始回复、采样、视觉网格及统计通过。128帧长片视觉输入已通过真实显存验证，评分请求最大allocated为18529.46 MiB。

两模型整体三维评分的中位数及Q1/Q3均为5，存在天花板现象；均分差仅作描述性观察，窗口评分独立保存。当前人评一致性已计算，四维α见[人评收表状态](../05-human-evaluation/README.md#当前收表与分析状态)；当前Qwen全量及其与人评的最终诊断对照仍待完成。

## 历史v1正式入口

单独准备个人Qwen环境：支持Qwen3VLForConditionalGeneration的固定Transformers、Torch、qwen-vl-utils（官方当前示例为0.0.14）、Pillow、NumPy、FFmpeg/ffprobe。安装、依赖检查、权重下载和GPU检查均使用独立sbatch；不更改C环境。下载前检查个人配额，固定 Qwen/Qwen3-VL-8B-Instruct 的40位revision，运行时只加载本地safetensors，不自动下载、不默认量化。必须用assets CPU入口记录已固定来源的本地文件，生成根目录assets.json；推理会核对revision和全部文件哈希：

```bash
sbatch 02-pilot/tc2/run_cpu.sh assets --root /absolute/local/Qwen3-VL-8B-Instruct \
  --revision Qwen/Qwen3-VL-8B-Instruct=FULL_MODEL_REVISION_SHA
```

每次请求直接抽取指定帧，按32倍数及像素预算缩放；传入原始帧索引/fps/total_num_frames视频metadata，禁用processor再缩放/抽帧，并核对视觉网格的实际帧数。视频单帧预算384×224、首图768×432；BF16、SDPA、batch1、do_sample=False、max_new_tokens512。首次使用新环境或配置时，须重新验证最长请求的真实显存。

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

## 人评顺序例外：诊断批次

2026-11-15，用户明确要求“同时运行诊断性Qwen，并记录人评顺序变更”，随后将C的历史LTX 2B结果一并纳入。该例外仅用于独立 `diagnostic` 批次：B、C各完整24例，共48个整体请求、48个长片窗口请求；输入预算、量表、一次格式修复和原始回复保留规则与上文一致。已完成评分不代表新13B结果；替换视频后须另建批次重新评分。

该模式不需要伪造或补填B的评分表，必须提供真实用户指示 `--diagnostic-reason`；冻结记录保存 `human_evaluation_order.status=waived_for_diagnostic_by_user`、原始指示和后续人评可能受影响的局限。正式/pilot历史工具仍要求B先提交相同匿名批次的三维CSV，不直接接收新v3 Excel；不能伪造旧批次收表凭据来解锁。诊断结果不并入四模型正式结果。新v3采用独立H3编号及原生帧率视频，人工与历史Qwen仅可由私有case/model和源视频哈希关联前三维，并披露输入版本差异。

```bash
sbatch 02-pilot/tc2/run_b_gpu.sh qwen --phase diagnostic \
  --manifest /absolute/private/bc-diagnostic/handoff/anonymous_clips.csv \
  --clip-map /absolute/private/bc-diagnostic/private/clip_map.csv \
  --model /absolute/local/Qwen3-VL-8B-Instruct --revision FULL_MODEL_REVISION_SHA \
  --diagnostic-reason '真实用户指示原文及日期' \
  --out /absolute/private/bc-diagnostic/qwen-diagnostic
```

`evaluate_diagnostic.sh` 先以长片验证VBench四项接口，再运行全部适用指标；随后以128帧请求验证Qwen真实显存和视觉接口，再进行全部整体/窗口评分。技术预检输出单独保留，不挑选或替换正式请求的回复。最后生成 `analysis-<attempt>/report.md`、`metrics.csv`、`summary.json`，按模型及长短组汇总，窗口不替代全片主分。
