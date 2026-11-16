# 人工视频评价

本文记录个人工作区中已运行的新版网页／Excel工具及其验收结果；`human_workbooks.py`、网页资源和对应新增作业入口尚未发布。公开仓库中的`human_eval.py`仍为历史三维工具，不能直接替代下述四维流程；使用新版命令需具备对应本地实现。

当前评分方法为 **human-v3-four-metrics-native-fps**，网页交互版本为 **human-browser-v2-scores-only**，填表规则为 **scores-only-v1**。四项指标仍是首图／主体外观保持、动作自然度、提示词遵循、背景细节保持。**D已补入原24个位置，当前ABCD共96条可评，0条待交付**，原编号、每人顺序及已填评分继续保留。按用户要求，网页已删除问题时间段、文字观察和评审声明输入，只要求四项评分；低分和背景NA无需补填说明。

最新分发目录为`/home/msai/lius0131/ai6132-b/human-evaluation/shared24-v3-20261115/releases/abcd-v1/`，**仅保留A/B/C/D四份完整评分ZIP**，每份约425MB、包含全部96条视频。按用户要求已删除两版共8个网页升级补丁；旧72条包、D增量包和历史验收凭据移到组织者私有归档，不再作为分发选项。完整包实际位于已授权的XJIANG026/liu，通过上述Home路径访问。当前入口见批次根目录`CURRENT_RELEASE.json`及`分发与后补D说明.md`。

D原片验收54790、95项软件回归54791、实际补入及四包离线浏览器验收54797均COMPLETED/0:0。随后用户补充LongCat_generation_records_24，54800核对310个文件及24例原片／首图哈希、prompt、seed=42、配置和16步日志全部一致，私有来源更新为verified_from_supplied_records。记录配置为16步、CFG=1、BF16；实际权重与完整执行源码未在接收端重算哈希。case_0001复用同参数技术试跑、三个独立共同试跑未完成及原formal_evaluation=false继续保留。来源更新不改变任何视频、评分表或ZIP，原包继续使用。所有测试分数均为SYNTHETIC，不是人评交卷。

## 当前收表与分析状态

截至2026-11-16 15:29（新加坡时间），只读接收54957已核验A/B/C/D四份表，每份96条全部有效，共384条评分记录、1536项数值。四维各384个有效数值、0个NA、0个缺失；原Excel文件及SHA256保留，未把软件测试中的合成评分作为交卷。

54961计算人评统计后，54970核验每模型18短／6长，54971修正帧数字符串分组并更新`human-reference/analysis-v2/`；旧分析保留。当前四维序数Krippendorff α依次为：首图／主体外观保持0.656077、动作自然度0.572817、提示词遵循0.582075、背景细节保持0.537587。分组修正未改变全体均值或α。计算凭据为个人`ai6132-review-evidence/qwen-repair-20261116/human-reference/`中的`INTAKE.json`、`analysis-v2/ANALYSIS.json`和`agreement.csv`。

这些结果确认收表完整及评审间一致性，不消除已见模型分数、缺陷讨论或分阶段交付的影响。当前Qwen v3.5原严格门槛仍失败、诊断全量运行中，最终人机对照及整合报告尚未完成；人评原分不输入Qwen、不用于改写其回复。

54802通过13项相关回归并核验清理结果：四份完整包仍为原哈希，各含96视频、24首图和本人96行空白表单，原视频、已发Excel、网页文件及私有评分状态均未改写。清理清单、历史凭据和完整包校验索引保留在`organizer_private/full-only-retirement-20261115/`；无需重新制作或重新评分。

## 直接在网页观看与评分

每位成员使用自己的完整评分包，解压后双击`viewer.html`即可离线使用，无需Python、服务器或网络。对应首图、完整视频和原始英文prompt在同一页展示；视频支持正常播放、暂停和慢放。

1. 先以1倍速看完整片，再按需暂停、慢放。四项分数独立选择，无默认分数。
2. 可选勾选实际看见的问题。网页保留人物外观／数量变化、主体变形、手部持续模糊、滑步、接触异常、卡顿／停滞、动作缺失、提示词细节遗漏、后段背景虚化／纹理丢失等选项；选项只记录观察，不自动改分，也不影响完成度。
3. 背景本来不可比较才选`NA`，原因选填。原先清晰、后来持续虚化应按退化程度评分，不用NA代替低分。
4. 阶段性使用“导出当前评分Excel”，允许尚未评分的整行留空。96条成功视频完成四项评分后即可“导出最终Excel”，不要求问题时间段、文字观察或评审声明。未交付行留空，技术失败四分留空。

导出的是真实`.xlsx`，沿用原96行、匿名编号、每人的随机顺序、链接、下拉校验、保护和量表，兼容`validate-workbook`收表检查。选中的可选问题写入“具体观察/备注”。新表单隐藏历史时间列和声明页，以便保留旧版导入资料；没有预填任何声明。导出文件通常位于浏览器下载目录，若需使用Excel内的相对视频链接，将它放回评分包根目录。

## 保存与恢复已有评分

网页尝试将草稿保存在本机浏览器，按批次及评审隔离；显示已完成数量和每条状态。浏览器对本地file页面的存储行为有差异，因此**每次结束下载JSON进度备份**。换浏览器、移动文件夹、使用无痕模式或清理缓存后，通过“导入进度/Excel”载入备份继续。[浏览器存储说明](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage)

已经填写旧Excel的成员，可直接选择原xlsx导入。网页核对评审、批次、96行顺序、匿名编号和prompt，保留分数、原时段、原观察及声明，再导出兼容的Excel。不同成员／批次、无效分数和公式会被拒绝；导入失败保留当前草稿。覆盖已有草稿前要求确认，可取消。多页实际修改同一评审草稿时会提示冲突，当前内容仍可下载备份。

JSON是恢复进度用的文件，最终交卷仍提交xlsx。原始提交、修订版和JSON备份分别保留，不覆盖首次提交。

## 只分发四份完整评分包

```text
评分包_A_HTML.zip                  # B、C、D分别对应自己的包
  viewer.html                     # 直接观看、评价、导出Excel
  rating_app.js
  workbook_io.js
  rating_config.js                 # 本人量表配置及空白Excel模板
  vendor/                         # 本地fflate 0.8.2、MIT许可及来源校验
  ratings_A.xlsx                  # 也可直接填写；现有原件升级时保留
  评分要求.txt
  order.js
  delivery_state.js
  anonymous_manifest.csv
  images/                         # 24张对应首图
  videos/                         # 当前96条完整ABCD视频
```

只向成员分发本人完整ZIP，完整解压后打开viewer.html；组织者私有映射及其他成员表单不在包内。

已有评分的成员先保存JSON进度备份或原Excel，将新完整包解压到新目录，再通过网页“导入进度/Excel”载入继续；批次、匿名编号和每人的顺序保持一致。保留原评分文件，不用复制分数或重新评价。

## 素材与解释范围

人工v3素材由已验收原片制作，统一768×432、H.264/yuv420p、CRF18，保持比例并补边，保留原生帧率、全部帧和完整时长，移除音轨与来源metadata，完整解码验收。ABC为24fps、D为15fps，首图统一768×432。网页复用原H3编号和ABC素材，D仅补入原预留位置，不重排或覆盖原Excel。

历史VBench/Qwen使用12fps副本，不能声称与人工v3逐帧同输入。A实际10steps、125/509帧的偏差仍保留，不补帧或改称符合129/513帧。统一显示尺寸下的评价不能直接推断原生分辨率的绝对清晰度。

本轮已在人工提交前获授权运行诊断Qwen并讨论缺陷，部分成员可能已见分数；该顺序变更和匿名性限制继续在组织者记录和报告中披露。新版不收集个人声明，也不将未填写解释为“否”。分阶段D交付及识别自己视频会降低匿名性，不能声称完全双盲。原计划3条独立共同人评校准没有完成凭据，不补造校准记录。

## 集群命令

所有Python、浏览器测试、视频处理都必须通过sbatch；每次先查mytcinfo。普通Excel工具使用个人Python3.11、openpyxl3.1.5、Pillow11.3.0和FFmpeg/ffprobe环境。浏览器测试使用独立human-web-test-env，生产评分包无需安装这些依赖。

```bash
export AI6132_CODE="$PWD"
export AI6132_CPU_ENV=/home/msai/lius0131/XJIANG026/liu/ai6132-evaluation/human-tools-env
export AI6132_FFMPEG=/home/msai/lius0131/.conda/envs/repostguard/bin/ffmpeg
export AI6132_FFPROBE=/home/msai/lius0131/.conda/envs/repostguard/bin/ffprobe
mytcinfo
sbatch --nodelist=TC2N08 --cpus-per-task=1 --mem=2G --time=00:20:00 \
  02-pilot/tc2/run_cpu.sh human-xlsx upgrade-web \
  --packet-root /path/to/existing/v3-packet-root --release-dir /path/to/new/web-release
```

工具在私有暂存区生成并检查四份完整包，确认状态／已发Excel未变后更新组织者的网页文件；不生成网页升级补丁。新建批次的内部凭据为WEB_READY.json；本次分发目录仅保留ZIP，对应原凭据已归入私有目录，并由CURRENT_RELEASE.json引用。首次prepare同样直接包含完整网页评分功能，所有命令均通过CPU作业提交。

D已完整接收并纳入全部评分包，无需再次运行complete-d。原始生成记录、来源追加及此前素材补入过程保留在私有记录中，组员只接收上面的完整ZIP。

## 收表

```bash
mytcinfo
sbatch --nodelist=TC2N08 02-pilot/tc2/run_cpu.sh human-xlsx validate-workbook \
  --packet-root /path/to/v3-packet-root --rater A \
  --workbook /path/to/submissions/ratings_A_current.xlsx \
  --report /path/to/checks/A_01.json --export-csv /path/to/exports/A_01.csv
```

新版`rating_policy=scores-only-v1`表单部分回收允许整行未评，已开始评分的成功行须四项齐全。最终交卷加`--final`，要求全部素材交付、成功项全评；问题选项和背景NA原因选填，不要求时间、备注或声明。旧版未带规则标记的Excel仍按原约定校验；可先导入新版网页，再导出新版表单。每次使用新报告和导出名。输出目录父级先mkdir -p。

四份最终表现已收齐；解盲和关联使用最新organizer_private/state_current.json，clip_map.csv是首发快照、D最初model_id为空。四维分别报告有效数、技术缺失、背景NA与一致性，不补零。与历史Qwen只比较共有前三维；当前四维诊断Qwen须待全量终结并复核后再关联，披露输入、时序敏感性和人评顺序局限。历史三维human_eval.py及CSV凭据仅作复核，新一轮默认使用四维网页／Excel。

本地ZIP库固定fflate0.8.2，随包附MIT许可及官方npm包完整性凭据；网页不从CDN加载。实际浏览器测试通过本地file://地址并禁用网络，覆盖视频、控件、低分/NA、Excel导入导出、进度恢复、并发冲突和D留位。所有测试评分保存在明确标记SYNTHETIC的证据目录，不能当成人工交卷。
