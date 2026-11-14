# A 的实际共同输入

`cases.csv` 和 `dataset/` 为 A 提供的原件，不覆盖、不重新抽样。24 个案例覆盖 11 类：单人、多人各 3，其余九类各 2；主体16、背景8；短18、长6，种子均为42。抽样过程未提供，不声称已验证随机抽样。

在仓库根目录设置个人 Python 3.10+ 环境（需 Pillow），查询当前 QoS 后提交：

```bash
export AI6132_CODE="$PWD"
export AI6132_CPU_ENV=/absolute/personal/conda/environment
mytcinfo
sbatch 02-pilot/tc2/run_cpu.sh dataset \
  --source 01-dataset/cases.csv --out 01-dataset/cases.canonical.csv
```

只创建新文件；若目标或 receipt 已存在则拒绝覆盖。`cases.canonical.csv` 的字段为 `case_id,category,image_path,prompt,seed,source_frame_num,target_frames`。文件名前缀01—24确定 `case_0001`—`case_0024`；类别、顺序、提示词和种子保留，源帧数128/512映射为原生129/513。不裁掉尾帧，也不插帧或续写。

`cases.canonical.receipt.json` 绑定源CSV、派生CSV和每张图片的SHA256、完整解码尺寸。图片需为共享16:9裁剪。生成后验证：

```bash
sbatch 02-pilot/tc2/run_cpu.sh dataset --verify 01-dataset/cases.canonical.csv
sbatch 02-pilot/tc2/run_cpu.sh validate-cases \
  --cases 01-dataset/cases.canonical.csv --phase formal
```

所有 Python 均在计算节点执行。C 使用 `run_ltx.sh formal ...` 默认读取全部11类、24个案例及逐案例129/513帧目标。所有模型必须消费同一 `case_id/category/prompt/seed/target_frames` 契约，交付实际原生规格；B 的 Hunyuan 入口见 [生成说明](../03-generation/hunyuan_video/README.md)。

共同试跑需另提供3个独立案例（可使用 `case_9001` 等 ID），采用相同字段，至少包含129与513两档；不得把正式案例或其复制图作为共同试跑。
