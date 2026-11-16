# 测试集

24 个案例取自 VBench-I2V 的 16:9 图片子集，由成员 A 选定，四个模型共用。

- **类别**：11 类。单人、多人各 3 例，动物、交通工具、食物、植物、其他主体、建筑、风景、室内、抽象各 2 例。前七类共 16 例为主体案例，后四类共 8 例为背景案例。
- **长度**：18 例生成短片（目标 129 帧），6 例生成长片（目标 513 帧，案例 01、04、07、09、19、23）。
- **种子**：全部为 42，每个案例每个模型只生成一次。

这是一个固定的子集，选取过程没有记录，不能当作随机抽样。

## 文件

| 文件 | 内容 |
| --- | --- |
| `cases.csv` | 成员 A 提供的原始清单：类别、图片、提示词、种子、帧数（128 或 512） |
| `dataset/` | 24 张参考图（不纳入版本控制） |
| `cases.canonical.csv` | 各生成入口读取的规范清单，字段为 `case_id,category,image_path,prompt,seed,source_frame_num,target_frames` |
| `cases.canonical.receipt.json` | 两份清单和每张图片的 SHA256 与尺寸，生成入口据此确认输入没有变化 |

规范清单由原始清单派生：图片文件名的前缀 01–24 对应 `case_0001`–`case_0024`，类别、顺序、提示词和种子不变，帧数 128／512 映射为模型可以直接生成的 129／513。

## 命令

```bash
export AI6132_CODE="$PWD"
export AI6132_CPU_ENV=/absolute/path/to/conda/env        # Python 3.10+，需要 Pillow

sbatch 02-pilot/tc2/run_cpu.sh dataset \
  --source 01-dataset/cases.csv --out 01-dataset/cases.canonical.csv   # 生成规范清单（目标已存在时拒绝覆盖）
sbatch 02-pilot/tc2/run_cpu.sh dataset --verify 01-dataset/cases.canonical.csv   # 校验清单和图片
```
