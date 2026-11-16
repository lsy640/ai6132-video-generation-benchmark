# AI6132 Video Generation Benchmark

English | [简体中文](README.zh-CN.md)

AI6132 Project 1 · Technical Review

We compare four open-source image-to-video models using the same reference images, English prompts, and random seed. The benchmark covers reference fidelity, motion naturalness, prompt adherence, background detail preservation, and inference cost. Evaluation combines ratings from four human reviewers, VBench-I2V metrics, and Qwen3-VL judgments, with an analysis of agreement across the three approaches.

Final report: [`group_11_report.pdf`](08-report/Review_of_Image_to_Video_Generation/group_11_report.pdf) ([LaTeX source](08-report/Review_of_Image_to_Video_Generation/main.tex)). Member contributions: [`group_11_contribution.pdf`](08-report/group_11_contribution.pdf).

## Experimental Design

The benchmark uses 24 fixed cases from the VBench-I2V 16:9 subset, spanning 11 categories: 16 subject cases and 8 background cases. Each model generates one video per case with seed 42, producing 96 videos in total. Each model contributes 18 short clips (target: 129 frames) and 6 long clips (target: 513 frames, generated in a single continuous pass). See the [canonical input manifest](01-dataset/cases.canonical.csv).

| Member | Model | Steps | Native Resolution / FPS | Frames (Short / Long) |
| --- | --- | --- | --- | --- |
| A | Wan2.2-TI2V-5B | 10 | 1248×704 / 24 | 125 / 509 |
| B | HunyuanVideo-1.5 480p I2V step-distilled | 8 | 848×480 / 24 | 129 / 513 |
| C | LTX-Video 13B v0.9.8 distilled | 7+3 | 768×432 / 24 | 129 / 513 |
| D | LongCat-Video 480p + CFG-step distilled LoRA | 16 | 832×480 / 15 | 129 / 513 |

Known protocol deviations: A uses 10 steps (the default is 50) and produces 4 fewer frames per clip than the target. D runs at 15 fps, resulting in longer playback durations for the same frame count. Step counts and resolutions differ across models, so the results compare these specific configurations.

## Main Results

Human ratings use a 1–5 scale. For each video, we take the median of the four reviewers' scores, then average these medians across the 24 videos per model.

| Model | Reference Fidelity | Motion Naturalness | Prompt Adherence | Background Preservation | Short-Clip Generation Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Wan 2.2 | 3.79 | 2.96 | 4.08 | 3.42 | 464 |
| HunyuanVideo 1.5 | 4.25 | 3.71 | 4.48 | 4.02 | 173 |
| LTX-Video | **4.67** | **3.92** | **4.71** | **4.75** | **25** |
| LongCat-Video | 4.00 | 3.69 | 4.15 | 3.88 | 447 |

- The four models perform similarly on short clips. Only LTX-Video remains stable on long clips; both Wan 2.2 and LongCat-Video fall below 2 for reference fidelity and background preservation.
- Ordinal Krippendorff's α ranges from 0.54 to 0.66 across the four reviewers, indicating moderate agreement.
- Qwen3-VL correlates with human ratings for reference fidelity (ρ=0.60) and prompt adherence (ρ=0.51), but shows no correlation for motion naturalness (ρ=−0.06). Its motion scores are used only to diagnose the evaluation method.
- VBench similarity scores cluster between 0.96 and 0.99 and do not reproduce the human rankings.

These findings are limited to the tested configurations, with one seed and 24 fixed cases. See the report for the full analysis and limitations.

## Repository Structure

| Path | Contents |
| --- | --- |
| [`01-dataset/`](01-dataset/README.md) | Manifest and reference images for the 24 cases |
| [`02-pilot/tc2/`](02-pilot/tc2/README.md) | TC2 environment setup scripts, SLURM job entry points, and generation records |
| [`03-generation/`](03-generation/) | Generation code, configurations, and handoff tools for the four models |
| [`ai6132_benchmark/`](ai6132_benchmark/) | Shared data validation, video processing, and VBench and Qwen evaluation implementations |
| [`04-vbench-evaluation/`](04-vbench-evaluation/README.md) | Preparation of evaluation copies and VBench-I2V evaluation |
| [`05-human-evaluation/`](05-human-evaluation/README.md) | HTML/Excel rating tools, submission validation, and four completed rating forms |
| [`06-vlm-evaluation/`](06-vlm-evaluation/README.md) | Qwen3-VL evaluation protocol |
| [`07-analysis/`](07-analysis/README.md) | Scripts for generating report figures |
| [`08-report/`](08-report/README.md) | LaTeX report, member contribution statement, and statistical tables |
| `video-output/` | Native videos and run records delivered by each member (excluded from version control) |

Original images, model weights, generated videos, and mappings between anonymous IDs and models are excluded from version control. See [PLAN.md](PLAN.md) for the experimental protocol and scoring rubric.

## Running the Benchmark

On NTU TC2, use the head node only for editing, Git operations, and job submission. Run all Python and GPU workloads on compute nodes through `sbatch`.

```bash
export AI6132_CODE="$PWD"
export AI6132_CPU_ENV=/absolute/path/to/conda/environment
sbatch 02-pilot/tc2/run_cpu.sh dataset --verify 01-dataset/cases.canonical.csv   # Validate inputs
sbatch 07-analysis/make_report_figures.sh                                     # Regenerate report figures
sbatch 08-report/build_report.sh                                             # Compile the report and contribution statement
```

For model generation commands, see [Wan2.2](03-generation/Wan2.2/BENCHMARK.md), [Hunyuan](03-generation/hunyuan_video/README.md), [LTX](03-generation/ltx_video/README.md), and [LongCat](03-generation/longcat_video/README.md). Evaluation workflows are documented in directories 04–06.
