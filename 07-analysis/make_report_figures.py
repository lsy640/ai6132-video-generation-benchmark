"""Build every figure used by the LaTeX report from the frozen analysis tables and native videos.

Run on a compute node (see make_report_figures.sh). Inputs are read-only; outputs go to
08-report/Review_of_Image_to_Video_Generation/figures/.
"""
import csv
import glob
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / '08-report/data'
OUT = ROOT / '08-report/Review_of_Image_to_Video_Generation/figures'
HUNYUAN = Path(sys.argv[1])          # directory holding case_XXXX/native.mp4
FFMPEG = sys.argv[2]
WIDTH = 6.3                          # inches; equals the report text width

MODELS = [('A', 'Wan 2.2'), ('B', 'HunyuanVideo'), ('C', 'LTX-Video'), ('D', 'LongCat-Video')]
COLOR = {'A': '#0072B2', 'B': '#E69F00', 'C': '#009E73', 'D': '#CC79A7'}   # Okabe-Ito, colour-blind safe
METRICS = [('reference_fidelity', 'Image alignment'), ('motion_naturalness', 'Motion naturalness'),
           ('prompt_adherence', 'Prompt adherence'), ('background_detail_retention', 'Background preservation')]
INK, MUTED, GRID = '#222222', '#666666', '#DDDDDD'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.6, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'text.color': INK, 'axes.labelcolor': INK, 'pdf.fonttype': 42, 'hatch.linewidth': 0.6,
})


def rows(name):
    with open(DATA / name, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def video(slot, case):
    number = int(case[-4:])
    if slot == 'A':
        return glob.glob(str(ROOT / f'video-output/wan22_10steps_output/{number:02d}_seed42_f*.mp4'))[0]
    if slot == 'B':
        return str(HUNYUAN / case / 'native.mp4')
    if slot == 'C':
        return str(ROOT / f'video-output/C_LTX_13B/delivery/native/{case}.mp4')
    return str(ROOT / f'video-output/LongCat_native_24/{case}.mp4')


def crop169(image, width=640):
    w, h = image.size
    if w * 9 > h * 16:
        new = h * 16 // 9
        image = image.crop(((w - new) // 2, 0, (w - new) // 2 + new, h))
    else:
        new = w * 9 // 16
        image = image.crop((0, (h - new) // 2, w, (h - new) // 2 + new))
    return image.resize((width, width * 9 // 16), Image.LANCZOS)


def frame(path, index):
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / 'frame.png'
        subprocess.run([FFMPEG, '-v', 'error', '-i', path, '-vf', f'select=eq(n\\,{index})',
                        '-frames:v', '1', str(target)], check=True)
        return crop169(Image.open(target).convert('RGB'))


def dataset_figure(cases):
    fig, axes = plt.subplots(4, 6, figsize=(WIDTH, WIDTH * 4 / 6 * 9 / 16 + 0.62))
    for ax, case in zip(axes.flat, cases):
        ax.imshow(crop169(Image.open(ROOT / '01-dataset' / case['image_path']).convert('RGB'), 480))
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        length = 'long' if case['target_frames'] == '513' else 'short'
        ax.set_title(f"{int(case['case_id'][-4:]):02d} · {case['category']} · {length}", fontsize=5.8, pad=2,
                     color=INK if length == 'short' else '#B03A2E',
                     fontweight='normal' if length == 'short' else 'bold')
    fig.subplots_adjust(left=0.004, right=0.996, top=0.965, bottom=0.004, wspace=0.03, hspace=0.20)
    fig.savefig(OUT / 'dataset_overview.jpg', dpi=300, pil_kwargs={'quality': 92})
    plt.close(fig)


def human_figure():
    table = {(r['member_slot'], r['metric'], r['target_frames']): float(r['mean_clip_median'])
             for r in rows('human_summary.csv')}
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 3.75), sharey=True)
    for ax, (key, label) in zip(axes.flat, METRICS):
        for i, (slot, _) in enumerate(MODELS):
            short, long_ = table[slot, key, '129'], table[slot, key, '513']
            ax.bar(i - 0.2, short, 0.36, color=COLOR[slot], edgecolor='white', linewidth=0.8, zorder=3)
            ax.bar(i + 0.2, long_, 0.36, color='white', edgecolor=COLOR[slot], linewidth=0.9, hatch='////', zorder=3)
            ax.text(i - 0.2, short + 0.06, f'{short:.2f}', ha='center', va='bottom', fontsize=6.5)
            ax.text(i + 0.2, long_ + 0.06, f'{long_:.2f}', ha='center', va='bottom', fontsize=6.5)
            ax.scatter([i], [table[slot, key, 'all']], marker='D', s=17, color=INK, edgecolor='white', linewidth=0.6, zorder=5)
        ax.set_title(label, loc='left', fontweight='bold')
        ax.set_xticks(range(4)); ax.set_xticklabels([name for _, name in MODELS])
        ax.set_ylim(1, 5.45); ax.set_yticks([1, 2, 3, 4, 5])
        ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0); ax.set_axisbelow(True)
        ax.tick_params(length=0)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel('Mean human score (1–5)')
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, edgecolor='white'),
               plt.Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=MUTED, hatch='////'),
               plt.Line2D([0], [0], color=INK, marker='D', markersize=4, linestyle='none')]
    fig.legend(handles, ['Short clips (18 per model)', 'Long clips (6 per model)', 'All 24 clips'],
               loc='upper center', ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.0),
               handlelength=1.6, columnspacing=1.8)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.875, bottom=0.065, wspace=0.05, hspace=0.42)
    fig.savefig(OUT / 'human_scores.pdf')
    plt.close(fig)


def qwen_figure():
    pairs = rows('qwen_human_pairs.csv')
    stats = {r['metric']: r for r in rows('qwen_human_comparison.csv') if r['member_slot'] == 'all'}
    fig, axes = plt.subplots(1, 4, figsize=(WIDTH, 2.1), sharey=True)
    for ax, (key, label) in zip(axes, METRICS):
        counts = Counter((float(r['human_median']), float(r['qwen'])) for r in pairs if r['metric'] == key)
        ax.plot([1, 5], [1, 5], color=MUTED, linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
        ax.scatter([k[0] for k in counts], [k[1] for k in counts], s=[6 + 7.5 * v for v in counts.values()],
                   color='#0072B2', alpha=0.8, edgecolor='white', linewidth=0.5, zorder=3, clip_on=False)
        ax.set_title(label, fontsize=8, fontweight='bold', pad=14)
        ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5)
        ax.set_xticks([1, 2, 3, 4, 5]); ax.set_yticks([1, 2, 3, 4, 5]); ax.set_aspect('equal')
        ax.grid(True, color=GRID, linewidth=0.4, zorder=0); ax.tick_params(length=0)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.text(0.5, 1.035, f"$\\rho={float(stats[key]['spearman_rho']):.2f}$,  MAE$\\,={float(stats[key]['mean_absolute_difference']):.2f}$",
                transform=ax.transAxes, ha='center', va='bottom', fontsize=7, color=MUTED)
        ax.set_xlabel('Human median')
    axes[0].set_ylabel('Qwen3-VL score')
    fig.subplots_adjust(left=0.065, right=0.99, top=0.83, bottom=0.19, wspace=0.1)
    fig.savefig(OUT / 'qwen_vs_human.pdf')
    plt.close(fig)


def qualitative_figure(case, long_clip, name):
    positions = [('Start', 0.0), ('1/3', 1 / 3), ('2/3', 2 / 3), ('End', 1.0)]
    fig, axes = plt.subplots(4, 4, figsize=(WIDTH, (WIDTH - 0.22) / 4 * 9 / 16 * 4 + 0.30))
    for r, (slot, model) in enumerate(MODELS):
        total = (509 if long_clip else 125) if slot == 'A' else (513 if long_clip else 129)
        for c, (title, fraction) in enumerate(positions):
            ax = axes[r, c]
            ax.imshow(frame(video(slot, case), round((total - 1) * fraction)))
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=8, pad=3)
            if c == 0:
                ax.set_ylabel(model, fontsize=7.5, labelpad=3, fontweight='bold')
    fig.subplots_adjust(left=0.035, right=0.999, top=0.945, bottom=0.003, wspace=0.02, hspace=0.03)
    fig.savefig(OUT / name, dpi=300, pil_kwargs={'quality': 90})
    plt.close(fig)



def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / '01-dataset/cases.canonical.csv', newline='', encoding='utf-8') as handle:
        cases = list(csv.DictReader(handle))
    dataset_figure(cases)
    human_figure()
    qwen_figure()
    qualitative_figure('case_0001', True, 'qualitative_long_running.jpg')
    qualitative_figure('case_0004', True, 'qualitative_long_boxing.jpg')
    qualitative_figure('case_0022', False, 'qualitative_short_gallery.jpg')
    for case, long_clip, name in [('case_0007', True, 'long_giraffe'), ('case_0009', True, 'long_train'),
                                  ('case_0019', True, 'long_waterfall'), ('case_0023', True, 'long_abstract'),
                                  ('case_0005', False, 'short_knitting'), ('case_0012', False, 'short_salmon'),
                                  ('case_0008', False, 'short_turtle')]:
        qualitative_figure(case, long_clip, f'qualitative_{name}.jpg')
    print('figures written to', OUT)


if __name__ == '__main__':
    main()
