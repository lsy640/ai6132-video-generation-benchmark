#!/usr/bin/env python3
"""Versioned four-metric Excel packets with stable slots for a late fourth model.

Run all commands through sbatch. Private model mappings stay outside public packets.
The older human_eval.py preserves the historical three-metric CSV contract only.
"""
import argparse
import base64
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import secrets
import shutil
import subprocess
import tempfile
import zipfile

from ai6132_benchmark.dataset import load_canonical
from ai6132_benchmark.io import read_csv, sha256, write_json
from ai6132_benchmark.media import binary, inspect_video

VERSION = 'human-v3-four-metrics-native-fps'
RATERS = ('A', 'B', 'C', 'D')
SCORES = ('reference_fidelity', 'motion_naturalness', 'prompt_adherence', 'background_detail_retention')
RATING_FIELDS = ('rater_id', 'clip_id', *SCORES, 'background_na_reason', 'evidence_time', 'comment')
HEADERS = ('观看序号', '匿名编号', '视频与提示词', '首图', '首图/主体外观保持', '动作自然度', '提示词遵循',
           '背景细节保持', '背景NA原因', '问题时间段', '具体观察/备注', '原始英文提示词', '首次分发素材批次')
DATA_START = 6
NA_REASONS = ('原背景已无法辨认', '画面无可评背景', '遮挡/视角变化无法比较', '提示词明确要求改变景深/背景', '其他（备注说明）')
WEB_VERSION = 'human-browser-v2-scores-only'
RATING_POLICY = 'scores-only-v1'
ISSUE_OPTIONS = {
    'reference_fidelity': ['人物外观/身份发生变化', '发型/衣着发生无故变化', '人物/物体数量发生变化',
                           '脸部/主体结构变形', '主体关键特征丢失', '场景与首图明显不一致', '其他'],
    'motion_naturalness': ['手部动作持续模糊', '脚部/局部动作难以辨认', '肢体/物体结构异常', '滑步/漂浮',
                           '接触/交互不合理', '卡顿/跳变', '动作重复/停滞', '应有动作基本缺失', '运动轨迹/速度不自然', '其他'],
    'prompt_adherence': ['主要动作未发生', '动作只完成一部分', '主体/物体不符合要求', '数量/顺序不符合要求',
                         '场景/属性要求未实现', '关键细节缺失', '其他'],
    'background_detail_retention': ['后段背景持续虚化', '背景纹理/边缘丢失', '背景结构扭曲', '背景漂移/无故换景',
                                    '背景闪烁/细节不稳定', '不合理的清晰度/对焦变化', '其他'],
}
RUBRICS = [
    ('首图/主体外观保持', 'reference_fidelity', (
        '主体或场景严重改变；人物被替换或已难辨认为原对象。',
        '多处或持续明显漂移，人物外观、衣着或主体结构大幅改变。',
        '关键特征大体保留，但有局部变形或明显短暂漂移。',
        '主要特征稳定，只有轻微、短暂且不影响辨认的偏差。',
        '全程主要外观与场景特征稳定，变化与动作/视角相容。')),
    ('动作自然度', 'motion_naturalness', (
        '严重卡顿/跳变/结构变形，或提示词明确要求的动作基本缺失。',
        '多处或持续不自然，滑步、肢体/物体异常显著影响观看。',
        '动作可辨，但存在短暂顿挫、重复、停滞或局部不自然。',
        '整体自然，局部有少量轻微问题，动作仍清楚连贯。',
        '动作持续连贯、局部细节可辨，速度、轨迹和接触符合情境。')),
    ('提示词遵循', 'prompt_adherence', (
        '主要要求缺失或明显违背。',
        '只完成少量要求，主要动作/主体/场景条件未实现。',
        '完成部分要求，但关键细节或动作不完整。',
        '主要要求完成，仍有少量次要细节缺失。',
        '主要主体、动作和场景要求均可在视频中确认。')),
    ('背景细节保持', 'background_detail_retention', (
        '最初可辨的背景结构/纹理几乎全部丢失，持续严重虚化或无故换景。',
        '大部分背景细节持续丢失，明显虚化、漂移或结构破坏。',
        '背景大体保留，但部分纹理丢失或后段明显变软。',
        '主要背景结构与细节保留，仅有少量局部/短暂损失。',
        '可比较的主要背景结构、边缘和纹理在全程基本保留。')),
]

INSTRUCTIONS = '''AI6132 人工视频评价 · 四维量表 v3

一、如何开始
1. 完整解压自己的评分包，保持Excel、viewer.html、images和videos的相对位置。
2. 打开viewer.html，阅读四维量表并直接评分；也保留Excel表单供需要时使用。
3. 按网页序号观看并评分；已填过Excel可先通过“导入进度/Excel”载入，沿用原编号和顺序。
4. 先以1倍速完整观看，再按需要暂停或慢放查看细节；不要只看开头、截图或最佳片段。
5. 完成四项分数即可；问题选项与背景NA原因均为选填，各维度独立，不要求固定分数比例。
6. 网页尝试自动保存本机浏览器草稿；每次结束请下载JSON进度备份。换浏览器、换目录或清理浏览器后可导入备份继续。
7. 点击“导出当前评分Excel”可阶段性交卷；D补齐并完成评分后用“导出最终Excel”。导出文件含原96行与四个工作表。

二、材料与后续补充
每人固定96行，目前72条视频可用，24条为后续交付预留。预留行暂不打分，不填0或NA。
收到后续素材补充包后，将其中视频和delivery_state.js合并到原评分包根目录，保留自己的Excel。
重新打开观看页，原编号即可继续评分，网页草稿/JSON备份和已有Excel沿用同一编号，不用重排。
“首次分发素材批次”只是初始交付快照，不决定后续能否打分。补充包不会包含或覆盖Excel。
确实生成/解码失败的项目四个分数均留空，按观看页技术状态记录。

三、评分原则
四项均为1至5整数，2和4是明确的中间水平。各档锚点见Excel“评分说明”。
首图/主体外观保持：跟踪人物数量、可见外观、发型、衣着和物体结构；仍是同类人物/同一场景不等于保持。
动作自然度：检查手脚结构、滑步、运动轨迹、物体接触、重复停滞和跳变。区分合理瞬时运动模糊与持续融化/无法辨认的动作。
明确要求动作却几乎不动应扣分；本应静止的场景不因运动幅度小自动扣分。
提示词遵循：只按可见内容判断，不能从提示词或动作姿势补全没有发生的动作。一次性动作不必在每段重复完成。
背景细节保持：对比首图/首段与中后段，观察背景结构、边缘、纹理和清晰度。主体清晰不能补偿背景细节持续消失。
背景原已无法辨认、无可评背景、被遮挡无法比较，或提示词明确要求改变背景/景深时，可填NA并选择原因。
合理的相机/视角/对焦变化应结合语境判断并说明；不要把正常遮挡或瞬时运动模糊一概判为严重退化。

四、提交
新版只要求四项评分，不收集问题时间段或个人评审声明。问题选项可选，不影响完成度，选择的内容会写入导出Excel的备注。
背景原先清晰、后段虚化应按退化程度打分，不能填NA代替低分；背景NA原因可以留空。
提交本人导出的xlsx，不复制他人的分数，不覆盖首次提交；修正版另存新文件。JSON用于恢复进度，不替代Excel交卷。
导出Excel通常在浏览器下载目录；如需点击Excel中的视频链接，将该文件放回原评分包根目录。
不猜测或记录模型来源，不查看其他人的评分、匿名映射或模型评审分数。
已知的人评前模型评分及讨论情况仍由组织者在研究局限中说明，不补造个人声明。
分阶段交付意味着后补素材的来源可能被推知；成员也可能识别自己生成的视频，本轮并非完全双盲。
'''


def write_table(path, fields, rows):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def link_or_copy(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def normalize_human(source, target, expected):
    """Full-duration, native-fps material; never fabricate missing-model video."""
    target = Path(target)
    if target.exists():
        raise ValueError('Refusing to overwrite an existing human-rating video')
    temporary = target.with_name(target.stem + '.partial.mp4')
    subprocess.run([binary('ffmpeg'), '-v', 'error', '-xerror', '-err_detect', 'explode',
                    '-filter_threads', '1', '-threads', '1', '-i', str(source),
                    '-map', '0:v:0', '-an', '-sn', '-dn', '-map_metadata', '-1', '-map_metadata:s:v', '-1',
                    '-metadata:s:v:0', 'handler_name=VideoHandler', '-map_chapters', '-1',
                    '-vf', 'setpts=PTS-STARTPTS,scale=768:432:force_original_aspect_ratio=decrease:force_divisible_by=2,pad=768:432:(ow-iw)/2:(oh-ih)/2,setsar=1',
                    '-vsync', '0', '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p',
                    '-threads', '1', '-movflags', '+faststart', '-n', str(temporary)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    actual = inspect_video(temporary)
    if ((actual['width'], actual['height'], actual['codec'], actual['pix_fmt']) != (768, 432, 'h264', 'yuv420p')
            or actual['frames'] != expected['frames'] or abs(actual['fps'] - expected['fps']) > 1e-6
            or abs(actual['duration'] - expected['duration']) > 1 / expected['fps'] + 1e-6):
        raise ValueError('Human material changed the native frame count/fps/duration')
    temporary.replace(target)
    return {k: v for k, v in actual.items() if k != 'timestamps'}


def string_cell(cell, value):
    cell.value = str(value)
    cell.data_type = 's'


def instructions_for(records):
    ready = sum(r['status'] == 'success' for r in records)
    pending = sum(r['status'] == 'pending_delivery' for r in records)
    failed = len(records) - ready - pending
    snapshot = f'每人固定{len(records)}行，当前{ready}条视频可用，{pending}条待交付，{failed}条技术失败。'
    snapshot += '待交付行不打分，不填0或NA。' if pending else '当前无需等待后补素材，可按原顺序完成全部评价。'
    return INSTRUCTIONS.replace('每人固定96行，目前72条视频可用，24条为后续交付预留。预留行暂不打分，不填0或NA。', snapshot)


def create_workbook(path, rater, batch_id, ordered):
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Protection, Side
    from openpyxl.worksheet.datavalidation import DataValidation
    from openpyxl.utils import get_column_letter
    wb = Workbook()
    ws = wb.active
    ws.title = '评分表'
    wb.properties.creator = 'AI6132'
    wb.properties.title = f'独立评分表 {rater} · 四维量表v3'
    navy, pale, gray = '17365D', 'FFF2CC', 'E7E6E6'
    ws.merge_cells('A1:M1')
    ws['A1'] = f'AI6132 独立人工评分 · 评审 {rater} · 四项指标'
    ws['A1'].font = Font(size=17, bold=True, color=navy)
    ws.merge_cells('A2:M2')
    ws['A2'] = f'固定{len(ordered)}行：当前{sum(r["status"] == "success" for r in ordered)}条可用，{sum(r["status"] == "pending_delivery" for r in ordered)}条待交付。仅需四项评分，问题选项和背景NA原因均为选填。'
    ws['A2'].alignment = Alignment(wrap_text=True)
    ws.row_dimensions[2].height = 32
    ws.merge_cells('A3:M3')
    ws['A3'] = '先阅读“评分说明”；点击视频链接查看视频与英文提示词。先1倍速看全片，再按需慢放，完成四项评分。'
    ws['A3'].alignment = Alignment(wrap_text=True)
    ws.row_dimensions[3].height = 30
    for column, text in enumerate(HEADERS, 1):
        cell = ws.cell(5, column, text)
        cell.fill = PatternFill('solid', fgColor=navy)
        cell.font = Font(color='FFFFFF', bold=True)
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
    ws.row_dimensions[5].height = 35
    widths = (8, 16, 20, 11, 14, 14, 14, 15, 28, 19, 45, 65, 16)
    for number, width in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(number)].width = width
    for index, item in enumerate(ordered, DATA_START):
        values = [index - DATA_START + 1, item['clip_id'], '观看视频与提示词', '打开首图',
                  None, None, None, None, None, None, None, item['prompt'],
                  '首批素材' if item.get('initial_status', item['status']) == 'success' else '后续补充']
        for column, value in enumerate(values, 1):
            cell = ws.cell(index, column)
            if isinstance(value, str):
                string_cell(cell, value)
            else:
                cell.value = value
            cell.alignment = Alignment(vertical='top', wrap_text=True)
            cell.font = Font(name='Calibri', size=11)
            cell.fill = PatternFill('solid', fgColor='F4F7FA' if index % 2 == 0 else 'FFFFFF')
            if 5 <= column <= 11:
                cell.protection = Protection(locked=False)
                cell.fill = PatternFill('solid', fgColor=pale)
        if item['status'] != 'success':
            ws.cell(index, 13).fill = PatternFill('solid', fgColor=gray)
        ws.cell(index, 3).hyperlink = f'viewer.html#{item["clip_id"]}'
        ws.cell(index, 4).hyperlink = item['image_path']
        for column in (3, 4):
            ws.cell(index, column).font = Font(color='0563C1', underline='single')
        ws.row_dimensions[index].height = 60
    end = DATA_START + len(ordered) - 1
    integer = DataValidation(type='whole', operator='between', formula1=1, formula2=5, allow_blank=True)
    integer.showErrorMessage, integer.errorStyle = True, 'stop'
    integer.errorTitle, integer.error = '分数无效', '请填写1至5整数；尚未评分的项目留空。'
    ws.add_data_validation(integer)
    integer.add(f'E{DATA_START}:G{end}')
    background = DataValidation(type='list', formula1='"1,2,3,4,5,NA"', allow_blank=True)
    background.showErrorMessage, background.errorStyle = True, 'stop'
    background.errorTitle, background.error = '背景分无效', '请输入1至5，或选择NA。'
    ws.add_data_validation(background)
    background.add(f'H{DATA_START}:H{end}')
    reasons = DataValidation(type='list', formula1='"' + ','.join(NA_REASONS) + '"', allow_blank=True)
    reasons.showErrorMessage = True
    ws.add_data_validation(reasons)
    reasons.add(f'I{DATA_START}:I{end}')
    for offset, (label, _, anchors) in enumerate(RUBRICS, 5):
        ws.cell(5, offset).comment = Comment('\n'.join(f'{n}分：{text}' for n, text in enumerate(anchors, 1)), 'AI6132')
    ws.freeze_panes = 'E6'
    # Retain old imports without exposing retired form fields in new exports.
    ws.column_dimensions['J'].hidden = True
    ws.auto_filter.ref = f'A5:M{end}'
    ws.sheet_view.zoomScale = 85
    ws.protection.sheet = True
    ws.protection.autoFilter = False
    ws.protection.selectLockedCells = False
    ws.protection.selectUnlockedCells = False
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.orientation = 'landscape'
    ws.page_setup.paperSize = ws.PAPERSIZE_A3
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.print_title_rows = '1:5'
    guide = wb.create_sheet('评分说明')
    guide.append(['四维评分量表v3', '1分', '2分', '3分', '4分', '5分'])
    for label, _, anchors in RUBRICS:
        guide.append([label, *anchors])
    guide.column_dimensions['A'].width = 23
    for column in 'BCDEF':
        guide.column_dimensions[column].width = 37
    for row in guide:
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical='top')
        guide.row_dimensions[row[0].row].height = 100 if row[0].row > 1 else 32
    guide.append(['完整评分要求'])
    for line in instructions_for(ordered).strip().split('\n'):
        guide.append([line])
        number = guide.max_row
        guide.merge_cells(start_row=number, start_column=1, end_row=number, end_column=6)
        guide.cell(number, 1).alignment = Alignment(wrap_text=True, vertical='top')
        guide.row_dimensions[number].height = 32 if len(line) > 45 else 22
    guide.freeze_panes = 'B2'
    declaration = wb.create_sheet('评审声明')
    for row in [
        ['评审ID（固定）', rater], ['姓名（可选）', None], ['评分日期（可选）', None],
        ['已看过模型评分（历史资料，可留空）', None],
        ['已讨论过具体视频缺陷（历史资料，可留空）', None],
        ['其他可能影响独立性的情况（可选）', None],
        ['历史资料说明', '本表仅兼容旧版导入资料，当前评分不要求填写。'],
    ]:
        declaration.append(row)
    declaration.column_dimensions['A'].width = 58
    declaration.column_dimensions['B'].width = 85
    for row in declaration:
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical='top')
        declaration.row_dimensions[row[0].row].height = 38
    for number in range(2, 7):
        declaration.cell(number, 2).protection = Protection(locked=False)
        declaration.cell(number, 2).fill = PatternFill('solid', fgColor=pale)
    yesno = DataValidation(type='list', formula1='"是,否"', allow_blank=True)
    yesno.showErrorMessage = True
    declaration.add_data_validation(yesno)
    yesno.add('B4:B5')
    declaration.protection.sheet = True
    declaration.sheet_state = 'hidden'
    meta = wb.create_sheet('表单信息')
    for row in [('protocol', VERSION), ('batch_id', batch_id), ('rater_id', rater), ('rows', len(ordered)), ('rating_policy', RATING_POLICY)]:
        meta.append(row)
    meta.column_dimensions['A'].width = 20
    meta.column_dimensions['B'].width = 65
    meta.protection.sheet = True
    wb.active = 0
    wb.save(path)


def public_rows(state):
    return [{key: record[key] for key in ('clip_id', 'video_path', 'image_path', 'prompt', 'status', 'failure_reason')}
            for record in state['records']]


def write_delivery_state(path, state):
    rows = {r['clip_id']: {'status': r['status'], 'failure_reason': r['failure_reason'],
                          'duration_seconds': r.get('human', {}).get('duration')} for r in state['records']}
    Path(path).write_text('window.DELIVERY_STATE=' + json.dumps({'batch_id': state['batch_id'], 'revision': state['revision'], 'clips': rows},
                                                             ensure_ascii=False).replace('</', '<\\/') + ';\n', encoding='utf-8')


def write_web_assets(packet, state, rater):
    """Embed a fresh blank workbook; never embed a rater's entered scores."""
    packet, source = Path(packet), Path(__file__).parent
    by_id = {r['clip_id']: r for r in state['records']}
    ordered = [{**next(x for x in public_rows(state) if x['clip_id'] == clip),
                'initial_status': 'pending_delivery' if by_id[clip]['member_slot'] == 'D' else 'success'}
               for clip in state['orders'][rater]]
    with tempfile.TemporaryDirectory(dir=packet, prefix='.web-template-') as tmp:
        blank = Path(tmp) / 'blank.xlsx'
        create_workbook(blank, rater, state['batch_id'], ordered)
        template = base64.b64encode(blank.read_bytes()).decode('ascii')
    config = {'version': WEB_VERSION, 'protocol': VERSION, 'rating_policy': RATING_POLICY, 'batch_id': state['batch_id'], 'rater': rater,
              'scores': list(SCORES), 'na_reasons': NA_REASONS, 'issue_options': ISSUE_OPTIONS,
              'durations': {r['clip_id']: r.get('human', {}).get('duration') for r in state['records']},
              'rubrics': [{'label': label, 'key': key, 'anchors': anchors} for label, key, anchors in RUBRICS],
              'template_base64': template}
    (packet / 'rating_config.js').write_text('window.RATING_CONFIG=' + json.dumps(config, ensure_ascii=False).replace('</', '<\\/') + ';\n', encoding='utf-8')
    for name in ('rating_viewer.html', 'rating_app.js', 'workbook_io.js'):
        shutil.copyfile(source / name, packet / ('viewer.html' if name == 'rating_viewer.html' else name))
    (packet / 'vendor').mkdir(exist_ok=True)
    for name in ('fflate-0.8.2.js', 'LICENSE.fflate', 'fflate.receipt.json'):
        shutil.copyfile(source / 'vendor' / name, packet / 'vendor' / name)


def upgrade_web(output, release_dir):
    """Prepare complete rating ZIPs only; issued Excel stays untouched."""
    output, release_dir = Path(output).resolve(), Path(release_dir).resolve()
    state_file = output / 'organizer_private/state_current.json'
    state_digest = sha256(state_file)
    state = json.loads(state_file.read_text(encoding='utf-8'))
    if state['protocol'] != VERSION or len(state['records']) != 96:
        raise ValueError('Not a 96-slot v3 packet')
    if release_dir.exists():
        raise ValueError('Use a new release directory; retain prior releases')
    before = {r: sha256(output / 'packets' / r / f'ratings_{r}.xlsx') for r in RATERS}
    by_id = {r['clip_id']: r for r in state['records']}
    release_dir.mkdir(parents=True, mode=0o700)
    files = ['viewer.html', 'rating_app.js', 'workbook_io.js', 'rating_config.js', '评分要求.txt',
             'vendor/fflate-0.8.2.js', 'vendor/LICENSE.fflate', 'vendor/fflate.receipt.json']
    receipt = {'version': WEB_VERSION, 'protocol': VERSION, 'batch_id': state['batch_id'],
               'state_sha256': state_digest, 'raters': {}, 'issued_workbooks_unchanged': False}
    with tempfile.TemporaryDirectory(prefix='web-candidate-', dir=output / 'organizer_private') as tmp:
        staged = Path(tmp)
        for rater in RATERS:
            packet = staged / rater
            packet.mkdir()
            ordered = [{**next(r for r in public_rows(state) if r['clip_id'] == clip),
                        'initial_status': 'pending_delivery' if by_id[clip]['member_slot'] == 'D' else 'success'}
                       for clip in state['orders'][rater]]
            for relative in sorted({r['image_path'] for r in state['records']} |
                                   {r['video_path'] for r in state['records'] if r['status'] == 'success'}):
                source = output / 'shared' / relative
                expected = next((r['video_sha256'] for r in state['records'] if r['video_path'] == relative and r['status'] == 'success'), None)
                if expected is None:
                    expected = state['workbooks'][rater]['immutable_files'][relative]
                if sha256(source) != expected:
                    raise ValueError('Media changed since the issued packet')
                link_or_copy(source, packet / relative)
            create_workbook(packet / f'ratings_{rater}.xlsx', rater, state['batch_id'], ordered)
            write_web_assets(packet, state, rater)
            (packet / '评分要求.txt').write_text(instructions_for(ordered), encoding='utf-8')
            (packet / 'order.js').write_text('window.RATING_PACKET=' + json.dumps({
                'rater': rater, 'batch_id': state['batch_id'],
                'rows': [{k: item[k] for k in ('clip_id', 'video_path', 'image_path', 'prompt', 'status', 'failure_reason')} for item in ordered]},
                ensure_ascii=False).replace('</', '<\\/') + ';\n', encoding='utf-8')
            write_delivery_state(packet / 'delivery_state.js', state)
            shutil.copyfile(output / 'packets' / rater / 'anonymous_manifest.csv', packet / 'anonymous_manifest.csv')
            report, ratings = workbook_values(packet / f'ratings_{rater}.xlsx', state, rater)
            if not report['valid'] or any(row[key] for row in ratings for key in SCORES):
                raise ValueError('New distribution template must be valid and blank')
            archives = {}
            for kind, name, members in [
                ('full', f'评分包_{rater}_HTML.zip', [p.relative_to(packet).as_posix() for p in packet.rglob('*') if p.is_file()]),
            ]:
                target = release_dir / name
                with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                    for relative in sorted(members):
                        archive.write(packet / relative, relative)
                with zipfile.ZipFile(target) as archive:
                    if archive.testzip() is not None:
                        raise ValueError('Web distribution archive corrupted')
                archives[kind] = {'path': str(target), 'bytes': target.stat().st_size, 'sha256': sha256(target)}
            receipt['raters'][rater] = {'archives': archives, 'interface_files': {name: sha256(packet / name) for name in files},
                                        'issued_workbook_sha256': before[rater], 'new_blank_template_sha256': sha256(packet / f'ratings_{rater}.xlsx')}
        if sha256(state_file) != state_digest or any(sha256(output / 'packets' / r / f'ratings_{r}.xlsx') != before[r] for r in RATERS):
            raise ValueError('Issued state/workbooks changed during staging; do not publish the candidate')
        backup = output / 'organizer_private' / ('web-upgrade-' + release_dir.name)
        backup.mkdir()
        for rater in RATERS:
            destination = output / 'packets' / rater
            for relative in files:
                target = destination / relative
                if target.exists():
                    previous = backup / rater / relative
                    previous.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(target, previous)
                target.parent.mkdir(parents=True, exist_ok=True)
                pending = target.with_suffix(target.suffix + '.web-tmp')
                shutil.copyfile(staged / rater / relative, pending)
                pending.replace(target)
            if sha256(destination / f'ratings_{rater}.xlsx') != before[rater]:
                raise ValueError('An issued Excel changed during UI update')
    receipt['issued_workbooks_unchanged'] = True
    write_json(release_dir / 'WEB_READY.json', receipt)
    return receipt


def assemble_packets(output, state):
    output = Path(output)
    metadata = {}
    by_id = {r['clip_id']: r for r in public_rows(state)}
    for rater in RATERS:
        packet = output / 'packets' / rater
        packet.mkdir(parents=True)
        for file in sorted((output / 'shared').rglob('*')):
            if file.is_file():
                link_or_copy(file, packet / file.relative_to(output / 'shared'))
        ordered = [by_id[clip] for clip in state['orders'][rater]]
        create_workbook(packet / f'ratings_{rater}.xlsx', rater, state['batch_id'], ordered)
        (packet / '评分要求.txt').write_text(instructions_for(ordered), encoding='utf-8')
        write_web_assets(packet, state, rater)
        (packet / 'order.js').write_text('window.RATING_PACKET=' + json.dumps({
            'rater': rater, 'batch_id': state['batch_id'], 'rows': ordered}, ensure_ascii=False).replace('</', '<\\/') + ';\n', encoding='utf-8')
        write_delivery_state(packet / 'delivery_state.js', state)
        write_table(packet / 'anonymous_manifest.csv', tuple(ordered[0]), ordered)
        metadata[rater] = {'template_sha256': sha256(packet / f'ratings_{rater}.xlsx'),
                           'immutable_files': {str(p.relative_to(packet)): sha256(p) for p in packet.rglob('*')
                                               if p.is_file() and p.suffix != '.xlsx'}}
    return metadata


def create_archives(output):
    output = Path(output)
    releases = output / 'releases'
    releases.mkdir(exist_ok=True)
    result = {}
    for rater in RATERS:
        packet = output / 'packets' / rater
        target = releases / f'评分包_{rater}.zip'
        if target.exists():
            raise ValueError('Retain previously released archives')
        with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for file in sorted(packet.rglob('*')):
                if file.is_file():
                    archive.write(file, file.relative_to(packet).as_posix())
        result[rater] = {'path': str(target), 'bytes': target.stat().st_size, 'sha256': sha256(target)}
    return result


def prepare(cases, cohort, output):
    from PIL import Image
    cases = Path(cases).resolve()
    cohort, output = Path(cohort).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Use a new output directory; never overwrite issued rating packets')
    canonical, case_receipt = load_canonical(cases, decode=True)
    completed = json.loads((cohort / 'COMPLETE.json').read_text(encoding='utf-8'))
    if completed['items'] != 72 or completed['phase'] != 'diagnostic':
        raise ValueError('Expected the validated complete A/B/C cohort of 72 videos')
    audits_file = cohort / 'private/generation_summary.json'
    if sha256(audits_file) != completed['files']['private/generation_summary.json']:
        raise ValueError('Frozen generation intake changed')
    audits = json.loads(audits_file.read_text(encoding='utf-8'))
    role_models = {'A': 'Wan2.2-A-delivery', 'B': 'HunyuanVideo-1.5-480P-I2V-step-distill', 'C': 'ltxv-13b-0.9.8-distilled'}
    indexed = {(r['case_id'], r['model_id']): r for r in audits}
    if len(indexed) != 72:
        raise ValueError('Intake has duplicate/missing cases')
    output.mkdir(parents=True, mode=0o700)
    private, shared = output / 'organizer_private', output / 'shared'
    private.mkdir(mode=0o700)
    (shared / 'images').mkdir(parents=True)
    (shared / 'videos').mkdir()
    secret_seed = secrets.randbits(128)
    rng = random.Random(secret_seed)
    keys = [(case['case_id'], role) for case in canonical for role in RATERS]
    rng.shuffle(keys)
    images = list(canonical)
    rng.shuffle(images)
    references = {}
    for number, case in enumerate(images, 1):
        name = f'images/ref_{number:04d}.png'
        source = cases.parent / case['image_path']
        with Image.open(source) as image:
            resized = image.convert('RGB').resize((768, 432), Image.Resampling.LANCZOS)
            clean = Image.new('RGB', resized.size)
            clean.paste(resized)
            clean.save(shared / name)
        references[case['case_id']] = name
    by_case = {r['case_id']: r for r in canonical}
    records = []
    state = {'protocol': VERSION, 'batch_id': secrets.token_hex(12), 'revision': 1,
             'case_receipt': case_receipt, 'cases_path': str(cases), 'cohort_path': str(cohort),
             'cohort_completion_sha256': sha256(cohort / 'COMPLETE.json'), 'secret_seed': secret_seed,
             'records': records, 'orders': {}, 'independent_calibration_receipt': None,
             'limitations': ['D pending slots identify the late-delivery source by timing',
                            'Some raters have already seen model scores or discussed defects',
                            'Human v3 preserves native fps, unlike the historical 12fps automatic copies',
                            'A delivered 10-step 125/509-frame videos; keep the declared protocol deviation']}
    write_json(private / 'build_started.json', {'protocol': VERSION, 'batch_id': state['batch_id'], 'expected_slots': 96})
    for number, (case_id, role) in enumerate(keys, 1):
        clip = f'H3_{state["batch_id"][:6]}_{number:04d}'
        case = by_case[case_id]
        record = {'clip_id': clip, 'case_id': case_id, 'member_slot': role,
                  'model_id': role_models.get(role), 'prompt': case['prompt'], 'seed': int(case['seed']),
                  'target_frames': int(case['target_frames']), 'image_path': references[case_id],
                  'video_path': f'videos/{clip}.mp4', 'status': 'pending_delivery', 'failure_reason': ''}
        if role != 'D':
            source_record = indexed[(case_id, role_models[role])]
            if source_record['handoff_status'] != 'success':
                raise ValueError('A/B/C intake must contain 72 successful videos')
            source = Path(source_record['native_video'])
            if not source.is_absolute():
                # C's received handoff uses portable relative paths.
                source = cases.parents[1] / 'video-output/C_LTX_13B/delivery' / source
            source = delivery_source(str(source), cases)
            if sha256(source) != source_record['native_sha256']:
                raise ValueError('Received native video changed after independent intake')
            actual = normalize_human(source, shared / record['video_path'], source_record['native'])
            record.update(status='success', source_video=str(source), source_sha256=source_record['native_sha256'],
                          video_sha256=sha256(shared / record['video_path']), native=source_record['native'], human=actual)
        records.append(record)
        print(json.dumps({'prepared': number, 'total': 96, 'status': record['status']}), flush=True)
    for rater in RATERS:
        order = [r['clip_id'] for r in records]
        seed = int.from_bytes(hashlib.sha256(f'{secret_seed}:{rater}'.encode()).digest(), 'big')
        random.Random(seed).shuffle(order)
        state['orders'][rater] = order
    state['workbooks'] = assemble_packets(output, state)
    write_json(private / 'state_v1.json', state)
    write_json(private / 'state_current.json', state)
    write_table(private / 'clip_map.csv', ('clip_id', 'case_id', 'member_slot', 'model_id'), records)
    archives = create_archives(output)
    receipt = {'protocol': VERSION, 'batch_id': state['batch_id'], 'rows_per_rater': 96,
               'available_videos': 72, 'pending_slots': 24, 'references': 24, 'raters': list(RATERS),
               'archives': archives, 'organizer_state_sha256': sha256(private / 'state_current.json')}
    write_json(output / 'READY.json', receipt)
    return receipt


def workbook_values(path, state, rater, final=False):
    """Validate identity/layout separately from editable scores and declarations."""
    from openpyxl import load_workbook
    if rater not in RATERS:
        raise ValueError('Unknown rater')
    wb = load_workbook(path, data_only=False, keep_links=False)
    if wb.sheetnames != ['评分表', '评分说明', '评审声明', '表单信息']:
        raise ValueError('Workbook sheets changed; use the issued template')
    meta = dict(wb['表单信息'].iter_rows(values_only=True))
    policy = meta.pop('rating_policy', None)
    if policy not in (None, RATING_POLICY):
        raise ValueError('Unsupported rating policy')
    scores_only = policy == RATING_POLICY
    if meta != {'protocol': VERSION, 'batch_id': state['batch_id'], 'rater_id': rater, 'rows': 96}:
        raise ValueError('Workbook/rater/batch identity differs from the issued template')
    ws = wb['评分表']
    if tuple(ws.cell(5, i).value for i in range(1, 14)) != HEADERS or ws.max_row != 101:
        raise ValueError('Score rows/headers changed')
    expected = {r['clip_id']: r for r in state['records']}
    errors, results = [], []
    for position, clip in enumerate(state['orders'][rater], DATA_START):
        record = expected[clip]
        original_phase = '后续补充' if record['member_slot'] == 'D' else '首批素材'
        fixed = {1: position - DATA_START + 1, 2: clip, 3: '观看视频与提示词', 4: '打开首图',
                 12: record['prompt'], 13: original_phase}
        if any(ws.cell(position, col).value != value for col, value in fixed.items()):
            raise ValueError(f'{clip}: immutable order/ID/prompt/source-phase cells changed')
        if (ws.cell(position, 3).hyperlink is None or ws.cell(position, 3).hyperlink.target != f'viewer.html#{clip}'
                or ws.cell(position, 4).hyperlink is None or ws.cell(position, 4).hyperlink.target != record['image_path']):
            raise ValueError(f'{clip}: material hyperlinks changed')
        cells = [ws.cell(position, col) for col in range(5, 12)]
        if any(cell.data_type == 'f' for cell in cells):
            raise ValueError(f'{clip}: enter literal observations/scores, not formulas')
        values = [cell.value for cell in cells]
        text = lambda value: '' if value is None else str(value).strip()
        result = dict(rater_id=rater, clip_id=clip,
                      **{field: text(value) for field, value in zip(SCORES, values[:4])},
                      background_na_reason=text(values[4]), evidence_time=text(values[5]), comment=text(values[6]))
        filled = any(text(value) for value in values)
        if record['status'] == 'pending_delivery':
            if filled:
                errors.append({'clip_id': clip, 'code': 'pending_material_has_scores_or_notes'})
            if final:
                errors.append({'clip_id': clip, 'code': 'material_still_pending'})
        elif record['status'] in ('generation_failed', 'decode_failed'):
            if any(text(value) for value in values[:4]):
                errors.append({'clip_id': clip, 'code': 'technical_failure_has_scores'})
            if final and not scores_only and not result['comment']:
                errors.append({'clip_id': clip, 'code': 'technical_failure_missing_comment'})
        elif (any(text(value) for value in values[:4]) if scores_only else filled) or final:
            numeric = []
            for number, value in enumerate(values[:4]):
                field = SCORES[number]
                if number == 3 and text(value) == 'NA':
                    if result['background_na_reason'] not in NA_REASONS and (result['background_na_reason'] or not scores_only):
                        errors.append({'clip_id': clip, 'code': 'missing_background_na_reason'})
                    elif not scores_only and result['background_na_reason'] == NA_REASONS[-1] and not result['comment']:
                        errors.append({'clip_id': clip, 'code': 'missing_other_na_explanation'})
                    continue
                valid = ((type(value) in (int, float) and 1 <= value <= 5 and int(value) == value)
                         or (type(value) is str and value in {'1', '2', '3', '4', '5'}))
                if not valid:
                    errors.append({'clip_id': clip, 'code': 'missing_or_invalid_score', 'field': field})
                else:
                    number_value = int(value)
                    result[field] = str(number_value)
                    numeric.append(number_value)
            if not scores_only and numeric and min(numeric) <= 3 and (not result['evidence_time'] or not result['comment']):
                errors.append({'clip_id': clip, 'code': 'low_score_requires_time_and_observation'})
            if result['background_detail_retention'] != 'NA' and result['background_na_reason']:
                errors.append({'clip_id': clip, 'code': 'na_reason_without_na_score'})
        results.append(result)
    declaration = wb['评审声明']
    if declaration['B1'].value != rater:
        raise ValueError('Rater declaration ID changed')
    statements = {'seen_model_scores': declaration['B4'].value,
                  'discussed_video_defects': declaration['B5'].value,
                  'other': declaration['B6'].value}
    if final and not scores_only:
        for key in ('seen_model_scores', 'discussed_video_defects'):
            if statements[key] not in ('是', '否'):
                errors.append({'code': 'missing_rater_declaration', 'field': key})
    return {'valid': not errors, 'rater_id': rater, 'final': final, 'rows': 96, 'rating_policy': policy or 'legacy-evidence-required',
            'pending_materials': sum(r['status'] == 'pending_delivery' for r in state['records']),
            'completed_rows': sum(all(r[k] for k in SCORES) for r in results),
            'errors': errors, 'declarations': statements}, results


def verify_packets(output):
    """Reopen workbooks and archives, verify every shared asset and portable link."""
    from openpyxl import load_workbook
    output = Path(output)
    state = json.loads((output / 'organizer_private/state_current.json').read_text(encoding='utf-8'))
    if state['protocol'] != VERSION or len(state['records']) != 96:
        raise ValueError('Invalid organizer state')
    references = set()
    for record in state['records']:
        references.add(record['image_path'])
        if record['status'] == 'success':
            if sha256(output / 'shared' / record['video_path']) != record['video_sha256']:
                raise ValueError('Shared video changed')
        elif record['status'] == 'pending_delivery' and (output / 'shared' / record['video_path']).exists():
            raise ValueError('Pending slot must not contain fabricated placeholder video')
    if len(references) != 24 or len({tuple(state['orders'][rater]) for rater in RATERS}) != 4:
        raise ValueError('Reference count or independent orders incorrect')
    for rater in RATERS:
        packet = output / 'packets' / rater
        report, rows = workbook_values(packet / f'ratings_{rater}.xlsx', state, rater)
        if not report['valid'] or any(r[field] for r in rows for field in SCORES):
            raise ValueError('Issued workbook contains scores or invalid structure')
        for record in state['records']:
            if not (packet / record['image_path']).is_file():
                raise ValueError('Reference hyperlink is not portable')
            if record['status'] == 'success' and sha256(packet / record['video_path']) != record['video_sha256']:
                raise ValueError('Packet video does not match the shared source')
        wb = load_workbook(packet / f'ratings_{rater}.xlsx')
        ws = wb['评分表']
        for row in range(6, 102):
            if any(ws.cell(row, col).protection.locked for col in range(5, 12)):
                raise ValueError('Editable scores/notes or future slots are locked')
        if ws.freeze_panes != 'E6' or len(ws.data_validations.dataValidation) != 3:
            raise ValueError('Workbook navigation/input validation missing')
        with zipfile.ZipFile(output / 'releases' / f'评分包_{rater}.zip') as archive:
            if archive.testzip() is not None:
                raise ValueError('Corrupt distribution archive')
            names = archive.namelist()
            if f'ratings_{rater}.xlsx' not in names or any('organizer_private' in n or 'clip_map' in n for n in names):
                raise ValueError('Distribution boundary incorrect')
            if any(n.endswith('.xlsx') and n != f'ratings_{rater}.xlsx' for n in names):
                raise ValueError('Another rater workbook was included')
            if hashlib.sha256(archive.read(f'ratings_{rater}.xlsx')).hexdigest() != state['workbooks'][rater]['template_sha256']:
                raise ValueError('Archived workbook differs from the issued blank template')
            if any(Path(n).is_absolute() or '..' in Path(n).parts for n in names):
                raise ValueError('Unsafe archive path')
            for relative, digest in state['workbooks'][rater]['immutable_files'].items():
                if hashlib.sha256(archive.read(relative)).hexdigest() != digest:
                    raise ValueError('Distribution archive differs from the verified packet')
    return {'status': 'passed', 'raters': 4, 'rows_per_rater': 96,
            'available_videos': sum(r['status'] == 'success' for r in state['records']),
            'pending_slots': sum(r['status'] == 'pending_delivery' for r in state['records']),
            'references': len(references), 'all_score_cells_blank': True,
            'all_future_score_cells_editable': True, 'independent_orders': True}


def complete_d(output, runs, patch_out, deviation_reason=None, unverified_source_reason=None):
    """Issue a media-only overlay. Never rewrite an issued or filled workbook."""
    with tempfile.TemporaryDirectory(prefix='d-staging-', dir=Path(output) / 'organizer_private') as staged:
        return _complete_d(output, runs, patch_out, deviation_reason, Path(staged), unverified_source_reason)


def _complete_d(output, runs, patch_out, deviation_reason, staged_root, unverified_source_reason):
    output, runs, patch_out = Path(output).resolve(), Path(runs).resolve(), Path(patch_out).resolve()
    private = output / 'organizer_private'
    state_file = private / 'state_current.json'
    old_digest = sha256(state_file)
    state = json.loads(state_file.read_text(encoding='utf-8'))
    canonical, receipt = load_canonical(state['cases_path'], decode=True)
    if receipt != state['case_receipt']:
        raise ValueError('Canonical inputs changed since human packet creation')
    _, rows, csv_digest = read_csv(runs)
    if len(rows) != 24 or {r['case_id'] for r in rows} != {r['case_id'] for r in canonical}:
        raise ValueError('D delivery must account for all 24 canonical cases')
    success_models = {r['model_id'] for r in rows if r['status'] == 'success'}
    issued_models = {r['model_id'] for r in state['records'] if r['member_slot'] == 'D' and r['status'] == 'success'}
    if len(success_models | issued_models) > 1:
        raise ValueError('Do not mix different D models in one set')
    if patch_out.exists():
        raise ValueError('Retain previous patches; use a new patch path')
    by_case = {r['case_id']: r for r in rows}
    written = []
    # All validation/encoding happens privately before any distributed material changes.
    # A failed encode can be retried from the unchanged issued state.
    before_workbooks = {r: sha256(output / 'packets' / r / f'ratings_{r}.xlsx') for r in RATERS}
    allowed_statuses = {'success', 'not_started', 'incomplete', 'interrupted', 'pending_delivery',
                        'technical_failure', 'generation_failed', 'decode_failed'}
    for row in rows:
        if row['status'] not in allowed_statuses:
            raise ValueError('Unknown D delivery status')
    for record in state['records']:
        if record['member_slot'] != 'D':
            continue
        row = by_case[record['case_id']]
        observed_seed = row.get('seed', '').strip()
        if observed_seed and int(observed_seed) != record['seed']:
            raise ValueError('D seed differs from the frozen shared input')
        if not observed_seed and not unverified_source_reason:
            raise ValueError('Missing D seed; an explicit unverified-source reason is required')
        if row.get('image_sha256') and row['image_sha256'] != receipt['images'][record['case_id']]['sha256']:
            raise ValueError('D input image differs from canonical image')
        if row['status'] in ('not_started', 'incomplete', 'interrupted', 'pending_delivery'):
            continue
        if record['status'] != 'pending_delivery':
            if row['status'] == 'success':
                source = delivery_source(row['native_video'], runs)
                if row['model_id'] != record['model_id'] or sha256(source) != record.get('source_sha256'):
                    raise ValueError('Already-issued D material changed; do not overwrite ratings inputs')
            continue
        if row['status'] == 'success':
            if (not row.get('weight_version') or not row.get('config')) and not unverified_source_reason:
                raise ValueError('D successful rows need a weight version and frozen configuration')
            source = delivery_source(row['native_video'], runs)
            source_digest = sha256(source)
            if row.get('video_sha256') and source_digest != row['video_sha256']:
                raise ValueError('D video differs from its generation receipt')
            native = inspect_video(source)
            if native['frames'] != record['target_frames'] and not deviation_reason:
                raise ValueError('D native frames differ; declare any diagnostic protocol exception explicitly')
            staged_video = staged_root / record['video_path']
            staged_video.parent.mkdir(parents=True, exist_ok=True)
            human = normalize_human(source, staged_video, native)
            record.update(status='success', model_id=row['model_id'], source_video=str(source), source_sha256=source_digest,
                          video_sha256=sha256(staged_video), native=native, human=human,
                          native_frame_deviation_reason=deviation_reason if native['frames'] != record['target_frames'] else None,
                          source_generation_provenance={'status': 'unverified' if unverified_source_reason else 'metadata_declared',
                              'reason': unverified_source_reason, 'observed_seed': int(observed_seed) if observed_seed else None,
                              'weight_version': row.get('weight_version') or None, 'config': row.get('config') or None,
                              'source_image_sha256': row.get('image_sha256') or None,
                              'case_association': row.get('source_case_association') or 'received_runs_csv'})
            written.append(record['video_path'])
        elif row['status'] in ('technical_failure', 'generation_failed', 'decode_failed'):
            record.update(status='decode_failed' if row['status'] == 'decode_failed' else 'generation_failed',
                          model_id=row.get('model_id'), failure_reason='decode_error' if row['status'] == 'decode_failed' else 'generation_error')
        else:
            raise ValueError('Unknown D delivery status')
    state.update(revision=state['revision'] + 1, previous_state_sha256=old_digest, latest_d_runs_sha256=csv_digest)
    # Patch assets have the exact same relative paths for all four recipients.
    patch_out.parent.mkdir(parents=True, exist_ok=True)
    state_js = staged_root / 'delivery_state.js'
    write_delivery_state(state_js, state)
    patch_temp = staged_root / 'patch.zip'
    with zipfile.ZipFile(patch_temp, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in written:
            archive.write(staged_root / relative, relative)
        archive.write(state_js, 'delivery_state.js')
        archive.writestr('补充素材说明.txt', '将本包内容合并到本人原评分包根目录，保留原Excel及所有已填分数。重新打开viewer.html，按原编号继续评分。\n')
    # Preflight every destination before publishing. Identical residual files from a
    # previously interrupted publication may be adopted, never silently replaced.
    destinations = []
    for relative in written:
        for root in [output / 'shared', *(output / 'packets' / r for r in RATERS)]:
            target = root / relative
            if target.exists() and sha256(target) != sha256(staged_root / relative):
                raise ValueError('Existing D material conflicts with the pending delivery')
            destinations.append((staged_root / relative, target))
    for source, target in destinations:
        if not target.exists():
            link_or_copy(source, target)
    for rater in RATERS:
        packet = output / 'packets' / rater
        shutil.copyfile(state_js, packet / 'delivery_state.js')
        if sha256(packet / f'ratings_{rater}.xlsx') != before_workbooks[rater]:
            raise ValueError('An issued workbook changed during media-only completion')
    shutil.copyfile(state_js, private / f'delivery_state_v{state["revision"]}.js')
    # Copy via a sibling temporary name because patch_out may be another filesystem.
    patch_publish = patch_out.with_suffix(patch_out.suffix + '.partial')
    if patch_publish.exists():
        raise ValueError('Retain the interrupted patch and use a new output path')
    shutil.copyfile(patch_temp, patch_publish)
    patch_publish.replace(patch_out)
    write_json(private / f'state_v{state["revision"]}.json', state)
    write_json(state_file, state)
    result = {'revision': state['revision'], 'added_videos': len(written),
              'remaining_pending': sum(r['status'] == 'pending_delivery' for r in state['records']),
              'workbooks_unchanged': True, 'patch': str(patch_out), 'sha256': sha256(patch_out)}
    write_json(private / f'patch_v{state["revision"]}.json', result)
    return result


def delivery_source(value, runs):
    """Resolve an intake path without traversing another member's home directory."""
    source = Path(value)
    if not source.is_absolute():
        source = runs.parent / source
    # Check the lexical path first: resolving a foreign path can itself traverse it.
    source = Path(os.path.abspath(source))
    lexical_roots = (Path('/home/msai/lius0131'), Path('/tmp'))
    if not any(source.is_relative_to(root) for root in lexical_roots):
        raise ValueError('Copy D videos into the authorized home before importing')
    resolved = source.resolve()
    roots = (*lexical_roots, Path('/home/msai/lius0131/XJIANG026/liu').resolve())
    if not any(resolved.is_relative_to(root) for root in roots):
        raise ValueError('D video resolves outside authorized storage')
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('prepare')
    build.add_argument('--cases', type=Path, required=True)
    build.add_argument('--cohort', type=Path, required=True)
    build.add_argument('--out', type=Path, required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('--packet-root', type=Path, required=True)
    upgrade = commands.add_parser('upgrade-web')
    upgrade.add_argument('--packet-root', type=Path, required=True)
    upgrade.add_argument('--release-dir', type=Path, required=True)
    late = commands.add_parser('complete-d')
    late.add_argument('--packet-root', type=Path, required=True)
    late.add_argument('--runs', type=Path, required=True)
    late.add_argument('--patch-out', type=Path, required=True)
    late.add_argument('--native-frame-deviation-reason')
    late.add_argument('--unverified-source-reason', help='Explicitly retain missing generation metadata as unknown; never waive a provided wrong seed')
    validate = commands.add_parser('validate-workbook')
    validate.add_argument('--packet-root', type=Path, required=True)
    validate.add_argument('--workbook', type=Path, required=True)
    validate.add_argument('--rater', choices=RATERS, required=True)
    validate.add_argument('--final', action='store_true')
    validate.add_argument('--report', type=Path, required=True)
    validate.add_argument('--export-csv', type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.cases, args.cohort, args.out)
    elif args.command == 'verify':
        result = verify_packets(args.packet_root)
    elif args.command == 'upgrade-web':
        result = upgrade_web(args.packet_root, args.release_dir)
    elif args.command == 'complete-d':
        result = complete_d(args.packet_root, args.runs, args.patch_out, args.native_frame_deviation_reason, args.unverified_source_reason)
    else:
        if args.report.exists() or (args.export_csv and args.export_csv.exists()):
            parser.error('Use new report/export paths; preserve previous submissions')
        state = json.loads((args.packet_root / 'organizer_private/state_current.json').read_text(encoding='utf-8'))
        result, rows = workbook_values(args.workbook, state, args.rater, args.final)
        write_json(args.report, result)
        if args.export_csv and result['valid']:
            write_table(args.export_csv, RATING_FIELDS, rows)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get('valid', True) else 1


if __name__ == '__main__':
    raise SystemExit(main())
