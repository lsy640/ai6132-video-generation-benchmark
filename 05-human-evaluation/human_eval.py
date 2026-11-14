#!/usr/bin/env python3
"""Prepare anonymous review packets and validate independent rating submissions."""

import argparse
import csv
import hashlib
import io
import json
import random
import re
import shutil
import sys
from pathlib import Path

MANIFEST_FIELDS = ["clip_id", "video_path", "image_path", "prompt", "status", "failure_reason"]
SCORES = ["reference_fidelity", "motion_naturalness", "prompt_adherence"]
RATING_FIELDS = ["rater_id", "clip_id", *SCORES, "comment"]
ORDER_FIELDS = ["order", "clip_id", "video_path", "image_path", "prompt", "status", "failure_reason"]
FAILURE_LABELS = {
    "generation_failed": "技术失败：未能生成视频",
    "decode_failed": "技术失败：视频无法解码",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
FAILURE_CODES = {
    "generation_failed": {"out_of_memory", "timeout", "generation_error", "missing_output", "other"},
    "decode_failed": {"decode_error", "empty_video", "other"},
}

INSTRUCTIONS = """AI6132 四人独立人工盲评

先用三条试跑视频校准量表，校准分数不并入正式评分。
正式评审按 order.csv 的顺序观看视频，同时查看对应首图与原始英文 prompt。
在本人的 ratings.csv 填写三个维度的整数 1–5 分；2、4 是相邻锚点的中间水平。
同分允许。prompt 未指定动作时，按动作与画面情境是否相容判断。

首图保持：
1：主要主体或场景明显改变，难以辨认为同一对象或场景。
3：关键特征大体保留，有局部变形或明显漂移。
5：主体身份与场景主要特征始终可辨，变化与动作或视角相容。

动作自然度：
1：明显卡顿、跳变、肢体变形，或应有动作基本缺失。
3：动作可辨，有短暂顿挫、重复或局部不自然。
5：动作连贯，速度和变化符合情境，未见明显跳变。

提示词遵循：
1：主要指令缺失或明显违背。
3：完成部分要求，关键细节或动作不完整。
5：主要主体、动作与场景要求均能在视频中确认。

可播放但质量差的视频仍需正常评分；不要把低质量当作技术失败。
generation_failed 或 decode_failed：三个分数全部留空，保留技术失败备注。
低分可在 comment 中说明可观察到的问题；不要猜测或填写模型名称。
视频和首图路径相对于本文件所在文件夹。

每位成员独立提交原始 ratings.csv。全部交卷前不查看他人的分数或模型映射，
不讨论具体视频的评分。发现漏填时由原评审本人补齐；不要替其他人补分。
CSV 用 UTF-8 编码；用表格软件导入时将文本列设为文本，保留 clip_id 和字段名。
"""


class InputError(ValueError):
    pass


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_packet(frozen, packet_root, rater=None):
    """Check retained or received files against a trusted preparation receipt."""
    if not isinstance(frozen, dict) or frozen.get("tool_version") != 2:
        raise InputError("需要 v2 preparation.json；旧评分包请保留原件，在新目录重新准备并核对素材")
    raters = frozen.get("raters")
    if not isinstance(raters, list):
        raise InputError("preparation.json 缺少评审列表")
    validate_raters(raters)
    if rater is not None and rater not in raters:
        raise InputError(f"未知 rater_id：{rater}")
    hashes = frozen.get("immutable_files_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise InputError("preparation.json 缺少评分包文件哈希")
    required = {f"raters/{name}/{filename}" for name in raters
                for filename in ("order.csv", "评分说明.txt")}
    if not required <= hashes.keys() or not any(name.startswith("assets/") for name in hashes):
        raise InputError("preparation.json 的评分包文件清单不完整")
    root = Path(packet_root).resolve()
    checked = 0
    for relative, expected in hashes.items():
        path = Path(relative)
        if (path.is_absolute() or ".." in path.parts or
                not path.parts or path.parts[0] not in {"assets", "raters"} or
                path.as_posix() != relative or
                (path.parts[0] == "assets" and len(path.parts) != 2) or
                (path.parts[0] == "raters" and
                 (len(path.parts) != 3 or path.parts[1] not in raters or
                  path.parts[2] not in {"order.csv", "评分说明.txt"})) or
                not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)):
            raise InputError(f"preparation.json 中有无效的冻结文件：{relative}")
        if rater is not None and path.parts[0] == "raters" and path.parts[1] != rater:
            continue
        target = (root / path).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise InputError(f"评分包文件缺失或越界：{relative}")
        if file_sha256(target) != expected:
            raise InputError(f"评分包文件已变化：{relative}")
        checked += 1
    return {"valid": True, "packet_verified": True, "checked_files": checked,
            "rater_id": rater, "packet_root": str(root)}


def read_csv(path, fields, snapshot=None):
    """Reject hidden fields and structurally damaged CSV before processing rows."""
    if snapshot is None:
        snapshot = Path(path).read_bytes()
    with io.StringIO(snapshot.decode("utf-8-sig"), newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames != fields:
            raise InputError(f"{path}: 表头必须按顺序为 {','.join(fields)}")
        rows = []
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise InputError(f"{path}: 第 {line} 行列数不匹配")
            rows.append(row)
    return rows


def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path, check_files=False, snapshot=None):
    path = Path(path)
    rows = read_csv(path, MANIFEST_FIELDS, snapshot=snapshot)
    if not rows:
        raise InputError("匿名清单为空；请等待 B 交付正式匿名视频清单。")
    ids = set()
    for row in rows:
        clip = row["clip_id"]
        if not re.fullmatch(r"clip_[0-9]{4,}", clip):
            raise InputError(f"clip_id 必须采用 clip_0001 这类中性编号：{clip!r}")
        if clip in ids:
            raise InputError(f"清单有重复 clip_id：{clip}")
        ids.add(clip)
        if row["status"] not in {"success", *FAILURE_LABELS}:
            raise InputError(f"{clip}: status 必须是 success / generation_failed / decode_failed")
        if not row["prompt"].strip() or not row["image_path"].strip():
            raise InputError(f"{clip}: 首图路径和原始 prompt 不能为空")
        if row["status"] == "success":
            if row["video_path"] != f"videos/{clip}.mp4":
                raise InputError(f"{clip}: 成功视频须先由 B 匿名为 videos/{clip}.mp4")
            if row["failure_reason"].strip():
                raise InputError(f"{clip}: 成功视频不能带技术失败原因")
        else:
            if row["video_path"]:
                raise InputError(f"{clip}: 技术失败不分发视频，请将 video_path 留空")
            if row["failure_reason"] not in FAILURE_CODES[row["status"]]:
                raise InputError(f"{clip}: 技术失败须用允许的中性原因代码，详细日志另行保管")
        image = (path.parent / row["image_path"]).resolve()
        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            raise InputError(f"{clip}: 首图格式须是 PNG/JPG/JPEG/WebP")
        if row["image_path"] != f"images/{clip}{image.suffix.lower()}":
            raise InputError(f"{clip}: 首图须先由 B 匿名为 images/{clip}.<图片扩展名>")
        if check_files:
            if not image.is_file():
                raise InputError(f"{clip}: 首图文件不存在：{image}")
            if row["status"] == "success":
                video = (path.parent / row["video_path"]).resolve()
                if video.suffix.lower() != ".mp4" or not video.is_file():
                    raise InputError(f"{clip}: 成功视频必须指向存在的 MP4 评分副本")
    return rows


def validate_raters(raters):
    if len(raters) != 4 or len(set(raters)) != 4:
        raise InputError("须提供四个互不重复的 rater_id")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", rater) for rater in raters):
        raise InputError("rater_id 仅支持 1–32 个字母、数字、下划线或连字符")


def rater_seed(seed, rater):
    data = f"AI6132-human-evaluation-v1\0{seed}\0{rater}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest(), "big")


def prepare(manifest, out, raters, seed, expected_clips, phase):
    validate_raters(raters)
    manifest = Path(manifest)
    manifest_snapshot = manifest.read_bytes()
    rows = load_manifest(manifest, check_files=True, snapshot=manifest_snapshot)
    if len(rows) != expected_clips:
        raise InputError(f"清单有 {len(rows)} 条，预期 {expected_clips} 条；请先核对批次")
    if phase == "formal" and expected_clips not in (48, 96):
        raise InputError("正式批次须为 48 或 96 条；校准请显式选择 --phase calibration")
    if phase == "calibration" and expected_clips != 3:
        raise InputError("按计划，校准批次须为 3 条")
    out = Path(out)
    if out.exists():
        raise InputError(f"输出目录已存在，拒绝覆盖原始评分或已冻结顺序：{out}")
    # Validate all inputs before creating output. Canonical IDs make row ordering
    # irrelevant to seeded permutations; input bytes are still hashed in the audit.
    canonical = sorted(rows, key=lambda row: row["clip_id"])
    out.mkdir(parents=True)
    assets = out / "assets"
    assets.mkdir()
    public = {}
    for row in canonical:
        clip = row["clip_id"]
        image = (manifest.parent / row["image_path"]).resolve()
        image_name = f"{clip}_reference{image.suffix.lower()}"
        shutil.copyfile(image, assets / image_name)
        video_name = ""
        if row["status"] == "success":
            video_name = f"{clip}.mp4"
            shutil.copyfile((manifest.parent / row["video_path"]).resolve(), assets / video_name)
        public[clip] = {
            "clip_id": clip,
            "video_path": f"../../assets/{video_name}" if video_name else "",
            "image_path": f"../../assets/{image_name}",
            "prompt": row["prompt"],
            "status": row["status"],
            # Detailed errors can reveal the generating model. Do not distribute them.
            "failure_reason": FAILURE_LABELS.get(row["status"], ""),
        }
    metadata = {
        "tool_version": 2,
        "python_version": sys.version.split()[0],
        "phase": phase,
        "seed": seed,
        "expected_clips": expected_clips,
        "manifest_sha256": hashlib.sha256(manifest_snapshot).hexdigest(),
        "raters": raters,
        "immutable_files_sha256": {},
    }
    for rater in raters:
        directory = out / "raters" / rater
        directory.mkdir(parents=True)
        order = list(public)
        random.Random(rater_seed(seed, rater)).shuffle(order)
        write_csv(directory / "order.csv", ORDER_FIELDS,
                  [{"order": index, **public[clip]} for index, clip in enumerate(order, 1)])
        write_csv(directory / "ratings.csv", RATING_FIELDS,
                  [{"rater_id": rater, "clip_id": clip,
                    **dict.fromkeys(SCORES, ""), "comment": public[clip]["failure_reason"]}
                   for clip in order])
        (directory / "评分说明.txt").write_text(INSTRUCTIONS, encoding="utf-8")
    immutable = list(assets.iterdir())
    immutable.extend(out / "raters" / rater / filename for rater in raters
                     for filename in ("order.csv", "评分说明.txt"))
    metadata["immutable_files_sha256"] = {
        path.relative_to(out).as_posix(): file_sha256(path) for path in sorted(immutable)
    }
    (out / "preparation.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def validate_submissions(manifest, submissions, raters, manifest_snapshot=None):
    validate_raters(raters)
    expected = {row["clip_id"]: row for row in load_manifest(manifest, snapshot=manifest_snapshot)}
    errors = []
    summary = {}
    if set(submissions) != set(raters):
        raise InputError("提交文件必须与四个指定 rater_id 一一对应")
    paths = [Path(submissions[rater]).resolve() for rater in raters]
    if len(set(paths)) != 4:
        raise InputError("四位评审须提交四个独立文件")
    for rater in raters:
        try:
            rows = read_csv(submissions[rater], RATING_FIELDS)
        except (InputError, OSError, UnicodeError, csv.Error) as exc:
            errors.append({"rater_id": rater, "code": "invalid_file", "detail": str(exc)})
            summary[rater] = {"rows": 0, "unique_expected_clips": 0}
            continue
        seen = set()
        for line, row in enumerate(rows, 2):
            clip = row["clip_id"]
            base = {"rater_id": rater, "clip_id": clip, "line": line}
            if row["rater_id"] != rater:
                errors.append({**base, "code": "wrong_rater"})
            if clip in seen:
                errors.append({**base, "code": "duplicate_clip"})
            seen.add(clip)
            if clip not in expected:
                errors.append({**base, "code": "unknown_clip"})
                continue
            if expected[clip]["status"] == "success":
                for field in SCORES:
                    value = row[field].strip()
                    if not value:
                        errors.append({**base, "code": "missing_score", "field": field})
                    elif value not in {"1", "2", "3", "4", "5"}:
                        errors.append({**base, "code": "invalid_score", "field": field})
            else:
                for field in SCORES:
                    if row[field].strip():
                        errors.append({**base, "code": "failed_clip_has_score", "field": field})
                if not row["comment"].strip():
                    errors.append({**base, "code": "missing_failure_reason"})
        for clip in sorted(set(expected) - seen):
            errors.append({"rater_id": rater, "clip_id": clip, "code": "missing_clip"})
        summary[rater] = {"rows": len(rows), "unique_expected_clips": len(seen & set(expected))}
    return {
        "valid": not errors,
        "expected_clips_per_rater": len(expected),
        "expected_total_records": len(expected) * 4,
        "summary": summary,
        "errors": errors,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="生成四份空评分表及独立随机顺序")
    prep.add_argument("--manifest", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--raters", nargs=4, default=["A", "B", "C", "D"])
    prep.add_argument("--seed", type=int, required=True)
    prep.add_argument("--expected-clips", type=int, required=True)
    prep.add_argument("--phase", choices=["formal", "calibration"], default="formal")
    check = commands.add_parser("validate", help="只校验，不修改任何原始评分")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--raters", nargs=4, default=["A", "B", "C", "D"])
    check.add_argument("--ratings", nargs=4, required=True, metavar="RATER=FILE")
    check.add_argument("--preparation", type=Path, required=True,
                       help="prepare 生成的 preparation.json，防止换用另一批清单")
    check.add_argument("--packet-root", type=Path,
                       help="完整评分包根目录；默认 preparation.json 所在目录")
    check.add_argument("--report", type=Path)
    packet = commands.add_parser("verify-packet", help="交接时核对实际收到的素材、顺序和说明")
    packet.add_argument("--preparation", type=Path, required=True, help="由准备方保留的可信 v2 记录")
    packet.add_argument("--packet-root", type=Path, required=True)
    packet.add_argument("--rater", help="只核对指定评审的目录和所有共享素材")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare(args.manifest, args.out, args.raters, args.seed,
                             args.expected_clips, args.phase)
            print(json.dumps({"output": str(args.out), **result}, ensure_ascii=False, indent=2))
            return 0
        frozen = json.loads(args.preparation.read_text(encoding="utf-8"))
        packet_result = verify_packet(frozen, args.packet_root or args.preparation.parent,
                                      args.rater if args.command == "verify-packet" else None)
        if args.command == "verify-packet":
            print(json.dumps(packet_result, ensure_ascii=False, indent=2))
            return 0
        submissions = {}
        for entry in args.ratings:
            rater, sep, filename = entry.partition("=")
            if not sep or not filename or rater in submissions:
                raise InputError("--ratings 使用四个不同的 RATER=FILE")
            submissions[rater] = filename
        manifest_snapshot = args.manifest.read_bytes()
        if frozen.get("manifest_sha256") != hashlib.sha256(manifest_snapshot).hexdigest():
            raise InputError("匿名清单与准备评分表时的文件不同；不要在正式评审中更换批次")
        if frozen.get("raters") != args.raters:
            raise InputError("rater_id 与 preparation.json 不一致")
        result = validate_submissions(args.manifest, submissions, args.raters, manifest_snapshot)
        result["packet_verified"] = True
        if frozen.get("expected_clips") != result["expected_clips_per_rater"]:
            raise InputError("preparation.json 中的预期视频数与清单不一致")
        content = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.report:
            # Never overwrite existing reports or any input submission.
            with args.report.open("x", encoding="utf-8") as stream:
                stream.write(content)
        print(content, end="")
        return 0 if result["valid"] else 1
    except (InputError, OSError, UnicodeError, csv.Error, json.JSONDecodeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
