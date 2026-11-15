#!/usr/bin/env python3
"""One fixed-seed native LTX clip per case; no automatic retries or rerolls."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from common import (MODEL_ID, MODEL_REVISION, TEXT_REVISION, UPSTREAM_COMMIT,
                    WEIGHT_FILES, digest, object_digest, read_manifest_snapshot, select_cases,
                    verify_assets, write_json)
from memory_adapter import STRATEGY as MEMORY_STRATEGY, install as install_memory_adapter

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE / "ltxv-13b-0.9.8-no-enhance.json"
CODE_FILES = {
    "runner": Path(__file__).resolve(), "common": HERE / "common.py",
    "memory_adapter": HERE / "memory_adapter.py",
    "dataset_contract": HERE.parents[1] / "ai6132_benchmark/dataset.py",
}
SOURCE_FILES = {**CODE_FILES, "pipeline_template": CONFIG_FILE}
CSV_FIELDS = [
    "case_id", "category", "model_id", "phase", "seed", "status", "failure_reason",
    "weight_version", "weight_revision", "text_encoder_revision", "upstream_commit",
    "config", "config_sha256", "case_fingerprint", "image_sha256",
    "native_video", "video_sha256", "native_width", "native_height", "native_frames",
    "native_fps", "native_duration_seconds", "native_codec", "elapsed_seconds",
    "worker_wall_seconds", "peak_vram_allocated_mib", "peak_vram_reserved_mib",
    "gpu_model", "gpu_uuid", "gpu_total_memory_mib", "cuda_visible_devices",
    "driver_version", "torch_version", "torch_cuda_version", "cudnn_version", "python_version",
    "offload_requested", "offload_effective", "started_at", "finished_at", "log_path",
    "target_frames", "source_frame_num",
    "memory_strategy", "memory_adapter_sha256", "memory_audit", "memory_audit_sha256",
    "conditioning_strength", "image_cond_noise_scale", "conditioning_start_frame",
    "source_receipt_sha256",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def verify_upstream(path: Path):
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f"Expected upstream commit {UPSTREAM_COMMIT}; found {commit}")
    changed = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"], text=True)
    if changed.strip():
        raise ValueError("Upstream has tracked modifications; use the clean pinned checkout")
    import yaml
    official = yaml.safe_load((path / "configs/ltxv-13b-0.9.8-distilled.yaml").read_text())
    official["prompt_enhancement_words_threshold"] = 0
    ours = json.loads(CONFIG_FILE.read_text())
    if ours != official:
        raise ValueError("Local pipeline must equal the official two-stage configuration except prompt enhancement=0")


def verify_worker_code(payload: dict) -> None:
    expected = payload.get("code_sha256", {})
    if set(expected) != set(CODE_FILES):
        raise ValueError("Worker request is missing its frozen code hashes")
    for name, path in CODE_FILES.items():
        if digest(path) != expected[name]:
            raise ValueError(f"Frozen worker code changed: {name}")
    if payload["record"].get("memory_adapter_sha256") != expected["memory_adapter"]:
        raise ValueError("Worker adapter hash differs from its run record")
    if payload["record"].get("model_id") != MODEL_ID:
        raise ValueError("Worker request is not for the selected 13B model")
    output = Path(payload["batch_root"])
    receipt_path = output / "source_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt != payload.get("source_receipt") or digest(receipt_path) != payload["record"].get("source_receipt_sha256"):
        raise ValueError("Frozen source receipt changed before worker start")
    verify_source_receipt(output, receipt)
    for name, sha in expected.items():
        if receipt["files"][name]["sha256"] != sha:
            raise ValueError(f"Frozen source receipt differs from worker code: {name}")
    if digest(Path(payload["config"])) != payload["record"].get("config_sha256"):
        raise ValueError("Frozen pipeline configuration changed before worker start")


def snapshot_sources() -> tuple[dict, dict[str, bytes]]:
    snapshots = {name: path.read_bytes() for name, path in SOURCE_FILES.items()}
    receipt = {"schema_version": 1, "files": {
        name: {"path": "source/" + SOURCE_FILES[name].relative_to(HERE.parents[1]).as_posix(),
               "sha256": hashlib.sha256(content).hexdigest()}
        for name, content in snapshots.items()
    }}
    return receipt, snapshots


def verify_source_receipt(output: Path, receipt: dict) -> None:
    if receipt.get("schema_version") != 1 or set(receipt.get("files", {})) != set(SOURCE_FILES):
        raise ValueError("Frozen source receipt has an unexpected file set")
    for name, path in SOURCE_FILES.items():
        item = receipt["files"][name]
        relative = "source/" + path.relative_to(HERE.parents[1]).as_posix()
        if item.get("path") != relative or digest(output / relative) != item.get("sha256"):
            raise ValueError(f"Frozen source snapshot missing or changed: {name}")


def save_source_snapshot(output: Path, receipt: dict, snapshots: dict[str, bytes], *, resume: bool) -> None:
    receipt_path = output / "source_receipt.json"
    if resume:
        if not receipt_path.is_file() or json.loads(receipt_path.read_text()) != receipt:
            raise ValueError("Frozen source receipt missing or changed")
        verify_source_receipt(output, receipt)
        return
    for name, content in snapshots.items():
        destination = output / receipt["files"][name]["path"]
        if destination.exists():
            if digest(destination) != receipt["files"][name]["sha256"]:
                raise ValueError(f"Existing source snapshot differs: {name}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(content)
    if receipt_path.exists() and json.loads(receipt_path.read_text()) != receipt:
        raise ValueError("Existing source receipt differs")
    write_json(receipt_path, receipt)
    verify_source_receipt(output, receipt)


def inference_arguments(payload: dict) -> dict:
    """Keep conditioning controls fixed and use this canonical case's frame count."""
    case, settings = payload["case"], payload["settings"]
    fixed = {"memory_strategy": MEMORY_STRATEGY, "conditioning_strength": 1.0,
             "image_cond_noise_scale": 0.15, "conditioning_start_frame": 0}
    if any(settings.get(key) != value for key, value in fixed.items()):
        raise ValueError("Worker conditioning/memory settings differ from the frozen 13B protocol")
    if settings["num_frames"] != case["target_frames"]:
        raise ValueError("Worker frame count differs from the canonical case")
    return {
        "prompt": case["prompt"], "pipeline_config": payload["config"],
        "output_path": str(Path(payload["record"]["log_path"]).parent / "native"),
        "seed": case["seed"], "height": settings["height"], "width": settings["width"],
        "num_frames": settings["num_frames"], "frame_rate": settings["frame_rate"],
        "offload_to_cpu": settings["offload_to_cpu"],
        "conditioning_media_paths": [case["image_path"]],
        "conditioning_start_frames": [settings["conditioning_start_frame"]],
        "conditioning_strengths": [settings["conditioning_strength"]],
        "image_cond_noise_scale": settings["image_cond_noise_scale"],
    }


def inspect_video(path: Path) -> dict:
    """Decode every frame; metadata-only checks would miss truncated files."""
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        if fps <= 0:
            raise ValueError("Output has no valid average frame rate")
        codec = stream.codec_context.name
        frames, sizes = 0, set()
        for frame in container.decode(stream):
            frames += 1
            sizes.add((frame.width, frame.height))
        if not frames or len(sizes) != 1:
            raise ValueError("Video has no frames or inconsistent frame dimensions")
        width, height = next(iter(sizes))
        return {"native_width": width, "native_height": height, "native_frames": frames,
                "native_fps": fps, "native_duration_seconds": frames / fps, "native_codec": codec}


def worker(payload_path: Path) -> int:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    case_dir = payload_path.parent
    case, settings = payload["case"], payload["settings"]
    result = dict(payload["record"])
    wall_start, infer_start, torch = time.perf_counter(), None, None
    adapter_handle = None
    audit_path = case_dir / "memory_audit.json"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        verify_worker_code(payload)
        infer_kwargs = inference_arguments(payload)
        if digest(Path(case["image_path"])) != case["image_sha256"]:
            raise ValueError("Shared first image changed after the batch was frozen")
        from PIL import Image
        with Image.open(case["image_path"]) as first_image:
            first_image.verify()
        with Image.open(case["image_path"]) as first_image:
            if first_image.width * settings["height"] != first_image.height * settings["width"]:
                raise ValueError("Shared first-image aspect ratio differs from output; ask A to freeze a common crop")
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("No allocated CUDA GPU; CPU/MPS fallback is disabled for this benchmark runner")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Allocated GPU does not support the frozen BF16 configuration")
        props = torch.cuda.get_device_properties(0)
        result.update({
            "gpu_model": props.name, "gpu_uuid": str(getattr(props, "uuid", "")),
            "gpu_total_memory_mib": props.total_memory / 1024**2,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "python_version": platform.python_version(),
            "offload_effective": False,
            "memory_strategy": MEMORY_STRATEGY,
        })
        try:
            result["driver_version"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True, timeout=10).splitlines()[0].strip()
        except (OSError, subprocess.SubprocessError, IndexError):
            result["driver_version"] = "unavailable"
        sys.path.insert(0, payload["upstream"])
        from ltx_video.inference import InferenceConfig, infer
        # install only registers the factory patch; all loading/encoding happens
        # inside infer and is included in the measured time and CUDA peak.
        adapter_handle = install_memory_adapter(case["prompt"], audit_path)
        # Every case starts a separate process: all loading, encoding and generation
        # allocations are included; no model state survives into the next sample.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        infer_start = time.perf_counter()
        infer(InferenceConfig(**infer_kwargs))
        torch.cuda.synchronize()
        result["elapsed_seconds"] = time.perf_counter() - infer_start
        files = list((case_dir / "native").glob("*.mp4"))
        if len(files) != 1:
            raise ValueError(f"Expected exactly one native MP4, found {len(files)}")
        video = files[0]
        result.update({"native_video": str(video), "video_sha256": digest(video)})
        result.update(inspect_video(video))
        actual = (result["native_width"], result["native_height"], result["native_frames"], result["native_fps"])
        expected = (settings["width"], settings["height"], settings["num_frames"], settings["frame_rate"])
        if actual != expected:
            raise ValueError(f"Native output specification mismatch: expected {expected}, got {actual}")
        result["status"], result["failure_reason"] = "success", ""
    except BaseException as exc:
        result["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "technical_failure"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        if adapter_handle is not None:
            try:
                adapter_handle.restore()
            except BaseException as exc:
                result["status"] = "technical_failure"
                result["failure_reason"] = f"Adapter audit finalization failed: {type(exc).__name__}: {exc}"
                traceback.print_exc()
        if audit_path.is_file():
            result.update(memory_audit=str(audit_path), memory_audit_sha256=digest(audit_path))
        if infer_start is not None and "elapsed_seconds" not in result:
            result["elapsed_seconds"] = time.perf_counter() - infer_start
        if torch is not None and torch.cuda.is_available():
            try:
                result["peak_vram_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
                result["peak_vram_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
            except RuntimeError:
                pass
        result["worker_wall_seconds"] = time.perf_counter() - wall_start
        result["finished_at"] = utc_now()
        write_json(case_dir / "result.json", result)
    return 0 if result["status"] == "success" else 1


def unstarted_record(case: dict, frozen: dict, output: Path) -> dict:
    config_path = output / "pipeline.json"
    model_id = frozen["model_id"]
    settings = frozen["settings"]
    return {"case_id": case["case_id"], "category": case["category"], "model_id": model_id,
            "phase": frozen["phase"], "seed": case["seed"], "status": "not_started",
            "failure_reason": "Not started in the selected batch",
            "weight_version": model_id, "weight_revision": MODEL_REVISION,
            "text_encoder_revision": TEXT_REVISION, "upstream_commit": UPSTREAM_COMMIT,
            "config": str(config_path), "config_sha256": digest(config_path),
            "case_fingerprint": object_digest({"case": case, "frozen": frozen}),
            "image_sha256": case["image_sha256"],
            "offload_requested": frozen["settings"]["offload_to_cpu"],
            "memory_strategy": settings.get("memory_strategy", ""),
            "memory_adapter_sha256": frozen.get("memory_adapter_sha256", ""),
            "memory_audit": (str(output / "cases" / case["case_id"] / "memory_audit.json")
                             if settings.get("memory_strategy") else ""),
            "conditioning_strength": settings.get("conditioning_strength", ""),
            "image_cond_noise_scale": settings.get("image_cond_noise_scale", ""),
            "conditioning_start_frame": settings.get("conditioning_start_frame", ""),
            "source_receipt_sha256": (digest(output / "source_receipt.json")
                                      if frozen.get("source_receipt") else ""),
            "target_frames": case['target_frames'],
            "source_frame_num": case['source_frame_num'],
            "log_path": str(output / "cases" / case["case_id"] / "inference.log")}


def records(output: Path) -> list[dict]:
    result = []
    for case_dir in sorted((output / "cases").glob("*")):
        completed, started = case_dir / "result.json", case_dir / "request.json"
        if completed.exists():
            result.append(json.loads(completed.read_text(encoding="utf-8")))
        elif started.exists():
            row = json.loads(started.read_text(encoding="utf-8"))["record"]
            row.update(status="incomplete", failure_reason="No terminal result: process may still run or was interrupted; never automatically rerun")
            result.append(row)
    selection_path = output / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    frozen = json.loads((output / "batch.json").read_text(encoding="utf-8"))
    selected = selection['case_ids']
    if selection != selection_receipt(frozen):
        raise ValueError('Selection receipt must match the complete canonical batch')
    cases = {case["case_id"]: case for case in frozen["cases"]}
    attempts = {row["case_id"]: row for row in result}
    if set(attempts) - set(selected):
        raise ValueError("Attempted cases are outside selection.json; refusing an incomplete handoff")
    return [attempts[case_id] if case_id in attempts else unstarted_record(cases[case_id], frozen, output)
            for case_id in selected]


def selection_receipt(frozen: dict) -> dict:
    return {'schema_version': 2, 'protocol': 'a-mixed-24-v1', 'phase': frozen['phase'],
            'cases_csv_sha256': frozen['cases_csv_sha256'],
            'case_ids': [case['case_id'] for case in frozen['cases']]}


def export_csv(output: Path):
    temporary = output / "runs.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records(output))
    temporary.replace(output / "runs.csv")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--image-root", type=Path, help="Default: cases.csv directory; never modifies shared images")
    p.add_argument("--phase", required=True, choices=["smoke", "pilot", "formal"],
                   help="smoke: 1 canonical diagnostic; pilot: 3 independent short/long cases; formal: all 24 A cases")
    p.add_argument("--validate-only", action="store_true", help="Validate manifest and files without GPU, weights or output writes")
    p.add_argument("--upstream", type=Path)
    p.add_argument("--assets", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=432)
    p.add_argument("--frame-rate", type=int, default=24)
    p.add_argument("--offload-to-cpu", action="store_true",
                   help="Record the request; the verified adapter uses sequential loading/final release, not CPU offload")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    settings = {"width": args.width, "height": args.height,
                "frame_rate": args.frame_rate, "offload_to_cpu": args.offload_to_cpu,
                "memory_strategy": MEMORY_STRATEGY, "conditioning_strength": 1.0,
                "image_cond_noise_scale": 0.15, "conditioning_start_frame": 0}
    if min(args.width, args.height, args.frame_rate) <= 0 or args.width * 9 != args.height * 16:
        raise ValueError('Use positive dimensions/fps and landscape 16:9')
    cases_path = args.cases.expanduser().resolve()
    all_cases, cases_csv_sha256 = read_manifest_snapshot(cases_path, args.image_root)
    chosen = select_cases(all_cases, args.phase)
    print(json.dumps({"phase": args.phase, "selected": len(chosen), "settings": settings,
                      "case_ids": [row["case_id"] for row in chosen]}, ensure_ascii=False))
    if args.validate_only:
        print("Manifest validation passed; no model inference or output generation performed.")
        return 0
    if not all([args.upstream, args.assets, args.output]):
        raise ValueError("Execution requires --upstream, --assets and --output")
    upstream, assets, output = [p.expanduser().resolve() for p in (args.upstream, args.assets, args.output)]
    verify_upstream(upstream)
    asset_manifest = verify_assets(assets)
    pipeline = json.loads(CONFIG_FILE.read_text())
    pipeline["checkpoint_path"] = str(assets / "ltx" / WEIGHT_FILES[0])
    pipeline["spatial_upscaler_model_path"] = str(assets / "ltx" / WEIGHT_FILES[1])
    pipeline["text_encoder_model_name_or_path"] = str(assets / "text")
    dependency_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    source_receipt, source_snapshots = snapshot_sources()
    code_sha256 = {name: source_receipt["files"][name]["sha256"] for name in CODE_FILES}
    frozen = {"schema_version": 3, "phase": args.phase, "model_id": MODEL_ID,
              "upstream_commit": UPSTREAM_COMMIT, "asset_manifest": asset_manifest,
              "cases_csv_sha256": cases_csv_sha256, "cases": all_cases,
              "dependency_freeze": dependency_freeze, "python_version": platform.python_version(),
              "runner_sha256": code_sha256["runner"], "common_sha256": code_sha256["common"],
              "memory_adapter_sha256": code_sha256["memory_adapter"],
              "dataset_contract_sha256": code_sha256["dataset_contract"],
              "source_receipt": source_receipt,
              "settings": settings, "pipeline": pipeline}
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another runner owns this output directory") from exc
        frozen_path = output / "batch.json"
        resume = frozen_path.exists()
        if frozen_path.exists():
            if json.loads(frozen_path.read_text(encoding="utf-8")) != frozen:
                raise ValueError("Frozen batch changed (manifest, images, prompt, seed, settings or weights). Do not mix configurations or reroll formal cases")
        else:
            if (output / "cases").exists() or (output / "runs.csv").exists():
                raise ValueError("Output has prior artifacts but no batch.json; refusing to overwrite")
        save_source_snapshot(output, source_receipt, source_snapshots, resume=resume)
        if not resume:
            write_json(frozen_path, frozen)
            (output / "environment.txt").write_text(
                f"Python: {sys.version}\nPlatform: {platform.platform()}\n\n" +
                dependency_freeze, encoding="utf-8")
        config_path = output / "pipeline.json"
        if config_path.exists() and json.loads(config_path.read_text()) != pipeline:
            raise ValueError("Saved pipeline configuration was changed")
        write_json(config_path, pipeline)
        selection_path = output / "selection.json"
        selected_ids = {row["case_id"] for row in chosen}
        selection = selection_receipt(frozen)
        if selection_path.exists():
            previous = json.loads(selection_path.read_text(encoding="utf-8"))
            if previous != selection:
                raise ValueError('Selection receipt must match the complete canonical batch')
        existing_ids = {path.name for path in (output / "cases").glob("*") if path.is_dir()}
        if not existing_ids <= selected_ids:
            raise ValueError("Existing attempts are outside the selected batch; do not shrink or mix batches")
        write_json(selection_path, selection)
        # Publish the complete selected denominator before launching any worker.
        # No case directories are created for not_started rows, so resume can start them.
        export_csv(output)
        interrupted = False
        try:
            for case in chosen:
                case_dir = output / "cases" / case["case_id"]
                if case_dir.exists():
                    result_path = case_dir / "result.json"
                    if result_path.exists():
                        old = json.loads(result_path.read_text())
                        if old.get("status") == "success":
                            video = Path(old["native_video"])
                            if not video.is_file() or digest(video) != old["video_sha256"]:
                                raise ValueError(f"Successful output missing/changed for {case['case_id']}; refusing regeneration")
                        print(f"SKIP {case['case_id']}: {old.get('status')}; no automatic retries", flush=True)
                    else:
                        if not (case_dir / "request.json").exists():
                            raise ValueError(f"Orphaned case directory without a request/result: {case_dir}; inspect before continuing")
                        print(f"SKIP {case['case_id']}: existing incomplete attempt; inspect its log", flush=True)
                    continue
                case_dir.mkdir(parents=True)
                record = unstarted_record(case, frozen, output)
                record.update(status="running", failure_reason="", started_at=utc_now())
                case_settings = dict(settings)
                case_settings['num_frames'] = case['target_frames']
                payload = {"case": case, "settings": case_settings, "upstream": str(upstream),
                           "config": str(config_path), "record": record, "code_sha256": code_sha256,
                           "batch_root": str(output), "source_receipt": source_receipt}
                request = case_dir / "request.json"
                write_json(request, payload)
                export_csv(output)
                print(f"RUN {case['case_id']} seed={case['seed']}", flush=True)
                with (case_dir / "inference.log").open("x", encoding="utf-8") as log:
                    code = subprocess.call([sys.executable, str(Path(__file__).resolve()), "--worker-json", str(request)],
                                           stdout=log, stderr=subprocess.STDOUT)
                if not (case_dir / "result.json").exists():
                    record.update(status="interrupted", failure_reason=f"Worker exited {code} without terminal metrics", finished_at=utc_now())
                    write_json(case_dir / "result.json", record)
                export_csv(output)
        except KeyboardInterrupt:
            interrupted = True
            print("Interrupted. Existing attempts are retained and will not be generated again.", file=sys.stderr)
        finally:
            export_csv(output)
        finished = {row["case_id"] for row in records(output) if row["case_id"] in selected_ids and row["status"] == "success"}
        print(f"Successful native clips: {len(finished)}/{len(chosen)}. Manifest: {output / 'runs.csv'}")
        return 0 if finished == selected_ids and not interrupted else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker-json":
        sys.exit(worker(Path(sys.argv[2])))
    try:
        sys.exit(main())
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
