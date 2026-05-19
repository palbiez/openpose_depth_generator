from __future__ import annotations

import argparse
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
SMPLIFYX_ROOT = PROJECT_ROOT / "smplify-x"
SMPLIFYX_MAIN = SMPLIFYX_ROOT / "smplifyx" / "main.py"
RENDER_RUNNER = PROJECT_ROOT / "render_blender_batch.py"
DEFAULT_PYTHON = Path("C:/Users/firew/Documents/python_scripts/venv312/Scripts/python.exe")
DEFAULT_BLENDER = Path("C:/Program Files/Blender Foundation/Blender 5.1/blender.exe")
DEFAULT_RESOLUTION = "512x768"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class CategoryConfig:
    name: str
    data_root: Path
    config_path: Path
    output_root: Path
    default_head_mode: str


@dataclass(frozen=True)
class PoseItem:
    category: CategoryConfig
    image_path: Path
    keypoint_path: Path
    pose_id: str

    @property
    def mesh_dir(self) -> Path:
        return self.category.output_root / "meshes" / self.pose_id

    @property
    def mesh_path(self) -> Path:
        return self.mesh_dir / "000.obj"


CATEGORIES = {
    "normal": CategoryConfig(
        name="normal",
        data_root=DATASET_ROOT / "poses_normal",
        config_path=SMPLIFYX_ROOT / "cfg_files" / "fit_smplx_normal.yaml",
        output_root=DATASET_ROOT / "poses_normal" / "smplx",
        default_head_mode="original",
    ),
    "complex": CategoryConfig(
        name="complex",
        data_root=DATASET_ROOT / "poses_complex",
        config_path=SMPLIFYX_ROOT / "cfg_files" / "fit_smplx_komplex.yaml",
        output_root=DATASET_ROOT / "poses_complex" / "smplx",
        default_head_mode="original",
    ),
}


def ensure_inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Refusing path outside {resolved_root}: {resolved}")
    return resolved


def safe_rmtree(path: Path, root: Path) -> None:
    resolved = ensure_inside(path, root)
    if resolved.exists():
        shutil.rmtree(resolved)


def unique_backup_path(path: Path) -> Path:
    if not path.exists():
        return path

    for index in range(2, 1000):
        candidate = path.with_name(f"{path.name}.{index}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Could not find unique backup path for {path}")


def remove_or_backup(path: Path, backup_root: Path | None, dataset_root: Path) -> str | None:
    resolved = ensure_inside(path, dataset_root)
    if not resolved.exists():
        return None

    if backup_root is not None:
        rel = resolved.relative_to(dataset_root.resolve())
        dst = unique_backup_path(backup_root / rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(resolved), str(dst))
        return str(dst)

    if resolved.is_dir():
        shutil.rmtree(resolved)
    else:
        resolved.unlink()
    return "deleted"


def render_output_paths(item: PoseItem, render_output_root: Path, passes: Sequence[str]) -> list[Path]:
    return [render_output_root / pass_name / f"{item.pose_id}.png" for pass_name in passes]


def clean_force_outputs(item: PoseItem, args: argparse.Namespace, backup_root: Path | None) -> list[dict[str, str]]:
    paths: list[Path] = []

    if args.force_fit and not args.skip_fit:
        paths.extend(
            [
                item.category.output_root / "meshes" / item.pose_id,
                item.category.output_root / "results" / item.pose_id,
                item.category.output_root / "images" / item.pose_id,
            ]
        )

    if args.force_render and not args.skip_render:
        paths.extend(render_output_paths(item, args.render_output_root, args.passes))

    cleaned: list[dict[str, str]] = []
    for path in paths:
        result = remove_or_backup(path, backup_root, DATASET_ROOT)
        if result is not None:
            cleaned.append({"path": str(path.resolve()), "result": result})

    return cleaned


def clean_render_pass_dirs(args: argparse.Namespace, backup_root: Path | None) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for pass_name in args.passes:
        path = args.render_output_root / pass_name
        result = remove_or_backup(path, backup_root, DATASET_ROOT)
        if result is not None:
            cleaned.append({"path": str(path.resolve()), "result": result})
        path.mkdir(parents=True, exist_ok=True)
    return cleaned


def parse_passes(raw: Sequence[str]) -> list[str]:
    passes: list[str] = []
    for value in raw:
        passes.extend(part.strip().lower() for part in value.split(",") if part.strip())

    valid = {"depth", "lineart", "normal"}
    invalid = sorted(set(passes) - valid)
    if invalid:
        raise ValueError(f"Unsupported render pass: {', '.join(invalid)}")

    out: list[str] = []
    for value in passes:
        if value not in out:
            out.append(value)
    return out or ["depth", "lineart", "normal"]


def render_outputs_exist(pose_id: str, output_root: Path, passes: Sequence[str]) -> bool:
    return all((output_root / pass_name / f"{pose_id}.png").exists() for pass_name in passes)


def discover_items(categories: Iterable[CategoryConfig]) -> tuple[list[PoseItem], list[dict[str, str]]]:
    items: list[PoseItem] = []
    missing: list[dict[str, str]] = []

    for category in categories:
        image_root = category.data_root / "images"
        keypoint_root = category.data_root / "keypoints"
        if not image_root.exists():
            missing.append({"category": category.name, "reason": "missing_image_root", "path": str(image_root)})
            continue

        for image_path in sorted(path for path in image_root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS):
            keypoint_path = keypoint_root / f"{image_path.stem}_keypoints.json"
            if not keypoint_path.exists():
                missing.append(
                    {
                        "category": category.name,
                        "reason": "missing_keypoints",
                        "image": str(image_path),
                        "expected_keypoints": str(keypoint_path),
                    }
                )
                continue
            items.append(
                PoseItem(
                    category=category,
                    image_path=image_path.resolve(),
                    keypoint_path=keypoint_path.resolve(),
                    pose_id=image_path.stem,
                )
            )

    return items, missing


def prepare_single_item_dataset(item: PoseItem, work_root: Path, force_clean: bool) -> Path:
    item_root = work_root / item.category.name / item.pose_id
    if force_clean:
        safe_rmtree(item_root, work_root)

    image_dir = item_root / "images"
    keypoint_dir = item_root / "keypoints"
    image_dir.mkdir(parents=True, exist_ok=True)
    keypoint_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(item.image_path, image_dir / item.image_path.name)
    shutil.copy2(item.keypoint_path, keypoint_dir / item.keypoint_path.name)
    return item_root


def run_command(
    cmd: Sequence[str],
    cwd: Path,
    log_path: Path,
    timeout: int | None,
    *,
    label: str,
    debug_output: bool,
    show_output: bool,
    heartbeat_seconds: float,
) -> tuple[int, float]:
    start = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write("$ " + " ".join(f'"{part}"' if " " in part else part for part in cmd) + "\n\n")
        log_file.flush()

        try:
            process = subprocess.Popen(
                list(cmd),
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except Exception as exc:
            log_file.write(f"\nEXCEPTION: {exc!r}\n")
            return 125, time.monotonic() - start

        output_queue: queue.Queue[str] = queue.Queue()

        def read_output() -> None:
            if process.stdout is None:
                return
            for line in process.stdout:
                output_queue.put(line)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()

        timed_out = False
        next_heartbeat = start + max(heartbeat_seconds, 1.0)

        while True:
            while True:
                try:
                    line = output_queue.get_nowait()
                except queue.Empty:
                    break
                log_file.write(line)
                if show_output:
                    print(f"    {line.rstrip()}", flush=True)

            log_file.flush()
            return_code = process.poll()
            now = time.monotonic()

            if timeout is not None and return_code is None and now - start >= timeout:
                timed_out = True
                process.kill()
                return_code = process.wait()

            if return_code is not None:
                break

            if debug_output and heartbeat_seconds > 0 and now >= next_heartbeat:
                print(f"  {label}: läuft seit {now - start:.0f}s, Log: {log_path}", flush=True)
                next_heartbeat = now + heartbeat_seconds

            time.sleep(0.2)

        reader.join(timeout=2.0)
        while True:
            try:
                line = output_queue.get_nowait()
            except queue.Empty:
                break
            log_file.write(line)
            if show_output:
                print(f"    {line.rstrip()}", flush=True)

        if timed_out:
            log_file.write(f"\nTIMEOUT after {timeout} seconds\n")
            return 124, time.monotonic() - start

        return int(process.returncode or 0), time.monotonic() - start


def run_smplifyx(item: PoseItem, item_dataset: Path, args: argparse.Namespace, log_path: Path) -> tuple[str, int, float]:
    cmd = [
        str(args.python_exe),
        str(SMPLIFYX_MAIN),
        "--config",
        str(item.category.config_path),
        "--data_folder",
        str(item_dataset),
        "--output_folder",
        str(item.category.output_root),
        "--gender",
        args.gender,
        "--max_persons",
        str(args.max_persons),
        "--use_cuda",
        str(args.use_cuda).lower(),
        "--visualize",
        "false",
        "--interactive",
        "false",
    ]

    if args.fit_maxiters is not None:
        cmd.extend(["--maxiters", str(args.fit_maxiters)])

    return_code, elapsed = run_command(
        cmd,
        cwd=SMPLIFYX_ROOT / "smplifyx",
        log_path=log_path,
        timeout=args.fit_timeout_seconds,
        label=f"fit {item.pose_id}",
        debug_output=args.debug_output,
        show_output=args.show_subprocess_output,
        heartbeat_seconds=args.subprocess_heartbeat_seconds,
    )

    if return_code != 0:
        return "fit_error", return_code, elapsed
    if not item.mesh_path.exists():
        return "mesh_missing", return_code, elapsed
    return "fit_ok", return_code, elapsed


def run_blender(item: PoseItem, args: argparse.Namespace, log_path: Path) -> tuple[str, int, float]:
    head_mode = args.head_mode
    if head_mode == "auto":
        head_mode = item.category.default_head_mode

    cmd = [
        sys.executable,
        str(RENDER_RUNNER),
        "--input-root",
        str(item.mesh_dir),
        "--output-root",
        str(args.render_output_root),
        "--passes",
        ",".join(args.passes),
        "--resolution",
        args.resolution,
        "--view",
        args.view,
        "--obj-axis-mode",
        args.obj_axis_mode,
        "--camera-mode",
        args.render_camera_mode,
        "--focal-length",
        str(args.render_focal_length),
        "--smplifyx-frame-padding",
        str(args.render_frame_padding),
        "--mesh-rotation",
        args.mesh_rotation,
        "--padding",
        str(args.padding),
        "--head-mode",
        head_mode,
        "--head-scale",
        str(args.head_scale),
        "--head-offset",
        args.head_offset,
        "--blender-exe",
        str(args.blender_exe),
    ]

    if args.force_render:
        cmd.append("--force")
    if args.auto_upright:
        cmd.append("--auto-upright")
    cmd.append("--smplifyx-auto-frame" if args.render_auto_frame else "--no-smplifyx-auto-frame")

    return_code, elapsed = run_command(
        cmd,
        cwd=PROJECT_ROOT,
        log_path=log_path,
        timeout=args.render_timeout_seconds,
        label=f"render {item.pose_id}",
        debug_output=args.debug_output,
        show_output=args.show_subprocess_output,
        heartbeat_seconds=args.subprocess_heartbeat_seconds,
    )

    if return_code != 0:
        return "render_error", return_code, elapsed
    if not render_outputs_exist(item.pose_id, args.render_output_root, args.passes):
        return "render_missing", return_code, elapsed
    return "render_ok", return_code, elapsed


def write_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def selected_categories(value: str) -> list[CategoryConfig]:
    if value == "all":
        return [CATEGORIES["normal"], CATEGORIES["complex"]]
    return [CATEGORIES[value]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenPose image/keypoint -> SMPLify-X mesh -> Blender render pipeline with per-item resume."
    )
    parser.add_argument("--category", choices=["all", "normal", "complex"], default="all")
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--blender-exe", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--render-output-root", type=Path, default=DATASET_ROOT / "rendered")
    parser.add_argument("--work-root", type=Path, default=DATASET_ROOT / "_pipeline_work")
    parser.add_argument("--log-root", type=Path, default=DATASET_ROOT / "pipeline_logs")
    parser.add_argument("--passes", nargs="+", default=["depth", "lineart", "normal"])
    parser.add_argument("--resolution", default=DEFAULT_RESOLUTION)
    parser.add_argument(
        "--render-camera-mode",
        choices=["smplifyx", "orthographic"],
        default="smplifyx",
        help="Use the saved SMPLify-X perspective camera from results/000.pkl or the old generic orthographic view.",
    )
    parser.add_argument(
        "--render-focal-length",
        type=float,
        default=5000.0,
        help="Perspective focal length in pixels for --render-camera-mode smplifyx.",
    )
    parser.add_argument(
        "--render-auto-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In SMPLify-X camera mode, scale/center the perspective render so the mesh stays in frame.",
    )
    parser.add_argument(
        "--render-frame-padding",
        type=float,
        default=1.04,
        help="Frame padding multiplier for --render-auto-frame.",
    )
    parser.add_argument("--view", choices=["front", "back", "left", "right", "top"], default="front")
    parser.add_argument(
        "--obj-axis-mode",
        choices=["smplx-y-up", "raw"],
        default="smplx-y-up",
        help="OBJ import axes for Blender rendering.",
    )
    parser.add_argument(
        "--mesh-rotation",
        default="0,0,0",
        help="Additional XYZ mesh rotation in degrees for Blender rendering.",
    )
    parser.add_argument("--padding", type=float, default=1.45, help="Orthographic render framing multiplier.")
    parser.add_argument("--gender", choices=["neutral", "female", "male"], default="female")
    parser.add_argument("--max-persons", type=int, default=1)
    parser.add_argument("--use-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fit-maxiters", type=int, help="Override SMPLify-X maxiters. Omit to use YAML configs.")
    parser.add_argument("--fit-timeout-seconds", type=int, default=900)
    parser.add_argument("--render-timeout-seconds", type=int, default=300)
    parser.add_argument("--head-mode", choices=["auto", "original", "proxy", "replace"], default="auto")
    parser.add_argument("--head-scale", type=float, default=0.85)
    parser.add_argument("--head-offset", default="0,0,0")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--pose-id",
        action="append",
        help="Run only the given pose id. Can be passed more than once.",
    )
    parser.add_argument("--start-after", help="Skip items until this pose id has been seen.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-fit", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--force-fit", action="store_true")
    parser.add_argument("--force-render", action="store_true")
    parser.add_argument(
        "--no-clean-force",
        action="store_true",
        help="Keep old force behavior and do not delete existing fit/render outputs before rerunning.",
    )
    parser.add_argument(
        "--backup-cleaned",
        action="store_true",
        help="Move cleaned force-run outputs to dataset/backup/pipeline_cleanups/<timestamp> instead of deleting them.",
    )
    parser.add_argument(
        "--auto-upright",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let the Blender renderer flip front/back renders when SMPL-X head landmarks are below the body center.",
    )
    parser.add_argument(
        "--debug-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print live start/heartbeat messages while SMPLify-X and Blender subprocesses run.",
    )
    parser.add_argument(
        "--show-subprocess-output",
        action="store_true",
        help="Also mirror SMPLify-X/Blender stdout to the console. Logs are always written either way.",
    )
    parser.add_argument(
        "--subprocess-heartbeat-seconds",
        type=float,
        default=10.0,
        help="Seconds between live heartbeat messages for long-running subprocesses.",
    )
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()
    args.passes = parse_passes(args.passes)
    args.python_exe = args.python_exe.resolve()
    args.blender_exe = args.blender_exe.resolve()
    args.render_output_root = args.render_output_root.resolve()
    args.work_root = args.work_root.resolve()
    args.log_root = args.log_root.resolve()
    return args


def main() -> int:
    args = parse_args()

    if not args.python_exe.exists():
        print(f"Python executable not found: {args.python_exe}", file=sys.stderr)
        return 2
    if not args.blender_exe.exists() and not args.skip_render:
        print(f"Blender executable not found: {args.blender_exe}", file=sys.stderr)
        return 2
    if not SMPLIFYX_MAIN.exists():
        print(f"SMPLify-X entry point not found: {SMPLIFYX_MAIN}", file=sys.stderr)
        return 2

    categories = selected_categories(args.category)
    items, missing = discover_items(categories)

    if args.pose_id:
        wanted_pose_ids = set(args.pose_id)
        items = [item for item in items if item.pose_id in wanted_pose_ids]

    if args.start_after:
        seen = False
        filtered: list[PoseItem] = []
        for item in items:
            if seen:
                filtered.append(item)
            elif item.pose_id == args.start_after:
                seen = True
        items = filtered

    if args.limit is not None:
        items = items[: args.limit]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_root = args.log_root / stamp
    events_path = run_log_root / "events.jsonl"
    summary_path = run_log_root / "summary.json"

    print(f"Items discovered: {len(items)}")
    print(f"Missing pairs:    {len(missing)}")
    print(f"Categories:       {', '.join(category.name for category in categories)}")
    print(f"Passes:           {', '.join(args.passes)}")
    print(f"Resolution:       {args.resolution}")
    print(f"Render camera:    {args.render_camera_mode}")
    print(f"Render focal:     {args.render_focal_length:g}")
    print(f"Render auto-fit:  {args.render_auto_frame} (padding {args.render_frame_padding:g})")
    print(f"Render view:      {args.view}")
    print(f"OBJ axis mode:    {args.obj_axis_mode}")
    print(f"Mesh rotation:    {args.mesh_rotation}")
    print(f"Render padding:   {args.padding}")
    print(f"Auto upright:     {args.auto_upright}")
    print(f"Logs:             {run_log_root}")

    clean_backup_root = DATASET_ROOT / "backup" / "pipeline_cleanups" / stamp if args.backup_cleaned else None
    clean_force = (args.force_fit or args.force_render) and not args.no_clean_force
    if clean_force:
        print("Force cleanup:    backup" if args.backup_cleaned else "Force cleanup:    delete")
        if clean_backup_root is not None:
            print(f"Cleanup backup:   {clean_backup_root}")

    if missing:
        missing_path = run_log_root / "missing_pairs.json"
        missing_path.parent.mkdir(parents=True, exist_ok=True)
        missing_path.write_text(json.dumps(missing, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Missing report:   {missing_path}")

    if args.dry_run:
        for item in items[:30]:
            mesh_state = "mesh-ok" if item.mesh_path.exists() else "mesh-missing"
            render_state = "render-ok" if render_outputs_exist(item.pose_id, args.render_output_root, args.passes) else "render-missing"
            print(f"[DRY] {item.category.name:7} {item.pose_id} [{mesh_state}, {render_state}]")
        if len(items) > 30:
            print(f"... {len(items) - 30} more")
        return 0

    counts = {
        "fit_ok": 0,
        "fit_skipped": 0,
        "fit_error": 0,
        "mesh_missing": 0,
        "render_ok": 0,
        "render_skipped": 0,
        "render_error": 0,
        "render_missing": 0,
    }

    args.work_root.mkdir(parents=True, exist_ok=True)
    run_log_root.mkdir(parents=True, exist_ok=True)

    global_cleanup: list[dict[str, str]] = []
    if clean_force and args.force_render and not args.skip_render:
        try:
            global_cleanup = clean_render_pass_dirs(args, clean_backup_root)
            if global_cleanup:
                print(f"Rendered cleanup: {len(global_cleanup)} pass folder(s)")
            else:
                print("Rendered cleanup: pass folders already empty/missing")
        except Exception as exc:
            print(f"Rendered cleanup failed: {exc}", file=sys.stderr)
            return 1

    for index, item in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] {item.category.name}: {item.pose_id}")
        item_log_root = run_log_root / item.category.name / item.pose_id
        event: dict[str, object] = {
            "index": index,
            "total": len(items),
            "category": item.category.name,
            "pose_id": item.pose_id,
            "image": str(item.image_path),
            "keypoints": str(item.keypoint_path),
            "mesh": str(item.mesh_path),
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }

        if clean_force:
            try:
                cleaned_outputs = clean_force_outputs(item, args, clean_backup_root)
                if cleaned_outputs:
                    event["cleaned_outputs"] = cleaned_outputs
                    print(f"  cleaned: {len(cleaned_outputs)} output path(s)")
            except Exception as exc:
                event["clean_status"] = "error"
                event["clean_error"] = repr(exc)
                write_jsonl(events_path, event)
                print(f"  clean: error ({exc})")
                counts["fit_error"] = counts.get("fit_error", 0) + 1
                continue

        fit_status = "fit_skipped"
        fit_code = 0
        fit_elapsed = 0.0

        if args.skip_fit:
            fit_status = "fit_skipped"
        elif item.mesh_path.exists() and not args.force_fit:
            fit_status = "fit_skipped"
        else:
            try:
                item_dataset = prepare_single_item_dataset(
                    item,
                    args.work_root,
                    force_clean=not args.keep_work,
                )
                print(f"  fit: start, Log: {item_log_root / 'smplifyx.log'}", flush=True)
                fit_status, fit_code, fit_elapsed = run_smplifyx(
                    item,
                    item_dataset,
                    args,
                    item_log_root / "smplifyx.log",
                )
            except Exception as exc:
                fit_status = "fit_error"
                fit_code = 125
                (item_log_root / "smplifyx.log").write_text(repr(exc), encoding="utf-8")

        counts[fit_status] = counts.get(fit_status, 0) + 1
        event.update(
            {
                "fit_status": fit_status,
                "fit_return_code": fit_code,
                "fit_elapsed_seconds": round(fit_elapsed, 3),
            }
        )

        if fit_status not in {"fit_ok", "fit_skipped"} or not item.mesh_path.exists():
            event["render_status"] = "not_attempted"
            write_jsonl(events_path, event)
            print(f"  fit: {fit_status} ({fit_elapsed:.1f}s)")
            continue

        render_status = "render_skipped"
        render_code = 0
        render_elapsed = 0.0

        if args.skip_render:
            render_status = "render_skipped"
        elif render_outputs_exist(item.pose_id, args.render_output_root, args.passes) and not args.force_render:
            render_status = "render_skipped"
        else:
            try:
                print(f"  render: start, Log: {item_log_root / 'blender.log'}", flush=True)
                render_status, render_code, render_elapsed = run_blender(
                    item,
                    args,
                    item_log_root / "blender.log",
                )
            except Exception as exc:
                render_status = "render_error"
                render_code = 125
                (item_log_root / "blender.log").write_text(repr(exc), encoding="utf-8")

        counts[render_status] = counts.get(render_status, 0) + 1
        event.update(
            {
                "render_status": render_status,
                "render_return_code": render_code,
                "render_elapsed_seconds": round(render_elapsed, 3),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        write_jsonl(events_path, event)
        print(f"  fit: {fit_status} ({fit_elapsed:.1f}s), render: {render_status} ({render_elapsed:.1f}s)")

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "items": len(items),
        "missing_pairs": len(missing),
        "counts": counts,
        "events": str(events_path),
        "global_cleanup": global_cleanup,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary: {summary_path}")

    return 1 if counts.get("fit_error") or counts.get("mesh_missing") or counts.get("render_error") or counts.get("render_missing") else 0


if __name__ == "__main__":
    raise SystemExit(main())
