from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOTS = [
    PROJECT_ROOT / "dataset" / "poses_normal" / "smplx" / "meshes",
    PROJECT_ROOT / "dataset" / "poses_complex" / "smplx" / "meshes",
    PROJECT_ROOT / "dataset" / "smplx" / "meshes",
]
BACKUP_INPUT_ROOT = PROJECT_ROOT / "dataset" / "backup" / "smplx" / "meshes"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "rendered"
BLENDER_SCRIPT = PROJECT_ROOT / "blender" / "render_smplx_passes.py"
DEFAULT_PASSES = ["depth", "lineart", "normal"]
DEFAULT_WINDOWS_BLENDER_EXE = Path("C:/Program Files/Blender Foundation/Blender 5.1/blender.exe")
DEFAULT_RESOLUTION = "512x768"


def parse_passes(raw: Sequence[str]) -> List[str]:
    values: List[str] = []
    for item in raw:
        values.extend(part.strip().lower() for part in item.split(",") if part.strip())

    valid = {"depth", "lineart", "normal"}
    invalid = sorted(set(values) - valid)
    if invalid:
        raise ValueError(f"Unsupported render pass: {', '.join(invalid)}")

    out: List[str] = []
    for value in values:
        if value not in out:
            out.append(value)

    return out or DEFAULT_PASSES


def parse_resolution(raw: str) -> tuple[int, int]:
    value = raw.lower().replace(" ", "")
    for separator in ("x", ",", ":"):
        if separator in value:
            width_raw, height_raw = value.split(separator, 1)
            width = int(width_raw)
            height = int(height_raw)
            break
    else:
        width = int(value)
        height = width

    if width <= 0 or height <= 0:
        raise ValueError("Resolution values must be positive")

    return width, height


def find_blender(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"Blender executable not found: {path}")

    if DEFAULT_WINDOWS_BLENDER_EXE.exists():
        return DEFAULT_WINDOWS_BLENDER_EXE

    from_path = shutil.which("blender")
    if from_path:
        return Path(from_path)

    candidates: List[Path] = []
    for base in (Path("C:/Program Files/Blender Foundation"), Path("C:/Program Files (x86)/Blender Foundation")):
        if base.exists():
            candidates.extend(base.glob("Blender */blender.exe"))

    if candidates:
        return sorted(candidates, reverse=True)[0]

    raise FileNotFoundError(
        "Could not find blender.exe. Pass --blender-exe or add Blender to PATH."
    )


def discover_obj_files(input_roots: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for root in input_roots:
        if not root.exists():
            continue
        files.extend(path for path in root.rglob("*.obj") if path.is_file())
    return sorted(set(path.resolve() for path in files))


def infer_category(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "poses_normal" in parts:
        return "normal"
    if "poses_complex" in parts:
        return "complex"
    if "backup" in parts:
        return "backup"
    return "unrouted"


def output_id_for(path: Path, used: set[str]) -> str:
    if path.name.lower() == "000.obj":
        base = path.parent.name
    else:
        base = f"{path.parent.name}_{path.stem}"

    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in base)
    candidate = safe
    idx = 2
    while candidate in used:
        candidate = f"{safe}_{idx}"
        idx += 1
    used.add(candidate)
    return candidate


def output_paths(output_root: Path, output_id: str, passes: Sequence[str]) -> dict[str, str]:
    return {
        pass_name: str((output_root / pass_name / f"{output_id}.png").resolve())
        for pass_name in passes
    }


def result_pkl_for_obj(obj_path: Path) -> Path | None:
    parts = list(obj_path.parts)
    lower_parts = [part.lower() for part in parts]
    try:
        meshes_idx = len(lower_parts) - 1 - lower_parts[::-1].index("meshes")
    except ValueError:
        return None

    result_parts = (
        parts[:meshes_idx]
        + ["results"]
        + parts[meshes_idx + 1 : -1]
        + [obj_path.with_suffix(".pkl").name]
    )
    return Path(*result_parts)


def build_manifest(
    obj_files: Sequence[Path],
    output_root: Path,
    passes: Sequence[str],
    force: bool,
) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    used: set[str] = set()

    for obj_path in obj_files:
        output_id = output_id_for(obj_path, used)
        outputs = output_paths(output_root, output_id, passes)

        if not force and all(Path(path).exists() for path in outputs.values()):
            continue

        pkl_path = result_pkl_for_obj(obj_path)
        items.append(
            {
                "id": output_id,
                "category": infer_category(obj_path),
                "obj_path": str(obj_path.resolve()),
                "pkl_path": str(pkl_path.resolve()) if pkl_path is not None and pkl_path.exists() else None,
                "outputs": outputs,
            }
        )

    return items


def write_manifest(output_root: Path, items: list[dict[str, object]], args: argparse.Namespace) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_dir = output_root / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"render_manifest_{stamp}.json"

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "passes": args.passes,
        "resolution": {
            "width": args.width,
            "height": args.height,
        },
        "camera_mode": args.camera_mode,
        "focal_length": args.focal_length,
        "smplifyx_auto_frame": args.smplifyx_auto_frame,
        "smplifyx_frame_padding": args.smplifyx_frame_padding,
        "view": args.view,
        "obj_axis_mode": args.obj_axis_mode,
        "mesh_rotation": args.mesh_rotation,
        "padding": args.padding,
        "head_mode": args.head_mode,
        "head_scale": args.head_scale,
        "head_offset": args.head_offset,
        "auto_upright": args.auto_upright,
        "items": items,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def run_blender(blender_exe: Path, manifest_path: Path, args: argparse.Namespace) -> int:
    cmd = [
        str(blender_exe),
        "--background",
        "--python",
        str(BLENDER_SCRIPT),
        "--",
        "--manifest",
        str(manifest_path),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--passes",
        ",".join(args.passes),
        "--view",
        args.view,
        "--camera-mode",
        args.camera_mode,
        "--focal-length",
        str(args.focal_length),
        "--smplifyx-frame-padding",
        str(args.smplifyx_frame_padding),
        "--obj-axis-mode",
        args.obj_axis_mode,
        "--mesh-rotation",
        args.mesh_rotation,
        "--padding",
        str(args.padding),
        "--line-thickness",
        str(args.line_thickness),
        "--normal-space",
        args.normal_space,
        "--head-mode",
        args.head_mode,
        "--head-scale",
        str(args.head_scale),
        "--head-offset",
        args.head_offset,
    ]

    if args.force:
        cmd.append("--force")
    if args.auto_upright:
        cmd.append("--auto-upright")
    cmd.append("--smplifyx-auto-frame" if args.smplifyx_auto_frame else "--no-smplifyx-auto-frame")

    print("Running Blender:")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd))
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render depth, lineart and normal passes from SMPL-X OBJ files via Blender."
    )
    parser.add_argument(
        "--input-root",
        action="append",
        type=Path,
        help="Mesh root to scan recursively. Can be passed more than once.",
    )
    parser.add_argument(
        "--include-backup",
        action="store_true",
        help="Also scan dataset/backup/smplx/meshes.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root folder for rendered pass outputs.",
    )
    parser.add_argument(
        "--passes",
        nargs="+",
        default=DEFAULT_PASSES,
        help="Passes to render: depth,lineart,normal. Comma-separated or space-separated.",
    )
    parser.add_argument(
        "--resolution",
        default=DEFAULT_RESOLUTION,
        help="Output resolution. Use one integer for square output or WIDTHxHEIGHT, e.g. 512x768.",
    )
    parser.add_argument(
        "--view",
        choices=["front", "back", "left", "right", "top"],
        default="front",
        help="Camera view used for all passes.",
    )
    parser.add_argument(
        "--camera-mode",
        choices=["orthographic", "smplifyx"],
        default="orthographic",
        help="Use a generic orthographic view or the saved SMPLify-X perspective camera from results/000.pkl.",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=5000.0,
        help="Perspective focal length in pixels for --camera-mode smplifyx.",
    )
    parser.add_argument(
        "--smplifyx-auto-frame",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In SMPLify-X camera mode, scale/center the perspective render so the mesh stays in frame.",
    )
    parser.add_argument(
        "--smplifyx-frame-padding",
        type=float,
        default=1.04,
        help="Frame padding multiplier for --smplifyx-auto-frame.",
    )
    parser.add_argument(
        "--obj-axis-mode",
        choices=["smplx-y-up", "raw"],
        default="smplx-y-up",
        help="OBJ import axes. smplx-y-up maps SMPLify-X Y-up OBJ files into Blender Z-up; raw keeps Blender-native Z-up.",
    )
    parser.add_argument(
        "--mesh-rotation",
        default="0,0,0",
        help="Additional XYZ rotation in degrees after OBJ import, e.g. 90,0,0.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=1.45,
        help="Orthographic framing multiplier.",
    )
    parser.add_argument(
        "--line-thickness",
        type=float,
        default=1.0,
        help="Freestyle line thickness in pixels.",
    )
    parser.add_argument(
        "--normal-space",
        choices=["camera", "world"],
        default="camera",
        help="Normal-map coordinate space.",
    )
    parser.add_argument(
        "--head-mode",
        choices=["original", "proxy", "replace"],
        default="original",
        help="Use the fitted head, add a proxy head, or replace the fitted head for missing-face-keypoint cases.",
    )
    parser.add_argument(
        "--head-scale",
        type=float,
        default=1.0,
        help="Scale multiplier for --head-mode proxy.",
    )
    parser.add_argument(
        "--head-offset",
        default="0,0,0",
        help="Manual XYZ offset in model units for --head-mode proxy, e.g. 0,0.03,-0.02.",
    )
    parser.add_argument(
        "--auto-upright",
        action="store_true",
        help="Flip front/back renders when SMPL-X head landmarks are below the body center.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Render only the first N pending OBJ files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render even when all requested output files already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print discovered/pending files and write the manifest.",
    )
    parser.add_argument(
        "--blender-exe",
        help="Path to blender.exe. If omitted, PATH and standard Windows install paths are checked.",
    )
    args = parser.parse_args()
    args.passes = parse_passes(args.passes)
    args.width, args.height = parse_resolution(args.resolution)
    return args


def main() -> int:
    args = parse_args()

    input_roots = [path.resolve() for path in (args.input_root or DEFAULT_INPUT_ROOTS)]
    if args.include_backup:
        input_roots.append(BACKUP_INPUT_ROOT.resolve())

    output_root = args.output_root.resolve()
    obj_files = discover_obj_files(input_roots)
    pending = build_manifest(obj_files, output_root, args.passes, args.force)

    if args.limit is not None:
        pending = pending[: args.limit]

    print(f"Input roots: {len(input_roots)}")
    for root in input_roots:
        print(f"  {root}")
    print(f"OBJ files discovered: {len(obj_files)}")
    print(f"Pending renders:      {len(pending)}")
    print(f"Output root:          {output_root}")
    print(f"Resolution:           {args.width}x{args.height}")
    print(f"Passes:               {', '.join(args.passes)}")
    print(f"Camera mode:          {args.camera_mode}")
    print(f"Focal length:         {args.focal_length:g}")
    print(f"SMPLify-X auto frame: {args.smplifyx_auto_frame} (padding {args.smplifyx_frame_padding:g})")
    print(f"OBJ axis mode:        {args.obj_axis_mode}")
    print(f"Mesh rotation:        {args.mesh_rotation}")
    print(f"Auto upright:         {args.auto_upright}")
    if args.camera_mode == "smplifyx":
        missing_pkl = sum(1 for item in pending if not item.get("pkl_path"))
        print(f"SMPLify-X PKL missing: {missing_pkl}")

    if not pending:
        return 0

    manifest_path = write_manifest(output_root, pending, args)
    print(f"Manifest:             {manifest_path}")

    if args.dry_run:
        for item in pending[:20]:
            print(f"[DRY] {item['id']} <- {item['obj_path']}")
        if len(pending) > 20:
            print(f"... {len(pending) - 20} more")
        return 0

    if not BLENDER_SCRIPT.exists():
        print(f"Missing Blender script: {BLENDER_SCRIPT}", file=sys.stderr)
        return 2

    blender_exe = find_blender(args.blender_exe)
    print(f"Blender:              {blender_exe}")
    return run_blender(blender_exe, manifest_path, args)


if __name__ == "__main__":
    raise SystemExit(main())
