from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_RENDER_ROOT = DATASET_ROOT / "rendered"
DEFAULT_TARGET_ROOT = Path("C:/EasyDiffusion/stable-diffusion/stable-diffusion-webui/models/openpose")
DEFAULT_BACKUP_ROOT = DATASET_ROOT / "release_backups"
PASS_NAMES = ("depth", "lineart", "normal")
SOURCE_DATASET_DIRS = ("rendered", "poses_complex", "poses_normal")


@dataclass
class ExportRow:
    status: str
    reason: str
    pass_name: str
    source_path: str
    target_path: str
    backup_path: str


@dataclass(frozen=True)
class OpenPoseName:
    relative_dir: Path
    pose_name: str


def parse_passes(raw: Sequence[str]) -> list[str]:
    values: list[str] = []
    for item in raw:
        values.extend(part.strip().lower() for part in item.split(",") if part.strip())
    invalid = sorted(set(values) - set(PASS_NAMES))
    if invalid:
        raise ValueError(f"Unsupported pass: {', '.join(invalid)}")
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out or list(PASS_NAMES)


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.name}.{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a unique path for {path}")


def build_openpose_index(root: Path) -> dict[str, OpenPoseName]:
    index: dict[str, OpenPoseName] = {}
    if not root.exists():
        return index

    for json_path in root.rglob("*_openpose.json"):
        try:
            relative_dir = json_path.parent.relative_to(root)
        except ValueError:
            continue
        pose_name = json_path.name[: -len("_openpose.json")]
        flat_id = "_".join([*relative_dir.parts, pose_name]) + "_bone_structure"
        index.setdefault(flat_id, OpenPoseName(relative_dir, pose_name))

    return index


def infer_name_from_existing_dirs(flat_id: str, openpose_root: Path) -> OpenPoseName | None:
    base = flat_id.removesuffix("_bone_structure")
    parts = base.split("_")
    if len(parts) < 5:
        return None

    category, gender, variant = parts[0], parts[1], parts[2]
    tail = "_".join(parts[3:])
    parent = openpose_root / category / gender / variant
    if parent.exists():
        candidates = sorted(
            (path.name for path in parent.iterdir() if path.is_dir()),
            key=len,
            reverse=True,
        )
        for subfolder in candidates:
            prefix = f"{subfolder}_"
            if tail.startswith(prefix) and len(tail) > len(prefix):
                return OpenPoseName(Path(category, gender, variant, subfolder), tail[len(prefix) :])

    return OpenPoseName(Path(category, gender, variant, parts[3]), "_".join(parts[4:]))


def output_relative_path(
    source_path: Path,
    pass_name: str,
    openpose_index: dict[str, OpenPoseName],
    openpose_root: Path,
) -> Path | None:
    flat_id = source_path.stem
    name = openpose_index.get(flat_id) or infer_name_from_existing_dirs(flat_id, openpose_root)
    if name is None:
        return None
    return name.relative_dir / f"{name.pose_name}_{pass_name}.png"


def copy_file(src: Path, dst: Path, overwrite: bool, dry_run: bool) -> tuple[str, str]:
    if dst.exists() and not overwrite:
        return "skipped", "exists"
    if dry_run:
        return "would_copy", ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return "copied", ""


def export_passes(args: argparse.Namespace, stamp: str) -> list[ExportRow]:
    openpose_index = build_openpose_index(args.openpose_root)
    rows: list[ExportRow] = []
    backup_openpose_root = args.backup_root / stamp / "openpose"

    for pass_name in args.passes:
        pass_dir = args.render_root / pass_name
        if not pass_dir.exists():
            continue

        for src in sorted(pass_dir.glob("*.png")):
            rel = output_relative_path(src, pass_name, openpose_index, args.openpose_root)
            if rel is None:
                rows.append(
                    ExportRow(
                        "error",
                        "cannot_map_filename",
                        pass_name,
                        str(src),
                        "",
                        "",
                    )
                )
                continue

            target_path = args.target_root / rel
            backup_path = backup_openpose_root / rel if args.backup_export else Path()

            status, reason = copy_file(src, target_path, args.overwrite, not args.apply)
            backup_status = ""
            backup_reason = ""
            if args.backup_export:
                backup_status, backup_reason = copy_file(src, backup_path, True, not args.apply)

            final_status = status
            final_reason = reason
            if args.backup_export and backup_status not in {"copied", "would_copy"}:
                final_status = "error"
                final_reason = f"backup_{backup_status}:{backup_reason}"

            rows.append(
                ExportRow(
                    final_status,
                    final_reason,
                    pass_name,
                    str(src),
                    str(target_path),
                    str(backup_path) if args.backup_export else "",
                )
            )

    return rows


def backup_source_dataset(args: argparse.Namespace, stamp: str) -> list[str]:
    copied: list[str] = []
    if not args.backup_source_dataset:
        return copied

    root = args.backup_root / stamp / "source_dataset"
    for dirname in SOURCE_DATASET_DIRS:
        src = DATASET_ROOT / dirname
        if not src.exists():
            continue
        dst = root / dirname
        copied.append(str(dst))
        if args.apply:
            shutil.copytree(src, dst, dirs_exist_ok=True)

    return copied


def write_reports(rows: list[ExportRow], backup_root: Path, stamp: str, source_backups: list[str]) -> tuple[Path, Path]:
    report_root = backup_root / stamp
    report_root.mkdir(parents=True, exist_ok=True)
    csv_path = report_root / "export_report.csv"
    json_path = report_root / "export_report.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ExportRow.__dataclass_fields__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": [asdict(row) for row in rows],
        "source_dataset_backups": source_backups,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export rendered pass PNGs back into the nested OpenPose model folder structure."
    )
    parser.add_argument("--render-root", type=Path, default=DEFAULT_RENDER_ROOT)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument(
        "--openpose-root",
        type=Path,
        default=DEFAULT_TARGET_ROOT,
        help="Existing OpenPose source tree used to disambiguate flattened names with underscores.",
    )
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument("--passes", nargs="+", default=list(PASS_NAMES))
    parser.add_argument("--apply", action="store_true", help="Actually copy files. Without this, only reports a dry run.")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backup-export", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backup-source-dataset", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.render_root = args.render_root.resolve()
    args.target_root = args.target_root.resolve()
    args.openpose_root = args.openpose_root.resolve()
    args.backup_root = args.backup_root.resolve()
    args.passes = parse_passes(args.passes)
    return args


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.apply:
        stamp = unique_path(args.backup_root / stamp).name

    rows = export_passes(args, stamp)
    source_backups = backup_source_dataset(args, stamp)
    csv_path, json_path = write_reports(rows, args.backup_root, stamp, source_backups)

    copied = sum(1 for row in rows if row.status == "copied")
    would_copy = sum(1 for row in rows if row.status == "would_copy")
    skipped = sum(1 for row in rows if row.status == "skipped")
    errors = [row for row in rows if row.status == "error"]

    print(f"Rows:        {len(rows)}")
    print(f"Copied:      {copied}")
    print(f"Would copy:  {would_copy}")
    print(f"Skipped:     {skipped}")
    print(f"Errors:      {len(errors)}")
    print(f"Target:      {args.target_root}")
    print(f"Backup root: {args.backup_root / stamp}")
    print(f"CSV:         {csv_path}")
    print(f"JSON:        {json_path}")
    if not args.apply:
        print("Dry run:     no files copied. Add --apply to export.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
