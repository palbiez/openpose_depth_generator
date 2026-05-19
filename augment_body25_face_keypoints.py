from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from prepare_head_pose_batches import (
    BODY_HEAD_IDS,
    count_points,
    ensure_inside,
    synthesize_face_keypoints,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_REPORT_ROOT = DATASET_ROOT / "face_keypoint_augments"


@dataclass
class AugmentRow:
    category: str
    pose_id: str
    keypoint_path: str
    status: str
    body_head_points: int
    face_points_before: int
    face_points_after: int
    backup_path: str


def zero_face() -> list[float]:
    return [0.0] * (68 * 3)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def backup_file(path: Path, backup_root: Path) -> Path:
    rel = path.resolve().relative_to(DATASET_ROOT.resolve())
    dst = backup_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        for idx in range(2, 10000):
            candidate = dst.with_name(f"{dst.stem}.{idx}{dst.suffix}")
            if not candidate.exists():
                dst = candidate
                break
    shutil.copy2(path, dst)
    return dst


def augment_file(path: Path, category: str, args: argparse.Namespace, backup_root: Path) -> AugmentRow:
    pose_id = path.name[: -len("_keypoints.json")]
    data = load_json(path)
    people = data.get("people", [])

    status = "unchanged"
    body_head_points = 0
    face_points_before = 0
    face_points_after = 0
    changed = False

    if not isinstance(people, list) or not people:
        status = "no_people"
    else:
        for person in people:
            if not isinstance(person, dict):
                continue
            pose = person.get("pose_keypoints_2d", [])
            face = person.get("face_keypoints_2d", [])
            if not isinstance(pose, list):
                pose = []
            if not isinstance(face, list):
                face = []

            body_head_points = max(body_head_points, count_points(pose, BODY_HEAD_IDS, args.keypoint_min_confidence))
            current_face_points = count_points(face, None, args.keypoint_min_confidence)
            face_points_before = max(face_points_before, current_face_points)

            if current_face_points > 0 and not args.overwrite_existing:
                face_points_after = max(face_points_after, current_face_points)
                continue

            synthetic = synthesize_face_keypoints(
                pose,
                args.face_confidence,
                args.keypoint_min_confidence,
            )
            if synthetic is None:
                synthetic = zero_face()
                status = "zero_face"
            else:
                status = "synthesized_face"
            person["face_keypoints_2d"] = synthetic
            face_points_after = max(face_points_after, count_points(synthetic, None, args.keypoint_min_confidence))
            changed = True

    backup_path = ""
    if changed and args.apply:
        backup_path = str(backup_file(path, backup_root).resolve())
        write_json(path, data)
    elif changed:
        status = f"dry_{status}"

    return AugmentRow(
        category=category,
        pose_id=pose_id,
        keypoint_path=str(path.resolve()),
        status=status,
        body_head_points=body_head_points,
        face_points_before=face_points_before,
        face_points_after=face_points_after,
        backup_path=backup_path,
    )


def write_report(rows: list[AugmentRow], report_root: Path, stamp: str) -> tuple[Path, Path]:
    report_root.mkdir(parents=True, exist_ok=True)
    csv_path = report_root / f"face_keypoint_augment_{stamp}.csv"
    json_path = report_root / f"face_keypoint_augment_{stamp}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AugmentRow.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    json_path.write_text(json.dumps([asdict(row) for row in rows], indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate OpenPose face_keypoints_2d from BODY_25 nose/eyes/ears anchors."
    )
    parser.add_argument("--categories", nargs="+", default=["poses_normal", "poses_complex"])
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--backup-root", type=Path, default=DATASET_ROOT / "backup" / "keypoints_before_face_augment")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--face-confidence", type=float, default=0.25)
    parser.add_argument("--keypoint-min-confidence", type=float, default=0.05)
    args = parser.parse_args()
    args.report_root = args.report_root.resolve()
    args.backup_root = args.backup_root.resolve()
    ensure_inside(args.report_root, DATASET_ROOT)
    ensure_inside(args.backup_root, DATASET_ROOT)
    for category in args.categories:
        if category not in {"poses_normal", "poses_complex"}:
            raise ValueError(f"Unsupported category: {category}")
    return args


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = args.backup_root / stamp
    rows: list[AugmentRow] = []

    for category in args.categories:
        keypoint_root = DATASET_ROOT / category / "keypoints"
        for path in sorted(keypoint_root.glob("*_keypoints.json")):
            rows.append(augment_file(path, category, args, backup_root))

    csv_path, json_path = write_report(rows, args.report_root, stamp)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1

    print(f"Files:   {len(rows)}")
    for status in sorted(counts):
        print(f"{status:18}: {counts[status]}")
    print(f"CSV:     {csv_path}")
    print(f"JSON:    {json_path}")
    if args.apply:
        print(f"Backup:  {backup_root}")
    else:
        print("Dry run: no files changed. Add --apply to write face_keypoints_2d.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
