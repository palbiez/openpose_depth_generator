from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_OUTPUT_ROOT = DATASET_ROOT / "head_pose_batches"
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp"]
BODY_HEAD_IDS = [0, 15, 16, 17, 18]  # nose, eyes, ears in BODY_25


@dataclass
class HeadBatchRow:
    pose_id: str
    category: str
    bucket: str
    body_head_points: int
    face_points: int
    image_path: str
    keypoint_path: str
    output_image_path: str
    output_keypoint_path: str
    synthesized_face: bool


def ensure_inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Refusing path outside {resolved_root}: {resolved}")
    return resolved


def point(points: Sequence[float], idx: int) -> tuple[float, float, float] | None:
    base = idx * 3
    if len(points) < base + 3:
        return None
    return float(points[base]), float(points[base + 1]), float(points[base + 2])


def valid_point(points: Sequence[float], idx: int, min_conf: float) -> tuple[float, float] | None:
    value = point(points, idx)
    if value is None:
        return None
    x, y, c = value
    if c <= min_conf:
        return None
    return x, y


def count_points(points: Sequence[float], ids: Iterable[int] | None, min_conf: float) -> int:
    if ids is None:
        return sum(1 for idx in range(len(points) // 3) if float(points[idx * 3 + 2]) > min_conf)
    return sum(1 for idx in ids if (point(points, idx) is not None and float(points[idx * 3 + 2]) > min_conf))


def normalize(vx: float, vy: float) -> tuple[float, float]:
    length = math.hypot(vx, vy)
    if length < 1e-6:
        return 1.0, 0.0
    return vx / length, vy / length


def add_local(
    out: list[float],
    idx: int,
    center: tuple[float, float],
    x_axis: tuple[float, float],
    y_axis: tuple[float, float],
    width: float,
    height: float,
    ux: float,
    uy: float,
    confidence: float,
) -> None:
    x = center[0] + x_axis[0] * ux * width + y_axis[0] * uy * height
    y = center[1] + x_axis[1] * ux * width + y_axis[1] * uy * height
    base = idx * 3
    out[base : base + 3] = [x, y, confidence]


def synthesize_face_keypoints(
    pose: Sequence[float],
    confidence: float,
    min_conf: float,
) -> list[float] | None:
    nose = valid_point(pose, 0, min_conf)
    reye = valid_point(pose, 15, min_conf)
    leye = valid_point(pose, 16, min_conf)
    rear = valid_point(pose, 17, min_conf)
    lear = valid_point(pose, 18, min_conf)

    anchors = [pt for pt in (nose, reye, leye, rear, lear) if pt is not None]
    if len(anchors) < 3:
        return None

    if reye is not None and leye is not None:
        x_axis = normalize(leye[0] - reye[0], leye[1] - reye[1])
        eye_center = ((reye[0] + leye[0]) * 0.5, (reye[1] + leye[1]) * 0.5)
        eye_dist = math.dist(reye, leye)
    elif rear is not None and lear is not None:
        x_axis = normalize(lear[0] - rear[0], lear[1] - rear[1])
        eye_center = ((rear[0] + lear[0]) * 0.5, (rear[1] + lear[1]) * 0.5)
        eye_dist = math.dist(rear, lear) * 0.45
    else:
        x_axis = (1.0, 0.0)
        eye_center = (
            sum(pt[0] for pt in anchors) / len(anchors),
            sum(pt[1] for pt in anchors) / len(anchors),
        )
        eye_dist = 32.0

    y_axis = (-x_axis[1], x_axis[0])
    if nose is not None:
        nose_vec = (nose[0] - eye_center[0], nose[1] - eye_center[1])
        if nose_vec[0] * y_axis[0] + nose_vec[1] * y_axis[1] < 0:
            y_axis = (-y_axis[0], -y_axis[1])

    ear_dist = math.dist(rear, lear) if rear is not None and lear is not None else 0.0
    width = max(ear_dist, eye_dist * 2.6, 24.0)
    if nose is not None:
        nose_depth = abs((nose[0] - eye_center[0]) * y_axis[0] + (nose[1] - eye_center[1]) * y_axis[1])
    else:
        nose_depth = width * 0.25
    height = max(width * 1.18, nose_depth * 3.0, 32.0)
    face_center = eye_center

    face = [0.0] * (68 * 3)

    # Face contour, useful only if use_face_contour is enabled later.
    for offset, idx in enumerate(range(17)):
        angle = math.pi * (0.15 + 0.70 * offset / 16.0)
        add_local(face, idx, face_center, x_axis, y_axis, width, height, math.cos(angle) * 0.48, math.sin(angle) * 0.48, confidence * 0.5)

    # Brows.
    for idx, ux in zip(range(17, 22), [-0.34, -0.27, -0.20, -0.13, -0.07]):
        add_local(face, idx, face_center, x_axis, y_axis, width, height, ux, -0.16, confidence)
    for idx, ux in zip(range(22, 27), [0.07, 0.13, 0.20, 0.27, 0.34]):
        add_local(face, idx, face_center, x_axis, y_axis, width, height, ux, -0.16, confidence)

    # Nose bridge and nostrils.
    for idx, uy in zip(range(27, 31), [-0.04, 0.04, 0.12, 0.20]):
        add_local(face, idx, face_center, x_axis, y_axis, width, height, 0.0, uy, confidence)
    for idx, ux in zip(range(31, 36), [-0.16, -0.08, 0.0, 0.08, 0.16]):
        add_local(face, idx, face_center, x_axis, y_axis, width, height, ux, 0.25, confidence)

    # Eyes.
    for start, eye, ux_center in ((36, reye, -0.20), (42, leye, 0.20)):
        center = eye if eye is not None else (
            face_center[0] + x_axis[0] * ux_center * width,
            face_center[1] + x_axis[1] * ux_center * width,
        )
        for local_idx, angle in enumerate([math.pi, 4 * math.pi / 3, 5 * math.pi / 3, 0.0, math.pi / 3, 2 * math.pi / 3]):
            x = center[0] + math.cos(angle) * width * 0.055 * x_axis[0] + math.sin(angle) * height * 0.025 * y_axis[0]
            y = center[1] + math.cos(angle) * width * 0.055 * x_axis[1] + math.sin(angle) * height * 0.025 * y_axis[1]
            base = (start + local_idx) * 3
            face[base : base + 3] = [x, y, confidence]

    # Mouth, intentionally low-detail and low confidence.
    for offset, idx in enumerate(range(48, 68)):
        angle = 2.0 * math.pi * offset / 20.0
        add_local(face, idx, face_center, x_axis, y_axis, width, height, math.cos(angle) * 0.18, 0.45 + math.sin(angle) * 0.065, confidence * 0.7)

    if nose is not None:
        base = 33 * 3
        face[base : base + 3] = [nose[0], nose[1], confidence]
    return face


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def image_for_pose(category_root: Path, pose_id: str) -> Path | None:
    for suffix in IMAGE_EXTS:
        path = category_root / "images" / f"{pose_id}{suffix}"
        if path.exists():
            return path
    return None


def classify_bucket(body_head_points: int, face_points: int, min_head_points: int) -> str:
    if body_head_points == 0:
        return "head_missing"
    if body_head_points < min_head_points:
        return "head_weak"
    if face_points == 0:
        return "body_head_only"
    return "face_ok"


def copy_or_write(
    src_image: Path | None,
    src_keypoint: Path,
    out_image: Path,
    out_keypoint: Path,
    args: argparse.Namespace,
) -> bool:
    if not args.apply:
        return False

    if src_image is not None:
        out_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_image, out_image)

    data = read_json(src_keypoint)
    synthesized = False
    if args.synthesize_face:
        people = data.get("people", [])
        if isinstance(people, list) and people:
            person = people[0]
            if isinstance(person, dict):
                pose = person.get("pose_keypoints_2d", [])
                face = person.get("face_keypoints_2d", [])
                face_points = count_points(face, None, args.keypoint_min_confidence) if isinstance(face, list) else 0
                if isinstance(pose, list) and face_points == 0:
                    synthetic = synthesize_face_keypoints(pose, args.face_confidence, args.keypoint_min_confidence)
                    if synthetic is not None:
                        person["face_keypoints_2d"] = synthetic
                        synthesized = True

    write_json(out_keypoint, data)
    return synthesized


def scan(args: argparse.Namespace, stamp: str) -> list[HeadBatchRow]:
    rows: list[HeadBatchRow] = []
    for category in args.categories:
        category_root = DATASET_ROOT / category
        keypoint_root = category_root / "keypoints"
        if not keypoint_root.exists():
            continue
        for keypoint_path in sorted(keypoint_root.glob("*_keypoints.json")):
            pose_id = keypoint_path.name[: -len("_keypoints.json")]
            image_path = image_for_pose(category_root, pose_id)
            data = read_json(keypoint_path)
            person = data.get("people", [{}])[0]
            if not isinstance(person, dict):
                continue
            pose = person.get("pose_keypoints_2d", [])
            face = person.get("face_keypoints_2d", [])
            if not isinstance(pose, list):
                pose = []
            if not isinstance(face, list):
                face = []
            body_head_points = count_points(pose, BODY_HEAD_IDS, args.keypoint_min_confidence)
            face_points = count_points(face, None, args.keypoint_min_confidence)
            bucket = classify_bucket(body_head_points, face_points, args.min_head_points)
            if bucket == "face_ok" and not args.include_face_ok:
                continue
            if bucket not in args.buckets:
                continue

            batch_root = args.output_root / stamp / bucket / category
            out_image = batch_root / "images" / image_path.name if image_path is not None else batch_root / "images" / f"{pose_id}.png"
            out_keypoint = batch_root / "keypoints" / keypoint_path.name
            synthesized = copy_or_write(image_path, keypoint_path, out_image, out_keypoint, args)
            rows.append(
                HeadBatchRow(
                    pose_id=pose_id,
                    category=category,
                    bucket=bucket,
                    body_head_points=body_head_points,
                    face_points=face_points,
                    image_path=str(image_path.resolve()) if image_path else "",
                    keypoint_path=str(keypoint_path.resolve()),
                    output_image_path=str(out_image.resolve()),
                    output_keypoint_path=str(out_keypoint.resolve()),
                    synthesized_face=synthesized,
                )
            )
    return rows


def write_report(rows: list[HeadBatchRow], output_root: Path, stamp: str) -> tuple[Path, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / stamp / "head_pose_batches.csv"
    json_path = output_root / stamp / "head_pose_batches.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(HeadBatchRow.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    json_path.write_text(json.dumps([asdict(row) for row in rows], indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create separate batches for poses with missing/weak head data or missing face_keypoints_2d."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--categories", nargs="+", default=["poses_normal", "poses_complex"])
    parser.add_argument(
        "--buckets",
        nargs="+",
        default=["head_missing", "head_weak", "body_head_only"],
        choices=["head_missing", "head_weak", "body_head_only", "face_ok"],
    )
    parser.add_argument("--include-face-ok", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Actually copy images/keypoints. Without this, only reports.")
    parser.add_argument(
        "--synthesize-face",
        action="store_true",
        help="Write approximate 68-point face_keypoints_2d from BODY_25 nose/eyes/ears when face data is empty.",
    )
    parser.add_argument("--face-confidence", type=float, default=0.2)
    parser.add_argument("--min-head-points", type=int, default=3)
    parser.add_argument("--keypoint-min-confidence", type=float, default=0.05)
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    ensure_inside(args.output_root, DATASET_ROOT)
    for category in args.categories:
        if category not in {"poses_normal", "poses_complex"}:
            raise ValueError(f"Unsupported category: {category}")
    return args


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = scan(args, stamp)
    csv_path, json_path = write_report(rows, args.output_root, stamp)

    counts: dict[str, int] = {}
    synthesized = 0
    for row in rows:
        counts[row.bucket] = counts.get(row.bucket, 0) + 1
        synthesized += int(row.synthesized_face)

    print(f"Rows:        {len(rows)}")
    for bucket in ("head_missing", "head_weak", "body_head_only", "face_ok"):
        print(f"{bucket:14}: {counts.get(bucket, 0)}")
    print(f"Synthesized: {synthesized}")
    print(f"CSV:         {csv_path}")
    print(f"JSON:        {json_path}")
    if args.apply:
        print(f"Output root: {args.output_root / stamp}")
    else:
        print("Dry run:     no batch files copied. Add --apply to write them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
