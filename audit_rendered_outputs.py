from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import shutil
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_RENDER_ROOT = DATASET_ROOT / "rendered"
DEFAULT_REPORT_ROOT = DATASET_ROOT / "render_audits"
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp"]
BODY_HEAD_IDS = [0, 15, 16, 17, 18]  # nose, eyes, ears in BODY_25


@dataclass
class RenderAudit:
    pose_id: str
    category: str
    status: str
    reasons: str
    review_reasons: str
    depth_path: str
    lineart_path: str
    normal_path: str
    keypoint_path: str
    image_path: str
    pkl_path: str
    foreground_ratio: float
    foreground_pixels: int
    bbox: str
    bbox_width: int
    bbox_height: int
    largest_component_ratio: float
    component_count: int
    border_touch_pixels: int
    lineart_ratio: float
    body_head_points: int
    face_points: int
    global_orient_norm: float
    global_orient_x: float
    global_orient_y: float
    global_orient_z: float


def parse_passes(raw: Sequence[str]) -> list[str]:
    values: list[str] = []
    for item in raw:
        values.extend(part.strip().lower() for part in item.split(",") if part.strip())
    valid = {"depth", "lineart", "normal"}
    invalid = sorted(set(values) - valid)
    if invalid:
        raise ValueError(f"Unsupported pass: {', '.join(invalid)}")
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out or ["depth", "lineart", "normal"]


def ensure_inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved_root = root.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"Refusing path outside {resolved_root}: {resolved}")
    return resolved


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(2, 10000):
        candidate = path.with_name(f"{path.stem}.{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create unique path for {path}")


def move_path(src: Path, dst: Path) -> Path:
    src = ensure_inside(src, DATASET_ROOT)
    dst = unique_path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return dst


def category_for_pose(pose_id: str) -> tuple[str, Path | None, Path | None]:
    for category in ("poses_normal", "poses_complex"):
        base = DATASET_ROOT / category
        keypoint_path = base / "keypoints" / f"{pose_id}_keypoints.json"
        image_path = next(
            (base / "images" / f"{pose_id}{suffix}" for suffix in IMAGE_EXTS if (base / "images" / f"{pose_id}{suffix}").exists()),
            None,
        )
        if keypoint_path.exists() or image_path is not None:
            return category, image_path, keypoint_path if keypoint_path.exists() else None
    return "unknown", None, None


def smplx_paths(category: str, pose_id: str) -> dict[str, Path]:
    if category not in {"poses_normal", "poses_complex"}:
        return {}
    root = DATASET_ROOT / category / "smplx"
    return {
        "mesh_dir": root / "meshes" / pose_id,
        "result_dir": root / "results" / pose_id,
        "smplx_image_dir": root / "images" / pose_id,
        "pkl_path": root / "results" / pose_id / "000.pkl",
    }


def point_count(points: Sequence[float], ids: Iterable[int] | None = None, min_conf: float = 0.05) -> int:
    if ids is None:
        count = len(points) // 3
        return sum(1 for idx in range(count) if float(points[idx * 3 + 2]) > min_conf)
    out = 0
    for idx in ids:
        base = idx * 3
        if len(points) >= base + 3 and float(points[base + 2]) > min_conf:
            out += 1
    return out


def keypoint_summary(keypoint_path: Path | None, min_conf: float) -> tuple[int, int]:
    if keypoint_path is None or not keypoint_path.exists():
        return 0, 0
    try:
        data = json.loads(keypoint_path.read_text(encoding="utf-8"))
        person = data.get("people", [{}])[0]
        pose = person.get("pose_keypoints_2d", [])
        face = person.get("face_keypoints_2d", [])
        return point_count(pose, BODY_HEAD_IDS, min_conf), point_count(face, None, min_conf)
    except Exception:
        return 0, 0


def pkl_orientation(pkl_path: Path | None) -> tuple[float, float, float, float]:
    if pkl_path is None or not pkl_path.exists():
        return math.nan, math.nan, math.nan, math.nan
    try:
        with pkl_path.open("rb") as handle:
            data = pickle.load(handle, encoding="latin1")
        orient = np.asarray(data.get("global_orient"), dtype=np.float32).reshape(-1)[:3]
        if len(orient) < 3 or not np.isfinite(orient).all():
            return math.nan, math.nan, math.nan, math.nan
        norm = float(np.linalg.norm(orient))
        return norm, float(orient[0]), float(orient[1]), float(orient[2])
    except Exception:
        return math.nan, math.nan, math.nan, math.nan


def image_mask_stats(path: Path, threshold: int, min_component_area: int) -> dict[str, object]:
    if not path.exists():
        return {
            "exists": False,
            "foreground_ratio": 0.0,
            "foreground_pixels": 0,
            "bbox": "",
            "bbox_width": 0,
            "bbox_height": 0,
            "largest_component_ratio": 0.0,
            "component_count": 0,
            "border_touch_pixels": 0,
        }

    arr = np.asarray(Image.open(path).convert("L"))
    mask = arr > threshold
    fg_pixels = int(mask.sum())
    ratio = float(fg_pixels / max(mask.size, 1))
    if fg_pixels == 0:
        return {
            "exists": True,
            "foreground_ratio": ratio,
            "foreground_pixels": 0,
            "bbox": "",
            "bbox_width": 0,
            "bbox_height": 0,
            "largest_component_ratio": 0.0,
            "component_count": 0,
            "border_touch_pixels": 0,
        }

    ys, xs = np.where(mask)
    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    components = connected_components(mask, min_component_area)
    largest = max(components, default=0)
    border_touch = int(mask[0, :].sum() + mask[-1, :].sum() + mask[:, 0].sum() + mask[:, -1].sum())
    return {
        "exists": True,
        "foreground_ratio": ratio,
        "foreground_pixels": fg_pixels,
        "bbox": f"{min_x},{min_y},{max_x},{max_y}",
        "bbox_width": max_x - min_x + 1,
        "bbox_height": max_y - min_y + 1,
        "largest_component_ratio": float(largest / max(fg_pixels, 1)),
        "component_count": len(components),
        "border_touch_pixels": border_touch,
    }


def connected_components(mask: np.ndarray, min_area: int) -> list[int]:
    height, width = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[int] = []

    for start_y, start_x in zip(*np.where(mask & ~visited)):
        if visited[start_y, start_x]:
            continue
        area = 0
        queue: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        while queue:
            y, x = queue.popleft()
            area += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        if area >= min_area:
            components.append(area)

    return components


def lineart_ratio(path: Path, threshold: int) -> float:
    if not path.exists():
        return 0.0
    arr = np.asarray(Image.open(path).convert("L"))
    return float((arr > threshold).sum() / max(arr.size, 1))


def output_path(render_root: Path, pass_name: str, pose_id: str) -> Path:
    return render_root / pass_name / f"{pose_id}.png"


def discover_pose_ids(render_root: Path) -> list[str]:
    pose_ids: set[str] = set()
    for pass_name in ("depth", "lineart", "normal"):
        folder = render_root / pass_name
        if folder.exists():
            pose_ids.update(path.stem for path in folder.glob("*.png"))
    return sorted(pose_ids)


def classify_pose(pose_id: str, args: argparse.Namespace) -> RenderAudit:
    render_root = args.render_root
    depth_path = output_path(render_root, "depth", pose_id)
    lineart_path = output_path(render_root, "lineart", pose_id)
    normal_path = output_path(render_root, "normal", pose_id)
    category, image_path, keypoint_path = category_for_pose(pose_id)
    paths = smplx_paths(category, pose_id)
    pkl_path = paths.get("pkl_path")

    depth_stats = image_mask_stats(depth_path, args.depth_threshold, args.min_component_area)
    line_ratio = lineart_ratio(lineart_path, args.lineart_threshold)
    body_head_points, face_points = keypoint_summary(keypoint_path, args.keypoint_min_confidence)
    orient_norm, orient_x, orient_y, orient_z = pkl_orientation(pkl_path)

    reject_reasons: list[str] = []
    review_reasons: list[str] = []

    def add_conditional(reason: str, reject: bool) -> None:
        if reject:
            reject_reasons.append(reason)
        else:
            review_reasons.append(reason)

    if not depth_stats["exists"]:
        reject_reasons.append("missing_depth")
    if not lineart_path.exists() and "lineart" in args.required_passes:
        reject_reasons.append("missing_lineart")
    if not normal_path.exists() and "normal" in args.required_passes:
        reject_reasons.append("missing_normal")
    if depth_stats["foreground_pixels"] < args.min_foreground_pixels:
        reject_reasons.append("empty_depth")
    if depth_stats["foreground_ratio"] < args.min_foreground_ratio:
        reject_reasons.append("tiny_depth")
    if depth_stats["foreground_ratio"] > args.max_foreground_ratio:
        add_conditional("huge_depth_blob", args.reject_huge_depth)
    if depth_stats["component_count"] > args.max_components:
        add_conditional("fragmented_depth", args.reject_fragmented_depth)
    if 0 < depth_stats["largest_component_ratio"] < args.min_largest_component_ratio:
        add_conditional("detached_depth_parts", args.reject_fragmented_depth)
    if depth_stats["border_touch_pixels"] > args.max_border_touch_pixels:
        add_conditional("cropped_or_touching_border", args.reject_cropped)
    if lineart_path.exists() and line_ratio < args.min_lineart_ratio:
        add_conditional("mostly_blank_lineart", args.reject_blank_lineart)
    if body_head_points < args.min_head_points:
        add_conditional("headless_or_weak_source_keypoints", args.reject_headless_source)
    elif face_points == 0:
        add_conditional("body_head_present_but_no_face_keypoints", args.reject_missing_face)
    if not pkl_path or not pkl_path.exists():
        review_reasons.append("missing_smplifyx_pkl")
    elif math.isnan(orient_norm):
        review_reasons.append("invalid_global_orient")
    elif orient_norm > args.orientation_review_threshold:
        add_conditional("large_global_orient", args.reject_large_orientation)

    if reject_reasons:
        status = "reject"
    elif body_head_points < args.min_head_points:
        status = "headless"
    elif review_reasons:
        status = "review"
    else:
        status = "ok"

    return RenderAudit(
        pose_id=pose_id,
        category=category,
        status=status,
        reasons=";".join(reject_reasons),
        review_reasons=";".join(review_reasons),
        depth_path=str(depth_path.resolve()) if depth_path.exists() else "",
        lineart_path=str(lineart_path.resolve()) if lineart_path.exists() else "",
        normal_path=str(normal_path.resolve()) if normal_path.exists() else "",
        keypoint_path=str(keypoint_path.resolve()) if keypoint_path else "",
        image_path=str(image_path.resolve()) if image_path else "",
        pkl_path=str(pkl_path.resolve()) if pkl_path and pkl_path.exists() else "",
        foreground_ratio=float(depth_stats["foreground_ratio"]),
        foreground_pixels=int(depth_stats["foreground_pixels"]),
        bbox=str(depth_stats["bbox"]),
        bbox_width=int(depth_stats["bbox_width"]),
        bbox_height=int(depth_stats["bbox_height"]),
        largest_component_ratio=float(depth_stats["largest_component_ratio"]),
        component_count=int(depth_stats["component_count"]),
        border_touch_pixels=int(depth_stats["border_touch_pixels"]),
        lineart_ratio=line_ratio,
        body_head_points=body_head_points,
        face_points=face_points,
        global_orient_norm=orient_norm,
        global_orient_x=orient_x,
        global_orient_y=orient_y,
        global_orient_z=orient_z,
    )


def write_reports(rows: list[RenderAudit], report_root: Path, stamp: str) -> tuple[Path, Path]:
    report_root.mkdir(parents=True, exist_ok=True)
    csv_path = report_root / f"render_audit_{stamp}.csv"
    json_path = report_root / f"render_audit_{stamp}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()) if rows else list(RenderAudit.__annotations__))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    json_path.write_text(json.dumps([asdict(row) for row in rows], indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path, json_path


def quarantine(rows: list[RenderAudit], args: argparse.Namespace, stamp: str) -> int:
    statuses = set(args.move_status)
    moved = 0
    root = args.quarantine_root / stamp
    for row in rows:
        if row.status not in statuses:
            continue
        pose_id = row.pose_id
        for pass_name in args.passes:
            src = output_path(args.render_root, pass_name, pose_id)
            if src.exists():
                dst = root / "rendered" / pass_name / src.name
                move_path(src, dst)
                moved += 1

        if args.move_source and row.category in {"poses_normal", "poses_complex"}:
            for src_raw in [row.image_path, row.keypoint_path]:
                if not src_raw:
                    continue
                src = Path(src_raw)
                if src.exists():
                    rel = src.resolve().relative_to(DATASET_ROOT.resolve())
                    move_path(src, root / "source" / rel)
                    moved += 1
            for src in smplx_paths(row.category, pose_id).values():
                if src.exists():
                    rel = src.resolve().relative_to(DATASET_ROOT.resolve())
                    move_path(src, root / "source" / rel)
                    moved += 1
    return moved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit rendered depth/lineart/normal outputs and optionally quarantine bad files."
    )
    parser.add_argument("--render-root", type=Path, default=DEFAULT_RENDER_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--quarantine-root", type=Path, default=DATASET_ROOT / "rejected_renders")
    parser.add_argument("--passes", nargs="+", default=["depth", "lineart", "normal"])
    parser.add_argument("--required-passes", nargs="+", default=["depth"])
    parser.add_argument("--apply", action="store_true", help="Move matching files. Without this, only writes reports.")
    parser.add_argument("--move-status", nargs="+", default=["reject"], choices=["reject", "headless", "review", "ok"])
    parser.add_argument("--move-source", action="store_true", help="Also move source image/keypoint and SMPLify-X folders.")
    parser.add_argument("--depth-threshold", type=int, default=8)
    parser.add_argument("--lineart-threshold", type=int, default=8)
    parser.add_argument("--min-foreground-pixels", type=int, default=512)
    parser.add_argument("--min-foreground-ratio", type=float, default=0.003)
    parser.add_argument("--max-foreground-ratio", type=float, default=0.68)
    parser.add_argument("--min-lineart-ratio", type=float, default=0.0005)
    parser.add_argument("--min-component-area", type=int, default=16)
    parser.add_argument("--max-components", type=int, default=6)
    parser.add_argument("--min-largest-component-ratio", type=float, default=0.86)
    parser.add_argument("--max-border-touch-pixels", type=int, default=4)
    parser.add_argument("--min-head-points", type=int, default=3)
    parser.add_argument("--keypoint-min-confidence", type=float, default=0.05)
    parser.add_argument("--orientation-review-threshold", type=float, default=2.0)
    parser.add_argument("--reject-large-orientation", action="store_true")
    parser.add_argument("--reject-missing-face", action="store_true")
    parser.add_argument("--reject-headless-source", action="store_true")
    parser.add_argument("--reject-huge-depth", action="store_true")
    parser.add_argument("--reject-fragmented-depth", action="store_true")
    parser.add_argument("--reject-cropped", action="store_true")
    parser.add_argument("--reject-blank-lineart", action="store_true")
    args = parser.parse_args()
    args.render_root = args.render_root.resolve()
    args.report_root = args.report_root.resolve()
    args.quarantine_root = args.quarantine_root.resolve()
    args.passes = parse_passes(args.passes)
    args.required_passes = parse_passes(args.required_passes)
    ensure_inside(args.render_root, DATASET_ROOT)
    ensure_inside(args.report_root, DATASET_ROOT)
    ensure_inside(args.quarantine_root, DATASET_ROOT)
    return args


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pose_ids = discover_pose_ids(args.render_root)
    rows = [classify_pose(pose_id, args) for pose_id in pose_ids]

    csv_path, json_path = write_reports(rows, args.report_root, stamp)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1

    print(f"Audited: {len(rows)}")
    for status in ("reject", "headless", "review", "ok"):
        print(f"{status:8}: {counts.get(status, 0)}")
    print(f"CSV:      {csv_path}")
    print(f"JSON:     {json_path}")

    if args.apply:
        moved = quarantine(rows, args, stamp)
        print(f"Moved:    {moved} file/folder item(s) to {args.quarantine_root / stamp}")
    else:
        print("Dry run:  no files moved. Add --apply to quarantine files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
