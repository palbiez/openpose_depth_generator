from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "dataset" / "bone_structure_qc" / "redrawn"
DEFAULT_REPORT = PROJECT_ROOT / "dataset" / "bone_structure_qc" / "report.jsonl"
DEFAULT_NORMALIZED_ROOT = PROJECT_ROOT / "dataset" / "bone_structure_qc" / "normalized_keypoints"
DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 512

BODY_25_LIMBS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (8, 12),
    (12, 13),
    (13, 14),
    (0, 15),
    (15, 17),
    (0, 16),
    (16, 18),
    (14, 19),
    (19, 20),
    (14, 21),
    (11, 22),
    (22, 23),
    (11, 24),
]

OPENPOSE_COLORS = [
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
    (180, 180, 180),
    (220, 220, 220),
    (180, 220, 255),
    (255, 220, 180),
    (220, 180, 255),
    (180, 255, 220),
]

JSON_SUFFIXES = (
    "_bone_structure_keypoints",
    "_keypoints",
    "_openpose",
)


def zero_pt() -> list[float]:
    return [0.0, 0.0, 0.0]


def pt(points: Sequence[float], idx: int) -> list[float]:
    base = idx * 3
    return [float(points[base]), float(points[base + 1]), float(points[base + 2])]


def best_of(a: Sequence[float], b: Sequence[float]) -> list[float]:
    if a[2] <= 0 and b[2] <= 0:
        return zero_pt()
    if a[2] <= 0:
        return [float(b[0]), float(b[1]), float(b[2])]
    if b[2] <= 0:
        return [float(a[0]), float(a[1]), float(a[2])]
    return [
        (float(a[0]) + float(b[0])) / 2.0,
        (float(a[1]) + float(b[1])) / 2.0,
        min(float(a[2]), float(b[2])),
    ]


def convert_18_to_25(points: Sequence[float]) -> list[float]:
    if len(points) != 18 * 3:
        raise ValueError(f"Expected 18 keypoints, got {len(points) // 3}")

    k25 = [zero_pt() for _ in range(25)]

    k25[0] = pt(points, 0)
    k25[1] = pt(points, 1)
    k25[2] = pt(points, 2)
    k25[3] = pt(points, 3)
    k25[4] = pt(points, 4)
    k25[5] = pt(points, 5)
    k25[6] = pt(points, 6)
    k25[7] = pt(points, 7)
    k25[9] = pt(points, 8)
    k25[10] = pt(points, 9)
    k25[11] = pt(points, 10)
    k25[12] = pt(points, 11)
    k25[13] = pt(points, 12)
    k25[14] = pt(points, 13)
    k25[15] = pt(points, 14)
    k25[16] = pt(points, 15)
    k25[17] = pt(points, 16)
    k25[18] = pt(points, 17)
    k25[8] = best_of(k25[9], k25[12])

    flat: list[float] = []
    for x, y, confidence in k25:
        flat.extend([x, y, confidence])
    return flat


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON payload must be an object")
    return data


def numeric_keypoints(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    out: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)):
            return None
        out.append(float(item))
    return out


def normalize_pose_keypoints(points: Sequence[float]) -> tuple[list[float] | None, str, str | None]:
    if len(points) % 3 != 0:
        return None, "invalid_length", f"value count {len(points)} is not divisible by 3"

    count = len(points) // 3
    if count == 18:
        return convert_18_to_25(points), "converted_18_to_25", None
    if count == 25:
        return [float(value) for value in points[: 25 * 3]], "body25", None

    return (
        None,
        "unsupported_point_count",
        f"pose_keypoints_2d has {count} points; expected 18 or 25. "
        "This usually means raw sampled skeleton/lineart points, not OpenPose BODY_18/BODY_25.",
    )


def people_from_payload(data: dict[str, Any]) -> list[dict[str, Any]]:
    people = data.get("people")
    if isinstance(people, list):
        return [person for person in people if isinstance(person, dict)]

    if "pose_keypoints_2d" in data or "keypoints" in data:
        return [data]

    return []


def strip_json_suffix(stem: str) -> str:
    for suffix in JSON_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def bone_stem_from_json(path: Path) -> str:
    stem = strip_json_suffix(path.stem)
    if stem.endswith("_bone_structure"):
        return stem
    return f"{stem}_bone_structure"


def relative_output_path(input_path: Path, root: Path | None, output_root: Path, suffix: str) -> Path:
    if root is not None:
        try:
            rel_parent = input_path.resolve().parent.relative_to(root.resolve())
        except ValueError:
            rel_parent = Path()
    else:
        rel_parent = Path()

    return output_root / rel_parent / f"{bone_stem_from_json(input_path)}{suffix}"


def canvas_size(data: dict[str, Any], reference_image: Path | None, width: int | None, height: int | None) -> tuple[int, int]:
    if reference_image is not None:
        with Image.open(reference_image) as image:
            return image.size

    raw_width = width or data.get("canvas_width") or data.get("width") or DEFAULT_WIDTH
    raw_height = height or data.get("canvas_height") or data.get("height") or DEFAULT_HEIGHT
    return int(raw_width), int(raw_height)


def valid_bbox(points: Sequence[float], min_confidence: float) -> dict[str, Any]:
    triples = [points[idx : idx + 3] for idx in range(0, len(points), 3)]
    valid = [(x, y, c) for x, y, c in triples if c > min_confidence]
    if not valid:
        return {"valid_points": 0}

    xs = [item[0] for item in valid]
    ys = [item[1] for item in valid]
    return {
        "valid_points": len(valid),
        "min_x": min(xs),
        "min_y": min(ys),
        "max_x": max(xs),
        "max_y": max(ys),
        "width": max(xs) - min(xs),
        "height": max(ys) - min(ys),
    }


def audit_points(points: Sequence[float], width: int, height: int, min_confidence: float) -> list[str]:
    warnings: list[str] = []
    bbox = valid_bbox(points, min_confidence)
    valid_count = int(bbox.get("valid_points", 0))
    if valid_count < 8:
        warnings.append(f"low_valid_keypoint_count:{valid_count}")
        return warnings

    if float(bbox["width"]) < width * 0.08:
        warnings.append("narrow_keypoint_bbox")
    if float(bbox["height"]) < height * 0.08:
        warnings.append("flat_keypoint_bbox")

    triples = [points[idx : idx + 3] for idx in range(0, len(points), 3)]
    border_points = 0
    for x, y, confidence in triples:
        if confidence <= min_confidence:
            continue
        if x <= 1 or y <= 1 or x >= width - 2 or y >= height - 2:
            border_points += 1

    if border_points >= 2:
        warnings.append(f"many_border_keypoints:{border_points}")

    bbox_diag = math.hypot(float(bbox["width"]), float(bbox["height"]))
    if bbox_diag > 1:
        triples = [points[idx : idx + 3] for idx in range(0, len(points), 3)]
        limb_lengths = []
        for a, b in BODY_25_LIMBS:
            if triples[a][2] <= min_confidence or triples[b][2] <= min_confidence:
                continue
            limb_lengths.append(math.hypot(triples[a][0] - triples[b][0], triples[a][1] - triples[b][1]))

        long_limbs = [length for length in limb_lengths if length > bbox_diag * 0.45]
        if len(long_limbs) >= max(4, len(limb_lengths) // 4):
            warnings.append(f"implausible_long_limb_count:{len(long_limbs)}/{len(limb_lengths)}")

    return warnings


def draw_body25(
    people: Sequence[Sequence[float]],
    width: int,
    height: int,
    line_width: int,
    point_radius: int,
    style: str,
    min_confidence: float,
) -> Image.Image:
    scale = 2
    image = Image.new("RGB", (width * scale, height * scale), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    white = (255, 255, 255)

    for points in people:
        triples = [points[idx : idx + 3] for idx in range(0, len(points), 3)]
        for limb_index, (a, b) in enumerate(BODY_25_LIMBS):
            if triples[a][2] <= min_confidence or triples[b][2] <= min_confidence:
                continue
            color = OPENPOSE_COLORS[limb_index % len(OPENPOSE_COLORS)] if style == "openpose_color" else white
            ax, ay, _ = triples[a]
            bx, by, _ = triples[b]
            draw.line(
                (
                    int(round(ax * scale)),
                    int(round(ay * scale)),
                    int(round(bx * scale)),
                    int(round(by * scale)),
                ),
                fill=color,
                width=max(1, int(line_width) * scale),
            )

        for point_index, (x, y, confidence) in enumerate(triples):
            if confidence <= min_confidence:
                continue
            color = OPENPOSE_COLORS[point_index % len(OPENPOSE_COLORS)] if style == "openpose_color" else white
            radius = max(1, int(point_radius)) * scale
            cx = int(round(x * scale))
            cy = int(round(y * scale))
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)

    if scale == 1:
        return image
    return image.resize((width, height), Image.Resampling.LANCZOS)


def draw_raw_diagnostic(
    people: Sequence[Sequence[float]],
    width: int,
    height: int,
    output_path: Path,
    min_confidence: float,
) -> None:
    image = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    colors = [(255, 255, 255), (255, 180, 40), (40, 220, 255), (255, 40, 180)]

    for points in people:
        triples = [points[idx : idx + 3] for idx in range(0, len(points), 3)]
        for index, (x, y, confidence) in enumerate(triples):
            if confidence <= min_confidence:
                continue
            color = colors[(index // 25) % len(colors)]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            draw.text((x + 4, y + 2), str(index), fill=color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def normalized_payload(data: dict[str, Any], normalized_people: Sequence[list[float]]) -> dict[str, Any]:
    payload = copy.deepcopy(data)
    people = people_from_payload(payload)
    if not people:
        payload["people"] = []
        people = payload["people"]

    for person, points in zip(people, normalized_people):
        person["pose_keypoints_2d"] = points
        person.setdefault("face_keypoints_2d", [])
        person.setdefault("hand_left_keypoints_2d", [])
        person.setdefault("hand_right_keypoints_2d", [])

    return payload


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_inputs(args: argparse.Namespace) -> list[Path]:
    inputs: list[Path] = []
    if args.json:
        inputs.extend(path.resolve() for path in args.json)
    if args.input_root:
        for root in args.input_root:
            inputs.extend(sorted(path.resolve() for path in root.rglob(args.glob) if path.is_file()))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in inputs:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def process_json(path: Path, args: argparse.Namespace, input_root: Path | None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "json": str(path),
        "status": "invalid",
        "warnings": [],
        "actions": [],
    }

    try:
        data = load_json(path)
        width, height = canvas_size(data, args.reference_image, args.width, args.height)
        record["canvas"] = {"width": width, "height": height}
        people = people_from_payload(data)
        if not people:
            record["error"] = "no_people"
            return record

        normalized_people: list[list[float]] = []
        raw_people: list[list[float]] = []
        point_counts: list[int] = []
        layouts: list[str] = []
        errors: list[str] = []
        warnings: list[str] = []

        for person in people:
            raw = numeric_keypoints(person.get("pose_keypoints_2d"))
            if raw is None:
                raw = numeric_keypoints(person.get("keypoints"))
            if raw is None:
                errors.append("person has no numeric pose_keypoints_2d/keypoints")
                continue

            raw_people.append(raw)
            point_counts.append(len(raw) // 3 if len(raw) % 3 == 0 else -1)
            normalized, layout, error = normalize_pose_keypoints(raw)
            layouts.append(layout)
            if error:
                errors.append(error)
                continue

            assert normalized is not None
            normalized_people.append(normalized)
            warnings.extend(audit_points(normalized, width, height, args.min_confidence))

        record["point_counts"] = point_counts
        record["layouts"] = layouts

        if args.write_diagnostics and raw_people:
            diagnostic_path = relative_output_path(
                path,
                input_root,
                args.output_root / "_diagnostics",
                "_raw_points.png",
            )
            draw_raw_diagnostic(raw_people, width, height, diagnostic_path, args.min_confidence)
            record["diagnostic"] = str(diagnostic_path)
            record["actions"].append("wrote_diagnostic")

        if errors:
            record["error"] = "; ".join(errors)
            return record

        severe_warnings = [warning for warning in warnings if warning.startswith("implausible_")]
        if severe_warnings and not args.allow_implausible:
            record["warnings"] = sorted(set(warnings))
            record["error"] = (
                "implausible BODY_25 geometry; not writing replacement image. "
                "Use --allow-implausible only for manual diagnostics."
            )
            return record

        output_path = args.output or relative_output_path(path, input_root, args.output_root, ".png")
        if output_path.exists() and not args.force:
            record["status"] = "ok"
            record["output"] = str(output_path)
            record["actions"].append("kept_existing_image")
        else:
            image = draw_body25(
                normalized_people,
                width,
                height,
                args.line_width,
                args.point_radius,
                args.style,
                args.min_confidence,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path)
            record["status"] = "ok"
            record["output"] = str(output_path)
            record["actions"].append("redrew_bone_structure")

        if args.write_normalized_json:
            normalized_root = args.normalized_json_root
            normalized_path = args.normalized_json_output or relative_output_path(
                path,
                input_root,
                normalized_root,
                "_keypoints.json",
            )
            normalized_path.parent.mkdir(parents=True, exist_ok=True)
            normalized_path.write_text(
                json.dumps(normalized_payload(data, normalized_people), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            record["normalized_json"] = str(normalized_path)
            record["actions"].append("wrote_normalized_json")

        record["warnings"] = sorted(set(warnings))
        return record
    except Exception as exc:
        record["error"] = repr(exc)
        return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit OpenPose JSON files and redraw bone-structure PNGs from valid BODY_18/BODY_25 keypoints."
    )
    parser.add_argument("--json", action="append", type=Path, help="Single JSON file. Can be passed more than once.")
    parser.add_argument("--input-root", action="append", type=Path, help="Root to scan recursively for JSON files.")
    parser.add_argument("--glob", default="*.json", help="Glob used with --input-root.")
    parser.add_argument("--output", type=Path, help="Output PNG for a single --json input.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--reference-image", type=Path, help="Optional image whose size should be reused.")
    parser.add_argument("--width", type=int, help="Fallback canvas width.")
    parser.add_argument("--height", type=int, help="Fallback canvas height.")
    parser.add_argument("--style", choices=["openpose_color", "white"], default="openpose_color")
    parser.add_argument("--line-width", type=int, default=4)
    parser.add_argument("--point-radius", type=int, default=4)
    parser.add_argument("--min-confidence", type=float, default=0.05)
    parser.add_argument("--write-normalized-json", action="store_true")
    parser.add_argument("--normalized-json-output", type=Path)
    parser.add_argument("--normalized-json-root", type=Path, default=DEFAULT_NORMALIZED_ROOT)
    parser.add_argument("--write-diagnostics", action="store_true")
    parser.add_argument(
        "--allow-implausible",
        action="store_true",
        help="Write redrawn images even when BODY_25 geometry looks semantically broken.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Return exit code 1 when any JSON is invalid.")
    args = parser.parse_args()

    args.output_root = args.output_root.resolve()
    args.report = args.report.resolve()
    args.normalized_json_root = args.normalized_json_root.resolve()
    if args.output is not None:
        args.output = args.output.resolve()
    if args.normalized_json_output is not None:
        args.normalized_json_output = args.normalized_json_output.resolve()
    if args.reference_image is not None:
        args.reference_image = args.reference_image.resolve()
    if args.input_root:
        args.input_root = [root.resolve() for root in args.input_root]

    if args.output and (not args.json or len(args.json) != 1):
        parser.error("--output can only be used with exactly one --json input")
    if args.normalized_json_output and (not args.json or len(args.json) != 1):
        parser.error("--normalized-json-output can only be used with exactly one --json input")

    return args


def main() -> int:
    args = parse_args()
    inputs = collect_inputs(args)
    if not inputs:
        print("No JSON inputs found.", file=sys.stderr)
        return 2

    records: list[dict[str, Any]] = []
    input_root = args.input_root[0] if args.input_root and len(args.input_root) == 1 else None
    for index, path in enumerate(inputs, start=1):
        record = process_json(path, args, input_root)
        records.append(record)
        status = record.get("status")
        point_counts = ",".join(str(value) for value in record.get("point_counts", []))
        suffix = f" ({point_counts} pts)" if point_counts else ""
        print(f"[{index}/{len(inputs)}] {status}: {path}{suffix}")
        if record.get("error"):
            print(f"  {record['error']}")

    write_jsonl(args.report, records)

    invalid = [record for record in records if record.get("status") != "ok"]
    print(f"Report:  {args.report}")
    print(f"Valid:   {len(records) - len(invalid)}")
    print(f"Invalid: {len(invalid)}")

    return 1 if args.strict and invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
