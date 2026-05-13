from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence


INPUT_DIR = Path(r"C:\Users\firew\Documents\python_scripts\openpose_pipeline\dataset\keypoints-18")
OUTPUT_DIR = Path(r"C:\Users\firew\Documents\python_scripts\openpose_pipeline\dataset\keypoints")


TARGET_KPTS = 25
TARGET_VALUES = TARGET_KPTS * 3


def zero_pt() -> List[float]:
    return [0.0, 0.0, 0.0]


def pt(points: Sequence[float], idx: int) -> List[float]:
    base = idx * 3
    return [float(points[base]), float(points[base + 1]), float(points[base + 2])]


def best_of(a: Sequence[float], b: Sequence[float]) -> List[float]:
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


def flatten_name(root: Path, file_path: Path) -> str:
    """
    standing/F/base/standing/file.json
    ->
    standing_F_base_standing_file.json
    """
    rel = file_path.relative_to(root)

    parts = list(rel.parts)

    filename = parts[-1]
    folders = parts[:-1]

    if folders:
        return "_".join(folders) + "_" + filename

    return filename


def normalize_to_25(points: Sequence[float]) -> List[float]:
    if len(points) % 3 != 0:
        raise ValueError(f"Invalid keypoint length: {len(points)}")

    num_kpts = len(points) // 3

    # Already >=25 -> clamp to first 25
    if num_kpts >= 25:
        return list(points[:TARGET_VALUES])

    # Convert 18 -> 25
    if num_kpts == 18:
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

        # Mid hip
        k25[8] = best_of(k25[9], k25[12])

        # Feet cannot be reconstructed
        for idx in [19, 20, 21, 22, 23, 24]:
            k25[idx] = zero_pt()

        flat: List[float] = []

        for x, y, c in k25:
            flat.extend([x, y, c])

        return flat

    raise ValueError(
        f"Unsupported keypoint count: {num_kpts}"
    )


def convert_file(src: Path, dst: Path) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))

    if "people" not in data:
        raise ValueError(f"{src} has no people list")

    for person in data["people"]:
        pose = person.get("pose_keypoints_2d", [])

        if isinstance(pose, list):
            person["pose_keypoints_2d"] = normalize_to_25(pose)

    dst.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(INPUT_DIR.rglob("*.json"))

    print(f"Found {len(files)} files")

    for src in files:
        flat_name = flatten_name(INPUT_DIR, src)

        dst = OUTPUT_DIR / flat_name

        try:
            convert_file(src, dst)
            print(f"[OK] {flat_name}")

        except Exception as e:
            print(f"[ERROR] {src}")
            print(f"        {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()