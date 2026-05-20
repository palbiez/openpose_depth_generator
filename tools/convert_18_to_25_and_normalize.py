from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "dataset" / "keypoints-18"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "keypoints"


def pt(points: Sequence[float], idx: int) -> List[float]:
    base = idx * 3
    return [float(points[base]), float(points[base + 1]), float(points[base + 2])]


def zero_pt() -> List[float]:
    return [0.0, 0.0, 0.0]


def best_of(a: Sequence[float], b: Sequence[float]) -> List[float]:
    if len(a) != 3 or len(b) != 3:
        return zero_pt()
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


def convert_18_to_25(k18: Sequence[float]) -> List[float]:
    if len(k18) != 18 * 3:
        raise ValueError(f"Expected 54 values (18*3), got {len(k18)}")

    k25 = [zero_pt() for _ in range(25)]

    k25[0] = pt(k18, 0)
    k25[1] = pt(k18, 1)
    k25[2] = pt(k18, 2)
    k25[3] = pt(k18, 3)
    k25[4] = pt(k18, 4)
    k25[5] = pt(k18, 5)
    k25[6] = pt(k18, 6)
    k25[7] = pt(k18, 7)
    k25[9] = pt(k18, 8)
    k25[10] = pt(k18, 9)
    k25[11] = pt(k18, 10)
    k25[12] = pt(k18, 11)
    k25[13] = pt(k18, 12)
    k25[14] = pt(k18, 13)
    k25[15] = pt(k18, 14)
    k25[16] = pt(k18, 15)
    k25[17] = pt(k18, 16)
    k25[18] = pt(k18, 17)

    k25[8] = best_of(k25[9], k25[12])

    # feet are not reconstructable from 18 points
    for idx in [19, 20, 21, 22, 23, 24]:
        k25[idx] = zero_pt()

    flat: List[float] = []
    for x, y, c in k25:
        flat.extend([x, y, c])
    return flat


def convert_file(src: Path, dst: Path) -> None:
    data = json.loads(src.read_text(encoding="utf-8"))

    if "people" not in data or not isinstance(data["people"], list):
        raise ValueError(f"{src}: no people list found")

    for person in data["people"]:
        pose = person.get("pose_keypoints_2d")
        if isinstance(pose, list) and len(pose) == 54:
            person["pose_keypoints_2d"] = convert_18_to_25(pose)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for src in sorted(INPUT_DIR.rglob("*.json")):
        rel = src.relative_to(INPUT_DIR)
        dst = OUTPUT_DIR / rel
        convert_file(src, dst)
        print(f"converted: {rel}")


if __name__ == "__main__":
    main()
