from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List, Optional


DATASET_ROOT = Path(r"C:\Users\firew\Documents\python_scripts\openpose_pipeline\dataset")

JSON_ROOTS = [
    DATASET_ROOT / "keypoints",
    DATASET_ROOT / "poses_normal" / "keypoints",
    DATASET_ROOT / "poses_complex" / "keypoints",
]

IMAGES_SRC = DATASET_ROOT / "images"

DEST_NORMAL_IMAGES = DATASET_ROOT / "poses_normal" / "images"
DEST_COMPLEX_IMAGES = DATASET_ROOT / "poses_complex" / "images"

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp"]

COMPLEX_KEYWORDS = {
    "kneeling",
    "all_fours",
    "lying",
    "split_leg",
    "suspended",
    "squatting",
    "dynamic_pose",
    "crouching",
    "prone",
    "bent_over",
}


def ensure_dirs() -> None:
    DEST_NORMAL_IMAGES.mkdir(parents=True, exist_ok=True)
    DEST_COMPLEX_IMAGES.mkdir(parents=True, exist_ok=True)


def json_base_from_name(name: str) -> str:
    stem = Path(name).stem
    for suffix in (
        "_bone_structure_keypoints",
        "_keypoints",
        "_openpose",
    ):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def json_text_fields(data: dict, src_path: Path) -> str:
    parts: List[str] = [src_path.as_posix(), src_path.stem]

    meta = data.get("meta", {})
    if isinstance(meta, dict):
        for key in ("source_name", "source_file", "folder", "pose", "variant", "subpose", "schema"):
            val = meta.get(key)
            if isinstance(val, str):
                parts.append(val)
        for key in ("attributes", "auto_attributes"):
            val = meta.get(key)
            if isinstance(val, list):
                parts.extend(str(v) for v in val if v is not None)

    return " ".join(parts).lower()


def classify_json(json_path: Path) -> str:
    """
    Returns:
        'poses_normal' or 'poses_complex'
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return "poses_complex"

    text = json_text_fields(data, json_path)

    if any(k in text for k in COMPLEX_KEYWORDS):
        return "poses_complex"

    people = data.get("people", [])
    if not isinstance(people, list) or not people:
        return "poses_complex"

    for person in people:
        pose = person.get("pose_keypoints_2d", [])
        if not isinstance(pose, list):
            return "poses_complex"
        if len(pose) % 3 != 0:
            return "poses_complex"

        n_kpts = len(pose) // 3
        if n_kpts not in {19, 25}:
            return "poses_complex"

        visible = sum(1 for c in pose[2::3] if float(c) > 0.0)
        if visible < 12:
            return "poses_complex"

    return "poses_normal"


def candidate_image_stems(json_path: Path) -> List[str]:
    stem = json_path.stem
    cands = []

    if stem.endswith("_bone_structure_keypoints"):
        cands.append(stem[: -len("_keypoints")])  # -> ..._bone_structure

    if stem.endswith("_keypoints"):
        cands.append(stem[: -len("_keypoints")])

    if stem.endswith("_openpose"):
        cands.append(stem[: -len("_openpose")] + "_bone_structure")

    # fallback: raw stem
    cands.append(stem)

    # de-dup while preserving order
    out = []
    seen = set()
    for s in cands:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def find_image_for_json(json_path: Path) -> Optional[Path]:
    stems = candidate_image_stems(json_path)

    for p in IMAGES_SRC.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        if p.stem in stems:
            return p

    return None


def move_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        dst.unlink()

    shutil.move(str(src), str(dst))
    return True


def collect_jsons() -> List[Path]:
    files: List[Path] = []
    for root in JSON_ROOTS:
        if root.exists():
            files.extend(root.rglob("*.json"))
    # de-dup
    uniq = []
    seen = set()
    for p in files:
        sp = str(p.resolve())
        if sp not in seen:
            uniq.append(p)
            seen.add(sp)
    return sorted(uniq)


def target_image_root(category: str) -> Path:
    return DEST_NORMAL_IMAGES if category == "poses_normal" else DEST_COMPLEX_IMAGES


def main() -> None:
    ensure_dirs()

    json_files = collect_jsons()
    print(f"Found {len(json_files)} JSON files")

    moved = 0
    missing = 0

    for jp in json_files:
        category = classify_json(jp)
        img = find_image_for_json(jp)

        if img is None:
            missing += 1
            print(f"[MISS] {jp.name} -> no image found")
            continue

        dst = target_image_root(category) / img.name

        if move_file(img, dst):
            moved += 1
            print(f"[OK]   {jp.name} -> {dst.parent.name}/{dst.name}")
        else:
            missing += 1
            print(f"[MISS] {jp.name} -> could not move image")

    print()
    print(f"Moved images: {moved}")
    print(f"Missing:      {missing}")


if __name__ == "__main__":
    main()