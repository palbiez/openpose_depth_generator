from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


DATASET_ROOT = Path(r"C:\Users\firew\Documents\python_scripts\openpose_pipeline\dataset")

KEYPOINTS_DIR = DATASET_ROOT / "keypoints"
IMAGES_DIR = DATASET_ROOT / "images"
SMPLX_DIR = DATASET_ROOT / "smplx"
SMPLX_IMAGES_DIR = SMPLX_DIR / "images"
SMPLX_MESHES_DIR = SMPLX_DIR / "meshes"
SMPLX_RESULTS_DIR = SMPLX_DIR / "results"
BACKUP_DIR = DATASET_ROOT / "backup"

VALID_KPT_COUNTS = {19, 25}


def json_base_from_name(name: str) -> str:
    stem = Path(name).stem
    for suffix in ("_bone_structure_keypoints", "_keypoints", "_openpose"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def find_image_for_base(base: str) -> Optional[Path]:
    # exact stem match is the intended case for your dataset
    for p in IMAGES_DIR.rglob("*.png"):
        if p.stem == base:
            return p
    for p in IMAGES_DIR.rglob("*.jpg"):
        if p.stem == base:
            return p
    for p in IMAGES_DIR.rglob("*.jpeg"):
        if p.stem == base:
            return p
    for p in IMAGES_DIR.rglob("*.webp"):
        if p.stem == base:
            return p
    return None


def find_jsons_for_base(base: str) -> List[Path]:
    hits: List[Path] = []
    for p in KEYPOINTS_DIR.rglob("*.json"):
        if json_base_from_name(p.name) == base:
            hits.append(p)
    return hits


def keypoint_count_ok(json_path: Path) -> bool:
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    people = data.get("people", [])
    if not isinstance(people, list) or not people:
        return False

    for person in people:
        pose = person.get("pose_keypoints_2d", [])
        if not isinstance(pose, list) or len(pose) % 3 != 0:
            return False
        if (len(pose) // 3) not in VALID_KPT_COUNTS:
            return False

    return True


def move_path(src: Path) -> bool:
    if not src.exists():
        return False

    rel = src.relative_to(DATASET_ROOT)
    dst = BACKUP_DIR / rel
    dst.parent.mkdir(parents=True, exist_ok=True)

    # avoid overwriting if rerun
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()

    shutil.move(str(src), str(dst))
    return True


def move_dir_if_exists(src_dir: Path) -> bool:
    if not src_dir.exists():
        return False
    return move_path(src_dir)


def move_related_for_base(base: str, reason: str) -> Tuple[int, int, int, int]:
    moved_json = 0
    moved_png = 0
    moved_smplx_images = 0
    moved_smplx_dirs = 0

    # JSONs
    for jp in find_jsons_for_base(base):
        if move_path(jp):
            moved_json += 1

    # PNG
    img = find_image_for_base(base)
    if img and move_path(img):
        moved_png += 1

    # SMPLX subfolders
    for root in (SMPLX_IMAGES_DIR, SMPLX_MESHES_DIR, SMPLX_RESULTS_DIR):
        d = root / base
        if d.exists():
            if move_path(d):
                if root == SMPLX_IMAGES_DIR:
                    moved_smplx_images += 1
                else:
                    moved_smplx_dirs += 1

    if moved_json or moved_png or moved_smplx_images or moved_smplx_dirs:
        print(
            f"[{reason}] {base} -> "
            f"json:{moved_json} png:{moved_png} "
            f"smplx_images:{moved_smplx_images} smplx_dirs:{moved_smplx_dirs}"
        )

    return moved_json, moved_png, moved_smplx_images, moved_smplx_dirs


def main() -> None:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    # 1) processed = every folder under smplx/meshes
    processed_bases: Set[str] = set()
    if SMPLX_MESHES_DIR.exists():
        for p in SMPLX_MESHES_DIR.iterdir():
            if p.is_dir():
                processed_bases.add(p.name)

    # 2) invalid JSONs = not 19 or 25 keypoints
    invalid_bases: Set[str] = set()
    for jp in KEYPOINTS_DIR.rglob("*.json"):
        if not keypoint_count_ok(jp):
            invalid_bases.add(json_base_from_name(jp.name))

    print(f"Processed bases: {len(processed_bases)}")
    print(f"Invalid bases:   {len(invalid_bases)}")

    # Move processed first
    for base in sorted(processed_bases):
        move_related_for_base(base, "processed")

    # Then move invalid ones
    for base in sorted(invalid_bases):
        move_related_for_base(base, "invalid-kpts")

    print("\nDone.")
    print(f"Backup root: {BACKUP_DIR}")


if __name__ == "__main__":
    main()