from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DATASET_ROOT = Path(r"C:\Users\firew\Documents\python_scripts\openpose_pipeline\dataset")

SRC_IMAGES = DATASET_ROOT / "images"
SRC_KEYPOINTS = DATASET_ROOT / "keypoints"
SRC_NORMAL = DATASET_ROOT / "normal"
SRC_LINEART = DATASET_ROOT / "lineart"
SRC_SMPLX = DATASET_ROOT / "smplx"

DEST_ROOT_NORMAL = DATASET_ROOT / "poses_normal"
DEST_ROOT_COMPLEX = DATASET_ROOT / "poses_complex"

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".webp"]
VALID_KPT_COUNTS = {19, 25}

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


def ensure_target_tree(root: Path) -> None:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "keypoints").mkdir(parents=True, exist_ok=True)
    (root / "normal").mkdir(parents=True, exist_ok=True)
    (root / "lineart").mkdir(parents=True, exist_ok=True)
    (root / "smplx" / "images").mkdir(parents=True, exist_ok=True)
    (root / "smplx" / "meshes").mkdir(parents=True, exist_ok=True)
    (root / "smplx" / "results").mkdir(parents=True, exist_ok=True)


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


def count_valid_keypoints(pose: List[float]) -> Tuple[int, int]:
    """
    Returns:
        (num_keypoints, num_conf_positive)
    """
    if len(pose) % 3 != 0:
        return 0, 0

    n = len(pose) // 3
    confs = pose[2::3]
    positive = sum(1 for c in confs if float(c) > 0.0)
    return n, positive


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

    # Keyword-first route
    if any(k in text for k in COMPLEX_KEYWORDS):
        return "poses_complex"

    people = data.get("people", [])
    if not isinstance(people, list) or not people:
        return "poses_complex"

    counts: List[int] = []
    visibles: List[int] = []

    for person in people:
        pose = person.get("pose_keypoints_2d", [])
        if not isinstance(pose, list):
            return "poses_complex"

        n_kpts, n_visible = count_valid_keypoints([float(x) for x in pose] if pose else [])
        if n_kpts == 0:
            return "poses_complex"

        counts.append(n_kpts)
        visibles.append(n_visible)

    # Only 19 or 25 is acceptable for normal routing
    if any(c not in VALID_KPT_COUNTS for c in counts):
        return "poses_complex"

    # If too few visible points, treat as complex
    if min(visibles) < 12:
        return "poses_complex"

    return "poses_normal"


def safe_replace_path(dst: Path) -> None:
    if dst.is_dir():
        shutil.rmtree(dst)
    elif dst.exists():
        dst.unlink()


def move_path(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        safe_replace_path(dst)

    shutil.move(str(src), str(dst))
    return True


def find_exact_file_by_stem(root: Path, base: str, exts: List[str]) -> Optional[Path]:
    """
    Finds a file whose stem matches base.
    Searches recursively.
    """
    if not root.exists():
        return None

    candidates = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and p.stem == base:
            candidates.append(p)

    if not candidates:
        return None

    return sorted(candidates)[0]


def find_exact_dir(root: Path, base: str) -> Optional[Path]:
    """
    Finds a directory named base below root. Searches recursively.
    """
    if not root.exists():
        return None

    candidates = []
    for p in root.rglob("*"):
        if p.is_dir() and p.name == base:
            candidates.append(p)

    if not candidates:
        return None

    return sorted(candidates)[0]


def move_smplx_bundle(base: str, dest_root: Path) -> None:
    """
    Moves:
      smplx/images/<base>
      smplx/meshes/<base>
      smplx/results/<base>
    into:
      dest_root/smplx/images/<base>
      dest_root/smplx/meshes/<base>
      dest_root/smplx/results/<base>
    """
    for sub in ("images", "meshes", "results"):
        src_dir = SRC_SMPLX / sub / base
        if src_dir.exists():
            dst_dir = dest_root / "smplx" / sub / base
            move_path(src_dir, dst_dir)


def move_related_assets(json_path: Path, category_root: Path) -> None:
    """
    Move json, image, normal, lineart and matching smplx subfolders for one pose.
    """
    base = json_base_from_name(json_path.name)

    # JSON -> keypoints
    dst_json = category_root / "keypoints" / json_path.name
    move_path(json_path, dst_json)

    # Image -> images
    img = find_exact_file_by_stem(SRC_IMAGES, base, IMAGE_EXTS)
    if img is not None:
        dst_img = category_root / "images" / img.name
        move_path(img, dst_img)

    # Normal -> normal
    normal_item = find_exact_file_by_stem(SRC_NORMAL, base, IMAGE_EXTS)
    if normal_item is not None:
        dst_normal = category_root / "normal" / normal_item.name
        move_path(normal_item, dst_normal)
    else:
        normal_dir = find_exact_dir(SRC_NORMAL, base)
        if normal_dir is not None:
            dst_normal_dir = category_root / "normal" / normal_dir.name
            move_path(normal_dir, dst_normal_dir)

    # Lineart -> lineart
    lineart_item = find_exact_file_by_stem(SRC_LINEART, base, IMAGE_EXTS)
    if lineart_item is not None:
        dst_lineart = category_root / "lineart" / lineart_item.name
        move_path(lineart_item, dst_lineart)
    else:
        lineart_dir = find_exact_dir(SRC_LINEART, base)
        if lineart_dir is not None:
            dst_lineart_dir = category_root / "lineart" / lineart_dir.name
            move_path(lineart_dir, dst_lineart_dir)

    # SMPLX bundle
    move_smplx_bundle(base, category_root)


def main() -> None:
    ensure_target_tree(DEST_ROOT_NORMAL)
    ensure_target_tree(DEST_ROOT_COMPLEX)

    json_files = sorted(SRC_KEYPOINTS.rglob("*.json"))
    print(f"Found {len(json_files)} JSON files")

    moved_normal = 0
    moved_complex = 0

    for jp in json_files:
        category = classify_json(jp)
        if category == "poses_normal":
            move_related_assets(jp, DEST_ROOT_NORMAL)
            moved_normal += 1
            print(f"[NORMAL]  {jp.name}")
        else:
            move_related_assets(jp, DEST_ROOT_COMPLEX)
            moved_complex += 1
            print(f"[COMPLEX] {jp.name}")

    print("\nDone.")
    print(f"Normal:  {moved_normal}")
    print(f"Complex: {moved_complex}")


if __name__ == "__main__":
    main()