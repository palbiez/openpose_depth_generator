from __future__ import annotations

import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# SOURCE ROOTS
KEYPOINTS_SRC = PROJECT_ROOT / "dataset" / "keypoints"

IMAGES_SRC = PROJECT_ROOT / "dataset" / "images"


# FLAT TARGETS
KEYPOINTS_DST = PROJECT_ROOT / "dataset" / "keypoints"

IMAGES_DST = PROJECT_ROOT / "dataset" / "images"


VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def flatten_name(root: Path, file_path: Path) -> str:
    """
    Example:
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


def copy_flattened(src_root: Path, dst_root: Path, pattern: str) -> None:
    dst_root.mkdir(parents=True, exist_ok=True)

    files = sorted(src_root.rglob(pattern))

    copied = 0
    skipped = 0

    for src in files:
        flat_name = flatten_name(src_root, src)
        dst = dst_root / flat_name

        if dst.exists():
            print(f"[SKIP EXISTS] {dst.name}")
            skipped += 1
            continue

        shutil.copy2(src, dst)
        copied += 1
        print(f"[COPY] {src} -> {dst.name}")

    print()
    print(f"Done: {src_root}")
    print(f"Copied: {copied}")
    print(f"Skipped: {skipped}")
    print()


def main() -> None:
    print("Flattening keypoints...")
    copy_flattened(KEYPOINTS_SRC, KEYPOINTS_DST, "*.json")

    print("Flattening images...")
    for ext in VALID_IMAGE_EXTS:
        copy_flattened(IMAGES_SRC, IMAGES_DST, f"*{ext}")


if __name__ == "__main__":
    main()
