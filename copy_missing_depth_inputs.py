from __future__ import annotations

import json
import shutil
from pathlib import Path


MISSING_LIST_PATH = Path(r"C:\Users\firew\Documents\python_scripts\openpose_pipeline\missing_depth_files.json")

SOURCE_ROOT = Path(r"C:\Users\firew\Documents\ComfyUI\models\openpose")
DEST_JSON_ROOT = Path(r"C:\Users\firew\Documents\python_scripts\openpose_pipeline\dataset\keypoints-18")
DEST_IMAGE_ROOT = Path(r"C:\Users\firew\Documents\python_scripts\openpose_pipeline\dataset\images")


def rename_json_filename(src_name: str) -> str:
    if not src_name.endswith("_openpose.json"):
        raise ValueError(f"Unexpected JSON filename format: {src_name}")
    return src_name.replace("_openpose.json", "_bone_structure_keypoints.json")


def source_image_filename(src_name: str) -> str:
    if not src_name.endswith("_openpose.json"):
        raise ValueError(f"Unexpected JSON filename format: {src_name}")
    return src_name.replace("_openpose.json", "_bone_structure.png")


def copy_item(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    DEST_JSON_ROOT.mkdir(parents=True, exist_ok=True)
    DEST_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)

    data = json.loads(MISSING_LIST_PATH.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("missing_depth_files.json must contain a list")

    copied_json = 0
    copied_png = 0
    missing_json = []
    missing_png = []

    for item in data:
        if not isinstance(item, dict) or "json_path" not in item:
            continue

        rel_json_path = Path(item["json_path"])
        src_json = SOURCE_ROOT / rel_json_path
        src_png = SOURCE_ROOT / rel_json_path.parent / source_image_filename(rel_json_path.name)

        dst_json = DEST_JSON_ROOT / rel_json_path.parent / rename_json_filename(rel_json_path.name)
        dst_png = DEST_IMAGE_ROOT / rel_json_path.parent / source_image_filename(rel_json_path.name)

        if copy_item(src_json, dst_json):
            copied_json += 1
        else:
            missing_json.append(str(src_json))

        if copy_item(src_png, dst_png):
            copied_png += 1
        else:
            missing_png.append(str(src_png))

    print(f"JSON copied: {copied_json}")
    print(f"PNG copied:  {copied_png}")

    if missing_json:
        print("\nMissing JSON files:")
        for p in missing_json:
            print(p)

    if missing_png:
        print("\nMissing PNG files:")
        for p in missing_png:
            print(p)


if __name__ == "__main__":
    main()