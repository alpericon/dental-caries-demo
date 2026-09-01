"""
Converts the DENTEX quadrant-enumeration-disease subset (COCO-style JSON,
diagnosis label = category_id_3) into a YOLOv8 dataset under data/.

Usage: python scripts/prepare_dentex_data.py

Expects the extracted DENTEX files at:
  downloads/extracted/training_data/quadrant-enumeration-disease/
    train_quadrant_enumeration_disease.json
    xrays/*.png

Produces:
  data/data.yaml
  data/images/{train,val}/*.png
  data/labels/{train,val}/*.txt
"""

import json
import random
import shutil
from pathlib import Path

SOURCE_DIR = Path("downloads/extracted/training_data/quadrant-enumeration-disease")
ANNOTATIONS_JSON = SOURCE_DIR / "train_quadrant_enumeration_disease.json"
IMAGES_DIR = SOURCE_DIR / "xrays"

OUT_DIR = Path("data")
VAL_FRACTION = 0.15
SEED = 42

# category_id_3 in the DENTEX JSON, in id order (0-indexed) -> our YOLO class ids
CLASS_NAMES = ["Impacted", "Caries", "Periapical Lesion", "Deep Caries"]


def main():
    if not ANNOTATIONS_JSON.exists():
        raise FileNotFoundError(
            f"Couldn't find {ANNOTATIONS_JSON}. Extract the DENTEX "
            "quadrant-enumeration-disease subset first."
        )

    with open(ANNOTATIONS_JSON) as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}

    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    image_ids = list(images_by_id.keys())
    random.Random(SEED).shuffle(image_ids)
    n_val = round(len(image_ids) * VAL_FRACTION)
    val_ids = set(image_ids[:n_val])

    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    for image_id, img in images_by_id.items():
        split = "val" if image_id in val_ids else "train"
        file_name = img["file_name"]
        width, height = img["width"], img["height"]

        src_image = IMAGES_DIR / file_name
        dst_image = OUT_DIR / "images" / split / file_name
        shutil.copyfile(src_image, dst_image)

        lines = []
        for ann in anns_by_image.get(image_id, []):
            class_id = ann["category_id_3"]
            x, y, w, h = ann["bbox"]  # COCO: top-left x, y, width, height
            x_center = (x + w / 2) / width
            y_center = (y + h / 2) / height
            norm_w = w / width
            norm_h = h / height
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        label_path = OUT_DIR / "labels" / split / (Path(file_name).stem + ".txt")
        label_path.write_text("\n".join(lines))

    yaml_content = (
        f"path: {OUT_DIR.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    (OUT_DIR / "data.yaml").write_text(yaml_content)

    n_train = len(image_ids) - n_val
    print(f"Done. {n_train} train images, {n_val} val images.")
    print(f"Wrote {OUT_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
