"""
Converts the DENTEX quadrant-enumeration-disease subset (COCO-style JSON,
diagnosis label = category_id_3) into a YOLOv8-seg dataset under data_seg/.

DENTEX is the only merged data source with polygon segmentation masks —
the Kaggle cavity set and CariesXrays are bbox-only — so this dataset is
DENTEX alone, not the full merged detection set. Periapical Lesion is
dropped and the remaining classes remapped, matching scripts/drop_class.py,
so class ids/order line up with data/data.yaml: Impacted, Caries, Deep Caries.

Usage: python scripts/prepare_dentex_segmentation.py

Expects the extracted DENTEX files at:
  downloads/extracted/training_data/quadrant-enumeration-disease/
    train_quadrant_enumeration_disease.json
    xrays/*.png

Produces:
  data_seg/data.yaml
  data_seg/images/{train,val}/*.png
  data_seg/labels/{train,val}/*.txt
"""

import json
import random
import shutil
from pathlib import Path

SOURCE_DIR = Path("downloads/extracted/training_data/quadrant-enumeration-disease")
ANNOTATIONS_JSON = SOURCE_DIR / "train_quadrant_enumeration_disease.json"
IMAGES_DIR = SOURCE_DIR / "xrays"

OUT_DIR = Path("data_seg")
VAL_FRACTION = 0.15
SEED = 42

# category_id_3 in the DENTEX JSON, in id order (0-indexed)
OLD_CLASS_NAMES = ["Impacted", "Caries", "Periapical Lesion", "Deep Caries"]
DROP_NAME = "Periapical Lesion"
NEW_CLASS_NAMES = [n for n in OLD_CLASS_NAMES if n != DROP_NAME]

drop_id = OLD_CLASS_NAMES.index(DROP_NAME)
ID_REMAP = {}
new_id = 0
for old_id, name in enumerate(OLD_CLASS_NAMES):
    if old_id == drop_id:
        ID_REMAP[old_id] = None
    else:
        ID_REMAP[old_id] = new_id
        new_id += 1


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

    total_instances = 0
    total_dropped = 0

    for image_id, img in images_by_id.items():
        split = "val" if image_id in val_ids else "train"
        file_name = img["file_name"]
        width, height = img["width"], img["height"]

        src_image = IMAGES_DIR / file_name
        dst_image = OUT_DIR / "images" / split / file_name
        shutil.copyfile(src_image, dst_image)

        lines = []
        for ann in anns_by_image.get(image_id, []):
            old_class = ann["category_id_3"]
            class_id = ID_REMAP[old_class]
            if class_id is None:
                total_dropped += 1
                continue

            for polygon in ann["segmentation"]:
                norm_coords = [
                    (coord / width) if i % 2 == 0 else (coord / height)
                    for i, coord in enumerate(polygon)
                ]
                coords_str = " ".join(f"{v:.6f}" for v in norm_coords)
                lines.append(f"{class_id} {coords_str}")
                total_instances += 1

        label_path = OUT_DIR / "labels" / split / (Path(file_name).stem + ".txt")
        label_path.write_text("\n".join(lines))

    yaml_content = (
        f"path: {OUT_DIR.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(NEW_CLASS_NAMES)}\n"
        f"names: {NEW_CLASS_NAMES}\n"
    )
    (OUT_DIR / "data.yaml").write_text(yaml_content)

    print(f"Processed {len(images_by_id)} images, {total_instances} polygon instances "
          f"({total_dropped} '{DROP_NAME}' instances dropped).")
    print(f"Classes: {NEW_CLASS_NAMES}")
    print(f"Wrote {OUT_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
