"""
Merges the CariesXrays dataset (Chen et al., AAAI 2024 — 6,000 panoramic
dental X-rays, VOC-XML format, single 'Decay' class) into the existing
data/ directory built by prepare_dentex_data.py.

NOTE: this dataset has no stated license and is sourced from real hospital
patient data. Used here for local prototyping only per an explicit decision
to accept that risk — see CLAUDE.md. Do not redistribute this data or any
model checkpoint trained on it beyond this local demo without confirming
reuse terms with the paper's authors.

Usage: python scripts/merge_cariesxrays.py

Expects the dataset already downloaded+extracted at:
  downloads/cariesxrays/Annotations/*.xml
  downloads/cariesxrays/JPEGImages/*.jpg
"""

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

SOURCE_DIR = Path("downloads/cariesxrays")
ANNOTATIONS_DIR = SOURCE_DIR / "Annotations"
IMAGES_DIR = SOURCE_DIR / "JPEGImages"

OUT_DIR = Path("data")
VAL_FRACTION = 0.15
SEED = 42

# this dataset's single class -> this project's class id (see prepare_dentex_data.py CLASS_NAMES)
CLASS_REMAP = {"Decay": 1}  # Decay -> Caries


def convert_annotation(xml_path: Path) -> str | None:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        class_id = CLASS_REMAP.get(name)
        if class_id is None:
            continue
        box = obj.find("bndbox")
        xmin = float(box.find("xmin").text)
        ymin = float(box.find("ymin").text)
        xmax = float(box.find("xmax").text)
        ymax = float(box.find("ymax").text)

        x_center = (xmin + xmax) / 2 / width
        y_center = (ymin + ymax) / 2 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    return "\n".join(lines)


def main():
    if not ANNOTATIONS_DIR.exists():
        raise FileNotFoundError(f"Couldn't find {ANNOTATIONS_DIR}. Extract the dataset first.")

    xml_paths = sorted(ANNOTATIONS_DIR.glob("*.xml"))
    random.Random(SEED).shuffle(xml_paths)
    n_val = round(len(xml_paths) * VAL_FRACTION)
    val_paths = set(xml_paths[:n_val])

    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    skipped = 0
    for xml_path in xml_paths:
        stem = xml_path.stem
        image_path = IMAGES_DIR / f"{stem}.jpg"
        if not image_path.exists():
            skipped += 1
            continue

        split = "val" if xml_path in val_paths else "train"
        new_stem = f"cariesxrays_{stem}"

        shutil.copyfile(image_path, OUT_DIR / "images" / split / f"{new_stem}.jpg")
        label_text = convert_annotation(xml_path)
        (OUT_DIR / "labels" / split / f"{new_stem}.txt").write_text(label_text)
        counts[split] += 1

    print(f"Merged {counts['train']} images into train, {counts['val']} into val.")
    if skipped:
        print(f"Skipped {skipped} annotations with no matching image.")


if __name__ == "__main__":
    main()
