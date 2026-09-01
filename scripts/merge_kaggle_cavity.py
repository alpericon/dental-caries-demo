"""
Merges the "Dental Cavity Radiograph Images" Kaggle dataset (real bitewing/
periapical X-rays, single 'Cavity' class, YOLOv8 format, CC BY 4.0) into the
existing data/ directory built by prepare_dentex_data.py.

Usage: python scripts/merge_kaggle_cavity.py

Expects the dataset already downloaded+unzipped at:
  downloads/kaggle_cavity/{train,valid,test}/{images,labels}/

The source dataset's single class (id 0, "Cavity") is remapped to this
project's "Caries" class id, matching CLASS_NAMES in prepare_dentex_data.py.
"""

import shutil
from pathlib import Path

SOURCE_DIR = Path("downloads/kaggle_cavity")
OUT_DIR = Path("data")

# source class id -> this project's class id (see prepare_dentex_data.py CLASS_NAMES)
CLASS_REMAP = {0: 1}  # Cavity -> Caries

# source split -> our split
SPLIT_MAP = {"train": "train", "valid": "val", "test": "val"}


def remap_label_lines(text: str) -> str:
    lines = []
    for line in text.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        parts[0] = str(CLASS_REMAP[int(parts[0])])
        lines.append(" ".join(parts))
    return "\n".join(lines)


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Couldn't find {SOURCE_DIR}. Download the dataset first.")

    counts = {"train": 0, "val": 0}
    for src_split, dst_split in SPLIT_MAP.items():
        images_dir = SOURCE_DIR / src_split / "images"
        labels_dir = SOURCE_DIR / src_split / "labels"
        if not images_dir.exists():
            continue

        dst_images = OUT_DIR / "images" / dst_split
        dst_labels = OUT_DIR / "labels" / dst_split
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        for image_path in images_dir.iterdir():
            if not image_path.is_file():
                continue
            label_path = labels_dir / (image_path.stem + ".txt")
            if not label_path.exists():
                continue

            # prefix to avoid any filename collision with DENTEX's train_N.png files
            new_stem = f"kaggle_{image_path.stem}"
            shutil.copyfile(image_path, dst_images / f"{new_stem}{image_path.suffix}")
            remapped = remap_label_lines(label_path.read_text())
            (dst_labels / f"{new_stem}.txt").write_text(remapped)
            counts[dst_split] += 1

    print(f"Merged {counts['train']} images into train, {counts['val']} into val.")


if __name__ == "__main__":
    main()
