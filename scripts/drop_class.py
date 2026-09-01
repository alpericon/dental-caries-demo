"""
Drops a class from the YOLO dataset under data/ and remaps remaining class
ids to be contiguous, updating data.yaml accordingly.

For this demo, Periapical Lesion is dropped: it only has ~117 instances
across the merged dataset (vs. 435-9777 for the other classes) and its
detection quality never got past ~0.32-0.38 mAP50 in any run, YOLO or
RT-DETR. Scope narrowed to Impacted / Caries / Deep Caries until more
periapical lesion data is available (e.g. via a future dentist partner).
See CLAUDE.md.

Usage: python scripts/drop_class.py
"""

from pathlib import Path

DATA_DIR = Path("data")
OLD_NAMES = ["Impacted", "Caries", "Periapical Lesion", "Deep Caries"]
DROP_NAME = "Periapical Lesion"

drop_id = OLD_NAMES.index(DROP_NAME)
NEW_NAMES = [n for n in OLD_NAMES if n != DROP_NAME]

# old class id -> new class id (or None if dropped)
ID_REMAP = {}
new_id = 0
for old_id, name in enumerate(OLD_NAMES):
    if old_id == drop_id:
        ID_REMAP[old_id] = None
    else:
        ID_REMAP[old_id] = new_id
        new_id += 1


def process_label_file(path: Path) -> None:
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    new_lines = []
    for line in lines:
        parts = line.split()
        old_class = int(parts[0])
        new_class = ID_REMAP[old_class]
        if new_class is None:
            continue
        parts[0] = str(new_class)
        new_lines.append(" ".join(parts))
    path.write_text("\n".join(new_lines))


def main():
    total_dropped = 0
    total_files = 0
    for split in ("train", "val"):
        labels_dir = DATA_DIR / "labels" / split
        for label_path in labels_dir.glob("*.txt"):
            before = label_path.read_text().count("\n") + 1 if label_path.stat().st_size else 0
            process_label_file(label_path)
            after = label_path.read_text().count("\n") + 1 if label_path.stat().st_size else 0
            total_dropped += max(before - after, 0)
            total_files += 1

    yaml_content = (
        f"path: {DATA_DIR.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(NEW_NAMES)}\n"
        f"names: {NEW_NAMES}\n"
    )
    (DATA_DIR / "data.yaml").write_text(yaml_content)

    print(f"Processed {total_files} label files, removed {total_dropped} '{DROP_NAME}' instances.")
    print(f"New classes: {NEW_NAMES}")
    print(f"Wrote {DATA_DIR / 'data.yaml'}")


if __name__ == "__main__":
    main()
