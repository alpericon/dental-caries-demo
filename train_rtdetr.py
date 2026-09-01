"""
Fine-tunes RT-DETR (transformer-based, anchor-free detector) on the same
dental caries dataset as train.py, to compare against the YOLO baseline.

Before running: data/data.yaml must exist (see train.py / README.md).
"""

from pathlib import Path

from ultralytics import RTDETR

DATA_YAML = Path("data/data.yaml")
RUN_NAME = "dental_caries_demo_rtdetr"


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Couldn't find {DATA_YAML}. Run scripts/prepare_dentex_data.py "
            "(and the merge scripts) first — see README.md."
        )

    model = RTDETR("rtdetr-l.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=4,           # RT-DETR-L is memory-hungry; batch=16 overflowed 8GB VRAM
        patience=20,       # stop early if validation loss plateaus
        name=RUN_NAME,
        project="runs/detect",
    )

    print(f"\nDone. Best weights: runs/detect/{RUN_NAME}/weights/best.pt")
    print(f"Validation samples: runs/detect/{RUN_NAME}/")


if __name__ == "__main__":
    main()
