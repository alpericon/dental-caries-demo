"""
Fine-tunes a YOLOv8 model on a dental caries detection dataset.

Before running: download a dataset in YOLOv8 format from Roboflow Universe
and unzip it into data/, so that data/data.yaml exists. See README.md.
"""

from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path("data/data.yaml")
RUN_NAME = "dental_caries_demo_m_3class_hires_1280"


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Couldn't find {DATA_YAML}. Download a dataset from Roboflow "
            "Universe in YOLOv8 format and unzip it into data/ first — "
            "see README.md step 2."
        )

    model = YOLO("yolov8m.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=1280,        # up from 640: Caries boxes are ~6x smaller in area than
                           # the other classes, so more resolution should help detect them
        batch=4,           # auto-batch picked 1 (targets only 60% VRAM, unstable
                           # batchnorm); the same probe measured batch=4 at 6.4GB/8GB
        patience=20,       # stop early if validation loss plateaus
        name=RUN_NAME,
        project="runs/detect",
    )

    print(f"\nDone. Best weights: runs/detect/{RUN_NAME}/weights/best.pt")
    print(f"Validation samples: runs/detect/{RUN_NAME}/")


if __name__ == "__main__":
    main()
