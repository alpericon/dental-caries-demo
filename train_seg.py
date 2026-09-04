"""
Fine-tunes a YOLOv8-seg model on the dental caries segmentation dataset.

DENTEX is the only merged data source with polygon segmentation masks, so
this dataset (data_seg/) is DENTEX alone — much smaller than the
7,102-image merged detection dataset. See scripts/prepare_dentex_segmentation.py.
"""

from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path("data_seg/data.yaml")
RUN_NAME = "dental_caries_seg_m_3class_1280"


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Couldn't find {DATA_YAML}. Run "
            "scripts/prepare_dentex_segmentation.py first."
        )

    model = YOLO("yolov8m-seg.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=1280,        # same reasoning as train.py: Caries instances are
                           # small relative to the other classes
        batch=4,           # same VRAM-safe value train.py had to hand-pick for
                           # this GPU at 1280px; the seg head adds memory overhead
                           # on top of detection, so this is not reduced further
                           # only if the first epoch actually OOMs
        patience=20,       # stop early if validation loss plateaus
        name=RUN_NAME,
        project="runs/segment",
    )

    print(f"\nDone. Best weights: runs/segment/{RUN_NAME}/weights/best.pt")
    print(f"Validation samples: runs/segment/{RUN_NAME}/")


if __name__ == "__main__":
    main()
