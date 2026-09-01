"""
Tiled inference (SAHI-style): instead of resizing a full panoramic X-ray down
to the model's input size (losing small lesions in the process), slice it
into overlapping tiles at native resolution, run detection on each tile, map
boxes back to original image coordinates, then merge overlapping detections
with NMS.

Implemented directly (no `sahi` package) to avoid its numpy 2.x pin, which
conflicts with ultralytics' numpy<2.0.0 requirement.

Usage: python scripts/tiled_inference.py <image_path>
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO

WEIGHTS_PATH = "runs/detect/dental_caries_demo_m_3class_hires_1280/weights/best.pt"
TILE_SIZE = 1280
OVERLAP = 0.2
CONF_THRESHOLD = 0.25
IOU_MERGE_THRESHOLD = 0.5


def make_tiles(width, height, tile_size, overlap):
    stride = int(tile_size * (1 - overlap))
    tiles = []
    for y in range(0, max(height - tile_size, 0) + 1, stride):
        for x in range(0, max(width - tile_size, 0) + 1, stride):
            tiles.append((x, y, min(x + tile_size, width), min(y + tile_size, height)))
    # ensure the far edges are covered even if the last stride overshoots
    if not tiles or tiles[-1][2] < width or tiles[-1][3] < height:
        tiles.append((max(0, width - tile_size), max(0, height - tile_size), width, height))
    return tiles


def tiled_predict(model, image_path, tile_size=TILE_SIZE, overlap=OVERLAP, conf=CONF_THRESHOLD):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    W, H = image.size

    tiles = make_tiles(W, H, tile_size, overlap)

    all_boxes, all_scores, all_classes = [], [], []
    for (x1, y1, x2, y2) in tiles:
        crop = image.crop((x1, y1, x2, y2))
        results = model.predict(crop, conf=conf, imgsz=tile_size, verbose=False)[0]
        for box in results.boxes:
            bx1, by1, bx2, by2 = box.xyxy[0].tolist()
            all_boxes.append([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1])
            all_scores.append(float(box.conf[0]))
            all_classes.append(int(box.cls[0]))

    if not all_boxes:
        return [], model.names

    boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
    scores_t = torch.tensor(all_scores, dtype=torch.float32)
    classes_t = np.array(all_classes)

    # NMS per class (merge duplicate detections in tile overlap regions)
    keep_indices = []
    for cls_id in set(all_classes):
        cls_mask = np.where(classes_t == cls_id)[0]
        cls_boxes = boxes_t[cls_mask]
        cls_scores = scores_t[cls_mask]
        keep = nms(cls_boxes, cls_scores, IOU_MERGE_THRESHOLD)
        keep_indices.extend(cls_mask[keep.numpy()].tolist())

    findings = []
    for i in keep_indices:
        findings.append(
            {
                "class": model.names[all_classes[i]],
                "confidence": round(all_scores[i], 3),
                "bbox_xyxy": [round(v, 1) for v in all_boxes[i]],
            }
        )
    findings.sort(key=lambda f: f["confidence"], reverse=True)
    return findings, model.names


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/tiled_inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    model = YOLO(WEIGHTS_PATH)

    print("=== Standard single-shot inference (imgsz=1280) ===")
    standard_results = model.predict(image_path, conf=CONF_THRESHOLD, imgsz=1280, verbose=False)[0]
    for box in standard_results.boxes:
        print(f"  {model.names[int(box.cls[0])]}: {float(box.conf[0]):.3f}")
    print(f"  ({len(standard_results.boxes)} findings)")

    print("\n=== Tiled inference (SAHI-style, native resolution) ===")
    tiled_findings, names = tiled_predict(model, image_path)
    for f in tiled_findings:
        print(f"  {f['class']}: {f['confidence']}")
    print(f"  ({len(tiled_findings)} findings)")


if __name__ == "__main__":
    main()
