"""
FastAPI backend for the dental caries detection demo.

Loads the model trained by train.py and exposes:
  POST /predict         — single image -> findings (with box coordinates)
  POST /predict-batch    — multiple images -> one result per image
  GET  /api/compare-examples — a fresh random sample of held-out images
                          each call, with both the ground-truth labels and
                          the model's predictions, for an accuracy comparison

Findings are returned down to a low confidence floor (see
MIN_CONFIDENCE below) rather than a single fixed cutoff, so the frontend
can offer a live confidence-threshold slider without re-calling the model
on every change — box drawing happens client-side (see app/static/boxes.js).
"""

import base64
import io
import random
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from ultralytics import YOLO

WEIGHTS_PATH = Path("runs/detect/dental_caries_demo_m_3class_hires_1280/weights/best.pt")
# A low floor, not the display default — lets the UI's confidence slider
# filter down to whatever the model actually returns. The slider's own
# starting position (0.25) lives in the frontend, not here.
MIN_CONFIDENCE = 0.05
INFERENCE_IMGSZ = 1280

# Used on the ground-truth comparison page, which has no slider of its own —
# matches the sliders' default elsewhere so "false positive" means the same
# thing across the whole app, not an artifact of a lower confidence floor.
DISPLAY_THRESHOLD = 0.25

# Same convention as this project's own mAP50 metric (see CLAUDE.md) —
# a predicted box counts as matching a ground-truth box at IoU >= 0.5.
IOU_MATCH_THRESHOLD = 0.5

DATA_IMAGES_DIR = Path("data/images/val")
DATA_LABELS_DIR = Path("data/labels/val")
CLASS_NAMES = ["Impacted", "Caries", "Deep Caries"]

# The ground-truth comparison page draws a fresh random sample from the
# full held-out validation set on every page load, rather than a fixed
# curated list — a random sample is a more honest test than hand-picking
# examples that happened to look good.
COMPARE_SAMPLE_SIZE = 10

# Known accuracy figures from the actual training run behind WEIGHTS_PATH
# (see CLAUDE.md for the full story — dataset merge, resolution tuning,
# and why each number is what it is). Not recomputed live: these are the
# validation-set results from training, reported here so the app itself
# carries the same honesty CLAUDE.md asks of any human presenting it,
# instead of relying on whoever's demoing it to remember to say them.
MODEL_INFO = {
    "run_name": "dental_caries_demo_m_3class_hires_1280",
    "architecture": "YOLOv8m",
    "image_size": 1280,
    "training_images": 7102,
    "data_sources": [
        {"name": "DENTEX", "note": "panoramic X-rays, CC-BY-NC-SA, research use"},
        {"name": "Kaggle cavity radiographs", "note": "bitewing/periapical X-rays, CC BY 4.0"},
        {
            "name": "CariesXrays",
            "note": (
                "panoramic X-rays, AAAI 2024 — no stated license, real hospital "
                "patient data, used for local prototyping only on an informed "
                "risk decision. Not for redistribution. See CLAUDE.md."
            ),
        },
    ],
    "overall_map50": 0.683,
    "per_class_map50": {
        "Impacted": 0.894,
        "Caries": 0.539,
        "Deep Caries": 0.616,
    },
    "limitations": [
        (
            "Caries is the weakest class (mAP50 0.539). Its boxes are ~6x "
            "smaller in area than the other classes on average, and ~87% of "
            "its training examples come from sources that never distinguished "
            "shallow from deep caries — both measurably blur the Caries/Deep "
            "Caries boundary, independent of any architecture choice."
        ),
        (
            "Periapical Lesion is not detected at all. It was dropped from "
            "training — the merged dataset only had ~117 labeled instances of "
            "it, and it never exceeded ~0.32 mAP50 in any run."
        ),
        (
            "This model has not been validated by a licensed dentist or on "
            "any clinical population. It was trained entirely on public "
            "research datasets — see the data sources above."
        ),
        (
            "Radiographic caries diagnosis has real, measured disagreement "
            "even between expert dentists (Cohen's kappa ≈0.66–0.70 "
            "between two dentists reading the same film). Some of what looks "
            "like model error is disagreement any two humans might also have."
        ),
    ],
}

app = FastAPI(title="Dental Caries Detection Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


@app.on_event("startup")
def load_model():
    global model
    if not WEIGHTS_PATH.exists():
        raise RuntimeError(
            f"No trained weights found at {WEIGHTS_PATH}. Run train.py "
            "first — see README.md."
        )
    model = YOLO(str(WEIGHTS_PATH))


def pil_to_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def run_prediction(image: Image.Image) -> list[dict]:
    """Runs the model on one PIL image, returns findings down to MIN_CONFIDENCE."""
    results = model.predict(image, conf=MIN_CONFIDENCE, imgsz=INFERENCE_IMGSZ, verbose=False)
    result = results[0]

    findings = []
    for box in result.boxes:
        findings.append(
            {
                "class": model.names[int(box.cls[0])],
                "confidence": round(float(box.conf[0]), 3),
                "bbox_xyxy": [round(float(v), 1) for v in box.xyxy[0]],
            }
        )
    findings.sort(key=lambda f: f["confidence"], reverse=True)  # most confident first
    return findings


async def read_upload_as_image(file: UploadFile) -> Image.Image:
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(400, f"'{file.filename}' isn't an image file.")
    image_bytes = await file.read()
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(400, f"Couldn't read '{file.filename}' as an image.")


@app.get("/api/model-info")
def model_info():
    return MODEL_INFO


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = await read_upload_as_image(file)
    return {"findings": run_prediction(image)}


@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        image = await read_upload_as_image(file)
        results.append({"filename": file.filename, "findings": run_prediction(image)})
    return {"results": results}


def read_ground_truth(label_path: Path, width: int, height: int) -> list[dict]:
    """Parses a YOLO-format label file into pixel-space findings (no drawing)."""
    findings = []
    if not label_path.exists():
        return findings

    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        cls_id, xc, yc, bw, bh = line.split()
        cls_id = int(cls_id)
        xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
        x1, y1 = (xc - bw / 2) * width, (yc - bh / 2) * height
        x2, y2 = (xc + bw / 2) * width, (yc + bh / 2) * height
        findings.append(
            {
                "class": CLASS_NAMES[cls_id],
                "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            }
        )
    return findings


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_findings(gt_findings: list[dict], pred_findings: list[dict]) -> dict:
    """Greedily pairs GT boxes with predicted boxes by IoU, highest first.

    Matching is class-agnostic on purpose: a prediction that overlaps a GT
    box well but picked the wrong class (e.g. Caries vs Deep Caries — see
    CLAUDE.md on why that boundary is genuinely blurry) is a different,
    more informative failure than missing the lesion entirely, and this
    lets the UI tell the two apart instead of just marking both "wrong."

    Mutates each finding in place with a "match" field, and returns a
    per-image summary.
    """
    candidate_pairs = []
    for gi, gt in enumerate(gt_findings):
        for pi, pred in enumerate(pred_findings):
            iou = compute_iou(gt["bbox_xyxy"], pred["bbox_xyxy"])
            if iou > 0:
                candidate_pairs.append((iou, gi, pi))
    candidate_pairs.sort(key=lambda p: -p[0])

    matched_gt_idx, matched_pred_idx = set(), set()
    for iou, gi, pi in candidate_pairs:
        if gi in matched_gt_idx or pi in matched_pred_idx or iou < IOU_MATCH_THRESHOLD:
            continue
        matched_gt_idx.add(gi)
        matched_pred_idx.add(pi)
        gt, pred = gt_findings[gi], pred_findings[pi]
        status = "matched" if gt["class"] == pred["class"] else "class_mismatch"
        gt["match"] = {"status": status, "iou": round(iou, 3), "other_class": pred["class"]}
        pred["match"] = {"status": status, "iou": round(iou, 3), "other_class": gt["class"]}

    for gi, gt in enumerate(gt_findings):
        if gi not in matched_gt_idx:
            gt["match"] = {"status": "missed", "iou": 0.0}
    for pi, pred in enumerate(pred_findings):
        if pi not in matched_pred_idx:
            pred["match"] = {"status": "extra", "iou": 0.0}

    matched_ious = [
        g["match"]["iou"] for g in gt_findings if g["match"]["status"] in ("matched", "class_mismatch")
    ]
    return {
        "gt_total": len(gt_findings),
        "matched": sum(1 for g in gt_findings if g["match"]["status"] == "matched"),
        "class_mismatch": sum(1 for g in gt_findings if g["match"]["status"] == "class_mismatch"),
        "missed": sum(1 for g in gt_findings if g["match"]["status"] == "missed"),
        "extra": sum(1 for p in pred_findings if p["match"]["status"] == "extra"),
        "mean_iou": round(sum(matched_ious) / len(matched_ious), 3) if matched_ious else None,
    }


@app.get("/api/compare-examples")
def compare_examples():
    if model is None:
        raise HTTPException(503, "Model not loaded yet.")

    all_images = [p.name for p in DATA_IMAGES_DIR.iterdir() if p.is_file()]
    sample = random.sample(all_images, min(COMPARE_SAMPLE_SIZE, len(all_images)))

    examples = []
    overall = {"gt_total": 0, "matched": 0, "class_mismatch": 0, "missed": 0, "extra": 0}
    overall_ious = []
    by_class = {name: {"gt_total": 0, "matched": 0, "class_mismatch": 0, "missed": 0, "extra": 0} for name in CLASS_NAMES}

    for filename in sample:
        image_path = DATA_IMAGES_DIR / filename
        label_path = DATA_LABELS_DIR / (Path(filename).stem + ".txt")
        if not image_path.exists():
            continue

        pil_image = Image.open(image_path).convert("RGB")
        width, height = pil_image.size

        gt_findings = read_ground_truth(label_path, width, height)
        pred_findings = [f for f in run_prediction(pil_image) if f["confidence"] >= DISPLAY_THRESHOLD]
        summary = match_findings(gt_findings, pred_findings)

        for key in overall:
            overall[key] += summary[key]
        overall_ious.extend(
            g["match"]["iou"] for g in gt_findings if g["match"]["status"] in ("matched", "class_mismatch")
        )
        for g in gt_findings:
            by_class[g["class"]]["gt_total"] += 1
            by_class[g["class"]][g["match"]["status"]] += 1
        for p in pred_findings:
            if p["match"]["status"] == "extra":
                by_class[p["class"]]["extra"] += 1

        examples.append(
            {
                "filename": filename,
                "original_image": pil_to_data_uri(pil_image),
                "ground_truth_findings": gt_findings,
                "predicted_findings": pred_findings,
                "summary": summary,
            }
        )

    overall["mean_iou"] = round(sum(overall_ious) / len(overall_ious), 3) if overall_ious else None
    overall["iou_threshold"] = IOU_MATCH_THRESHOLD
    overall["by_class"] = by_class
    overall["sample_size"] = len(examples)

    return {"examples": examples, "overall_summary": overall}


# Serve the demo UI at the root path
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
