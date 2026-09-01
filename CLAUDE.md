# Dental caries detection — project context

This file is read automatically by Claude Code at the start of a session.
It captures the reasoning behind this project, not just the code, so
nothing gets lost when picking this back up.

## What this is

A demo application that detects likely caries (cavities) in dental
X-rays (bitewing/periapical) and displays them as bounding boxes with
confidence scores. Purpose right now: show local dentists a working
prototype to get their feedback and, ideally, recruit one as an informal
clinical advisor.

**This is a research prototype, not a medical device.** A trained
model exists and the demo app runs end-to-end (see "Current state"
below), but it has not been clinically validated in any way.

## The bigger picture (why this project is scoped the way it is)

The long-term goal is a real clinical tool, built solo, currently using
only public datasets. That combination matters for every decision below:

- A solo developer can build the software and a research prototype.
  A real clinical tool additionally needs a licensed dentist as
  clinical advisor, eventually a regulatory affairs consultant, and a
  quality management system — none of which are code. Don't let the
  code or the UI imply more clinical readiness than actually exists.
- Public datasets (DENTEX, Roboflow Universe sets, etc.) are typically
  research-only licensed. Fine for this demo. Not fine for a commercial
  product without checking/relicensing — flag this if it comes up.
  One training source (CariesXrays, from an AAAI 2024 paper) has **no
  stated license at all** and is real hospital patient data — it was
  used anyway on an explicit, informed decision to accept that risk
  for local prototyping only. Do not redistribute the raw CariesXrays
  data itself. The "local demo only" scope for *checkpoints* trained on
  it was later deliberately widened, on an explicit decision, to also
  include the private GitHub repo this project pushes to (still not
  public, still not commercial use) — see the repo's commit history
  around when `best.pt` was added. If this project ever moves toward a
  real product, the underlying CariesXrays licensing question still
  needs to be resolved first (contact the authors, or drop that data
  source and retrain) regardless of that widened scope.
- Scope is deliberately narrow: 2D X-ray finding detection, framed as
  decision support (a dentist reviews every finding), not autonomous
  diagnosis. Dental MRI was explicitly ruled out for v1 — no standard
  dataset or clinical precedent yet.
- Current model detects 3 classes: **Impacted, Caries, Deep Caries**.
  Periapical Lesion was in scope originally but got dropped from
  training — the merged dataset only had ~117 instances of it (vs.
  hundreds-to-thousands for the other classes) and it never performed
  above ~0.32 mAP50 in any run. Re-add it once there's more labeled
  data for it specifically (e.g. via a dentist partner), rather than
  training on what we have now and overclaiming coverage.

Full phase roadmap (for context — only Phases 1–2 are active work right now):

- **Phase 0** — scope, dentist advisor outreach, license audit (in progress)
- **Phase 1** — data + research prototype (in progress — this repo)
- **Phase 2** — MVP demo app for the dentist meeting (in progress — this repo)
- **Phase 3** — real clinic data partnership, IRB-approved validation (not started)
- **Phase 4** — regulatory pathway, FDA 510(k) or EU MDR (not started — needs funding/team)
- **Phase 5** — production deployment, HIPAA/GDPR, PACS/EHR integration (not started)

## Current state of the code

```
train.py                    — YOLOv8 fine-tuning script (currently yolov8m,
                               imgsz=1280). Expects data/data.yaml.
train_rtdetr.py              — RT-DETR variant of the same, used once to
                               confirm the bottleneck was data, not
                               architecture (see below). Not the active model.
scripts/prepare_dentex_data.py — converts DENTEX's COCO-style annotations
                               into data/ (YOLOv8 format).
scripts/merge_kaggle_cavity.py — merges in a Kaggle bitewing/periapical
                               cavity dataset (CC BY 4.0), remapped to Caries.
scripts/merge_cariesxrays.py — merges in the CariesXrays dataset (see
                               licensing note above), remapped to Caries.
scripts/drop_class.py        — removes Periapical Lesion from data/ and
                               remaps the remaining 3 classes.
scripts/tiled_inference.py   — SAHI-style tiled inference experiment.
                               Tried and shelved: naive tiling changes the
                               apparent object scale relative to what the
                               model was trained on (whole-image letterbox
                               resize) and made results worse, not better.
                               Would need tile-aware retraining to pay off.
                               Not used by the app.
app/main.py                  — FastAPI backend: /predict, /predict-batch,
                               /api/compare-examples (random 10-image
                               sample each call, IoU-matched against
                               ground truth), /api/model-info. Points at
                               the current best model. Verified end-to-end
                               via real browser sessions (Playwright), not
                               just curl/TestClient.
app/labeling.py               — separate router (mounted at /api/labeling)
                               for the data labeling tool — see below.
app/static/index.html,        — demo UI: single-image upload, batch
  batch.html, compare.html,     upload, ground-truth comparison (with
  about.html                    per-class IoU stats), and a Model Info
                               page surfacing the same limitations listed
                               here (so the honesty lives in the app, not
                               just in this file). Dark radiology-viewer
                               styling, confidence-threshold sliders,
                               click-to-zoom on every image, persistent
                               "not a medical device" disclaimer.
app/static/labeling.html,     — the data labeling tool: create a project
  annotate.html, annotator.js   (detection or segmentation, user-defined
                               classes), upload images, label on a canvas
                               (annotator.js — hand-rolled, no external
                               annotation library), export a YOLO-format
                               zip. Filesystem-based storage under
                               labeling_projects/ (gitignored — it's
                               user-generated data), mirroring the same
                               images/+labels/ convention data/ uses.
                               Built to close the exact gap CLAUDE.md
                               flags below: no way to create new labeled
                               data (e.g. periapical lesion examples, or
                               segmentation masks for the classes DENTEX
                               already has polygons for — see "Immediate
                               next steps").
app/static/style.css, boxes.js, zoom.js — shared styling and client-side
                               box-drawing/zoom-viewer JS used across all
                               of the above pages.
colab_train.ipynb            — Colab fallback for training on a different
                               GPU. Turned out not to beat the local RTX
                               3070 Ti for this workload once its CPU
                               dataloader bottleneck was fixed (see below).
```

**Trained model:** `runs/detect/dental_caries_demo_m_3class_hires_1280/weights/best.pt`
— yolov8m, imgsz=1280, 3 classes (Impacted/Caries/Deep Caries), trained on
7,102 images merged from DENTEX + a Kaggle cavity set + CariesXrays.
Overall mAP50 0.683 (Impacted 0.894, Caries 0.539, Deep Caries 0.616).
This is the best of several runs tried — see git history / session notes
for the progression (yolov8n → yolov8m → +more data → +higher resolution)
and for why RT-DETR and naive tiled inference were tried and didn't help
further. Caries is the weakest class, for two identified, evidenced
reasons: its boxes are ~6x smaller in area than the other classes
(small-object detection is harder), and ~87% of its training instances
come from sources that never distinguished shallow from deep caries,
which measurably blurs the Caries/Deep Caries boundary. Both match
findings independently reported in the published literature for this
task, not just this project's own data (see literature review artifact
from the 2026-08-28 session if picking this back up).

## Immediate next steps

1. Pick 5–10 clean held-out examples (from `data/images/val/`) to
   rehearse the dentist demo with — not done yet as of this writing.
2. Run the demo (`uvicorn app.main:app --reload`) and walk through it
   once end-to-end before the actual meeting.
3. In the room: be upfront that Periapical Lesion isn't covered yet,
   and that Caries specifically has known accuracy limits — see the
   confusion-matrix findings above. Don't oversell the numbers.
4. After the demo: use it to open a real conversation with a dentist
   about becoming an advisor (Phase 3 depends on this) — and about
   getting real periapical-lesion-labeled data if they're willing. The
   labeling tool (`app/static/labeling.html`) is now built for exactly
   this — a dentist partner could label directly in the browser, no
   separate tooling needed, and export a YOLO-format zip to merge into
   `data/` the same way the other sources were.

## Conventions to keep

- Never let the UI or any output imply clinical validity it doesn't
  have. The disclaimer stays visible — don't remove or shrink it while
  iterating on design.
- Favor YOLOv8 nano/small for iteration speed at this stage; accuracy
  optimization comes later, after there's a real clinical partner to
  validate against.
- No auto-retraining in production, ever — model updates should be
  deliberate, versioned, and validated (MLflow or similar, once this
  moves past the demo stage).
- If a task starts drifting toward "let's just ship this to real
  patients" — pump the brakes and flag the regulatory/data-partnership
  gap explicitly rather than quietly building past it.
