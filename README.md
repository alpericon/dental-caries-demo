# Dental Caries Detection — Demo

A research prototype that detects likely cavities in bitewing/periapical dental
X-rays and displays them with bounding boxes and confidence scores. Built to
demo the core idea to dentists before investing in a full clinical pipeline.

**This is not a medical device and must not be used for real diagnosis.**
It exists to show the concept and gather feedback from dental professionals.

## What's in here

```
dental-caries-demo/
├── data/                        # merged, converted dataset (not included — see step 2)
├── scripts/
│   ├── prepare_dentex_data.py    # DENTEX COCO -> YOLOv8 format
│   ├── merge_kaggle_cavity.py    # merges in a Kaggle cavity radiograph set
│   ├── merge_cariesxrays.py      # merges in the CariesXrays dataset
│   └── drop_class.py             # drops a sparse class, remaps the rest
├── train.py                     # fine-tunes YOLOv8 on data/
├── app/
│   ├── main.py                   # FastAPI backend serving the trained model
│   └── static/index.html         # single-page demo UI
├── requirements.txt
└── README.md
```

A trained model already exists at
`runs/detect/dental_caries_demo_m_3class_hires_1280/weights/best.pt`
(3 classes: Impacted, Caries, Deep Caries — see CLAUDE.md for the full
story of how it got there and what its known weak points are). Skip to
step 4 if you just want to run the app against it.

## 1. Set up your environment

Run this on your own machine (needs internet access to install packages and
download a dataset — this won't work inside a sandboxed environment with
restricted network access).

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get a dataset

The current model is trained on three merged public sources — not a single
Roboflow download. To reproduce it from scratch:

1. **DENTEX** (panoramic X-rays, caries/deep caries/periapical lesion/impacted
   labels, CC-BY-NC-SA): download `training_data.zip` from
   [Zenodo record 7812323](https://zenodo.org/records/7812323), extract just
   the `quadrant-enumeration-disease` subset, then run
   `python scripts/prepare_dentex_data.py`.
2. **Kaggle cavity radiographs** (CC BY 4.0): download the
   `dental-cavity-radiograph-images` Kaggle dataset into
   `downloads/kaggle_cavity/`, then run `python scripts/merge_kaggle_cavity.py`.
3. **CariesXrays** (AAAI 2024, panoramic X-rays) — **no stated license, real
   hospital patient data**. Used here for local prototyping only, on an
   explicit informed decision; see CLAUDE.md before touching this one.
   Download from the Google Drive link in the
   [GitHub repo](https://github.com/Binz-Chen/AAAI2024_CariesXrays) into
   `downloads/cariesxrays/`, then run `python scripts/merge_cariesxrays.py`.
4. Optionally, `python scripts/drop_class.py` to remove a class that's too
   data-sparse to be worth training on (currently used to drop Periapical
   Lesion — see CLAUDE.md for why).

Each merge script appends to `data/images/{train,val}` and
`data/labels/{train,val}` and rewrites `data/data.yaml`, so the order above
matters (DENTEX first, since it establishes the class list).

If you'd rather use a single simpler dataset instead, any Roboflow Universe
YOLOv8-format caries dataset unzipped into `data/` works too — just make
sure `data/data.yaml` ends up pointing at it.

## 3. Train a model

```bash
python train.py
```

Currently configured for yolov8m at imgsz=1280 (upgraded from an initial
yolov8n/640px baseline once the dataset grew past ~1,000 images — see
CLAUDE.md for the reasoning and the runs that got tried along the way,
including why a Colab GPU and an RT-DETR architecture swap were tried and
didn't end up being used). Expect a few hours on a modern GPU; drop `imgsz`
back to 640 and the model to `yolov8n.pt` for much faster iteration if
you're just testing the pipeline.

This saves weights to `runs/detect/<name from train.py>/weights/best.pt`.
Sanity-check the results in that same folder — `confusion_matrix_normalized.png`
in particular is worth a look before trusting the numbers.

## 4. Run the demo app

```bash
uvicorn app.main:app --reload
```

Open http://localhost:8000 in a browser, upload an X-ray image, and you'll
see the original next to the AI-annotated version with confidence scores.

## 5. Before presenting to dentists

- Hold out 5–10 X-rays with clear, confirmed findings and test the demo on
  them ahead of time. Don't rely on a live, unrehearsed upload for a
  first impression.
- Be upfront in the room: this is a research prototype trained on public
  data, not a validated clinical tool. You're there to show direction and
  ask for their read on where it's wrong, not to claim it's ready.
- Know the current model's actual numbers going in (mAP50 0.683 overall;
  Impacted 0.894, Caries 0.539, Deep Caries 0.616) and its known weak
  spots — Caries is the least reliable class, and Periapical Lesion isn't
  covered at all right now. See CLAUDE.md for why. Don't get caught
  overselling a number you haven't looked at.
- Bring specific questions for them: Does the confidence threshold feel
  right? What would they want flagged that this misses? Would they trust
  a tool like this enough to use it as a second opinion? Would they be
  willing to help label more periapical-lesion examples?
