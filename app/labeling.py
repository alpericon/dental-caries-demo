"""
Data labeling tool: create labeling projects, upload X-rays, draw
detection or segmentation labels via a canvas UI, export in YOLO format.

Storage is filesystem-based under LABELING_DIR, mirroring the same
images/+labels/ convention this project already uses for data/:

  labeling_projects/<project_id>/
    project.json   — {id, name, type, classes, created_at, images: {...}}
    images/<image_id>.jpg
    labels/<image_id>.txt   — YOLO (detection) or YOLO-seg (segmentation)
                              format; absent/empty means not yet labeled.

project.json is the single source of truth for metadata — "labeled"
status is derived from whether a non-empty label file exists, not
duplicated into project.json itself.
"""

import io
import json
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel

LABELING_DIR = Path("labeling_projects")

router = APIRouter()


class CreateProjectRequest(BaseModel):
    name: str
    type: Literal["detection", "segmentation"]
    classes: list[str]


class Shape(BaseModel):
    class_id: int
    # Pixel coordinates in the original image's space.
    # detection: exactly [x1, y1, x2, y2]
    # segmentation: flat [x1, y1, x2, y2, ..., xn, yn], at least 3 points
    points: list[float]


class SaveLabelsRequest(BaseModel):
    shapes: list[Shape]


def project_dir(project_id: str) -> Path:
    d = LABELING_DIR / project_id
    if not (d / "project.json").exists():
        raise HTTPException(404, f"Project '{project_id}' not found.")
    return d


def load_project(project_id: str) -> dict:
    return json.loads((project_dir(project_id) / "project.json").read_text())


def save_project(project_id: str, data: dict) -> None:
    (LABELING_DIR / project_id / "project.json").write_text(json.dumps(data, indent=2))


def is_labeled(d: Path, image_id: str) -> bool:
    label_path = d / "labels" / f"{image_id}.txt"
    return label_path.exists() and label_path.stat().st_size > 0


@router.post("/projects")
def create_project(req: CreateProjectRequest):
    if not req.name.strip():
        raise HTTPException(400, "Project name can't be empty.")
    classes = [c.strip() for c in req.classes if c.strip()]
    if not classes:
        raise HTTPException(400, "At least one class is required.")

    project_id = uuid.uuid4().hex[:12]
    d = LABELING_DIR / project_id
    (d / "images").mkdir(parents=True)
    (d / "labels").mkdir(parents=True)

    data = {
        "id": project_id,
        "name": req.name.strip(),
        "type": req.type,
        "classes": classes,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "images": {},
    }
    save_project(project_id, data)
    return data


@router.get("/projects")
def list_projects():
    if not LABELING_DIR.exists():
        return {"projects": []}

    projects = []
    for d in sorted(LABELING_DIR.iterdir()):
        if not (d / "project.json").exists():
            continue
        data = json.loads((d / "project.json").read_text())
        labeled = sum(1 for image_id in data["images"] if is_labeled(d, image_id))
        projects.append(
            {
                "id": data["id"],
                "name": data["name"],
                "type": data["type"],
                "classes": data["classes"],
                "image_count": len(data["images"]),
                "labeled_count": labeled,
                "created_at": data["created_at"],
            }
        )
    projects.sort(key=lambda p: p["created_at"], reverse=True)
    return {"projects": projects}


@router.get("/projects/{project_id}")
def get_project(project_id: str):
    d = project_dir(project_id)
    data = load_project(project_id)
    images = [
        {"id": image_id, **meta, "labeled": is_labeled(d, image_id)}
        for image_id, meta in data["images"].items()
    ]
    return {**data, "images": images}


@router.delete("/projects/{project_id}")
def delete_project(project_id: str):
    shutil.rmtree(project_dir(project_id))
    return {"ok": True}


@router.post("/projects/{project_id}/images")
async def upload_images(project_id: str, files: list[UploadFile] = File(...)):
    d = project_dir(project_id)
    data = load_project(project_id)

    added = []
    for file in files:
        if file.content_type is None or not file.content_type.startswith("image/"):
            continue
        image_bytes = await file.read()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            continue

        image_id = uuid.uuid4().hex[:12]
        image.save(d / "images" / f"{image_id}.jpg", format="JPEG", quality=95)
        (d / "labels" / f"{image_id}.txt").touch()

        meta = {"filename": file.filename or f"{image_id}.jpg", "width": image.width, "height": image.height}
        data["images"][image_id] = meta
        added.append({"id": image_id, **meta, "labeled": False})

    save_project(project_id, data)
    return {"added": added}


@router.get("/projects/{project_id}/images/{image_id}")
def get_image(project_id: str, image_id: str):
    d = project_dir(project_id)
    data = load_project(project_id)
    if image_id not in data["images"]:
        raise HTTPException(404, "Image not found.")
    path = d / "images" / f"{image_id}.jpg"
    return Response(content=path.read_bytes(), media_type="image/jpeg")


@router.get("/projects/{project_id}/images/{image_id}/labels")
def get_labels(project_id: str, image_id: str):
    d = project_dir(project_id)
    data = load_project(project_id)
    meta = data["images"].get(image_id)
    if meta is None:
        raise HTTPException(404, "Image not found.")

    width, height = meta["width"], meta["height"]
    is_detection = data["type"] == "detection"
    label_path = d / "labels" / f"{image_id}.txt"

    shapes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            class_id = int(parts[0])
            nums = [float(v) for v in parts[1:]]
            if is_detection:
                # stored as YOLO (xc, yc, w, h) — convert back to (x1, y1, x2, y2)
                xc, yc, w, h = nums
                points = [
                    (xc - w / 2) * width,
                    (yc - h / 2) * height,
                    (xc + w / 2) * width,
                    (yc + h / 2) * height,
                ]
            else:
                points = [v * width if i % 2 == 0 else v * height for i, v in enumerate(nums)]
            shapes.append({"class_id": class_id, "points": points})
    return {"shapes": shapes}


@router.put("/projects/{project_id}/images/{image_id}/labels")
def save_labels(project_id: str, image_id: str, req: SaveLabelsRequest):
    d = project_dir(project_id)
    data = load_project(project_id)
    meta = data["images"].get(image_id)
    if meta is None:
        raise HTTPException(404, "Image not found.")

    width, height = meta["width"], meta["height"]
    is_detection = data["type"] == "detection"

    lines = []
    for shape in req.shapes:
        if is_detection:
            if len(shape.points) != 4:
                raise HTTPException(400, "Detection shapes need exactly 4 points (x1,y1,x2,y2).")
            x1, y1, x2, y2 = shape.points
            xc = (x1 + x2) / 2 / width
            yc = (y1 + y2) / 2 / height
            w = abs(x2 - x1) / width
            h = abs(y2 - y1) / height
            lines.append(f"{shape.class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        else:
            if len(shape.points) < 6 or len(shape.points) % 2 != 0:
                raise HTTPException(400, "Segmentation shapes need at least 3 points.")
            normed = [v / width if i % 2 == 0 else v / height for i, v in enumerate(shape.points)]
            coords = " ".join(f"{v:.6f}" for v in normed)
            lines.append(f"{shape.class_id} {coords}")

    (d / "labels" / f"{image_id}.txt").write_text("\n".join(lines))
    return {"ok": True}


@router.get("/projects/{project_id}/export")
def export_project(project_id: str):
    d = project_dir(project_id)
    data = load_project(project_id)

    buf = io.BytesIO()
    used_names: set[str] = set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("classes.txt", "\n".join(data["classes"]))

        for image_id, meta in data["images"].items():
            stem = Path(meta["filename"]).stem or image_id
            name, suffix = stem, 1
            while name in used_names:
                name = f"{stem}_{suffix}"
                suffix += 1
            used_names.add(name)

            image_path = d / "images" / f"{image_id}.jpg"
            label_path = d / "labels" / f"{image_id}.txt"
            zf.write(image_path, f"images/{name}.jpg")
            zf.writestr(f"labels/{name}.txt", label_path.read_text() if label_path.exists() else "")

    filename = f"{data['name'].replace(' ', '_')}.zip"
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
