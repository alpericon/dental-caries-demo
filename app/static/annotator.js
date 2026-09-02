// Canvas-based labeling engine, shared by annotate.html for both
// detection (box) and segmentation (polygon) modes.
//
// Usage:
//   const annotator = createAnnotator({
//     container, imageUrl, mode, classes, initialShapes, onChange,
//   });
//   annotator.setActiveClass(1);
//   annotator.getShapes();       // -> [{class_id, points}], points in
//                                //    original-image pixel space
//   annotator.destroy();
//
// Shapes: {class_id, points}
//   detection:    points = [x1, y1, x2, y2]
//   segmentation: points = [x1, y1, x2, y2, ..., xn, yn]  (flat, >=3 pts)
//
// The canvas is a fixed-size viewport (matches the container, doesn't
// resize with zoom) — scroll to zoom toward the cursor, hold Space and
// drag to pan. Same interaction convention as the app's zoom.js viewer,
// so it should feel familiar.

const ANNOTATOR_PALETTE = [
  "#4C8DFF", "#2FE0D0", "#FFB020", "#FF5C5C",
  "#B57EDC", "#F4A9C9", "#8BD17C", "#F2D06B",
];

function annotatorColor(classId) {
  return ANNOTATOR_PALETTE[classId % ANNOTATOR_PALETTE.length];
}

const CLOSE_VERTEX_RADIUS_PX = 10; // screen-space radius for "click near start to close"
const MIN_ZOOM = 1;
const MAX_ZOOM = 12;

function createAnnotator({ container, imageUrl, mode, classes, initialShapes, onChange, onZoomChange }) {
  const canvas = document.createElement("canvas");
  canvas.style.display = "block";
  canvas.style.cursor = "crosshair";
  container.innerHTML = "";
  container.appendChild(canvas);
  const ctx = canvas.getContext("2d");

  let img = null;
  let baseScale = 1; // fit-to-viewport scale at zoom=1
  let zoom = 1;
  let panX = 0;
  let panY = 0;
  let shapes = (initialShapes || []).map((s) => ({ ...s, points: [...s.points] }));
  let activeClass = 0;
  let selectedIndex = -1;
  let dragBox = null; // {x1,y1,x2,y2} in image-pixel space, while drawing a detection box
  let polygonInProgress = null; // [ [x,y], ... ] in image-pixel space, while drawing a polygon
  let cursorImagePoint = null; // [x,y], for the live preview line in segmentation mode
  let draggingVertex = null; // {shapeIndex, pointIndex}, for adjusting a polygon vertex
  let spaceHeld = false;
  let isPanning = false;
  let panStartScreen = null;
  let panStartOffset = null;

  function scale() {
    return baseScale * zoom;
  }

  function emitChange() {
    if (onChange) onChange(getShapes());
  }

  function sizeCanvasToContainer() {
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight || Math.round(container.clientWidth * 0.6);
  }

  function toImagePoint(evt) {
    const rect = canvas.getBoundingClientRect();
    const sx = (evt.clientX - rect.left) * (canvas.width / rect.width);
    const sy = (evt.clientY - rect.top) * (canvas.height / rect.height);
    return [(sx - panX) / scale(), (sy - panY) / scale()];
  }

  function toScreen(x, y) {
    return [x * scale() + panX, y * scale() + panY];
  }

  function distance(a, b) {
    return Math.hypot(a[0] - b[0], a[1] - b[1]);
  }

  function pointInBox(pt, box) {
    const [x1, y1, x2, y2] = box;
    const minX = Math.min(x1, x2), maxX = Math.max(x1, x2);
    const minY = Math.min(y1, y2), maxY = Math.max(y1, y2);
    return pt[0] >= minX && pt[0] <= maxX && pt[1] >= minY && pt[1] <= maxY;
  }

  function pointInPolygon(pt, points) {
    // ray casting; points is flat [x1,y1,x2,y2,...]
    let inside = false;
    const n = points.length / 2;
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const xi = points[i * 2], yi = points[i * 2 + 1];
      const xj = points[j * 2], yj = points[j * 2 + 1];
      const intersect =
        yi > pt[1] !== yj > pt[1] && pt[0] < ((xj - xi) * (pt[1] - yi)) / (yj - yi) + xi;
      if (intersect) inside = !inside;
    }
    return inside;
  }

  function hitTest(imagePoint) {
    for (let i = shapes.length - 1; i >= 0; i--) {
      const s = shapes[i];
      if (mode === "detection" ? pointInBox(imagePoint, s.points) : pointInPolygon(imagePoint, s.points)) {
        return i;
      }
    }
    return -1;
  }

  function vertexHitTest(imagePoint) {
    for (let si = 0; si < shapes.length; si++) {
      const pts = shapes[si].points;
      for (let pi = 0; pi < pts.length / 2; pi++) {
        const v = [pts[pi * 2], pts[pi * 2 + 1]];
        if (distance(toScreen(...v), toScreen(...imagePoint)) < CLOSE_VERTEX_RADIUS_PX) {
          return { shapeIndex: si, pointIndex: pi };
        }
      }
    }
    return null;
  }

  function drawLabel(x, y, text, color) {
    ctx.font = "600 12px 'IBM Plex Mono', monospace";
    const w = ctx.measureText(text).width;
    ctx.fillStyle = color;
    ctx.fillRect(x, Math.max(0, y - 16), w + 8, 16);
    ctx.fillStyle = "#06231F";
    ctx.fillText(text, x + 4, Math.max(11, y - 4));
  }

  function redraw() {
    if (!img) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const [ix, iy] = toScreen(0, 0);
    ctx.drawImage(img, ix, iy, img.naturalWidth * scale(), img.naturalHeight * scale());

    shapes.forEach((s, i) => {
      const color = annotatorColor(s.class_id);
      const label = classes[s.class_id] || `class ${s.class_id}`;
      ctx.lineWidth = i === selectedIndex ? 3 : 2;
      ctx.strokeStyle = color;

      if (mode === "detection") {
        const [sx1, sy1] = toScreen(s.points[0], s.points[1]);
        const [sx2, sy2] = toScreen(s.points[2], s.points[3]);
        ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);
        drawLabel(sx1, sy1, label, color);
      } else {
        ctx.beginPath();
        for (let p = 0; p < s.points.length; p += 2) {
          const [sx, sy] = toScreen(s.points[p], s.points[p + 1]);
          if (p === 0) ctx.moveTo(sx, sy);
          else ctx.lineTo(sx, sy);
        }
        ctx.closePath();
        ctx.stroke();
        ctx.fillStyle = color + "33";
        ctx.fill();
        const [lx, ly] = toScreen(s.points[0], s.points[1]);
        drawLabel(lx, ly, label, color);
        if (i === selectedIndex) {
          for (let p = 0; p < s.points.length; p += 2) {
            const [sx, sy] = toScreen(s.points[p], s.points[p + 1]);
            ctx.beginPath();
            ctx.arc(sx, sy, 4, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
          }
        }
      }
    });

    if (dragBox) {
      const [sx1, sy1] = toScreen(dragBox.x1, dragBox.y1);
      const [sx2, sy2] = toScreen(dragBox.x2, dragBox.y2);
      ctx.lineWidth = 2;
      ctx.strokeStyle = annotatorColor(activeClass);
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(sx1, sy1, sx2 - sx1, sy2 - sy1);
      ctx.setLineDash([]);
    }

    if (polygonInProgress && polygonInProgress.length) {
      ctx.lineWidth = 2;
      ctx.strokeStyle = annotatorColor(activeClass);
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      polygonInProgress.forEach(([x, y], idx) => {
        const [sx, sy] = toScreen(x, y);
        if (idx === 0) ctx.moveTo(sx, sy);
        else ctx.lineTo(sx, sy);
      });
      if (cursorImagePoint) {
        const [sx, sy] = toScreen(...cursorImagePoint);
        ctx.lineTo(sx, sy);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      polygonInProgress.forEach(([x, y]) => {
        const [sx, sy] = toScreen(x, y);
        ctx.beginPath();
        ctx.arc(sx, sy, 4, 0, Math.PI * 2);
        ctx.fillStyle = annotatorColor(activeClass);
        ctx.fill();
      });
    }
  }

  function commitDragBox() {
    if (dragBox) {
      const w = Math.abs(dragBox.x2 - dragBox.x1);
      const h = Math.abs(dragBox.y2 - dragBox.y1);
      if (w > 4 && h > 4) {
        shapes.push({
          class_id: activeClass,
          points: [
            Math.min(dragBox.x1, dragBox.x2),
            Math.min(dragBox.y1, dragBox.y2),
            Math.max(dragBox.x1, dragBox.x2),
            Math.max(dragBox.y1, dragBox.y2),
          ],
        });
        emitChange();
      }
    }
    dragBox = null;
  }

  function closePolygon() {
    if (polygonInProgress && polygonInProgress.length >= 3) {
      shapes.push({ class_id: activeClass, points: polygonInProgress.flat() });
      emitChange();
    }
    polygonInProgress = null;
    cursorImagePoint = null;
  }

  function setZoom(newZoom, screenPivot) {
    const prevScale = scale();
    zoom = Math.min(Math.max(newZoom, MIN_ZOOM), MAX_ZOOM);
    const newScale = scale();
    const [px, py] = screenPivot || [canvas.width / 2, canvas.height / 2];
    panX = px - (px - panX) * (newScale / prevScale);
    panY = py - (py - panY) * (newScale / prevScale);
    clampPan();
    redraw();
    if (onZoomChange) onZoomChange(Math.round(zoom * 100));
  }

  function clampPan() {
    // keep at least a small sliver of the image on screen at all times
    const w = img.naturalWidth * scale();
    const h = img.naturalHeight * scale();
    const margin = 60;
    panX = Math.min(Math.max(panX, -w + margin), canvas.width - margin);
    panY = Math.min(Math.max(panY, -h + margin), canvas.height - margin);
  }

  function onWheel(evt) {
    evt.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const px = (evt.clientX - rect.left) * (canvas.width / rect.width);
    const py = (evt.clientY - rect.top) * (canvas.height / rect.height);
    const factor = evt.deltaY < 0 ? 1.2 : 1 / 1.2;
    setZoom(zoom * factor, [px, py]);
  }

  function onMouseDown(evt) {
    if (spaceHeld || evt.button === 1) {
      isPanning = true;
      panStartScreen = [evt.clientX, evt.clientY];
      panStartOffset = [panX, panY];
      canvas.style.cursor = "grabbing";
      return;
    }

    const pt = toImagePoint(evt);

    if (mode === "segmentation") {
      const vhit = vertexHitTest(pt);
      if (vhit && !polygonInProgress) {
        draggingVertex = vhit;
        return;
      }
      if (polygonInProgress && polygonInProgress.length >= 3) {
        if (distance(toScreen(...polygonInProgress[0]), toScreen(...pt)) < CLOSE_VERTEX_RADIUS_PX) {
          closePolygon();
          redraw();
          return;
        }
      }
      if (polygonInProgress) {
        polygonInProgress.push(pt);
        redraw();
        return;
      }
      const hit = hitTest(pt);
      if (hit >= 0) {
        selectedIndex = hit;
        redraw();
        return;
      }
      polygonInProgress = [pt];
      redraw();
      return;
    }

    // detection mode
    const hit = hitTest(pt);
    if (hit >= 0) {
      selectedIndex = hit;
      redraw();
      return;
    }
    selectedIndex = -1;
    dragBox = { x1: pt[0], y1: pt[1], x2: pt[0], y2: pt[1] };
    redraw();
  }

  function onMouseMove(evt) {
    if (isPanning) {
      panX = panStartOffset[0] + (evt.clientX - panStartScreen[0]);
      panY = panStartOffset[1] + (evt.clientY - panStartScreen[1]);
      redraw();
      return;
    }

    const pt = toImagePoint(evt);
    if (draggingVertex) {
      shapes[draggingVertex.shapeIndex].points[draggingVertex.pointIndex * 2] = pt[0];
      shapes[draggingVertex.shapeIndex].points[draggingVertex.pointIndex * 2 + 1] = pt[1];
      redraw();
      return;
    }
    if (dragBox) {
      dragBox.x2 = pt[0];
      dragBox.y2 = pt[1];
      redraw();
      return;
    }
    if (polygonInProgress) {
      cursorImagePoint = pt;
      redraw();
    }
  }

  function onMouseUp() {
    if (isPanning) {
      isPanning = false;
      clampPan();
      canvas.style.cursor = spaceHeld ? "grab" : "crosshair";
      redraw();
      return;
    }
    if (draggingVertex) {
      draggingVertex = null;
      emitChange();
      return;
    }
    if (dragBox) {
      commitDragBox();
      redraw();
    }
  }

  function onKeyDown(evt) {
    if (evt.key === " " && !spaceHeld) {
      spaceHeld = true;
      canvas.style.cursor = "grab";
      evt.preventDefault();
    } else if (evt.key === "Escape") {
      polygonInProgress = null;
      cursorImagePoint = null;
      redraw();
    } else if (evt.key === "Enter") {
      closePolygon();
      redraw();
    } else if (evt.key === "Delete" || evt.key === "Backspace") {
      if (selectedIndex >= 0 && document.activeElement === canvas) {
        shapes.splice(selectedIndex, 1);
        selectedIndex = -1;
        emitChange();
        redraw();
      }
    }
  }

  function onKeyUp(evt) {
    if (evt.key === " ") {
      spaceHeld = false;
      canvas.style.cursor = "crosshair";
    }
  }

  function onResize() {
    sizeCanvasToContainer();
    if (img) {
      baseScale = Math.min(canvas.width / img.naturalWidth, canvas.height / img.naturalHeight, 1);
      redraw();
    }
  }

  canvas.tabIndex = 0; // so it can receive keydown
  canvas.addEventListener("mousedown", onMouseDown);
  canvas.addEventListener("mousemove", onMouseMove);
  canvas.addEventListener("wheel", onWheel, { passive: false });
  window.addEventListener("mouseup", onMouseUp);
  canvas.addEventListener("keydown", onKeyDown);
  canvas.addEventListener("keyup", onKeyUp);
  window.addEventListener("resize", onResize);

  function loadImage(url, newShapes) {
    return new Promise((resolve) => {
      const image = new Image();
      image.onload = () => {
        img = image;
        shapes = (newShapes || []).map((s) => ({ ...s, points: [...s.points] }));
        selectedIndex = -1;
        polygonInProgress = null;
        dragBox = null;
        sizeCanvasToContainer();
        baseScale = Math.min(canvas.width / img.naturalWidth, canvas.height / img.naturalHeight, 1);
        zoom = 1;
        // center the fitted image in the viewport
        panX = (canvas.width - img.naturalWidth * baseScale) / 2;
        panY = (canvas.height - img.naturalHeight * baseScale) / 2;
        redraw();
        resolve();
      };
      image.src = url;
    });
  }

  function getShapes() {
    return shapes.map((s) => ({ class_id: s.class_id, points: [...s.points] }));
  }

  function destroy() {
    window.removeEventListener("mouseup", onMouseUp);
    window.removeEventListener("resize", onResize);
    container.innerHTML = "";
  }

  const ready = loadImage(imageUrl, initialShapes);

  return {
    ready,
    setActiveClass(idx) {
      activeClass = idx;
    },
    deleteSelected() {
      if (selectedIndex >= 0) {
        shapes.splice(selectedIndex, 1);
        selectedIndex = -1;
        emitChange();
        redraw();
      }
    },
    deleteAt(index) {
      shapes.splice(index, 1);
      if (selectedIndex === index) selectedIndex = -1;
      emitChange();
      redraw();
    },
    selectAt(index) {
      selectedIndex = index;
      redraw();
    },
    zoomIn() { setZoom(zoom * 1.4); },
    zoomOut() { setZoom(zoom / 1.4); },
    resetZoom() {
      zoom = 1;
      if (img) {
        panX = (canvas.width - img.naturalWidth * baseScale) / 2;
        panY = (canvas.height - img.naturalHeight * baseScale) / 2;
      }
      redraw();
      if (onZoomChange) onZoomChange(100);
    },
    getZoomPercent() { return Math.round(zoom * 100); },
    getShapes,
    loadImage,
    destroy,
  };
}
