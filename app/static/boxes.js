// Shared client-side box drawing. Findings come back from the API as
// {class, confidence?, bbox_xyxy} in original-image pixel coordinates —
// this draws them onto a copy of the source image at native resolution
// (so the resulting data URI stays crisp under the zoom viewer too).
const CLASS_COLORS = {
  Impacted: "#4C8DFF",
  Caries: "#2FE0D0",
  "Deep Caries": "#FFB020",
};
const GROUND_TRUTH_COLOR = "#FF5C5C";

// Classes with materially lower measured accuracy — see the Model Info
// page. Surfaced inline wherever a class name is shown, not just there,
// so the caveat travels with the finding instead of needing a separate lookup.
const WEAK_CLASSES = { Caries: "Lower-accuracy class (mAP50 0.539) — see Model Info" };

function classBadge(className) {
  const note = WEAK_CLASSES[className];
  if (!note) return "";
  return `<span class="weak-badge" title="${note}">&#9679;</span>`;
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

// boxes: [{bbox_xyxy, color, label}]
function compositeBoxes(sourceImg, boxes) {
  const canvas = document.createElement("canvas");
  canvas.width = sourceImg.naturalWidth;
  canvas.height = sourceImg.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(sourceImg, 0, 0);

  // scale line/font weight relative to a ~1400px-wide baseline so boxes
  // look right whether the source is a small crop or a full panoramic
  const scale = Math.max(0.9, Math.min(2.2, sourceImg.naturalWidth / 1400));
  const lineWidth = 2.5 * scale;
  const fontSize = Math.round(15 * scale);
  const padY = Math.round(6 * scale);

  boxes.forEach(({ bbox_xyxy, polygon, color, label }) => {
    const [x1, y1] = bbox_xyxy;
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = color;

    if (polygon && polygon.length >= 3) {
      // segmentation mask: translucent fill + solid outline
      ctx.beginPath();
      ctx.moveTo(polygon[0][0], polygon[0][1]);
      for (let i = 1; i < polygon.length; i++) ctx.lineTo(polygon[i][0], polygon[i][1]);
      ctx.closePath();
      ctx.fillStyle = color + "40"; // ~25% alpha
      ctx.fill();
      ctx.stroke();
    } else {
      const [, , x2, y2] = bbox_xyxy;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }

    ctx.font = `600 ${fontSize}px 'IBM Plex Mono', monospace`;
    const textWidth = ctx.measureText(label).width;
    const labelH = fontSize + padY;
    ctx.fillStyle = color;
    ctx.fillRect(x1, Math.max(0, y1 - labelH), textWidth + padY * 2, labelH);
    ctx.fillStyle = "#06231F";
    ctx.fillText(label, x1 + padY * 0.7, Math.max(fontSize, y1 - padY * 0.6));
  });

  return canvas.toDataURL("image/png");
}

function findingsToBoxes(findings, threshold) {
  return findings
    .filter((f) => f.confidence === undefined || f.confidence >= threshold)
    .map((f) => ({
      bbox_xyxy: f.bbox_xyxy,
      polygon: f.polygon,
      color: CLASS_COLORS[f.class] || "#FFFFFF",
      label: f.confidence !== undefined ? `${f.class} ${(f.confidence * 100).toFixed(0)}%` : f.class,
    }));
}
