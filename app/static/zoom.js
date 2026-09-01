// Shared click-to-zoom image viewer. Works on any <img> inside a
// .viewer-stage, .card-image, or .col-image container, on any page that
// includes this script — no per-page wiring needed, even for images
// inserted dynamically after this script runs (event delegation).
(function () {
  const overlay = document.createElement("div");
  overlay.className = "zoom-overlay";
  overlay.innerHTML = `
    <button class="zoom-close" aria-label="Close">&times;</button>
    <div class="zoom-hint">Scroll to zoom &middot; drag to pan &middot; double-click to reset &middot; Esc to close</div>
    <div class="zoom-wrap">
      <img class="zoom-img" alt="">
    </div>
  `;
  document.body.appendChild(overlay);

  const wrap = overlay.querySelector(".zoom-wrap");
  const zoomImg = overlay.querySelector(".zoom-img");
  const closeBtn = overlay.querySelector(".zoom-close");

  const MIN_SCALE = 1;
  const MAX_SCALE = 8;
  let scale = 1;
  let tx = 0;
  let ty = 0;
  let isPanning = false;
  let panStartX = 0;
  let panStartY = 0;
  let startTx = 0;
  let startTy = 0;

  function updateTransform() {
    zoomImg.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
    zoomImg.style.cursor = scale > MIN_SCALE ? "grab" : "zoom-in";
  }

  function resetView() {
    scale = 1;
    tx = 0;
    ty = 0;
    updateTransform();
  }

  function zoomAt(clientX, clientY, targetScale) {
    const rect = wrap.getBoundingClientRect();
    const mouseX = clientX - rect.left;
    const mouseY = clientY - rect.top;
    const prevScale = scale;
    scale = Math.min(Math.max(targetScale, MIN_SCALE), MAX_SCALE);
    tx = mouseX - (mouseX - tx) * (scale / prevScale);
    ty = mouseY - (mouseY - ty) * (scale / prevScale);
    if (scale === MIN_SCALE) {
      tx = 0;
      ty = 0;
    }
    updateTransform();
  }

  function open(src) {
    zoomImg.src = src;
    resetView();
    overlay.classList.add("open");
    document.body.style.overflow = "hidden";
  }

  function close() {
    overlay.classList.remove("open");
    document.body.style.overflow = "";
  }

  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) close();
  });
  closeBtn.addEventListener("click", close);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && overlay.classList.contains("open")) close();
  });

  wrap.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.2 : 1 / 1.2;
      zoomAt(e.clientX, e.clientY, scale * factor);
    },
    { passive: false }
  );

  zoomImg.addEventListener("mousedown", (e) => {
    if (scale <= MIN_SCALE) return;
    e.preventDefault();
    isPanning = true;
    panStartX = e.clientX;
    panStartY = e.clientY;
    startTx = tx;
    startTy = ty;
    zoomImg.style.cursor = "grabbing";
  });
  window.addEventListener("mousemove", (e) => {
    if (!isPanning) return;
    tx = startTx + (e.clientX - panStartX);
    ty = startTy + (e.clientY - panStartY);
    updateTransform();
  });
  window.addEventListener("mouseup", () => {
    if (!isPanning) return;
    isPanning = false;
    zoomImg.style.cursor = scale > MIN_SCALE ? "grab" : "zoom-in";
  });

  zoomImg.addEventListener("dblclick", (e) => {
    if (scale > MIN_SCALE) {
      resetView();
    } else {
      zoomAt(e.clientX, e.clientY, 3);
    }
  });

  // Delegated: catches images already on the page AND ones added later
  // (batch results, comparison examples fetched after this script runs).
  document.addEventListener("click", (e) => {
    const target = e.target;
    if (
      target.tagName === "IMG" &&
      target.closest(".viewer-stage, .card-image, .col-image")
    ) {
      open(target.src);
    }
  });
})();
