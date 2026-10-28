"use strict";

document.addEventListener("DOMContentLoaded", () => {
  const dialog = document.getElementById("roi-viewer");
  const viewport = document.getElementById("roi-viewer-viewport");
  const image = document.getElementById("roi-viewer-image");
  const message = document.getElementById("roi-viewer-message");
  const zoomValue = document.getElementById("roi-zoom-value");
  const zoomIn = document.getElementById("roi-zoom-in");
  const zoomOut = document.getElementById("roi-zoom-out");
  let scale = 1, panX = 0, panY = 0, fitMode = true, drag = null;
  const maxScale = 8;
  const ready = () => dialog.open && !image.hidden && image.naturalWidth > 0;
  const fitScale = () => Math.min(maxScale, viewport.clientWidth / image.naturalWidth,
    viewport.clientHeight / image.naturalHeight);
  const minScale = () => Math.min(0.05, fitScale());

  function render() {
    const limitX = Math.max(0, (image.naturalWidth * scale - viewport.clientWidth) / 2);
    const limitY = Math.max(0, (image.naturalHeight * scale - viewport.clientHeight) / 2);
    panX = Math.max(-limitX, Math.min(limitX, panX));
    panY = Math.max(-limitY, Math.min(limitY, panY));
    image.style.transform = `translate(-50%, -50%) translate(${panX}px, ${panY}px) scale(${scale})`;
    zoomValue.textContent = `${Math.round(scale * 100)}%`;
    zoomOut.disabled = scale <= minScale();
    zoomIn.disabled = scale >= maxScale;
    viewport.classList.toggle("can-pan", limitX > 0 || limitY > 0);
  }

  function fit() {
    if (!ready()) return;
    fitMode = true;
    scale = fitScale();
    panX = panY = 0;
    render();
  }

  function zoom(nextScale, clientX, clientY) {
    if (!ready()) return;
    fitMode = false;
    const rect = viewport.getBoundingClientRect();
    const x = clientX == null ? 0 : clientX - rect.left - viewport.clientWidth / 2;
    const y = clientY == null ? 0 : clientY - rect.top - viewport.clientHeight / 2;
    const previousScale = scale;
    scale = Math.max(minScale(), Math.min(maxScale, nextScale));
    panX = x - (x - panX) * scale / previousScale;
    panY = y - (y - panY) * scale / previousScale;
    render();
  }

  document.querySelectorAll("[data-roi-image]").forEach(button => {
    button.addEventListener("click", () => {
      const source = document.getElementById(button.dataset.roiImage);
      if (source.hidden || !source.getAttribute("src")) return;
      const caption = button.closest("figure").querySelector("figcaption").textContent;
      document.getElementById("roi-viewer-title").textContent = `${caption} · 확대 보기`;
      image.alt = `${caption} 확대 영상`;
      image.hidden = true;
      message.textContent = "이미지를 불러오는 중입니다.";
      message.hidden = false;
      zoomIn.disabled = zoomOut.disabled = true;
      zoomValue.textContent = "—";
      fitMode = true;
      panX = panY = 0;
      dialog.showModal();
      document.body.classList.add("roi-viewer-open");
      image.src = source.currentSrc || source.src;
    });
  });

  image.addEventListener("load", () => {
    if (!dialog.open || !image.getAttribute("src")) return;
    image.hidden = false;
    message.hidden = true;
    fit();
  });
  image.addEventListener("error", () => {
    if (!dialog.open || !image.getAttribute("src")) return;
    image.hidden = true;
    message.textContent = "이미지를 표시할 수 없습니다. 창을 닫고 다시 측정하세요.";
    message.hidden = false;
    zoomIn.disabled = zoomOut.disabled = true;
  });
  zoomIn.addEventListener("click", () => zoom(scale * 1.25));
  zoomOut.addEventListener("click", () => zoom(scale / 1.25));
  document.getElementById("roi-zoom-fit").addEventListener("click", fit);
  document.getElementById("roi-zoom-actual").addEventListener("click", () => zoom(1));
  document.getElementById("roi-viewer-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", event => {
    const rect = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right ||
        event.clientY < rect.top || event.clientY > rect.bottom)) dialog.close();
  });
  dialog.addEventListener("close", () => {
    endDrag();
    image.hidden = true;
    image.removeAttribute("src");
    document.body.classList.remove("roi-viewer-open");
  });
  dialog.addEventListener("keydown", event => {
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.key === "+" || event.key === "=") zoom(scale * 1.25);
    else if (event.key === "-") zoom(scale / 1.25);
    else if (event.key === "0") fit();
    else return;
    event.preventDefault();
  });
  viewport.addEventListener("wheel", event => {
    event.preventDefault();
    zoom(scale * Math.exp(-Math.max(-100, Math.min(100, event.deltaY)) * 0.002),
      event.clientX, event.clientY);
  }, {passive: false});
  viewport.addEventListener("dblclick", fit);
  viewport.addEventListener("pointerdown", event => {
    if (!ready() || event.button !== 0 || !viewport.classList.contains("can-pan")) return;
    event.preventDefault();
    drag = {id: event.pointerId, x: event.clientX, y: event.clientY};
    viewport.setPointerCapture(event.pointerId);
    viewport.classList.add("is-panning");
  });
  viewport.addEventListener("pointermove", event => {
    if (!drag || drag.id !== event.pointerId) return;
    panX += event.clientX - drag.x;
    panY += event.clientY - drag.y;
    drag.x = event.clientX;
    drag.y = event.clientY;
    render();
  });
  function endDrag() {
    if (drag && viewport.hasPointerCapture(drag.id)) viewport.releasePointerCapture(drag.id);
    drag = null;
    viewport.classList.remove("is-panning");
  }
  ["pointerup", "pointercancel", "lostpointercapture"].forEach(type => {
    viewport.addEventListener(type, endDrag);
  });
  new ResizeObserver(() => {
    if (ready()) {
      if (fitMode) fit();
      else render();
    }
  }).observe(viewport);
});
