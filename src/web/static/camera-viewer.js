"use strict";

document.addEventListener("DOMContentLoaded", () => {
  const dialog = document.getElementById("camera-viewer");
  const viewport = document.getElementById("camera-viewer-viewport");
  const image = document.getElementById("camera-viewer-image");
  const zoomValue = document.getElementById("camera-zoom-value");
  const zoomIn = document.getElementById("camera-zoom-in");
  const zoomOut = document.getElementById("camera-zoom-out");
  const empty = document.getElementById("camera-viewer-empty");
  const status = document.getElementById("camera-viewer-status");
  let scale = 1, panX = 0, panY = 0, fitMode = true, drag = null;
  let initialized = false;
  const maxScale = 8;
  const ready = () => dialog.open && !image.hidden && image.naturalWidth > 0;
  const fitScale = () => Math.min(1, viewport.clientWidth / image.naturalWidth,
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

  function showFrame(source) {
    if (!source) {
      endDrag();
      image.hidden = true;
      image.removeAttribute("src");
      empty.hidden = false;
      zoomIn.disabled = zoomOut.disabled = true;
      return;
    }
    if (image.getAttribute("src") !== source) image.src = source;
  }

  document.querySelectorAll("[data-viewer-image]").forEach(button => {
    button.addEventListener("click", () => {
      const source = document.getElementById(button.dataset.viewerImage);
      dialog.dataset.source = source.id;
      initialized = false;
      fitMode = true;
      panX = panY = 0;
      document.getElementById("camera-viewer-title").textContent = source.alt;
      image.alt = `${source.alt} · 실시간 영상`;
      empty.textContent = source.id === "roi-preview"
        ? "ROI 추출 후 실시간 영상을 표시합니다."
        : "카메라 영상 수신을 기다리는 중입니다.";
      status.textContent = "실시간 영상 연결 중…";
      document.getElementById("camera-viewer-help").textContent =
        `${source.id === "roi-preview" ? "최근 추출 ROI를 실시간 영상에 적용 · 위치가 바뀌면 ROI를 다시 추출하세요. · " : "실시간 카메라 영상 · "}마우스 휠: 확대/축소 · 드래그: 이동 · 더블 클릭: 화면 맞춤`;
      dialog.showModal();
      document.body.classList.add("camera-viewer-open");
      showFrame(source.getAttribute("src") ? source.currentSrc || source.src : null);
    });
  });

  document.addEventListener("camera-preview-update", event => {
    if (!dialog.open) return;
    const data = event.detail;
    const roi = dialog.dataset.source === "roi-preview";
    const source = roi ? data.roi_image : data.image;
    const frameInfo = (data.frame_info || "").split("\n")[0];
    status.textContent = roi ? `${data.roi_frame_info || "ROI 영상 수신 대기"} · ${frameInfo}` : frameInfo;
    if (!source) empty.textContent = roi ? data.roi_frame_info || "ROI 영상 수신 대기" : frameInfo;
    showFrame(source);
  });
  image.addEventListener("load", () => {
    if (!dialog.open || !image.getAttribute("src")) return;
    image.hidden = false;
    empty.hidden = true;
    if (!initialized || fitMode) fit();
    else render();
    initialized = true;
  });
  image.addEventListener("error", () => {
    if (!dialog.open || !image.getAttribute("src")) return;
    showFrame(null);
    status.textContent = "영상을 표시하지 못했습니다. 다음 프레임을 기다립니다.";
  });
  zoomIn.addEventListener("click", () => zoom(scale * 1.25));
  zoomOut.addEventListener("click", () => zoom(scale / 1.25));
  document.getElementById("camera-zoom-fit").addEventListener("click", fit);
  document.getElementById("camera-zoom-actual").addEventListener("click", () => zoom(1));
  document.getElementById("camera-viewer-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", event => {
    const rect = dialog.getBoundingClientRect();
    if (event.target === dialog && (event.clientX < rect.left || event.clientX > rect.right ||
        event.clientY < rect.top || event.clientY > rect.bottom)) dialog.close();
  });
  dialog.addEventListener("close", () => {
    endDrag();
    delete dialog.dataset.source;
    image.hidden = true;
    image.removeAttribute("src");
    empty.hidden = false;
    document.body.classList.remove("camera-viewer-open");
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
