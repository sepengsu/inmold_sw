"use strict";

const $ = id => document.getElementById(id);
const state = {page: 1, pages: 1, lighting: [0, 0, 0, 0], previewUrl: null, previewBusy: false, cameraEnabled: false};
const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const mode = () => document.querySelector('input[name="mode"]:checked').value;
const fmt = (value, digits = 2) => Number(value).toFixed(digits);

async function request(url, options = {}) {
  const response = await fetch(url, {cache: "no-store", ...options});
  if (!response.ok) {
    let message = `HTTP ${response.status}`;
    try { message = (await response.json()).detail || message; } catch (_) {}
    throw new Error(message);
  }
  return response.status === 204 ? null : response.json();
}
const post = (url, data) => request(url, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(data)});
const showError = (id, error) => { $(id).textContent = `⚠ ${error.message || error}`; };

function showImage(id, source) {
  const image = $(id);
  const placeholder = image.parentElement.querySelector(".image-placeholder");
  image.hidden = !source;
  if (source) image.src = source;
  else image.removeAttribute("src");
  if (placeholder) placeholder.hidden = !!source;
  const roiButton = image.closest("[data-roi-image]");
  if (roiButton) roiButton.disabled = !source;
}

function renderStatus(status) {
  state.cameraEnabled = status.camera_enabled;
  const transfer = status.transfer;
  const card = (name, level, label, details) => `<div class="system-status-card"><span class="status-light ${level}" aria-hidden="true"></span><div class="system-status-content"><div class="system-status-heading"><b>${name}</b><span class="status-chip ${level}">${label}</span></div>${details.filter(Boolean).map(detail => `<div class="system-status-detail">${detail}</div>`).join("")}</div></div>`;
  const cameraLevel = status.camera_connected ? "ok" : status.camera_enabled ? "error" : "wait";
  const cameraLabel = status.camera_connected ? "정상" : status.camera_enabled ? "오류" : "대기";
  const modelLevel = status.model === "LOADED" ? "ok" : status.model === "로딩 대기" ? "wait" : "error";
  const lightingLevel = status.light.includes("ACK 확인") ? "ok" : status.light.includes("연결 대기") || status.light.includes("파일 검사") || status.light.includes("제어 대기") ? "wait" : "error";
  const tcpLevel = transfer?.enabled ? "ok" : transfer?.error ? "error" : "wait";
  const cameraSummary = status.camera_connected ? "카메라 연결됨" : status.camera_enabled ? "카메라 연결에 실패했습니다" : "파일 검사 모드";
  const modelSummary = status.model === "LOADED" ? "AI 모델 로드 완료" : escapeHtml(status.model);
  const lightingSummary = status.light.includes("·") ? status.light.split("·").at(-1).trim() : status.light;
  const tcpSummary = transfer?.enabled ? "자동 전송 사용 가능" : transfer?.error ? "수신 장치에 연결할 수 없습니다" : "연결 상태를 확인하는 중입니다";
  const detailRows = [
    status.roi ? `ROI: ${escapeHtml(status.roi)}` : "",
    status.threshold == null ? "" : `PIXEL: ${escapeHtml(status.threshold)}`,
    status.positive_classes?.length ? `대상 픽셀: ${escapeHtml(status.positive_classes.join(" + "))}` : "",
    !status.camera_connected && status.camera_enabled ? `카메라 오류: ${escapeHtml(status.camera)}` : "",
    transfer?.error ? `TCP 오류: ${escapeHtml(transfer.error)}` : ""
  ].filter(Boolean);
  $("system-status").innerHTML = [
    card("CAMERA", cameraLevel, cameraLabel, [cameraSummary, `${status.pixel_format === "Mono8" ? "Mono8" : "BayerGR12"} · 노출 ${escapeHtml(status.exposure_us)} µs`]),
    card("MODEL", modelLevel, modelLevel === "ok" ? "정상" : modelLevel === "wait" ? "대기" : "오류", [modelSummary, status.inference_device ? `${escapeHtml(status.inference_device_name || status.inference_device)} (${escapeHtml(status.inference_device)})` : ""]),
    card("LIGHTING", lightingLevel, lightingLevel === "ok" ? "정상" : lightingLevel === "wait" ? "대기" : "오류", [`${escapeHtml(status.light_port)} · ${escapeHtml(status.light_baudrate)} bps`, escapeHtml(lightingSummary)]),
    transfer ? card("TCP", tcpLevel, tcpLevel === "ok" ? "연결됨" : tcpLevel === "wait" ? "대기" : "오류", [`수신: ${escapeHtml(transfer.host)}:${escapeHtml(transfer.port)}`, `송신: ${escapeHtml(transfer.local_host)}`, tcpSummary, `최근 처리: ${escapeHtml(transfer.last_result)}`, transfer.last_sent_measurement ? `측정 시각: ${escapeHtml(transfer.last_sent_measurement)}` : ""]) : "",
    detailRows.length ? `<div class="system-status-extra"><b>세부 메시지</b>${detailRows.map(detail => `<div>${detail}</div>`).join("")}</div>` : ""
  ].join("");
}

function renderHistory(records) {
  const rows = Array.from({length: 5}, (_, index) => {
    const record = records[index];
    if (!record) return `<div class="history-row empty"><span>${index + 1}</span><b>--.--%</b><b>--.--%</b><b>--.--%</b></div>`;
    const label = record.metadata?.conditions?.inspection_mode === "certification" ? "국소 모드" : "일반 모드";
    return `<div class="history-entry"><div class="history-row"><span>${index + 1}</span><b>${fmt(record.p1)}%</b><b>${fmt(record.p2)}%</b><b>${fmt(record.p3)}%</b></div><div class="history-meta"><span>${record.source === "camera" ? "카메라" : "파일"}</span><span>${label}</span><span>${record.id ? "저장됨" : "미저장"}</span><span>소요 시간 ${record.elapsed_seconds == null ? "--" : fmt(record.elapsed_seconds, 3) + "초"}</span><time>${escapeHtml(record.measured_at)}</time></div></div>`;
  });
  $("history").innerHTML = `<div class="history-head"><span>NO</span><span>P1</span><span>P2</span><span>P3</span></div>${rows.join("")}`;
}

function updateMode() {
  const certification = mode() === "certification";
  $("overview-caption").textContent = "검사 영상 · P1~P3 ROI 위치";
  $("mode-hint").textContent = certification ? "흰 종이 안의 P1~P3 사각형 ROI를 찾아 회전 보정 후, 각 256×256px로 측정합니다." : "기존 Gabor ROI 및 P1~P3 측정 절차를 사용합니다.";
  $("measurement-info").innerHTML = certification ? '<div class="section-label">국소 측정 영역</div><div class="measurement-spec"><div><span>P1 · P2 · P3</span><b>각 256 × 256px</b></div><div><span>검출 순서</span><b>왼쪽 → 오른쪽</b></div><div><span>점수 분모</span><b>사각형 ROI 전체 · 검정 제외</b></div></div><div class="hint">흰 종이 내부에서 크기·정렬·간격이 맞는 사각형 ROI 3개를 선택하고 그 안의 원본 픽셀을 보존합니다.</div>' : '<div class="section-label">측정 영역</div><div class="measurement-spec"><div><span>P1 · P2 · P3</span><b>256 × 512px</b></div><div><span>가로 중심 간격</span><b>547px</b></div><div><span>세로 위치</span><b>ROI 중앙</b></div></div><div class="hint">측정 결과와 원본·결과 영상은 자동 저장됩니다.</div>';
  ["왼쪽", "가운데", "오른쪽"].forEach((position, index) => {
    $(`p${index + 1}-label`).textContent = `P${index + 1} · ${position} · ${certification ? "256×256" : "256×512"}`;
  });
  $("calculation-title").textContent = `${certification ? "국소" : "일반"} · 함침율 계산`;
  $("calculation-basis").textContent = certification
    ? "전체: 사각형 ROI 전체의 검정 제외 픽셀\n빨간색: 결함 픽셀"
    : "전체: 256 × 512 = 131,072\n빨간색: 결함 픽셀";
}

function updateSource() {
  const file = $("source").value === "file";
  $("file-input").hidden = !file;
  $("file-preview-box").hidden = !file || !$("file-input").files.length;
}

function updateLightingValues(values) {
  if (!values) return;
  state.lighting = Array.from(values, Number);
  state.lighting.forEach((value, index) => {
    $(`light-${index + 1}`).value = value;
    $(`light-value-${index + 1}`).textContent = `${value}%`;
  });
  updateLightingButton();
}
function currentLighting() { return [1, 2, 3, 4].map(i => Number($(`light-${i}`).value)); }
function updateLightingButton() {
  state.lighting = currentLighting();
  $("open-lighting").textContent = `4개 조명 설정 · 평균 ${fmt(state.lighting.reduce((a, b) => a + b, 0) / 4, 0)}%`;
}

function renderRoi(roi) {
  if (roi.threshold != null) {
    $("roi-threshold").value = roi.threshold;
    $("roi-value").textContent = Number(roi.threshold).toFixed(2);
  }
  $("roi-level").value = roi.level || "custom";
  $("roi-details").textContent = (roi.details || "").replace(/\*\*/g, "");
  if (roi.message) $("roi-message").textContent = roi.message.replace(/\*\*/g, "");
}

async function previewRoi(parameter, value) {
  const result = await request(`/api/roi/preview?${parameter}=${encodeURIComponent(value)}`);
  renderRoi(result);
}

function resetInspection() {
  showImage("result-image", null);
  [1, 2, 3].forEach(i => {
    showImage(`p${i}-image`, null);
    $(`p${i}-calculation-score`).textContent = "— %";
    $(`p${i}-defect-pixels`).textContent = "—";
    $(`p${i}-total-pixels`).textContent = "—";
    $(`p${i}-equation`).textContent = "측정 후 계산식이 표시됩니다.";
  });
  $("result").innerHTML = '<div class="result-number">--.--<small>%</small></div><div class="hint">검사 입력을 선택하고 측정을 시작하세요.</div>';
}

function renderCalculations(payload) {
  [1, 2, 3].forEach((point, index) => {
    const defects = payload.defect_pixels[index];
    const total = payload.valid_pixels[index];
    const score = [payload.record.p1, payload.record.p2, payload.record.p3][index];
    const count = value => Number(value).toLocaleString("ko-KR");
    $(`p${point}-calculation-score`).textContent = `${fmt(score)}%`;
    $(`p${point}-defect-pixels`).textContent = count(defects);
    $(`p${point}-total-pixels`).textContent = count(total);
    $(`p${point}-equation`).textContent = `(1 − ${count(defects)} / ${count(total)}) × 100 = ${fmt(score)}%`;
  });
}

function renderResult(payload) {
  const record = payload.record;
  const modeLabel = payload.mode === "certification" ? "국소 모드" : "일반 모드";
  const detail = payload.mode === "certification" ? "흰 종이 내부 ROI 3개 · 원본 사각형 전체 크롭 · 회전 보정 후 256×256px · 원본 검정 픽셀만 점수 제외" : "각 256×512px · 중심 간격 547px";
  const extras = [];
  if (payload.transfer) extras.push(payload.transfer.message);
  if (payload.positive_classes?.length > 1) extras.push(`대상 픽셀: ${payload.positive_classes.join(" + ")}`);
  if (payload.valid_pixels) extras.push(`유효 픽셀 · ${payload.valid_pixels.map((n, i) => `P${i + 1} ${n.toLocaleString("ko-KR")}`).join(" · ")}`);
  if (payload.rotation_degrees?.length === 3 && payload.rotation_degrees.every(n => n != null)) extras.push(`회전 보정 · ${payload.rotation_degrees.map((n, i) => `P${i + 1} ${Number(n) >= 0 ? "+" : ""}${Number(n).toFixed(1)}°`).join(" · ")}`);
  if (payload.roi_message) extras.push(`자동 ROI · ${payload.roi_message} · P1~P3 측정 영역만 추론`);
  $("result").innerHTML = `<div class="mode-result">${modeLabel}</div><div class="result-number">${fmt(record.average)}<small>%</small></div><div class="result-caption">P1~P3 평균 함침율</div><div class="result-caption">${detail}</div><div class="result-caption inspection-duration">측정 소요 시간 ${fmt(record.elapsed_seconds, 3)}초</div><div class="result-caption">${escapeHtml(record.measured_at)}</div><div class="result-caption">${escapeHtml(payload.saved_message)}</div>${extras.map(item => `<div class="result-caption">${escapeHtml(item)}</div>`).join("")}<div class="point-scores"><div>P1~P3 함침율</div>${[record.p1, record.p2, record.p3].map((value, i) => `<div class="score-row"><span>P${i + 1}</span><b>${fmt(value)}%</b></div>`).join("")}</div>`;
  showImage("result-image", payload.images.result);
  [1, 2, 3].forEach(i => showImage(`p${i}-image`, payload.images[`p${i}`]));
  renderCalculations(payload);
  renderHistory(payload.history);
}

async function inspect() {
  const button = $("inspect");
  button.disabled = true;
  button.textContent = "측정 중…";
  document.querySelectorAll('input[name="mode"]').forEach(input => { input.disabled = true; });
  resetInspection();
  try {
    const data = new FormData();
    data.append("source", $("source").value);
    data.append("mode", mode());
    data.append("lighting", currentLighting().join(","));
    if ($("source").value === "file" && $("file-input").files[0]) data.append("image", $("file-input").files[0]);
    renderResult(await request("/api/inspect", {method: "POST", body: data}));
    renderStatus(await request("/api/status"));
  } catch (error) {
    $("result").innerHTML = `<div class="result-state error">ERROR</div><div class="result-caption">${escapeHtml(error.message)}</div>`;
  } finally {
    button.disabled = false;
    button.textContent = "▶  측정 시작";
    document.querySelectorAll('input[name="mode"]').forEach(input => { input.disabled = false; });
  }
}

async function refreshStatistics(page = 1) {
  const params = new URLSearchParams({source: $("statistics-source").value, page});
  if ($("statistics-start").value) params.set("start", $("statistics-start").value);
  if ($("statistics-end").value) params.set("end", $("statistics-end").value);
  try {
    const data = await request(`/api/statistics?${params}`);
    state.page = data.page; state.pages = data.pages;
    const summary = data.summary;
    const number = (key, suffix = "%") => summary[key] == null ? "—" : `${fmt(summary[key], 3)}${suffix}`;
    $("statistics-summary").innerHTML = `<div class="statistics-cards"><div class="statistics-card"><span>측정 수</span><strong>${summary.count.toLocaleString("ko-KR")}건</strong></div><div class="statistics-card"><span>평균 함침율</span><strong>${number("average")}</strong></div><div class="statistics-card"><span>평균 소요 시간</span><strong>${number("elapsed_seconds", "초")}</strong></div></div><div class="statistics-points">P1 평균 ${number("p1")} · P2 평균 ${number("p2")} · P3 평균 ${number("p3")} · 측정 평균 최솟값 ${number("minimum")} · 최댓값 ${number("maximum")}</div>`;
    $("statistics-rows").replaceChildren();
    data.rows.forEach(record => {
      const row = document.createElement("tr");
      const cells = [record.measured_at, record.source === "camera" ? "카메라" : "파일", record.p1, record.p2, record.p3, record.average, record.elapsed_seconds];
      cells.forEach((cell, index) => { const td = document.createElement("td"); td.textContent = index < 2 ? cell : cell == null ? "—" : fmt(cell, 3); row.append(td); });
      row.addEventListener("click", () => showDetail(record.id));
      $("statistics-rows").append(row);
    });
    $("statistics-page-label").textContent = `${data.page} / ${data.pages} 페이지 · 전체 ${summary.count.toLocaleString("ko-KR")}건`;
    $("statistics-previous").disabled = data.page <= 1;
    $("statistics-next").disabled = data.page >= data.pages;
    $("statistics-message").textContent = data.rows.length ? "행을 선택하면 저장된 원본과 결과를 볼 수 있습니다." : "조회 조건에 해당하는 측정 이력이 없습니다.";
    $("statistics-detail-text").textContent = "측정 항목을 선택하세요.";
    ["original", "result", "p1", "p2", "p3"].forEach(key => { const image = $(`detail-${key}`); image.hidden = true; image.removeAttribute("src"); });
    $("detail-metadata").textContent = "";
  } catch (error) { showError("statistics-message", error); }
}

async function showDetail(id) {
  try {
    const data = await request(`/api/statistics/${encodeURIComponent(id)}`);
    const record = data.record;
    $("statistics-detail-text").textContent = `${record.measured_at} · 평균 ${fmt(record.average, 3)}% · 소요 시간 ${record.elapsed_seconds == null ? "—" : fmt(record.elapsed_seconds, 3) + "초"} · P1 ${fmt(record.p1, 3)}% · P2 ${fmt(record.p2, 3)}% · P3 ${fmt(record.p3, 3)}%`;
    ["original", "result", "p1", "p2", "p3"].forEach(key => {
      const image = $(`detail-${key}`);
      image.src = data.images[key] || "";
      image.hidden = !data.images[key];
    });
    $("detail-metadata").textContent = JSON.stringify(record.metadata, null, 2);
  } catch (error) { showError("statistics-message", error); }
}

async function refreshPreview() {
  if ($("inspection-page").hidden || !state.cameraEnabled || state.previewBusy) return;
  state.previewBusy = true;
  try {
    const response = await fetch("/api/preview", {cache: "no-store"});
    if (response.status === 204) { showImage("camera-preview", null); return; }
    if (!response.ok) return;
    const url = URL.createObjectURL(await response.blob());
    showImage("camera-preview", url);
    if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = url;
  } catch (_) {} finally { state.previewBusy = false; }
}

async function initialize() {
  try {
    const data = await request("/api/bootstrap");
    renderStatus(data.status);
    $("pixel-format").value = data.status.pixel_format;
    $("source").value = data.source; updateSource();
    renderHistory(data.history);
    renderRoi(data.roi);
    $("lighting-config-message").textContent = data.lighting.message;
    $("lighting-groups").innerHTML = data.lighting.groups.map((group, index) => `<div class="lighting-group-card"><div class="lighting-group-header"><span>LIGHT ${String(index + 1).padStart(2, "0")}</span><strong>${escapeHtml(group.name)}</strong><code>${escapeHtml(group.channels)}</code><small>${group.channels.includes("+") ? "두 채널 같은 밝기로 제어" : "단일 채널 제어"}</small></div><div class="brightness-value"><span>설정 밝기</span><strong id="light-value-${index + 1}">0%</strong></div><input id="light-${index + 1}" type="range" min="0" max="100" step="1"><button data-group="${index + 1}">이 조명 적용</button><div id="receipt-${index + 1}" class="hint">최근 ACK: 미확인</div></div>`).join("");
    updateLightingValues(data.lighting.values);
    [1, 2, 3, 4].forEach(i => $(`light-${i}`).addEventListener("input", () => { $(`light-value-${i}`).textContent = `${$(`light-${i}`).value}%`; updateLightingButton(); }));
    document.querySelectorAll("[data-group]").forEach(button => button.addEventListener("click", () => applyLighting(Number(button.dataset.group)).catch(e => showError("lighting-message", e))));
  } catch (error) { $("system-status").textContent = `시작 실패: ${error.message}`; }
  updateMode();
}

async function applyLighting(group = null) {
  const result = await post("/api/lighting/apply", {values: currentLighting(), group});
  $("lighting-message").textContent = result.message;
  Object.entries(result.receipts).forEach(([index, receipt]) => { if (receipt !== "최근 ACK: 미확인") $(`receipt-${index}`).textContent = receipt; });
  if (group === null && result.success) $("lighting-modal").hidden = true;
  renderStatus(await request("/api/status"));
}

document.addEventListener("DOMContentLoaded", () => {
  initialize();
  document.querySelectorAll('input[name="mode"]').forEach(input => input.addEventListener("change", () => { updateMode(); resetInspection(); }));
  $("source").addEventListener("change", updateSource);
  $("file-input").addEventListener("change", () => { const file = $("file-input").files[0]; showImage("file-preview", file ? URL.createObjectURL(file) : null); updateSource(); });
  $("inspect").addEventListener("click", inspect);
  $("reconnect").addEventListener("click", async () => { try { renderStatus(await post("/api/camera/reconnect", {})); } catch (e) { showError("pixel-format-message", e); } });
  $("apply-pixel-format").addEventListener("click", async () => { try { const result = await post("/api/camera/pixel-format", {pixel_format: $("pixel-format").value}); renderStatus(result.status); $("pixel-format").value = result.status.pixel_format; $("pixel-format-message").textContent = result.message; } catch (e) { showError("pixel-format-message", e); } });
  $("roi-level").addEventListener("change", async () => { try { await previewRoi("level", $("roi-level").value); } catch (e) { showError("roi-message", e); } });
  $("roi-threshold").addEventListener("input", async () => { $("roi-value").textContent = $("roi-threshold").value; try { await previewRoi("threshold", $("roi-threshold").value); } catch (e) { showError("roi-message", e); } });
  $("save-roi").addEventListener("click", async () => { try { renderRoi(await post("/api/roi", {threshold: Number($("roi-threshold").value)})); } catch (e) { showError("roi-message", e); } });
  $("load-roi").addEventListener("click", async () => { try { renderRoi(await request("/api/roi")); } catch (e) { showError("roi-message", e); } });
  $("open-lighting").addEventListener("click", () => { $("lighting-modal").hidden = false; });
  $("close-lighting").addEventListener("click", () => { $("lighting-modal").hidden = true; });
  $("save-lighting").addEventListener("click", async () => { try { $("lighting-config-message").textContent = (await post("/api/lighting/config", {values: currentLighting()})).message; } catch (e) { showError("lighting-config-message", e); } });
  $("load-lighting").addEventListener("click", async () => { try { const data = await request("/api/lighting"); updateLightingValues(data.values); $("lighting-config-message").textContent = data.message; } catch (e) { showError("lighting-config-message", e); } });
  $("apply-lighting").addEventListener("click", () => applyLighting().catch(e => showError("lighting-message", e)));
  $("open-statistics").addEventListener("click", () => { $("inspection-page").hidden = true; $("statistics-page").hidden = false; refreshStatistics(1); });
  $("back-inspection").addEventListener("click", async () => { $("statistics-page").hidden = true; $("inspection-page").hidden = false; try { renderHistory(await request("/api/history")); } catch (_) {} });
  ["statistics-start", "statistics-end", "statistics-source", "statistics-refresh"].forEach(id => $(id).addEventListener("change", () => refreshStatistics(1)));
  $("statistics-refresh").addEventListener("click", () => refreshStatistics(1));
  $("statistics-previous").addEventListener("click", () => refreshStatistics(state.page - 1));
  $("statistics-next").addEventListener("click", () => refreshStatistics(state.page + 1));
  $("statistics-export").addEventListener("click", async () => { const params = new URLSearchParams({source: $("statistics-source").value}); if ($("statistics-start").value) params.set("start", $("statistics-start").value); if ($("statistics-end").value) params.set("end", $("statistics-end").value); try { const response = await fetch(`/api/statistics/export/csv?${params}`); if (!response.ok) throw new Error((await response.json()).detail); const blob = await response.blob(); const url = URL.createObjectURL(blob); const link = document.createElement("a"); link.href = url; link.download = /filename="?([^";]+)/.exec(response.headers.get("content-disposition") || "")?.[1] || "inspection.csv"; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); $("statistics-message").textContent = "CSV 생성 완료 · 선택한 기간의 전체 기록이 포함됩니다."; } catch (e) { showError("statistics-message", e); } });
  $("clear-history").addEventListener("click", async () => { try { const data = await post("/api/statistics/clear", {confirmed: $("clear-confirm").checked}); $("clear-confirm").checked = false; await refreshStatistics(1); renderHistory([]); $("statistics-message").textContent = data.message; } catch (e) { showError("statistics-message", e); } });
  setInterval(refreshPreview, 200);
  setInterval(async () => { try { renderStatus(await request("/api/status")); } catch (_) {} }, 1000);
});
