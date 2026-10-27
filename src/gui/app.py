from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from html import escape
from pathlib import Path
import sqlite3
from time import monotonic
from typing import TYPE_CHECKING
from xml.etree import ElementTree

import gradio as gr
from src.gui.controls import brightness_slider, roi_threshold_controls
from src.gui.statistics import StatisticsController, TABLE_HEADERS, summary_html
from src.inspection_history import InspectionHistoryStore, InspectionRecord, LOCAL_TIMEZONE
from src.hardware.camera_control import CameraControl, CameraError
from src.hardware.camera_config import DEFAULT_CAMERA_CONFIG, load_camera_settings, save_camera_pixel_format
from src.hardware.lighting_config import LightingConfigStore
from src.hardware.led_control import (
    BAUD_RATE,
    COM_PORT,
    LIGHT_GROUPS,
    OUTPUT_MAX,
    LightCommunicationError,
    LightTransmission,
    connect_light,
    disconnect_light,
    get_light_status,
    set_light_group_brightness,
    set_light_groups_brightness,
)
from src.roi.threshold_config import RoiThresholdConfigStore
from src.roi.quality_display import roi_score_message

if TYPE_CHECKING:
    from src.gui.inspection import InspectionEngine

APP_DIR = Path(__file__).parent
TEMPLATE_DIR = APP_DIR / "templates"
CSS_PATH = TEMPLATE_DIR / "style.css"
FRAGMENTS_TEMPLATE_PATH = TEMPLATE_DIR / "fragments.html"
PROJECT_ROOT = APP_DIR.resolve().parents[1]
MAX_HISTORY = 5
INSPECTION_MODES = {
    "normal": "일반 모드",
    "certification": "공인인증 평가 모드",
}
lighting_config = LightingConfigStore()
INITIAL_LIGHTING_VALUES, INITIAL_LIGHTING_MESSAGE = lighting_config.initial_for_ui()
DEFAULT_BRIGHTNESS = sum(INITIAL_LIGHTING_VALUES) / len(LIGHT_GROUPS)
LIGHT_COUNT = len(LIGHT_GROUPS)
DEFAULT_LIGHTING_LABEL = f"4개 조명 설정 · 평균 {DEFAULT_BRIGHTNESS:.0f}%"
history_store = InspectionHistoryStore()
statistics_controller = StatisticsController(lambda: history_store)


camera_control = CameraControl()
inspection_engine: InspectionEngine | None = None
model_error: str | None = None
camera_enabled = True
roi_config_path: Path | None = None
active_camera_config_path = DEFAULT_CAMERA_CONFIG


def renderFragment(name: str, **values: object) -> str:
    root = ElementTree.parse(FRAGMENTS_TEMPLATE_PATH).getroot()
    fragment = root.find(f"template[@id='{name}']")
    if fragment is None:
        raise ValueError(f"Unknown template fragment: {name}")
    content = "".join(
        ElementTree.tostring(child, encoding="unicode", method="html")
        for child in fragment
    )
    return content.format(**values)


def formatInspectionDuration(elapsed_seconds: float | None) -> str:
    return "--" if elapsed_seconds is None else f"{elapsed_seconds:.3f}초"


def renderHistory(records: Sequence[InspectionRecord]) -> str:
    rows: list[str] = []
    for index in range(MAX_HISTORY):
        position = index + 1
        if index >= len(records):
            rows.append(renderFragment("history_empty_row", position=position))
            continue

        record = records[index]
        rows.append(
            renderFragment(
                "history_row",
                position=position,
                p1=record.p1,
                p2=record.p2,
                p3=record.p3,
                timestamp=escape(record.measured_at),
                source=escape(record.source),
                mode=escape(INSPECTION_MODES.get(
                    record.metadata.get("conditions", {}).get("inspection_mode"), "")),
                duration=formatInspectionDuration(record.elapsed_seconds),
                saved="저장됨" if record.record_id else "미저장",
            )
        )
    return renderFragment("history_list", rows="".join(rows))


def loadHistory() -> tuple:
    try:
        records = history_store.query(limit=MAX_HISTORY)
        return renderHistory(records), records
    except (OSError, ValueError, sqlite3.Error) as error:
        return f'<div class="hint">이력 불러오기 실패: {escape(str(error))}</div>', []


def calculateLighting(
    light_1: float,
    light_2: float,
    light_3: float,
    light_4: float,
) -> tuple[float, str]:
    average = sum((light_1, light_2, light_3, light_4)) / LIGHT_COUNT
    label = f"4개 조명 설정 · 평균 {average:.0f}%"
    return average, label


def loadLightingConfig() -> tuple:
    values, message = lighting_config.load_for_ui()
    if values is None:
        return *(gr.skip() for _ in range(LIGHT_COUNT + 2)), message
    average, label = calculateLighting(*values)
    return *values, average, label, message


def saveLightingConfig(light_1: float, light_2: float, light_3: float, light_4: float) -> str:
    return lighting_config.save_for_ui((light_1, light_2, light_3, light_4))


def roiThresholdConfig() -> RoiThresholdConfigStore:
    config = getattr(inspection_engine, "config", {})
    return RoiThresholdConfigStore(config.get("auto_roi", {}).get("boundary_config") or roi_config_path)


def lightingReceipt(group: int, transmissions: Sequence[LightTransmission]) -> object:
    fixture = LIGHT_GROUPS[group - 1]
    accepted = [sent for sent in transmissions if sent.channel in fixture.channels]
    if not accepted:
        return gr.skip()
    complete = len(accepted) == len(fixture.channels)
    state = "최근 ACK 확인" if complete else "부분 적용 · ACK 확인"
    values = " · ".join(f"CH{sent.channel} {sent.percent:g}% ({sent.value}/255)" for sent in accepted)
    return f"**{state}**\n\n{values}"


def lightingFailure(error: LightCommunicationError | ValueError) -> str:
    accepted = getattr(error, "acknowledged", ())
    detail = ""
    if accepted:
        detail = " · 이번 요청의 ACK 확인: " + ", ".join(f"CH{sent.channel}" for sent in accepted)
        detail += ". 이후 채널은 전송하지 않았습니다."
    return f"⚠ 전송 실패: {error}{detail}"


def applyLighting(
    light_1: float,
    light_2: float,
    light_3: float,
    light_4: float,
) -> tuple:
    """Send four physical fixture settings across all six controller channels."""
    values = (light_1, light_2, light_3, light_4)
    average, label = calculateLighting(*values)
    try:
        transmissions = set_light_groups_brightness(*values)
    except (LightCommunicationError, ValueError) as error:
        receipts = tuple(lightingReceipt(group, getattr(error, "acknowledged", ())) for group in range(1, LIGHT_COUNT + 1))
        return average, label, gr.update(visible=True), lightingFailure(error), *receipts

    frames = " · ".join(sent.packet.hex(" ").upper() for sent in transmissions)
    message = f"✓ 4개 조명 · CH1~CH6 ACK 확인 ({COM_PORT}, {BAUD_RATE} bps): {frames}"
    receipts = tuple(lightingReceipt(group, transmissions) for group in range(1, LIGHT_COUNT + 1))
    return average, label, gr.update(visible=False), message, *receipts


def applyLightGroup(
    group: int,
    light_1: float,
    light_2: float,
    light_3: float,
    light_4: float,
) -> tuple:
    """Apply the selected physical fixture, including both outputs for panels."""
    values = (light_1, light_2, light_3, light_4)
    average, label = calculateLighting(*values)
    if not 1 <= group <= LIGHT_COUNT:
        return average, label, "⚠ 알 수 없는 조명입니다.", gr.skip()
    fixture = LIGHT_GROUPS[group - 1]
    try:
        sent = set_light_group_brightness(group, values[group - 1])
    except (LightCommunicationError, ValueError) as error:
        receipt = lightingReceipt(group, getattr(error, "acknowledged", ()))
        return average, label, f"{fixture.name} ({fixture.channel_label}) · {lightingFailure(error)}", receipt

    message = (
        f"✓ {fixture.name} ({fixture.channel_label}) 적용 완료 · 모두 ACK 확인 "
        f"({COM_PORT}, {BAUD_RATE} bps): " + " · ".join(item.packet.hex(" ").upper() for item in sent)
    )
    return average, label, message, lightingReceipt(group, sent)


def showLighting() -> object:
    return gr.update(visible=True)


def hideLighting() -> object:
    return gr.update(visible=False)


def renderStatus() -> str:
    camera = camera_control.get_status()
    light = get_light_status()
    light_state = "RS232 OPEN · 제어 대기" if light.connected else "연결 대기"
    if light.ack_verified:
        light_state = "RS232 OPEN · ACK 확인"
    elif light.error:
        light_state = light.error
    lighting_detail = (f"{escape(COM_PORT)} · {BAUD_RATE} bps · {escape(light_state)}"
                       if camera_enabled else "파일 검사 모드 · 연결하지 않음")
    camera_state = "READY" if camera.connected else escape(camera.error or "연결 대기")
    if not camera_enabled:
        camera_state = "파일 검사 모드"
    elif (camera.connected and (camera.last_frame_at is None
                               or monotonic() - camera.last_frame_at > 2.0)):
        camera_state = "프레임 수신 대기"
    model_state = "LOADED" if inspection_engine is not None else escape(model_error or "로딩 대기")
    detail = ""
    if inspection_engine is not None:
        roi = inspection_engine.config["roi"]
        if inspection_engine.config.get("auto_roi", {}).get("enabled"):
            last = inspection_engine.last_roi_detection
            state = escape(last["quality_status"]) if last else "측정 시 추출"
            detail = f'<div>ROI: 자동 Gabor 경계 · {state}</div>'
        else:
            detail = (f'<div>ROI: {roi["width"]} × {roi["height"]} '
                      f'@ ({roi["x"]}, {roi["y"]})</div>')
        detail += f'<div>PIXEL: {inspection_engine.checkpoint["threshold"]:g}</div>'
        positive_classes = inspection_engine.checkpoint.get("positive_classes", ("Exposure",))
        detail += f'<div>대상 픽셀: {escape(" + ".join(positive_classes))}</div>'
    return (
        '<div class="section-label">SYSTEM STATUS</div><div class="system-list">'
        f'<div>CAMERA: {camera_state}</div>'
        f'<div>촬영 형식: {"GrayScale · Mono8" if camera_control.settings.pixel_format == "Mono8" else "RGB Color · BayerGR12"}</div>'
        f'<div>노출 설정: {camera_control.settings.exposure_us:g} µs</div><div>MODEL: {model_state}</div>'
        f'<div>LIGHTING: {lighting_detail}</div>'
        f'{detail}</div>'
    )


def getCameraPreview() -> tuple[object, str]:
    frame = camera_control.get_latest_frame()
    return frame, renderStatus()


def reconnectCamera() -> str:
    global camera_enabled
    camera_enabled = True
    camera_control.start()
    return renderStatus()


def applyCameraPixelFormat(pixel_format: str) -> tuple[str, str]:
    if pixel_format not in ("Mono8", "BayerGR12"):
        return renderStatus(), "⚠ 촬영 형식을 선택하세요."
    try:
        save_camera_pixel_format(pixel_format, active_camera_config_path)
        camera_control.set_pixel_format(pixel_format)
        if camera_enabled and not camera_control.is_running:
            camera_control.start()
    except (ValueError, OSError, CameraError) as error:
        return renderStatus(), f"⚠ 촬영 형식 적용 실패: {error}"
    label = "GrayScale (Mono8)" if pixel_format == "Mono8" else "RGB Color (BayerGR12)"
    connection = "카메라 재연결 중" if camera_enabled else "다음 카메라 연결 시 적용"
    return renderStatus(), f"✓ {label} 저장 완료 · {connection}"


def initialSource() -> str:
    return "camera" if camera_enabled else "file"


def modeDescription(mode: str) -> str:
    if mode == "certification":
        return ("<div class=\"hint mode-hint\">공인인증 평가 모드 선택됨 · "
                "어두운 영역 3개를 검출하고 회전 보정해 각 크롭을 256×256 영상 전체로 확대해 추론합니다. "
                "점수 분모에서 원본 크롭의 검정 픽셀을 제외합니다. "
                "인증 판정 기준은 아직 적용되지 않습니다.</div>")
    return "<div class=\"hint mode-hint\">일반 모드 선택됨 · 기존 측정 절차를 사용합니다.</div>"


def modePresentation(mode: str) -> tuple:
    if mode == "certification":
        return (modeDescription(mode), renderFragment("measurement_info_pce"),
                *(gr.update(label=f"P{index} · {position} · 256×256")
                  for index, position in enumerate(("왼쪽", "가운데", "오른쪽"), 1)))
    return (modeDescription(mode), renderFragment("measurement_info"),
            *(gr.update(label=f"P{index} · {position} · 256×512")
              for index, position in enumerate(("왼쪽", "가운데", "오른쪽"), 1)))


def _inspectionError(message: str, records: list[InspectionRecord]) -> tuple:
    caption = escape(message).replace("\n", "<br>")
    result = (
        '<div class="summary"><div class="result-state error">ERROR</div>'
        f'<div class="result-caption">{caption}</div></div>'
    )
    return result, renderHistory(records), records, None, None, None, None


def inspectCamera(
    history_records: list[InspectionRecord],
    source: str = "camera",
    uploaded_frame: object = None,
    light_1: float | None = None, light_2: float | None = None,
    light_3: float | None = None, light_4: float | None = None,
    inspection_mode: str = "normal",
) -> tuple:
    started_at = monotonic()
    if inspection_mode not in INSPECTION_MODES:
        return _inspectionError("검사 모드를 선택하세요.", history_records)
    if source not in ("camera", "file"):
        return _inspectionError("검사 입력을 선택하세요.", history_records)
    frame = camera_control.get_latest_frame() if source == "camera" else uploaded_frame
    if frame is None:
        if source == "file":
            return _inspectionError("검사할 이미지 파일을 선택하세요.", history_records)
        status = camera_control.get_status()
        return _inspectionError(status.error or "최근 2초 이내 수신된 카메라 프레임이 없습니다.", history_records)
    if inspection_engine is None:
        return _inspectionError(model_error or "모델이 로드되지 않았습니다.", history_records)
    try:
        inference = (inspection_engine.inspect_certification(frame)
                     if inspection_mode == "certification" else inspection_engine.inspect(frame))
    except (ValueError, RuntimeError, OSError) as error:
        return _inspectionError(str(error), history_records)
    scores = inference.scores
    timestamp = datetime.now(LOCAL_TIMEZONE).isoformat(timespec="milliseconds")
    elapsed_seconds = monotonic() - started_at
    metadata = {
        "roi": getattr(inference, "roi", None),
        "regions": list(getattr(inference, "regions", ())),
        "roi_detection": getattr(inference, "roi_detection", None),
        "model": getattr(inspection_engine, "model_info", getattr(inspection_engine, "checkpoint", {})),
        "conditions": {"source": source, "inspection_mode": inspection_mode},
        "frame_shape": list(frame.shape),
    }
    valid_pixels = getattr(inference, "valid_pixels", None)
    if inspection_mode == "certification" and valid_pixels is not None:
        metadata["conditions"]["valid_pixels"] = list(map(int, valid_pixels))
        metadata["conditions"]["denominator"] = "nonblack_rectified_crop_pixels"
    if source == "camera":
        metadata["conditions"].update(
            exposure_us=camera_control.settings.exposure_us,
            pixel_format=camera_control.settings.pixel_format,
            camera_serial=camera_control.serial_number,
            lighting_ui_percent=dict(zip((g.config_key for g in LIGHT_GROUPS),
                                          (light_1, light_2, light_3, light_4))),
        )
    record = InspectionRecord(*map(float, scores), measured_at=timestamp,
                              source=source, elapsed_seconds=elapsed_seconds, metadata=metadata)
    saved_message = "이력 자동 저장 완료"
    try:
        record = history_store.save(record, frame, inference.annotated_frame)
    except (OSError, ValueError, sqlite3.Error) as error:
        saved_message = f"이력 저장 실패: {error}"
    updated_records = [record, *history_records][:MAX_HISTORY]
    if record.record_id:
        try:
            updated_records = history_store.query(limit=MAX_HISTORY)
        except (OSError, ValueError, sqlite3.Error):
            pass  # A saved result remains visible if the subsequent recent-history read fails.
    result_html = renderFragment(
            "result",
            average=inference.average,
            timestamp=escape(timestamp),
            duration=formatInspectionDuration(elapsed_seconds),
            saved_message=escape(saved_message),
            p1=scores[0],
            p2=scores[1],
            p3=scores[2],
            measurement_detail=("각 검출 영역을 회전 보정 후 256×256px로 확대 · 검정 픽셀은 분모에서 제외"
                                if inspection_mode == "certification"
                                else "각 256×512px · 중심 간격 547px"),
        )
    result_html = (f'<div class="mode-result">{escape(INSPECTION_MODES[inspection_mode])}</div>'
                   + result_html)
    positive_classes = getattr(inspection_engine, "checkpoint", {}).get("positive_classes", ("Exposure",))
    if len(positive_classes) > 1:
        result_html += (f'<div class="result-caption">대상 픽셀: '
                        f'{escape(" + ".join(positive_classes))}</div>')
    if inspection_mode == "certification" and valid_pixels is not None:
        result_html += (f'<div class="result-caption">유효 픽셀 · P1 {valid_pixels[0]:,} · '
                        f'P2 {valid_pixels[1]:,} · P3 {valid_pixels[2]:,}</div>')
        angles = [region.get("rotation_deg") for region in getattr(inference, "regions", ())]
        if all(angle is not None for angle in angles):
            result_html += ('<div class="result-caption">회전 보정 · '
                            + ' · '.join(f'P{index} {angle:+.1f}°'
                                       for index, angle in enumerate(angles, 1)) + '</div>')
    detection = getattr(inference, "roi_detection", None)
    if detection is not None:
        result_html += (f'<div class="result-caption">자동 ROI '
                        f'· {escape(roi_score_message(detection))} · P1~P3 측정 영역만 추론</div>')
    return (
        result_html,
        renderHistory(updated_records),
        updated_records,
        inference.annotated_frame,
        *inference.crops,
    )


def statisticsView(start_date=None, end_date=None, source="all", page=1):
    return (*statistics_controller.query(start_date, end_date, source, page),
            "측정 항목을 선택하세요.", None, None, None, None, None, {})


def showStatistics(start_date=None, end_date=None, source="all"):
    return (gr.update(visible=False), gr.update(visible=True),
            *statisticsView(start_date, end_date, source), gr.update(value=False))


def previousStatistics(start_date, end_date, source, page):
    return statisticsView(start_date, end_date, source, page - 1)


def nextStatistics(start_date, end_date, source, page):
    return statisticsView(start_date, end_date, source, page + 1)


def clearHistory(confirmed: bool, start_date=None, end_date=None, source="all") -> tuple:
    """Clear all saved measurements, then refresh archive and recent history."""
    if not confirmed:
        skipped = ["삭제 확인을 선택한 뒤 다시 누르세요." if index == 5 else gr.skip()
                   for index in range(14)]
        return (*skipped, gr.skip(), gr.skip(), gr.skip())
    try:
        count = history_store.clear_all()
        message = f"전체 측정 이력 {count:,}건과 저장 영상·CSV를 초기화했습니다."
    except (OSError, ValueError, sqlite3.Error) as error:
        message = f"초기화 실패 또는 일부 파일 삭제 실패: {error}"
    view = list(statisticsView(start_date, end_date, source))
    view[5] = message
    history_html, records = loadHistory()
    return (*view, history_html, records, gr.update(value=False))


theme = gr.themes.Base(
    font=[gr.themes.Font(name) for name in ("Malgun Gothic", "Segoe UI", "sans-serif")],
    font_mono=[gr.themes.Font(name) for name in ("Consolas", "monospace")],
).set(
    body_background_fill="#1e1e1e",
    body_background_fill_dark="#1e1e1e",
    background_fill_primary="#252526",
    background_fill_primary_dark="#252526",
    background_fill_secondary="#2d2d30",
    background_fill_secondary_dark="#2d2d30",
    border_color_primary="#3c3c3c",
    border_color_primary_dark="#3c3c3c",
    table_text_color="#cccccc",
    table_text_color_dark="#cccccc",
    table_even_background_fill="#252526",
    table_even_background_fill_dark="#252526",
    table_odd_background_fill="#2d2d30",
    table_odd_background_fill_dark="#2d2d30",
    table_border_color="#3c3c3c",
    table_border_color_dark="#3c3c3c",
    table_row_focus="#173b57",
    table_row_focus_dark="#173b57",
    body_text_color="#cccccc",
    body_text_color_dark="#cccccc",
    block_background_fill="#252526",
    block_background_fill_dark="#252526",
    block_border_width="0px",
    block_radius="0px",
    input_background_fill="#3c3c3c",
    input_background_fill_dark="#3c3c3c",
    input_border_color="#3c3c3c",
    input_border_color_focus="#007fd4",
    button_primary_background_fill="#0e639c",
    button_primary_background_fill_hover="#1177bb",
    button_primary_text_color="#ffffff",
    block_label_background_fill="#252526",
    block_label_background_fill_dark="#252526",
    block_label_text_color="#cccccc",
    block_label_text_color_dark="#cccccc",
    button_secondary_background_fill="#2d2d30",
    button_secondary_background_fill_dark="#2d2d30",
    button_secondary_background_fill_hover="#3c3c3c",
    button_secondary_text_color="#cccccc",
    button_secondary_text_color_dark="#cccccc",
    button_secondary_border_color="#3c3c3c",
    checkbox_background_color="#2d2d30",
    checkbox_label_background_fill="#2d2d30",
    checkbox_label_text_color="#cccccc",
    checkbox_label_text_color_selected="#ffffff",
)


with gr.Blocks(title="Vision Workbench", analytics_enabled=False) as demo:
    history_records = gr.State(value=[])
    average_brightness = gr.State(value=DEFAULT_BRIGHTNESS)
    with gr.Row(equal_height=False, elem_classes="main-grid") as inspection_page:
        with gr.Column(scale=2, elem_classes="side"):
            with gr.Column(scale=0, elem_classes=["ui-block", "mode-panel"]):
                gr.HTML('<div class="section-label">검사 모드</div>')
                inspection_mode = gr.Radio(
                    choices=[(label, key) for key, label in INSPECTION_MODES.items()],
                    value="normal", show_label=False, elem_classes="inspection-mode",
                )
                mode_hint = gr.HTML(modeDescription("normal"))
            with gr.Column(scale=0, elem_classes="ui-block"):
                gr.HTML(renderFragment("capture_label"))
                camera_preview = gr.Image(
                    label="실시간 프리뷰", type="numpy", interactive=False,
                    elem_classes="asset-frame", show_label=False,
                    height=160, buttons=[],
                )
                reconnect_button = gr.Button("카메라 연결 / 재시도", size="sm")
                camera_pixel_format = gr.Radio(
                    choices=[("GrayScale · Mono8", "Mono8"),
                             ("RGB Color · BayerGR12", "BayerGR12")],
                    value=camera_control.settings.pixel_format,
                    label="카메라 촬영 형식", elem_classes="input-source",
                )
                pixel_format_apply = gr.Button("촬영 형식 적용 및 저장", size="sm")
                pixel_format_status = gr.Markdown(
                    "형식 변경 시 카메라를 재연결합니다. 검사 입력은 두 형식 모두 RGB 배열로 처리합니다.")
                input_source = gr.Radio(
                    choices=[("실시간 카메라", "camera"), ("이미지 파일", "file")],
                    value="camera", label="검사 입력", elem_classes="input-source",
                )
                file_input = gr.Image(
                    label="검사 이미지", type="numpy", image_mode="RGB",
                    sources=["upload"], height=140, visible=False,
                    elem_classes="file-input", buttons=[],
                )
            with gr.Column(
                scale=0,
                elem_classes=["ui-block", "control-group"],
            ):
                gr.HTML(renderFragment("lighting_label"))
                lighting_button = gr.Button(
                    DEFAULT_LIGHTING_LABEL,
                    elem_classes="lighting-open-button",
                )
                gr.HTML(renderFragment("lighting_hint"))
            with gr.Column(scale=0, elem_classes=["ui-block", "control-group"]):
                gr.HTML('<div class="section-label">자동 ROI 설정</div>')
                roi_threshold_controls(demo, roiThresholdConfig)
            with gr.Column(scale=0, elem_classes=["ui-block", "measurement-info"]):
                measurement_info = gr.HTML(renderFragment("measurement_info"))

        with gr.Column(scale=6, elem_classes="workbench"):
            with gr.Column(elem_classes="ui-block"):
                gr.HTML('<div class="editor-tab">검사 영상 <span>P1~P3 검출 영역은 빨간색</span></div>')
                inspection_output = gr.Image(
                    label="마지막 검사 결과", type="numpy", interactive=False,
                    show_label=False, elem_classes="inspection-image",
                    height=320,
                )
                with gr.Row(elem_classes="region-images"):
                    p1_image = gr.Image(label="P1 · 왼쪽 · 256×512", type="numpy", interactive=False, height=180, elem_classes="p1")
                    p2_image = gr.Image(label="P2 · 가운데 · 256×512", type="numpy", interactive=False, height=180, elem_classes="p2")
                    p3_image = gr.Image(label="P3 · 오른쪽 · 256×512", type="numpy", interactive=False, height=180, elem_classes="p3")
                inspect_button = gr.Button("측정 시작", variant="primary", elem_classes="overview-inspect-button")

        with gr.Column(scale=2, elem_classes="results"):
            with gr.Column(elem_classes="ui-block"):
                gr.HTML(renderFragment("result_label"))
                result = gr.HTML(renderFragment("result_initial"))
            with gr.Column(elem_classes="ui-block"):
                with gr.Row(elem_classes="history-header"):
                    gr.HTML(renderFragment("history_label"))
                    statistics_button = gr.Button("통계 보기", size="sm", scale=0, min_width=80)
                history = gr.HTML(renderHistory([]))
            with gr.Column(elem_classes="ui-block"):
                system_status = gr.HTML(renderStatus())

    with gr.Column(visible=False, elem_classes="statistics-page") as statistics_page:
        with gr.Row(elem_classes="statistics-header"):
            gr.HTML('<div><div class="section-label">MEASUREMENT ARCHIVE</div><h1>통계 보기</h1>'
                    '<p>저장된 측정값과 결과 이미지를 기간별로 확인하세요.</p></div>')
            statistics_back = gr.Button("검사 화면으로", scale=0, min_width=130)
        with gr.Row(elem_classes="statistics-filters"):
            statistics_start = gr.DateTime(include_time=False, type="string", timezone="Asia/Seoul", label="시작일")
            statistics_end = gr.DateTime(include_time=False, type="string", timezone="Asia/Seoul", label="종료일")
            statistics_source = gr.Dropdown(choices=[("전체 입력", "all"), ("카메라", "camera"), ("이미지 파일", "file")],
                                            value="all", label="검사 입력")
            statistics_refresh = gr.Button("조회", scale=0, min_width=80)
            statistics_export = gr.Button("CSV 내보내기", variant="primary", scale=0, min_width=135)
        with gr.Accordion("전체 측정 이력 초기화", open=False):
            gr.Markdown("모든 측정 기록, 저장된 원본·결과 이미지, CSV 내보내기 파일을 삭제합니다. "
                        "현재 조회 필터와 관계없이 전체 이력에 적용됩니다.")
            history_clear_confirm = gr.Checkbox(
                label="전체 측정 이력과 저장 파일 삭제를 확인했습니다.", value=False)
            history_clear_button = gr.Button("전체 이력 초기화", variant="stop", size="sm")
        statistics_summary = gr.HTML(summary_html({}))
        statistics_status = gr.Markdown("", elem_classes="statistics-status")
        statistics_csv = gr.File(label="CSV 다운로드", interactive=False, file_types=[".csv"], visible=False)
        statistics_table = gr.Dataframe(headers=TABLE_HEADERS, datatype=["str", "str", *["number"] * 5],
                                        type="array", interactive=False, row_count=0,
                                        label="측정 이력 · 행을 선택해 상세 보기", max_height=420,
                                        column_widths=[260, 90, 100, 100, 100, 110, 150])
        statistics_ids = gr.State([])
        statistics_page_number = gr.State(1)
        with gr.Row(elem_classes="statistics-pagination"):
            statistics_previous = gr.Button("이전", size="sm", scale=0, min_width=80)
            statistics_page_label = gr.Markdown("1 / 1 페이지")
            statistics_next = gr.Button("다음", size="sm", scale=0, min_width=80)
        with gr.Column(elem_classes="statistics-detail"):
            statistics_detail = gr.Markdown("측정 항목을 선택하세요.")
            with gr.Row():
                statistics_original = gr.Image(label="저장 원본", interactive=False, height=300)
                statistics_result = gr.Image(label="저장 결과", interactive=False, height=300)
            with gr.Row(elem_classes="archive-regions"):
                statistics_p1 = gr.Image(label="P1", interactive=False, height=230)
                statistics_p2 = gr.Image(label="P2", interactive=False, height=230)
                statistics_p3 = gr.Image(label="P3", interactive=False, height=230)
            with gr.Accordion("촬영·검사 조건", open=False):
                statistics_metadata = gr.JSON(label="저장된 측정 조건")

    preview_timer = gr.Timer(value=0.2)
    preview_timer.tick(getCameraPreview, outputs=[camera_preview, system_status],
                       show_progress="hidden", queue=False, api_visibility="private")
    reconnect_button.click(reconnectCamera, outputs=system_status, show_progress="hidden")
    demo.load(lambda: camera_control.settings.pixel_format, outputs=camera_pixel_format,
              queue=False, api_visibility="private")
    pixel_format_apply.click(applyCameraPixelFormat, inputs=camera_pixel_format,
                             outputs=[system_status, pixel_format_status],
                             show_progress="hidden", concurrency_id="camera-settings",
                             concurrency_limit=1)
    demo.load(initialSource, outputs=input_source, queue=False)
    input_source.change(lambda source: gr.update(visible=source == "file"),
                        inputs=input_source, outputs=file_input, queue=False)
    inspection_mode.change(modePresentation, inputs=inspection_mode,
                           outputs=[mode_hint, measurement_info, p1_image, p2_image, p3_image],
                           queue=False, api_visibility="private")

    with gr.Column(visible=False, elem_classes="lighting-modal") as lighting_modal:
        with gr.Column(elem_classes="lighting-dialog"):
            gr.HTML(renderFragment("lighting_modal_header"))
            with gr.Row():
                lighting_save = gr.Button("조명값 저장", size="sm")
                lighting_load = gr.Button("저장값 불러오기", size="sm")
            lighting_config_status = gr.Markdown(INITIAL_LIGHTING_MESSAGE, elem_classes="lighting-config-status")
            light_controls, light_apply_buttons, light_receipts = [], [], []
            with gr.Row(elem_classes="lighting-group-grid"):
                for group, fixture in enumerate(LIGHT_GROUPS, 1):
                    with gr.Column(min_width=0, elem_classes="lighting-group-card"):
                        gr.HTML(renderFragment(
                            "lighting_group_header", number=group, fixture_name=fixture.name,
                            channels=fixture.channel_label,
                            mode="두 채널 같은 밝기로 제어" if len(fixture.channels) > 1 else "단일 채널 제어",
                        ))
                        level = brightness_slider(f"{fixture.name} 밝기 (%)", INITIAL_LIGHTING_VALUES[group - 1])
                        apply = gr.Button("이 조명 적용", size="sm")
                        receipt = gr.Markdown("최근 ACK: **미확인**", elem_classes="lighting-group-receipt")
                        light_controls.append(level)
                        light_apply_buttons.append(apply)
                        light_receipts.append(receipt)
            lighting_close = gr.Button(
                "설정 적용 후 닫기",
                variant="primary",
                elem_classes="lighting-close-button",
            )
            lighting_cancel = gr.Button("닫기", size="sm")
            serial_status = gr.Markdown(
                f"통신 설정: `{COM_PORT}` · `{BAUD_RATE}` bps · 8N1 · 출력 0–`{OUTPUT_MAX}`",
                elem_classes="serial-status",
            )

    lighting_button.click(
        showLighting,
        outputs=lighting_modal,
        show_progress="hidden",
    )
    lighting_close.click(
        applyLighting,
        inputs=light_controls,
        outputs=[average_brightness, lighting_button, lighting_modal, serial_status, *light_receipts],
        show_progress="hidden",
        api_name="lighting_all",
        concurrency_id="lighting", concurrency_limit=1,
    )
    lighting_cancel.click(hideLighting, outputs=lighting_modal, queue=False)
    config_outputs = [*light_controls, average_brightness, lighting_button, lighting_config_status]
    lighting_save.click(
        saveLightingConfig, inputs=light_controls, outputs=lighting_config_status,
        api_name="lighting_config_save", show_progress="hidden",
        concurrency_id="lighting", concurrency_limit=1,
    )
    lighting_load.click(
        loadLightingConfig, outputs=config_outputs, api_name="lighting_config_load",
        show_progress="hidden", concurrency_id="lighting", concurrency_limit=1,
    )
    demo.load(loadLightingConfig, outputs=config_outputs, api_visibility="private", show_progress="hidden",
              concurrency_id="lighting", concurrency_limit=1)

    for light_control in light_controls:
        light_control.change(
            calculateLighting,
            inputs=light_controls,
            outputs=[average_brightness, lighting_button],
            show_progress="hidden",
            queue=False,
        )

    def groupCallback(group: int):
        def apply(left_lower_panel, left_upper_bar, right_upper_bar, right_upper_panel):
            return applyLightGroup(group, left_lower_panel, left_upper_bar, right_upper_bar, right_upper_panel)
        return apply

    for group, (light_control, apply_button, receipt) in enumerate(
        zip(light_controls, light_apply_buttons, light_receipts, strict=True), start=1
    ):
        callback = groupCallback(group)
        event_options = dict(
            inputs=light_controls,
            outputs=[average_brightness, lighting_button, serial_status, receipt],
            show_progress="hidden",
            concurrency_id="lighting", concurrency_limit=1,
        )
        apply_button.click(callback, api_name=f"lighting_group_{group}", **event_options)

    inspect_button.click(
        inspectCamera,
        inputs=[history_records, input_source, file_input, *light_controls, inspection_mode],
        outputs=[result, history, history_records, inspection_output,
                 p1_image, p2_image, p3_image],
        api_name="inspect", concurrency_id="inspection", concurrency_limit=1,
    )
    demo.load(loadHistory, outputs=[history, history_records], api_name="history_load",
              show_progress="hidden")
    statistics_filters = [statistics_start, statistics_end, statistics_source]
    statistics_details = [statistics_detail, statistics_original, statistics_result,
                          statistics_p1, statistics_p2, statistics_p3, statistics_metadata]
    statistics_outputs = [statistics_summary, statistics_table, statistics_ids, statistics_page_number,
                          statistics_page_label, statistics_status, statistics_csv, *statistics_details]
    statistics_event_options = dict(concurrency_id="history-statistics", concurrency_limit=1)
    statistics_button.click(showStatistics, inputs=statistics_filters,
        outputs=[inspection_page, statistics_page, *statistics_outputs, history_clear_confirm],
        api_name="statistics_open",
        **statistics_event_options)
    statistics_back.click(lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[inspection_page, statistics_page], queue=False).then(
        loadHistory, outputs=[history, history_records], show_progress="hidden")
    statistics_refresh.click(statisticsView, inputs=statistics_filters, outputs=statistics_outputs,
                              api_name="statistics_query", **statistics_event_options)
    for component in statistics_filters:
        component.change(statisticsView, inputs=statistics_filters, outputs=statistics_outputs,
                         show_progress="hidden", api_visibility="private", **statistics_event_options)
    statistics_previous.click(previousStatistics, inputs=[*statistics_filters, statistics_page_number],
                               outputs=statistics_outputs, api_visibility="private", **statistics_event_options)
    statistics_next.click(nextStatistics, inputs=[*statistics_filters, statistics_page_number],
                           outputs=statistics_outputs, api_visibility="private", **statistics_event_options)
    statistics_export.click(statistics_controller.export, inputs=statistics_filters,
        outputs=[statistics_csv, statistics_status], api_name="history_csv",
        concurrency_id="inspection", concurrency_limit=1)
    history_clear_button.click(clearHistory, inputs=[history_clear_confirm, *statistics_filters],
        outputs=[*statistics_outputs, history, history_records, history_clear_confirm],
        api_visibility="private", concurrency_id="inspection", concurrency_limit=1)
    statistics_table.select(statistics_controller.detail, inputs=statistics_ids,
        outputs=statistics_details, api_name="history_detail", **statistics_event_options)


def launchApp(
    *, config_path: Path = PROJECT_ROOT / "config/preprocess.yaml",
    checkpoint_path: Path = PROJECT_ROOT / "model/best.pt", device: str = "cpu",
    batch_size: int = 8, camera_serial: str | None = None, use_camera: bool = True,
    inbrowser: bool = True, server_name: str = "127.0.0.1", server_port: int = 7860,
    camera_config_path: Path = DEFAULT_CAMERA_CONFIG,
) -> None:
    global inspection_engine, model_error, camera_control, camera_enabled, roi_config_path, active_camera_config_path
    camera_settings = load_camera_settings(camera_config_path)
    active_camera_config_path = camera_config_path
    camera_enabled = use_camera
    camera_control = CameraControl(serial_number=camera_serial, settings=camera_settings)
    inspection_engine = None
    roi_config_path = None
    try:
        from src.gui.inspection import InspectionEngine
        from src.preprocess import load_config

        roi_config_path = load_config(config_path)["auto_roi"]["boundary_config"]
        inspection_engine = InspectionEngine(config_path, checkpoint_path, device=device,
                                             batch_size=batch_size)
        model_error = None
    except (ImportError, ValueError, RuntimeError, OSError, KeyError) as error:
        model_error = str(error)
    try:
        if use_camera:
            try:
                connect_light()
            except LightCommunicationError:
                pass  # Connection errors remain visible in the system status.
            camera_control.start()
        demo.launch(inbrowser=inbrowser, server_name=server_name, server_port=server_port,
                    theme=theme, css_paths=[CSS_PATH],
                    head=renderFragment("head"))
    finally:
        camera_control.stop()
        disconnect_light()


if __name__ == "__main__":
    launchApp()
