"""Live Basler preview and explicit RS232 light controls, without model loading."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from math import isfinite
import os
from pathlib import Path
import socket
from threading import RLock
from time import monotonic
import urllib.request
import json
import webbrowser
from uuid import uuid4

import gradio as gr
import cv2
import numpy as np
from serial.tools import list_ports

from src.hardware.camera_control import BaslerCameraSettings, CameraControl, CameraError
from src.hardware.camera_config import (
    DEFAULT_CAMERA_CONFIG, load_camera_settings, save_camera_exposure, validate_exposure_us,
)
from src.hardware.lighting_config import LightingConfigStore
from src.hardware.led_control import (
    ACK, BAUD_RATE, COM_PORT, LIGHT_GROUPS, LightCommunicationError, LightController,
    build_brightness_packet,
)
from src.preprocess.auto_roi import AutoRoiResult, detect_auto_roi, draw_roi_boundaries
from src.gui.controls import brightness_slider, roi_threshold_controls
from src.roi.threshold_config import RoiThresholdConfigStore
from src.roi.quality_display import roi_quality_message


CSS_PATH = Path(__file__).parent / "src/gui/templates/cam_test.css"
DEFAULT_CAPTURE_DIR = Path(__file__).resolve().parent / "data" / "cam_test"
CAPTURE_TIMEZONE = timezone(timedelta(hours=9))


def capture_name_stem(name: str) -> str:
    """Accept a Windows filename, optionally ending in .png; never accept a path."""
    if not isinstance(name, str):
        raise ValueError("이미지 이름은 문자열로 입력하세요.")
    name = name.strip()
    if not name:
        return ""
    if name.lower().endswith(".png"):
        name = name[:-4]
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}
    if (not name or name.endswith((".", " "))
            or any(character in '<>:"/\\|?*' or ord(character) < 32 for character in name)
            or name.split(".")[0].upper() in reserved):
        raise ValueError("올바른 이미지 파일 이름을 입력하세요. 경로·특수문자·예약된 이름은 사용할 수 없습니다.")
    return name


class CamTestController:
    def __init__(
        self, serial_number: str | None = None, port: str = COM_PORT,
        baudrate: int = BAUD_RATE, settings: BaslerCameraSettings | None = None,
        roi_config: str | Path | None = None,
        lighting_config: str | Path | None = None,
        camera_config: str | Path = DEFAULT_CAMERA_CONFIG,
        capture_dir: str | Path = DEFAULT_CAPTURE_DIR,
    ) -> None:
        self.camera_config_path = Path(camera_config).expanduser().resolve()
        self.capture_dir = Path(capture_dir).expanduser().resolve()
        self.settings = settings if settings is not None else load_camera_settings(self.camera_config_path)
        self.camera = CameraControl(serial_number, settings=self.settings)
        self.light = LightController(port, baudrate)
        self._lock = RLock()
        self._fps_sample = (monotonic(), 0)
        self._measured_fps = 0.0
        self._camera_enabled = False
        self._next_camera_retry = 0.0
        self.roi_config = roi_config
        self.roi_threshold_config = RoiThresholdConfigStore(roi_config)
        self.roi_result: AutoRoiResult | None = None
        self._light_requested: tuple[int, ...] = ()
        self._light_acknowledged = ()
        self.lighting_config = LightingConfigStore() if lighting_config is None else LightingConfigStore(lighting_config)

    def start(self) -> None:
        self._camera_enabled = True
        self._next_camera_retry = monotonic() + 3
        self.camera.start()
        try:
            self.light.connect()
        except LightCommunicationError:
            pass  # Keep the camera usable; the serial error is shown in the UI.

    def close(self) -> None:
        with self._lock:
            self._camera_enabled = False
            try:
                self.camera.stop()
            finally:
                self.light.disconnect()

    def camera_status(self) -> str:
        status = self.camera.get_status()
        age = None if status.last_frame_at is None else monotonic() - status.last_frame_at
        state = "READY" if status.connected and age is not None and age <= 2 else "대기"
        if status.running and not status.error and not status.connected:
            state = "연결 중"
        if status.error:
            state = f"오류: {status.error}"
        elif status.connected and (age is None or age > 2):
            state = "새 프레임 수신 대기"
        return (f"**CAMERA: {state}**\n\n{status.description}\n\n"
                f"노출 설정: {self.settings.exposure_us:g} µs ({self.settings.exposure_us / 1000:g} ms)")

    def light_status(self) -> str:
        status = self.light.get_status()
        state = "RS232 OPEN · 제어 대기" if status.connected else "연결 해제"
        if status.ack_verified:
            state = "RS232 OPEN · ACK 확인"
        if status.error:
            state = f"오류: {status.error}"
        return f"**LIGHT: {self.light.port} · {self.light.baudrate} bps · 8N1**\n\n{state}"

    def preview(self, show_roi: bool = True) -> tuple:
        with self._lock:
            status = self.camera.get_status()
            if self._camera_enabled and not status.running and monotonic() >= self._next_camera_retry:
                # GigE control ownership may outlive a recently stopped process.
                self._next_camera_retry = monotonic() + 3
                self.camera.start()
                status = self.camera.get_status()
            frame = self.camera.get_latest_camera_frame()
            camera_status, light_status = self.camera_status(), self.light_status()
            if (not status.connected or frame is None or status.last_frame_at is None
                    or monotonic() - status.last_frame_at > 2):
                return None, camera_status, light_status, "최근 수신된 프레임이 없습니다."
            now = monotonic()
            elapsed = now - self._fps_sample[0]
            if elapsed >= 1:
                self._measured_fps = max(0, status.frames_received - self._fps_sample[1]) / elapsed
                self._fps_sample = (now, status.frames_received)
            rgb = frame.display_rgb
            mean = float(rgb.mean())
            stats = (
                f"{frame.width} × {frame.height} · {frame.pixel_format} · "
                f"수신 {status.frames_received}장 · {self._measured_fps:.1f} fps\n\n"
                f"RGB 평균 {mean:.2f}/255 · 원본 {frame.raw_array.dtype} "
                f"범위 {frame.raw_array.min()}–{frame.raw_array.max()}"
            )
            if mean < 5:
                stats += "\n\n영상이 매우 어둡습니다. 조명·렌즈·노출을 확인하세요."
            if self.roi_result is not None:
                stats += f"\n\nROI: {self.roi_result.metadata['quality_status']} · 최근 추출 경계"
                if show_roi and self.roi_result.metadata["roi_detected"]:
                    rgb = draw_roi_boundaries(rgb, self.roi_result.metadata)
            return rgb, camera_status, light_status, stats

    def save_image(self, name: str = "") -> tuple:
        """Save Color and Mono PNGs from one fresh, unannotated RGB snapshot."""
        try:
            stem = capture_name_stem(name)
        except ValueError as error:
            return gr.update(value=None, visible=False), f"이미지 저장 실패: {error}"
        with self._lock:
            status = self.camera.get_status()
            frame = self.camera.get_latest_frame()
            if (not status.connected or frame is None or status.last_frame_at is None
                    or monotonic() - status.last_frame_at > 2):
                return gr.update(value=None, visible=False), "이미지 저장 실패: 최근 수신된 카메라 프레임이 없습니다."
            snapshot = frame.copy()
            captured_at = datetime.now(CAPTURE_TIMEZONE)
        created_paths = []
        try:
            if snapshot.dtype != np.uint8 or snapshot.ndim != 3 or snapshot.shape[2] != 3 or not snapshot.size:
                raise ValueError("저장할 프레임은 uint8 RGB 영상이어야 합니다.")
            # PNG encoding and disk I/O must not hold up the live preview.
            encoded_images = []
            for label, conversion in (("Color", cv2.COLOR_RGB2BGR), ("Mono", cv2.COLOR_RGB2GRAY)):
                ok, encoded = cv2.imencode(".png", cv2.cvtColor(snapshot, conversion))
                if not ok:
                    raise OSError(f"{label} PNG 인코딩에 실패했습니다.")
                encoded_images.append(encoded)
            folder = self.capture_dir / captured_at.strftime("%Y-%m-%d")
            folder.mkdir(parents=True, exist_ok=True)
            stem = stem or f"cam_{captured_at:%Y%m%d_%H%M%S_%f}_{uuid4().hex[:12]}"
            suffix = 0
            while True:
                pair_stem = f"{stem}{'_' + str(suffix) if suffix else ''}"
                paths = [folder / f"{pair_stem}_{label}.png" for label in ("Color", "Mono")]
                with ExitStack() as stack:
                    streams = []
                    try:
                        for path in paths:
                            streams.append(stack.enter_context(path.open("xb")))
                            created_paths.append(path)
                    except FileExistsError:
                        stack.close()
                        for path in created_paths:
                            path.unlink(missing_ok=True)
                        created_paths.clear()
                        suffix += 1
                        continue
                    for stream, encoded in zip(streams, encoded_images, strict=True):
                        stream.write(encoded.tobytes())
                        stream.flush()
                        os.fsync(stream.fileno())
                break
        except (OSError, ValueError, cv2.error) as error:
            for path in created_paths:
                path.unlink(missing_ok=True)
            return gr.update(value=None, visible=False), f"이미지 저장 실패: {error}"
        height, width = snapshot.shape[:2]
        return (gr.update(value=[str(path) for path in paths], visible=True),
                f"✓ Color · Mono 이미지 2장 저장 완료 · **{width} × {height}px · PNG**\n\n"
                + "\n\n".join(f"**{label}**: `{path}`" for label, path in zip(("Color", "Mono"), paths)))

    def extract_roi(self) -> tuple:
        with self._lock:
            camera = self.camera
            settings = self.settings
            frame = camera.get_latest_frame()
        if frame is None:
            return None, "ROI 추출 실패: 최근 수신된 카메라 프레임이 없습니다."
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # Expensive detection must not hold the live-preview/device lock.
            detected = detect_auto_roi(gray, self.roi_config, require_accepted=False)
            with self._lock:
                if (self.camera is not camera or self.settings != settings
                        or not self.camera.get_status().connected):
                    return None, "ROI 추출 중 카메라 연결 또는 촬영 설정이 변경됐습니다. 다시 추출하세요."
                self.roi_result = detected
            info = detected.metadata
            if not info["roi_detected"]:
                return None, f"ROI 검출 실패 · {info['quality_status']}\n\n{roi_quality_message(info)}"
            masked = np.where(detected.mask[:, :, None], frame, 0)
            box = detected.bounds
            message = (f"**ROI: {info['quality_status']}**\n\n{roi_quality_message(info)}\n\n"
                       f"영역 {box['width']} × {box['height']} @ ({box['x']}, {box['y']}) · "
                       f"유효 {int(detected.mask.sum()):,}픽셀 · 원본 좌표/해상도 유지")
            if not detected.accepted:
                message += "\n\n**검토 필요 — 위 보류 사유로 자동 검사에 사용하지 않습니다.**"
            else:
                message += "\n\n초록색 경계: 품질 기준 통과. 검사 프로그램은 측정 프레임에서 다시 추출합니다."
            return masked, message
        except (ValueError, OSError, RuntimeError) as error:
            return None, f"ROI 추출 실패: {error}"

    def apply_exposure(self, exposure: float) -> tuple:
        with self._lock:
            try:
                exposure = validate_exposure_us(exposure)
                applied = self.camera.set_exposure_us(exposure)
            except (CameraError, ValueError, TypeError) as error:
                return gr.skip(), self.camera_status(), f"**노출 적용 실패:** {error} · 저장값은 유지됩니다."
            self.settings = replace(self.settings, exposure_us=applied)
            self.roi_result = None
            state = "카메라 적용 완료" if self.camera.get_status().connected else "다음 카메라 연결 시 적용"
            try:
                save_camera_exposure(applied, self.camera_config_path)
            except (OSError, ValueError) as error:
                message = f"**{state}: {applied:g} µs · config 저장 실패:** {error} · 다시 적용 및 저장하세요."
            else:
                message = f"✓ {state}: **{applied:g} µs** ({applied / 1000:g} ms) · 저장 완료: `{self.camera_config_path}`"
                if applied != exposure:
                    message += f"\n\n요청값 {exposure:g} µs를 카메라가 지원하는 값으로 조정했습니다."
            return applied, self.camera_status(), message

    def load_exposure(self) -> tuple:
        with self._lock:
            try:
                settings = load_camera_settings(self.camera_config_path)
            except ValueError as error:
                return gr.skip(), f"**노출 불러오기 실패:** {error}"
            return settings.exposure_us, f"불러오기 완료: `{self.camera_config_path}` · **노출 적용 및 저장**을 눌러 카메라에 적용하세요."

    def apply_camera(self, serial_number: str, gain: float, fps: float) -> str:
        try:
            gain, fps = float(gain), float(fps)
            if not all(isfinite(value) for value in (gain, fps)) or gain < 0 or fps <= 0:
                raise ValueError("FPS는 양수, 게인은 0 이상이어야 합니다.")
            with self._lock:
                settings = replace(self.settings, gain=gain, frame_rate_fps=fps)
                self.camera.stop()
                self.roi_result = None
                self.settings = settings
                self.camera = CameraControl(serial_number.strip() or None, settings=settings)
                self._fps_sample = (monotonic(), 0)
                self._measured_fps = 0
                self._camera_enabled = True
                self._next_camera_retry = monotonic() + 3
                self.camera.start()
                return self.camera_status()
        except (ValueError, TypeError) as error:
            return f"**카메라 설정 오류:** {error}"

    def disconnect_camera(self) -> str:
        with self._lock:
            self._camera_enabled = False
            self.camera.stop()
            self.roi_result = None
            return self.camera_status()

    def connect_light(self, port: str, baudrate: int) -> str:
        try:
            port = port.strip()
            if not port or int(baudrate) <= 0:
                raise ValueError("포트와 전송 속도를 확인하세요.")
            with self._lock:
                if (port, int(baudrate)) != (self.light.port, self.light.baudrate):
                    self.light.disconnect()
                    self.light = LightController(port, int(baudrate))
                self.light.connect()
                self._light_requested = ()
                self._light_acknowledged = ()
                return f"{port} 연결 완료. 밝기 명령은 전송하지 않았습니다."
        except (LightCommunicationError, ValueError, TypeError) as error:
            return f"연결 실패: {error}"

    def disconnect_light(self) -> str:
        with self._lock:
            self.light.disconnect()
            self._light_requested = ()
            self._light_acknowledged = ()
            return "RS232 연결 해제. 밝기 변경 명령은 전송하지 않았습니다."

    def send_light(self, channels: tuple[tuple[int, float], ...], port: str, baudrate: int) -> str:
        with self._lock:
            self._light_requested = tuple(channel for channel, _ in channels)
            self._light_acknowledged = ()
            try:
                # Validate the entire request before any channel is transmitted.
                packets = tuple(build_brightness_packet(channel, float(value)) for channel, value in channels)
                if (port.strip(), int(baudrate)) != (self.light.port, self.light.baudrate):
                    raise ValueError("포트 설정이 바뀌었습니다. 먼저 RS232 연결을 누르세요.")
                sent = self.light.send(packets)
                self._light_acknowledged = sent
                return "\n".join(
                    f"ACK 확인 · CH{item.channel}: {item.percent:g}% ({item.value}/255) · "
                    f"TX {item.packet.hex(' ').upper()} · RX {ACK.hex(' ').upper()}"
                    for item in sent
                )
            except (LightCommunicationError, ValueError, TypeError) as error:
                self._light_acknowledged = getattr(error, "acknowledged", ())
                result = f"전송 실패: {error}"
                if self._light_acknowledged:
                    result += "\n이번 요청의 ACK 확인: "
                    result += ", ".join(f"CH{sent.channel}" for sent in self._light_acknowledged)
                    result += ". 이후 채널은 전송하지 않았습니다."
                return result

    def send_light_group(self, group: int, value: float, port: str, baudrate: int) -> str:
        if not 1 <= group <= len(LIGHT_GROUPS):
            return "전송 실패: 알 수 없는 조명입니다."
        return self.send_light(tuple((channel, value) for channel in LIGHT_GROUPS[group - 1].channels), port, baudrate)

    def send_light_groups(self, values: tuple[float, ...], port: str, baudrate: int) -> str:
        if len(values) != len(LIGHT_GROUPS):
            return "전송 실패: 네 조명의 밝기를 지정하세요."
        channels = tuple(
            (channel, value)
            for fixture, value in zip(LIGHT_GROUPS, values, strict=True)
            for channel in fixture.channels
        )
        return self.send_light(channels, port, baudrate)

    def light_receipts(self) -> tuple:
        with self._lock:
            notes = []
            for fixture in LIGHT_GROUPS:
                if not set(fixture.channels).intersection(self._light_requested):
                    notes.append(gr.skip())
                    continue
                accepted = [sent for sent in self._light_acknowledged if sent.channel in fixture.channels]
                if not accepted:
                    notes.append("이번 적용: **ACK 미확인**")
                    continue
                state = "최근 ACK 확인" if len(accepted) == len(fixture.channels) else "부분 적용 · ACK 확인"
                values = " · ".join(f"CH{sent.channel} {sent.percent:g}% ({sent.value}/255)" for sent in accepted)
                notes.append(f"**{state}**\n\n{values}")
            return tuple(notes)

    def turn_off_lights(self, port: str, baudrate: int) -> tuple:
        with self._lock:
            result = self.send_light_groups((0,) * len(LIGHT_GROUPS), port, baudrate)
            accepted = {sent.channel for sent in self._light_acknowledged}
            values = tuple(0 if set(fixture.channels).issubset(accepted) else gr.skip() for fixture in LIGHT_GROUPS)
            return result, *self.light_receipts(), *values


def build_ui(controller: CamTestController) -> gr.Blocks:
    ports = sorted({COM_PORT, controller.light.port, *(port.device for port in list_ports.comports())})
    initial_lighting_values, initial_lighting_message = controller.lighting_config.initial_for_ui()
    with gr.Blocks(title="Cam_test", analytics_enabled=False) as demo:
        gr.Markdown("# Cam_test\n실시간 Basler 카메라 · 4개 조명 / RS232 6채널 연동 제어")
        with gr.Row():
            with gr.Column(scale=3):
                preview = gr.Image(
                    label="실시간 카메라", type="numpy", format="jpeg", interactive=False,
                    height=570, buttons=[],
                )
                with gr.Row():
                    capture_name = gr.Textbox(value="", label="이미지 이름", placeholder="예: Case01_시편01",
                                              info="_Color.png / _Mono.png 자동 추가 · 같은 이름은 번호 추가 · 비우면 촬영 시각 사용",
                                              scale=3)
                    capture_button = gr.Button("Color · Mono 두 장 저장", variant="primary", scale=1)
                capture_status = gr.Markdown("버튼을 누르면 같은 카메라 프레임을 원본 해상도의 Color PNG와 흑백 Mono PNG 두 장으로 저장합니다.",
                                              elem_classes="cam-capture-status")
                capture_file = gr.File(label="Color · Mono 이미지 다운로드", interactive=False,
                                       file_count="multiple", file_types=[".png"], visible=False)
                exposure = gr.Number(value=controller.settings.exposure_us, minimum=1, step=100, label="카메라 노출 시간 (µs)")
                with gr.Row():
                    exposure_apply = gr.Button("노출 적용 및 저장", variant="primary")
                    exposure_load = gr.Button("저장된 노출 불러오기")
                gr.Markdown("10000 µs = 10 ms. 실제 촬영 노출을 변경합니다. 연결 중에는 재연결 없이 적용하고, 연결 해제 상태에서는 다음 연결에 사용합니다.")
                exposure_config_status = gr.Markdown(f"시작 노출: **{controller.settings.exposure_us:g} µs** · 공통 config: `{controller.camera_config_path}`")
                frame_info = gr.Markdown("첫 프레임을 기다리는 중입니다.")
                with gr.Accordion("자동 ROI 추출 · Gabor / DP / RANSAC", open=True):
                    roi_threshold_controls(demo, lambda: controller.roi_threshold_config)
                    show_roi = gr.Checkbox(value=True, label="실시간 화면에 최근 ROI 경계 표시")
                    roi_button = gr.Button("현재 프레임에서 ROI 추출", variant="primary")
                    roi_status = gr.Markdown("아직 ROI를 추출하지 않았습니다. 추출 중에도 실시간 영상은 갱신됩니다.")
                    roi_preview = gr.Image(label="마지막 ROI 추출 프레임 (카메라 원본)", type="numpy", format="png", interactive=False, height=260)
            with gr.Column(scale=2):
                camera_status = gr.Markdown(controller.camera_status())
                with gr.Accordion("카메라 연결 / 게인·FPS 설정", open=False):
                    serial_number = gr.Textbox(value=controller.camera.serial_number or "", label="카메라 일련번호 (비우면 자동 선택)")
                    with gr.Row():
                        gain = gr.Number(value=controller.settings.gain, minimum=0, label="게인")
                        fps = gr.Number(value=controller.settings.frame_rate_fps, minimum=1, label="촬영 FPS")
                    gr.Markdown("설정값은 카메라가 지원하는 범위로 제한됩니다. 적용 시 카메라를 다시 연결합니다.")
                    with gr.Row():
                        camera_connect = gr.Button("카메라 연결 / 설정 적용", variant="primary")
                        camera_disconnect = gr.Button("카메라 연결 해제")
                gr.Markdown("### 조명 제어 · 4개 조명 / 6채널")
                light_status = gr.Markdown(controller.light_status())
                with gr.Row():
                    port = gr.Dropdown(choices=ports, value=controller.light.port, allow_custom_value=True, label="RS232 포트")
                    baudrate = gr.Dropdown(choices=sorted({9600, 19200, 38400, 57600, 115200, controller.light.baudrate}), value=controller.light.baudrate, label="전송 속도 (bps)")
                with gr.Row():
                    light_connect = gr.Button("RS232 연결")
                    light_disconnect = gr.Button("RS232 연결 해제")
                gr.Markdown("슬라이더는 전송할 설정값입니다. **이 조명 적용 / 4개 조명 적용**을 눌러 전송합니다. 면조명은 두 채널을 같은 밝기로 제어합니다.")
                with gr.Row():
                    lighting_save = gr.Button("조명값 저장", size="sm")
                    lighting_load = gr.Button("저장값 불러오기", size="sm")
                lighting_config_status = gr.Markdown(initial_lighting_message, elem_classes="cam-light-config-status")
                levels, group_buttons, group_receipts = [], [], []
                with gr.Row(elem_classes="cam-light-grid"):
                    for group, fixture in enumerate(LIGHT_GROUPS, 1):
                        with gr.Column(min_width=0, elem_classes="cam-light-card"):
                            mode = "두 채널 같은 밝기로 제어" if len(fixture.channels) > 1 else "단일 채널 제어"
                            gr.HTML(
                                f'<div class="cam-light-header"><span>LIGHT {group:02d}</span>'
                                f'<strong>{fixture.name}</strong><code>{fixture.channel_label}</code><small>{mode}</small></div>'
                            )
                            level = brightness_slider(f"{fixture.name} 밝기 (%)", initial_lighting_values[group - 1])
                            apply = gr.Button("이 조명 적용", size="sm")
                            note = gr.Markdown("최근 ACK: **미확인**", elem_classes="cam-light-receipt")
                        levels.append(level)
                        group_buttons.append(apply)
                        group_receipts.append(note)
                with gr.Row():
                    apply_all = gr.Button("4개 조명 적용", variant="primary")
                    all_off = gr.Button("전체 끄기 (0%)", variant="stop")
                gr.Markdown("최근 ACK는 명령 수신 기록이며 장치에서 읽어온 실제 광량이 아닙니다.")
                receipt = gr.Textbox(label="RS232 전송 결과 / ACK", value="아직 밝기 명령을 전송하지 않았습니다.", lines=4, interactive=False)

        def apply_group(group: int, value: float, selected_port: str, baud: int) -> tuple:
            result = controller.send_light_group(group, value, selected_port, baud)
            return result, *controller.light_receipts()

        def apply_groups(a: float, b: float, c: float, d: float, selected_port: str, baud: int) -> tuple:
            result = controller.send_light_groups((a, b, c, d), selected_port, baud)
            return result, *controller.light_receipts()

        def connect_lights(selected_port: str, baud: int) -> tuple:
            result = controller.connect_light(selected_port, baud)
            notes = ["최근 ACK: **미확인**"] * len(LIGHT_GROUPS) if result.startswith(f"{selected_port.strip()} 연결 완료") else [gr.skip()] * len(LIGHT_GROUPS)
            return result, *notes

        def disconnect_lights() -> tuple:
            return controller.disconnect_light(), *(["최근 ACK: **미확인**"] * len(LIGHT_GROUPS))

        def save_lighting_config(a: float, b: float, c: float, d: float) -> str:
            return controller.lighting_config.save_for_ui((a, b, c, d))

        def load_lighting_config() -> tuple:
            values, message = controller.lighting_config.load_for_ui()
            if values is None:
                return *(gr.skip() for _ in LIGHT_GROUPS), message
            return *values, message

        config_outputs = [*levels, lighting_config_status]
        lighting_save.click(save_lighting_config, inputs=levels, outputs=lighting_config_status,
                            api_name="lighting_config_save", concurrency_id="light", concurrency_limit=1)
        lighting_load.click(load_lighting_config, outputs=config_outputs,
                            api_name="lighting_config_load", concurrency_id="light", concurrency_limit=1)
        demo.load(load_lighting_config, outputs=config_outputs, api_visibility="private", show_progress="hidden",
                  concurrency_id="light", concurrency_limit=1)

        light_outputs = [receipt, *group_receipts]
        for group, (apply, level) in enumerate(zip(group_buttons, levels, strict=True), 1):
            apply.click(
                lambda value, port, baud, selected=group: apply_group(selected, value, port, baud),
                inputs=[level, port, baudrate], outputs=light_outputs,
                api_name=f"light_group_{group}", concurrency_id="light", concurrency_limit=1,
            )
        timer = gr.Timer(value=0.2)
        timer.tick(controller.preview, inputs=show_roi, outputs=[preview, camera_status, light_status, frame_info], queue=False, show_progress="hidden", api_name="preview")
        demo.load(controller.preview, inputs=show_roi, outputs=[preview, camera_status, light_status, frame_info], queue=False, show_progress="hidden", api_visibility="private")
        roi_button.click(controller.extract_roi, outputs=[roi_preview, roi_status], api_name="extract_roi", concurrency_id="roi", concurrency_limit=1)
        capture_button.click(controller.save_image, inputs=capture_name, outputs=[capture_file, capture_status],
                             api_name="save_image", concurrency_id="capture", concurrency_limit=1)
        exposure_apply.click(controller.apply_exposure, inputs=exposure, outputs=[exposure, camera_status, exposure_config_status],
                             api_name="exposure_apply", concurrency_id="camera", concurrency_limit=1)
        exposure_load.click(controller.load_exposure, outputs=[exposure, exposure_config_status],
                            api_name="exposure_load", concurrency_id="camera", concurrency_limit=1)
        demo.load(controller.load_exposure, outputs=[exposure, exposure_config_status], api_visibility="private", show_progress="hidden",
                  concurrency_id="camera", concurrency_limit=1)
        camera_connect.click(controller.apply_camera, inputs=[serial_number, gain, fps], outputs=camera_status, concurrency_id="camera", concurrency_limit=1)
        camera_disconnect.click(controller.disconnect_camera, outputs=camera_status, concurrency_id="camera", concurrency_limit=1)
        light_connect.click(connect_lights, inputs=[port, baudrate], outputs=light_outputs, api_name="connect_light", concurrency_id="light", concurrency_limit=1)
        light_disconnect.click(disconnect_lights, outputs=light_outputs, api_name="disconnect_light", concurrency_id="light", concurrency_limit=1)
        apply_all.click(apply_groups, inputs=[*levels, port, baudrate], outputs=light_outputs, api_name="light_all", concurrency_id="light", concurrency_limit=1)
        all_off.click(controller.turn_off_lights, inputs=[port, baudrate], outputs=[*light_outputs, *levels], api_name="light_off", concurrency_id="light", concurrency_limit=1)
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Cam_test — 실시간 카메라 및 RS232 라이트 테스트")
    parser.add_argument("--camera-serial", default="24346077")
    parser.add_argument("--light-port", default=COM_PORT)
    parser.add_argument("--baudrate", type=int, default=BAUD_RATE)
    parser.add_argument("--roi-config", type=Path, help="자동 ROI 경계 설정 YAML")
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG, help="카메라 시작 설정 YAML")
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR, help="카메라 PNG 저장 폴더")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or args.baudrate <= 0:
        parser.error("포트는 1~65535, 전송 속도는 양수여야 합니다.")
    url = f"http://{args.host}:{args.port}"
    try:
        with socket.socket() as probe:
            probe.bind((args.host, args.port))
    except OSError:
        try:
            with urllib.request.urlopen(url + "/config", timeout=2) as response:
                existing = json.load(response)
        except (OSError, ValueError):
            existing = {}
        if existing.get("title") == "Cam_test":
            print(f"Cam_test가 이미 실행 중입니다: {url}", flush=True)
            if not args.no_browser:
                webbrowser.open(url)
            return
        parser.error(f"{args.port} 포트를 다른 서버가 사용 중입니다. 기존 서버를 종료하세요.")
    try:
        controller = CamTestController(args.camera_serial or None, args.light_port, args.baudrate,
                                       roi_config=args.roi_config, camera_config=args.camera_config,
                                       capture_dir=args.capture_dir)
    except ValueError as error:
        parser.error(str(error))
    try:
        controller.start()
        demo = build_ui(controller)
        demo.launch(server_name=args.host, server_port=args.port, inbrowser=not args.no_browser,
                    theme=gr.themes.Soft(), css_paths=[CSS_PATH], footer_links=[])
    finally:
        controller.close()


if __name__ == "__main__":
    main()
