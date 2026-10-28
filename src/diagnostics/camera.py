"""Live Basler preview and explicit RS232 light controls, without model loading."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from math import isfinite
import os
from pathlib import Path
from threading import RLock
from time import monotonic
from uuid import uuid4

import cv2
import numpy as np

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
from src.roi.threshold_config import RoiThresholdConfigStore
from src.roi.quality_display import roi_quality_message


DEFAULT_CAPTURE_DIR = Path(__file__).resolve().parents[2] / "data" / "cam_test"
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
        camera: CameraControl | None = None, light: LightController | None = None,
        camera_enabled: bool = False, on_camera_replace=None, on_light_replace=None,
    ) -> None:
        self.camera_config_path = Path(camera_config).expanduser().resolve()
        self.capture_dir = Path(capture_dir).expanduser().resolve()
        self.settings = settings if settings is not None else load_camera_settings(self.camera_config_path)
        self.camera = camera or CameraControl(serial_number, settings=self.settings)
        self.light = light or LightController(port, baudrate)
        self.on_camera_replace = on_camera_replace
        self.on_light_replace = on_light_replace
        self._lock = RLock()
        self._fps_sample = (monotonic(), 0)
        self._measured_fps = 0.0
        self._camera_enabled = camera_enabled
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
            return None, f"이미지 저장 실패: {error}"
        with self._lock:
            status = self.camera.get_status()
            frame = self.camera.get_latest_frame()
            if (not status.connected or frame is None or status.last_frame_at is None
                    or monotonic() - status.last_frame_at > 2):
                return None, "이미지 저장 실패: 최근 수신된 카메라 프레임이 없습니다."
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
            return None, f"이미지 저장 실패: {error}"
        height, width = snapshot.shape[:2]
        return ([str(path) for path in paths],
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
                return None, self.camera_status(), f"**노출 적용 실패:** {error} · 저장값은 유지됩니다."
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
                return None, f"**노출 불러오기 실패:** {error}"
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
                if self.on_camera_replace is not None:
                    self.on_camera_replace(self.camera)
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
                    if self.on_light_replace is not None:
                        self.on_light_replace(self.light)
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
                    notes.append(None)
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
            values = tuple(0 if set(fixture.channels).issubset(accepted) else None for fixture in LIGHT_GROUPS)
            return result, *self.light_receipts(), *values

