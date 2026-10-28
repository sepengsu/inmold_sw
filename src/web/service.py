"""UI-independent workbench operations shared by FastAPI endpoints."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sqlite3
from threading import RLock
from time import monotonic

import cv2
import numpy as np

from src.hardware.camera_config import DEFAULT_CAMERA_CONFIG, load_camera_settings, save_camera_pixel_format
from src.hardware.camera_control import CameraControl
from src.hardware.lighting_config import LightingConfigStore
from src.hardware.led_control import (
    BAUD_RATE, COM_PORT, LIGHT_GROUPS, LightCommunicationError, LightController,
    build_brightness_packet,
)
from src.inspection_history import InspectionHistoryStore, InspectionRecord, LOCAL_TIMEZONE
from src.preprocess import load_config
from src.roi.quality_display import roi_score_message
from src.roi.quality_profiles import linked_roi_settings, roi_level, roi_settings_summary
from src.roi.threshold_config import RoiThresholdConfigStore


def image_bytes(rgb: np.ndarray, extension: str = ".png") -> bytes:
    ok, data = cv2.imencode(extension, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError("영상 인코딩에 실패했습니다.")
    return data.tobytes()


def record_data(record: InspectionRecord) -> dict:
    return {
        "id": record.record_id, "measured_at": record.measured_at,
        "source": record.source, "p1": record.p1, "p2": record.p2,
        "p3": record.p3, "average": record.average,
        "elapsed_seconds": record.elapsed_seconds, "metadata": record.metadata,
    }


class Workbench:
    def __init__(self, *, config_path: Path, checkpoint_path: Path, device: str,
                 batch_size: int, camera_serial: str | None, use_camera: bool,
                 camera_config_path: Path, history_root: Path | None = None,
                 load_model: bool = True, auto_connect_light: bool | None = None):
        self.lock = RLock()
        self.camera_enabled = use_camera
        self.auto_connect_light = use_camera if auto_connect_light is None else auto_connect_light
        self.camera_config_path = camera_config_path
        self.camera = CameraControl(camera_serial, settings=load_camera_settings(camera_config_path))
        self.light = LightController()
        self.history = InspectionHistoryStore(history_root) if history_root else InspectionHistoryStore()
        self.lighting_config = LightingConfigStore()
        self.roi_config = RoiThresholdConfigStore()
        self.engine = None
        self.model_error = "진단 모드 · 모델 미사용" if not load_model else None
        if load_model:
            try:
                from src.inspection.engine import InspectionEngine

                self.roi_config = RoiThresholdConfigStore(
                    load_config(config_path)["auto_roi"]["boundary_config"])
                self.engine = InspectionEngine(config_path, checkpoint_path, device=device, batch_size=batch_size)
            except (ImportError, ValueError, RuntimeError, OSError, KeyError) as error:
                self.model_error = str(error)

    def start(self) -> None:
        if self.auto_connect_light:
            try:
                self.light.connect()
            except LightCommunicationError:
                pass
        if self.camera_enabled:
            self.camera.start()

    def stop(self) -> None:
        self.camera.stop()
        self.light.disconnect()

    def status(self) -> dict:
        camera = self.camera.get_status()
        light = self.light.get_status()
        camera_state = "파일 검사 모드" if not self.camera_enabled else (
            "READY" if camera.connected else camera.error or "연결 대기")
        if (self.camera_enabled and camera.connected and
                (camera.last_frame_at is None or monotonic() - camera.last_frame_at > 2)):
            camera_state = "프레임 수신 대기"
        light_state = ("파일 검사 모드 · 연결하지 않음" if not self.camera_enabled else
                       light.error or ("RS232 OPEN · ACK 확인" if light.ack_verified else
                                       "RS232 OPEN · 제어 대기" if light.connected else "연결 대기"))
        roi_state = ""
        threshold = None
        classes = []
        if self.engine:
            config = self.engine.config
            if config.get("auto_roi", {}).get("enabled"):
                last = self.engine.last_roi_detection
                roi_state = "자동 Gabor 경계 · " + (str(last["quality_status"]) if last else "측정 시 추출")
            else:
                roi = config["roi"]
                roi_state = f'{roi["width"]} × {roi["height"]} @ ({roi["x"]}, {roi["y"]})'
            threshold = self.engine.checkpoint["threshold"]
            classes = list(self.engine.checkpoint.get("positive_classes", ("Exposure",)))
        return {"camera": camera_state, "camera_connected": camera.connected,
                "camera_enabled": self.camera_enabled, "pixel_format": self.camera.settings.pixel_format,
                "exposure_us": self.camera.settings.exposure_us,
                "model": "LOADED" if self.engine else self.model_error or "로딩 대기",
                "light": light_state, "light_port": COM_PORT, "light_baudrate": BAUD_RATE,
                "roi": roi_state, "threshold": threshold, "positive_classes": classes}

    def bootstrap(self) -> dict:
        levels, lighting_message = self.lighting_config.initial_for_ui()
        threshold, roi_message, level, details = self.roi_config.settings_for_ui()
        return {"status": self.status(), "source": "camera" if self.camera_enabled else "file",
                "lighting": {"values": levels, "message": lighting_message,
                             "groups": [{"name": g.name, "channels": g.channel_label} for g in LIGHT_GROUPS]},
                "roi": {"threshold": threshold, "level": level, "details": details, "message": roi_message},
                "history": [record_data(r) for r in self.history.query(limit=5)]}

    def preview(self) -> bytes | None:
        frame = self.camera.get_latest_frame()
        return None if frame is None else image_bytes(frame, ".jpg")

    def reconnect(self) -> dict:
        with self.lock:
            self.camera_enabled = True
            self.camera.start()
            return self.status()

    def set_pixel_format(self, value: str) -> dict:
        if value not in ("Mono8", "BayerGR12"):
            raise ValueError("촬영 형식은 Mono8 또는 BayerGR12여야 합니다.")
        with self.lock:
            save_camera_pixel_format(value, self.camera_config_path)
            self.camera.set_pixel_format(value)
            if self.camera_enabled and not self.camera.is_running:
                self.camera.start()
            connection = "카메라 재연결 중" if self.camera_enabled else "다음 카메라 연결 시 적용"
            return {"status": self.status(), "message": f"✓ {value} 저장 완료 · {connection}"}

    def roi_settings(self) -> dict:
        value, message, level, details = self.roi_config.settings_for_ui()
        return {"threshold": value, "message": message, "level": level, "details": details}

    def save_roi(self, value: float) -> dict:
        with self.lock:
            threshold, message = self.roi_config.save_for_ui(value)
            if threshold is None:
                raise ValueError(message)
            config = linked_roi_settings(threshold)
            return {"threshold": threshold, "message": message,
                    "level": roi_level(config), "details": roi_settings_summary(config)}

    def load_lighting(self) -> dict:
        values, message = self.lighting_config.load_for_ui()
        return {"values": values, "message": message}

    def save_lighting(self, values: list[float]) -> dict:
        return {"message": self.lighting_config.save_for_ui(values)}

    @staticmethod
    def _receipt(group: int, sent) -> str:
        fixture = LIGHT_GROUPS[group - 1]
        accepted = [item for item in sent if item.channel in fixture.channels]
        if not accepted:
            return "최근 ACK: 미확인"
        prefix = "최근 ACK 확인" if len(accepted) == len(fixture.channels) else "부분 적용 · ACK 확인"
        return prefix + " · " + " · ".join(
            f"CH{item.channel} {item.percent:g}% ({item.value}/255)" for item in accepted)

    def apply_lighting(self, values: list[float], group: int | None = None) -> dict:
        with self.lock:
            try:
                selected = tuple(enumerate(values, 1)) if group is None else ((group, values[group - 1]),)
                sent = self.light.send(tuple(build_brightness_packet(channel, percent)
                                            for fixture, percent in selected
                                            for channel in LIGHT_GROUPS[fixture - 1].channels))
                message = (f"✓ 4개 조명 · CH1~CH6 ACK 확인 ({COM_PORT}, {BAUD_RATE} bps)" if group is None else
                           f"✓ {LIGHT_GROUPS[group - 1].name} 적용 완료 · ACK 확인 ({COM_PORT}, {BAUD_RATE} bps)")
                receipts = {str(i): self._receipt(i, sent) for i in range(1, len(LIGHT_GROUPS) + 1)}
                return {"message": message, "receipts": receipts, "success": True}
            except (LightCommunicationError, ValueError) as error:
                acknowledged = getattr(error, "acknowledged", ())
                detail = (" · 이번 요청의 ACK 확인: "
                          + ", ".join(f"CH{sent.channel}" for sent in acknowledged)
                          + ". 이후 채널은 전송하지 않았습니다."
                          if acknowledged else "")
                return {"message": f"⚠ 전송 실패: {error}{detail}", "success": False,
                        "receipts": {str(i): self._receipt(i, acknowledged)
                                     for i in range(1, len(LIGHT_GROUPS) + 1)}}

    def inspect(self, source: str, mode: str, uploaded: np.ndarray | None,
                lighting_values: list[float]) -> dict:
        with self.lock:
            started = monotonic()
            if mode not in ("normal", "certification"):
                raise ValueError("검사 모드를 선택하세요.")
            if source not in ("camera", "file"):
                raise ValueError("검사 입력을 선택하세요.")
            frame = self.camera.get_latest_frame() if source == "camera" else uploaded
            if frame is None:
                raise ValueError("검사할 이미지 파일을 선택하세요." if source == "file" else
                                 self.camera.get_status().error or "최근 2초 이내 수신된 카메라 프레임이 없습니다.")
            if self.engine is None:
                raise ValueError(self.model_error or "모델이 로드되지 않았습니다.")
            inference = (self.engine.inspect_certification(frame) if mode == "certification"
                         else self.engine.inspect(frame))
            timestamp = datetime.now(LOCAL_TIMEZONE).isoformat(timespec="milliseconds")
            elapsed = monotonic() - started
            metadata = {
                "roi": getattr(inference, "roi", None),
                "regions": list(getattr(inference, "regions", ())),
                "roi_detection": getattr(inference, "roi_detection", None),
                "model": getattr(self.engine, "model_info", self.engine.checkpoint),
                "conditions": {"source": source, "inspection_mode": mode},
                "frame_shape": list(frame.shape),
            }
            valid = getattr(inference, "valid_pixels", None)
            if mode == "certification" and valid is not None:
                metadata["conditions"].update(valid_pixels=list(map(int, valid)),
                                               denominator="nonblack_rectified_crop_pixels")
            if source == "camera":
                metadata["conditions"].update(
                    exposure_us=self.camera.settings.exposure_us,
                    pixel_format=self.camera.settings.pixel_format,
                    camera_serial=self.camera.serial_number,
                    lighting_ui_percent=dict(zip((g.config_key for g in LIGHT_GROUPS), lighting_values)),
                )
            record = InspectionRecord(*map(float, inference.scores), measured_at=timestamp,
                                      source=source, elapsed_seconds=elapsed, metadata=metadata)
            saved_message = "이력 자동 저장 완료"
            try:
                record = self.history.save(record, frame, inference.annotated_frame)
            except (OSError, ValueError, sqlite3.Error) as error:
                saved_message = f"이력 저장 실패: {error}"
            result = {"record": record_data(record), "mode": mode, "saved_message": saved_message,
                      "valid_pixels": valid, "positive_classes": self.engine.checkpoint.get("positive_classes", ()),
                      "rotation_degrees": [r.get("rotation_deg") for r in getattr(inference, "regions", ())],
                      "roi_message": (roi_score_message(inference.roi_detection)
                                      if getattr(inference, "roi_detection", None) else None),
                      "history": [],
                      "images": {"result": image_bytes(inference.annotated_frame),
                                 **{f"p{i}": image_bytes(crop) for i, crop in enumerate(inference.crops, 1)}}}
            try:
                result["history"] = [record_data(r) for r in self.history.query(limit=5)]
            except (OSError, ValueError, sqlite3.Error):
                result["history"] = [record_data(record)]
            return result

    def statistics(self, start: str | None, end: str | None, source: str, page: int) -> dict:
        from math import ceil

        summary = self.history.summary(start, end, source)
        pages = max(1, ceil(summary["count"] / 50))
        page = min(pages, max(1, page))
        rows = self.history.query(start, end, source, limit=50, offset=(page - 1) * 50)
        return {"summary": summary, "rows": [record_data(r) for r in rows],
                "page": page, "pages": pages}

    def detail(self, record_id: str) -> dict:
        record = self.history.get(record_id)
        result = self.history.read_image(record.result_path)
        crops = {}
        certification = record.metadata.get("conditions", {}).get("inspection_mode") == "certification"
        for i, region in enumerate(record.metadata.get("regions", [])[:3], 1):
            if certification and "quad" in region:
                from src.PCE.inspection import rectify_region
                crop = rectify_region(result, region)
            else:
                x, y, width, height = (region[key] for key in ("x", "y", "width", "height"))
                crop = result[y:y + height, x:x + width].copy()
            if certification:
                crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LINEAR)
            crops[f"p{i}"] = image_bytes(crop)
        return {"record": record_data(record), "images": {
            "original": image_bytes(self.history.read_image(record.original_path)),
            "result": image_bytes(result), **crops}}
