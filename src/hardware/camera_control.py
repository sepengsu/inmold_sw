"""Basler pylon camera-control module for the Vision Workbench.

This module is adapted to the Basler-camera control project supplied with this
workspace. It owns one pypylon camera, continuously acquires only the newest
frame, and exposes safe RGB copies for Gradio and inference code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from threading import Event, Lock, RLock, Thread
from time import monotonic, time_ns
from typing import Any

import numpy as np


class CameraError(RuntimeError):
    """Raised for a camera connection, configuration, or acquisition failure."""


CAMERA_PIXEL_FORMATS = {"grayscale": "Mono8", "color": "BayerGR12"}


@dataclass(frozen=True, slots=True)
class BaslerCameraSettings:
    """Acquisition settings applied before streaming starts."""
    pixel_format: str = "BayerGR12"
    use_max_resolution: bool = True
    exposure_us: float = 10_000.0
    gain: float = 0.0
    limit_frame_rate: bool = True
    frame_rate_fps: float = 10.0


@dataclass(frozen=True, slots=True)
class CameraFrame:
    """A frame and its Basler transport metadata."""

    display_rgb: np.ndarray
    raw_array: np.ndarray
    width: int
    height: int
    pixel_format: str
    block_id: int | None
    camera_timestamp: int | None
    captured_at_ns: int


@dataclass(frozen=True, slots=True)
class CameraStatus:
    running: bool
    connected: bool
    description: str
    serial_number: str | None
    frames_received: int
    last_frame_at: float | None
    error: str | None


def _load_pylon() -> tuple[Any, Any]:
    """Import pypylon lazily so importing the app works before driver setup."""
    try:
        from pypylon import genicam, pylon
    except ImportError as error:
        raise CameraError(
            "pypylon을 불러오지 못했습니다. Basler pylon Camera Software Suite와 "
            "Python 패키지 pypylon을 설치하세요."
        ) from error
    return pylon, genicam


def list_devices() -> list[dict[str, str]]:
    """Return Basler devices visible to the pylon transport layer."""
    pylon, _ = _load_pylon()
    try:
        return [
            {
                "model_name": info.GetModelName(),
                "serial_number": info.GetSerialNumber(),
                "user_name": info.GetUserDefinedName(),
                "device_class": info.GetDeviceClass(),
            }
            for info in pylon.TlFactory.GetInstance().EnumerateDevices()
        ]
    except Exception as error:
        raise CameraError(f"Basler 카메라 목록을 가져오지 못했습니다: {error}") from error


class CameraControl:
    """Thread-safe live Basler camera acquisition service.

    ``serial_number=None`` selects pylon's first available device. Calling
    :meth:`get_latest_frame` returns an RGB NumPy copy that is safe to submit to
    a model or a ``gr.Image(type='numpy')`` component.
    """

    def __init__(
        self,
        serial_number: str | None = None,
        *,
        settings: BaslerCameraSettings | None = None,
        retrieve_timeout_ms: int = 200,
    ) -> None:
        if retrieve_timeout_ms <= 0:
            raise ValueError("retrieve_timeout_ms must be greater than zero")

        self.serial_number = serial_number
        self.settings = settings or BaslerCameraSettings()
        self.retrieve_timeout_ms = retrieve_timeout_ms
        self._camera: Any | None = None
        self._converter: Any | None = None
        self._frame: CameraFrame | None = None
        self._lock = Lock()
        self._device_lock = RLock()
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._connected = False
        self._description = "Basler camera"
        self._error: str | None = None
        self._frames_received = 0
        self._last_frame_at: float | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Open, configure, and start the camera acquisition thread."""
        if self.is_running:
            return
        with self._lock:
            self._frame = None
            self._last_frame_at = None
            self._error = None
        self._stop_event.clear()
        self._thread = Thread(
            target=self._capture_loop,
            name="basler-camera-capture",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout_seconds: float = 3.0) -> None:
        """Stop acquisition and release the Basler device."""
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout_seconds)
        self._close_camera()

    def get_latest_frame(self, max_age_seconds: float = 2.0) -> np.ndarray | None:
        """Return a fresh RGB copy; disconnected or stalled streams have no frame."""
        with self._lock:
            if (not self._connected or self._frame is None or self._last_frame_at is None
                    or monotonic() - self._last_frame_at > max_age_seconds):
                return None
            return self._frame.display_rgb.copy()

    def get_latest_camera_frame(self) -> CameraFrame | None:
        """Return an immutable metadata record with copied image arrays."""
        with self._lock:
            if self._frame is None:
                return None
            frame = self._frame
            return CameraFrame(
                display_rgb=frame.display_rgb.copy(),
                raw_array=frame.raw_array.copy(),
                width=frame.width,
                height=frame.height,
                pixel_format=frame.pixel_format,
                block_id=frame.block_id,
                camera_timestamp=frame.camera_timestamp,
                captured_at_ns=frame.captured_at_ns,
            )

    def get_status(self) -> CameraStatus:
        with self._lock:
            return CameraStatus(
                running=self.is_running,
                connected=self._connected,
                description=self._description,
                serial_number=self.serial_number,
                frames_received=self._frames_received,
                last_frame_at=self._last_frame_at,
                error=self._error,
            )

    def set_exposure_us(self, exposure_us: float) -> float:
        """Set sensor exposure during acquisition, or retain it for the next start."""
        if isinstance(exposure_us, bool) or not isinstance(exposure_us, (int, float)):
            raise ValueError("노출 시간(µs)은 유한한 양수여야 합니다.")
        try:
            exposure = float(exposure_us)
        except (ValueError, OverflowError) as error:
            raise ValueError("노출 시간(µs)은 유한한 양수여야 합니다.") from error
        if not isfinite(exposure) or exposure <= 0:
            raise ValueError("노출 시간(µs)은 유한한 양수여야 합니다.")
        with self._device_lock:
            with self._lock:
                connected = self._connected
                if self.is_running and not connected:
                    raise CameraError("카메라 연결이 완료된 뒤 노출을 적용하세요.")
            applied = exposure
            if connected:
                try:
                    camera = self._require_camera()
                    self._set_enum(camera, "ExposureAuto", "Off", required=False)
                    applied = self._set_number(camera, ("ExposureTime", "ExposureTimeAbs"), applied)
                except Exception as error:
                    raise CameraError(f"노출 시간을 적용하지 못했습니다: {error}") from error
            self.settings = replace(self.settings, exposure_us=applied)
            with self._lock:
                # Do not present a cached frame from the previous exposure setting.
                self._frame = None
                self._last_frame_at = None
            return applied

    def set_pixel_format(self, pixel_format: str) -> None:
        """Switch acquisition format while the camera is idle, then reconnect."""
        if pixel_format not in CAMERA_PIXEL_FORMATS.values():
            raise ValueError("촬영 형식은 Mono8 또는 BayerGR12여야 합니다.")
        if pixel_format == self.settings.pixel_format:
            return
        was_running = self.is_running
        self.stop()
        self.settings = replace(self.settings, pixel_format=pixel_format)
        if was_running:
            self.start()

    def _capture_loop(self) -> None:
        try:
            with self._device_lock:
                self._open_and_configure()
                self._start_grabbing()
            while not self._stop_event.is_set():
                with self._device_lock:
                    frame = self._retrieve_frame()
                    if frame is None:
                        continue
                    with self._lock:
                        self._frame = frame
                        self._frames_received += 1
                        self._last_frame_at = monotonic()
                        self._connected = True
                        self._error = None
        except CameraError as error:
            self._set_error(str(error))
        except Exception as error:
            self._set_error(f"카메라 수신 중 예상하지 못한 오류: {error}")
        finally:
            self._close_camera()

    def _open_and_configure(self) -> None:
        pylon, _ = _load_pylon()
        factory = pylon.TlFactory.GetInstance()
        try:
            if self.serial_number:
                info = next(
                    (
                        candidate
                        for candidate in factory.EnumerateDevices()
                        if candidate.GetSerialNumber() == self.serial_number
                    ),
                    None,
                )
                if info is None:
                    raise CameraError(
                        f"일련번호 {self.serial_number} Basler 카메라를 찾지 못했습니다."
                    )
                device = factory.CreateDevice(info)
            else:
                device = factory.CreateFirstDevice()

            self._camera = pylon.InstantCamera(device)
            self._camera.Open()
            info = self._camera.GetDeviceInfo()
            with self._lock:
                self._description = f"{info.GetModelName()} / S/N {info.GetSerialNumber()}"
                self.serial_number = info.GetSerialNumber()

            self._configure_transport(info.GetDeviceClass())
            self._apply_settings()
            converter = pylon.ImageFormatConverter()
            converter.OutputPixelFormat = pylon.PixelType_RGB8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            self._converter = converter
        except CameraError:
            raise
        except Exception as error:
            raise CameraError(f"Basler 카메라를 열거나 설정하지 못했습니다: {error}") from error

    def _apply_settings(self) -> None:
        camera = self._require_camera()
        settings = self.settings
        self._set_enum(camera, "PixelFormat", settings.pixel_format)
        if settings.use_max_resolution:
            self._set_node_to_max(camera, "Width")
            self._set_node_to_max(camera, "Height")
        self._set_enum(camera, "ExposureAuto", "Off", required=False)
        self._set_number(camera, ("ExposureTime", "ExposureTimeAbs"), settings.exposure_us)
        self._set_enum(camera, "GainAuto", "Off", required=False)
        self._set_number(camera, ("Gain", "GainRaw"), settings.gain)
        self._set_bool(camera, "AcquisitionFrameRateEnable", settings.limit_frame_rate, required=False)
        if settings.limit_frame_rate:
            self._set_number(
                camera,
                ("AcquisitionFrameRate", "AcquisitionFrameRateAbs"),
                settings.frame_rate_fps,
                required=False,
            )

    def _configure_transport(self, device_class: str) -> None:
        """Use the source project's safe defaults for GigE packet transport."""
        camera = self._require_camera()
        if "GigE" not in device_class:
            return
        try:
            self._ensure_minimum(camera.GetInstantCameraNodeMap(), ("MaxNumBuffer",), 20)
            self._ensure_minimum(camera.GetStreamGrabberNodeMap(), ("FrameRetention",), 500)
            self._ensure_minimum(camera.GetNodeMap(), ("GevSCPD",), 1_000)
        except Exception:
            # Transport nodes differ by camera and pylon transport layer.
            pass

    def _start_grabbing(self) -> None:
        pylon, _ = _load_pylon()
        camera = self._require_camera()
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        with self._lock:
            self._connected = True

    def _retrieve_frame(self) -> CameraFrame | None:
        pylon, _ = _load_pylon()
        camera = self._require_camera()
        result = camera.RetrieveResult(self.retrieve_timeout_ms, pylon.TimeoutHandling_Return)
        if result is None or not result.IsValid():
            return None
        try:
            if not result.GrabSucceeded():
                self._set_error(f"프레임 수신 실패: {result.ErrorCode} {result.ErrorDescription}")
                return None
            if self._converter is None:
                raise CameraError("RGB 변환기가 초기화되지 않았습니다.")
            raw_array = result.GetArray().copy()
            if self.settings.pixel_format == "Mono8":
                # Keep the inspection and history interfaces consistently RGB.
                display_rgb = np.repeat(raw_array[:, :, np.newaxis], 3, axis=2)
            else:
                display_rgb = self._converter.Convert(result).GetArray().copy()
            return CameraFrame(
                display_rgb=display_rgb,
                raw_array=raw_array,
                width=int(result.GetWidth()),
                height=int(result.GetHeight()),
                pixel_format=self.settings.pixel_format,
                block_id=int(result.GetBlockID()),
                camera_timestamp=int(result.GetTimeStamp()),
                captured_at_ns=time_ns(),
            )
        finally:
            result.Release()

    def _close_camera(self) -> None:
        with self._device_lock:
            camera = self._camera
            with self._lock:
                self._connected = False
                self._frame = None
            if camera is None:
                return
            try:
                if camera.IsGrabbing():
                    camera.StopGrabbing()
                if camera.IsOpen():
                    camera.Close()
            finally:
                self._camera = None
                self._converter = None
                with self._lock:
                    self._connected = False

    def _require_camera(self) -> Any:
        if self._camera is None or not self._camera.IsOpen():
            raise CameraError("Basler 카메라가 연결되어 있지 않습니다.")
        return self._camera

    @staticmethod
    def _get_node(camera: Any, name: str, *, writable: bool = False) -> Any | None:
        _, genicam = _load_pylon()
        try:
            node = camera.GetNodeMap().GetNode(name)
            if node is None or not genicam.IsAvailable(node):
                return None
            return node if not writable or genicam.IsWritable(node) else None
        except Exception:
            return None

    def _set_enum(self, camera: Any, name: str, value: str, *, required: bool = True) -> bool:
        node = self._get_node(camera, name, writable=True)
        if node is None:
            if required:
                raise CameraError(f"카메라가 쓰기 가능한 {name} 기능을 지원하지 않습니다.")
            return False
        if value not in tuple(node.Symbolics):
            if required:
                raise CameraError(f"{name}={value}을 지원하지 않습니다. 사용 가능: {', '.join(node.Symbolics)}")
            return False
        node.SetValue(value)
        return True

    def _set_number(self, camera: Any, names: tuple[str, ...], value: float, *, required: bool = True) -> float | None:
        _, genicam = _load_pylon()
        for name in names:
            node = self._get_node(camera, name, writable=True)
            if node is None:
                continue
            bounded = min(max(value, float(node.Min)), float(node.Max))
            is_integer = node.GetNode().GetPrincipalInterfaceType() == genicam.intfIInteger
            node.SetValue(int(round(bounded)) if is_integer else bounded)
            return float(node.Value)
        if required:
            raise CameraError(f"카메라가 쓰기 가능한 {', '.join(names)} 기능을 지원하지 않습니다.")
        return None

    def _set_bool(self, camera: Any, name: str, value: bool, *, required: bool = True) -> bool:
        node = self._get_node(camera, name, writable=True)
        if node is None:
            if required:
                raise CameraError(f"카메라가 쓰기 가능한 {name} 기능을 지원하지 않습니다.")
            return False
        node.SetValue(value)
        return True

    def _set_node_to_max(self, camera: Any, name: str) -> None:
        node = self._get_node(camera, name, writable=True)
        if node is None:
            raise CameraError(f"카메라가 쓰기 가능한 {name} 기능을 지원하지 않습니다.")
        node.SetValue(node.Max)

    def _ensure_minimum(self, node_map: Any, names: tuple[str, ...], minimum: float) -> None:
        _, genicam = _load_pylon()
        for name in names:
            try:
                node = node_map.GetNode(name)
                if node is None or not genicam.IsAvailable(node) or not genicam.IsWritable(node):
                    continue
                target = min(max(float(node.Value), minimum, float(node.Min)), float(node.Max))
                is_integer = node.GetNode().GetPrincipalInterfaceType() == genicam.intfIInteger
                node.SetValue(int(round(target)) if is_integer else target)
                return
            except Exception:
                continue

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._connected = False
            self._error = message


if __name__ == "__main__":
    for device in list_devices():
        print(device)
