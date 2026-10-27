"""Load and atomically save shared camera settings without accessing devices."""

from collections.abc import Mapping
from math import isfinite
import os
from pathlib import Path
import tempfile

import yaml

from .camera_control import BaslerCameraSettings, CAMERA_PIXEL_FORMATS


DEFAULT_CAMERA_CONFIG = Path(__file__).resolve().parents[2] / "config/camera.yaml"


def validate_exposure_us(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("노출 시간(µs)은 0보다 큰 숫자여야 합니다.")
    try:
        exposure = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError("노출 시간(µs)은 유한한 양수여야 합니다.") from error
    if not isfinite(exposure) or exposure <= 0:
        raise ValueError("노출 시간(µs)은 유한한 양수여야 합니다.")
    return exposure


def load_camera_settings(path: str | Path = DEFAULT_CAMERA_CONFIG) -> BaslerCameraSettings:
    path = Path(path).expanduser().resolve()
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"카메라 설정 파일을 읽지 못했습니다: {path} · {error}") from error
    if (not isinstance(data, Mapping) or "exposure_us" not in data
            or set(data) - {"exposure_us", "pixel_format"}):
        raise ValueError(f"{path}: exposure_us와 선택적 pixel_format을 지정하세요.")
    try:
        exposure = validate_exposure_us(data["exposure_us"])
    except ValueError as error:
        raise ValueError(f"{path}: exposure_us · {error}") from error
    pixel_format = data.get("pixel_format", BaslerCameraSettings().pixel_format)
    if pixel_format not in CAMERA_PIXEL_FORMATS.values():
        raise ValueError(f"{path}: pixel_format은 Mono8 또는 BayerGR12여야 합니다.")
    return BaslerCameraSettings(exposure_us=exposure, pixel_format=pixel_format)


def save_camera_exposure(exposure_us: float, path: str | Path = DEFAULT_CAMERA_CONFIG) -> None:
    exposure = validate_exposure_us(exposure_us)
    path = Path(path).expanduser().resolve()
    pixel_format = load_camera_settings(path).pixel_format if path.exists() else BaslerCameraSettings().pixel_format
    _save_camera_settings(exposure, pixel_format, path)


def save_camera_pixel_format(pixel_format: str, path: str | Path = DEFAULT_CAMERA_CONFIG) -> None:
    if pixel_format not in CAMERA_PIXEL_FORMATS.values():
        raise ValueError("촬영 형식은 Mono8 또는 BayerGR12여야 합니다.")
    path = Path(path).expanduser().resolve()
    exposure = load_camera_settings(path).exposure_us if path.exists() else BaslerCameraSettings().exposure_us
    _save_camera_settings(exposure, pixel_format, path)


def _save_camera_settings(exposure: float, pixel_format: str, path: Path) -> None:
    data = {"exposure_us": int(exposure) if exposure.is_integer() else exposure,
            "pixel_format": pixel_format}
    document = "# 본 UI와 Cam_test가 공유하는 카메라 설정\n"
    document += "# 카메라 연결 시 자동 노출을 끄고 노출 시간(µs)과 촬영 형식을 적용합니다.\n"
    document += yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
