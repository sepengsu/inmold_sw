"""Shared lighting presets; loading and saving never communicate with hardware."""

from collections.abc import Mapping, Sequence
from math import isfinite
import os
from pathlib import Path
import tempfile

import yaml

from .led_control import LIGHT_GROUPS


DEFAULT_LIGHTING_CONFIG = Path(__file__).resolve().parents[2] / "config/lighting.yaml"
DEFAULT_LIGHTING_VALUES = (0.0,) * len(LIGHT_GROUPS)


def validate_lighting_values(values: Sequence[float]) -> tuple[float, ...]:
    if len(values) != len(LIGHT_GROUPS):
        raise ValueError("네 조명의 밝기를 모두 지정하세요.")
    normalized = []
    for fixture, value in zip(LIGHT_GROUPS, values, strict=True):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{fixture.name}: 밝기는 0~100 사이의 숫자여야 합니다.")
        try:
            percent = float(value)
        except (ValueError, OverflowError) as error:
            raise ValueError(f"{fixture.name}: 밝기는 0~100 사이의 숫자여야 합니다.") from error
        if not isfinite(percent) or not 0 <= percent <= 100:
            raise ValueError(f"{fixture.name}: 밝기는 0~100 사이여야 합니다.")
        normalized.append(percent)
    return tuple(normalized)


class LightingConfigStore:
    def __init__(self, path: str | Path = DEFAULT_LIGHTING_CONFIG) -> None:
        self.path = Path(path).expanduser().resolve()

    def load(self) -> tuple[float, ...]:
        try:
            data = yaml.safe_load(self.path.read_text(encoding="utf-8-sig"))
        except (yaml.YAMLError, UnicodeError) as error:
            raise ValueError(f"조명 설정 파일 형식 오류: {error}") from error
        if not isinstance(data, Mapping) or set(data) != {"version", "brightness_percent"}:
            raise ValueError("조명 설정에 version과 brightness_percent를 지정하세요.")
        if type(data["version"]) is not int or data["version"] != 1:
            raise ValueError("지원하지 않는 조명 설정 버전입니다.")
        levels = data["brightness_percent"]
        keys = tuple(fixture.config_key for fixture in LIGHT_GROUPS)
        if not isinstance(levels, Mapping) or set(levels) != set(keys):
            raise ValueError("brightness_percent에 네 조명의 설정 키를 정확히 지정하세요.")
        return validate_lighting_values(tuple(levels[key] for key in keys))

    def save(self, values: Sequence[float]) -> None:
        validated = validate_lighting_values(values)
        data = {
            "version": 1,
            "brightness_percent": {
                fixture.config_key: int(value) if value.is_integer() else value
                for fixture, value in zip(LIGHT_GROUPS, validated, strict=True)
            },
        }
        comments = "# 본 UI와 Cam_test가 공유하는 조명 설정 (%)\n"
        comments += "# 불러오기·저장은 장치에 전송하지 않습니다. 적용 버튼으로 전송하세요.\n"
        comments += "".join(f"# {fixture.config_key}: {fixture.name} ({fixture.channel_label})\n" for fixture in LIGHT_GROUPS)
        text = comments + yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            # Replace only after a complete write so another UI never reads half a file.
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=self.path.parent,
                prefix=f".{self.path.name}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def load_for_ui(self) -> tuple[tuple[float, ...] | None, str]:
        try:
            values = self.load()
        except (OSError, ValueError) as error:
            return None, f"⚠ 불러오기 실패: {self.path} · {error}"
        return values, f"✓ 불러오기 완료: {self.path} · 화면 설정값만 변경했습니다. 장치에는 전송하지 않았습니다."

    def initial_for_ui(self) -> tuple[tuple[float, ...], str]:
        values, message = self.load_for_ui()
        if values is None:
            return DEFAULT_LIGHTING_VALUES, message + " · 화면은 기본값 0%로 시작합니다."
        return values, message

    def save_for_ui(self, values: Sequence[float]) -> str:
        try:
            self.save(values)
        except (OSError, ValueError) as error:
            return f"⚠ 저장 실패: {self.path} · {error}"
        return f"✓ 저장 완료: {self.path} · 두 UI가 다음 로드 때 사용합니다. 장치에는 전송하지 않았습니다."
