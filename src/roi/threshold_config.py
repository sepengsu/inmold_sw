"""Persist linked ROI quality settings while preserving unrelated detector settings."""

from math import isfinite
import os
from pathlib import Path
import tempfile

import yaml

from .roi_boundary_quality import DEFAULT_CONFIG, load_config
from .quality_profiles import ROI_LEVEL_LABELS, linked_roi_settings, roi_level, roi_settings_summary


def validate_roi_threshold(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("ROI 품질 임계값은 0~1 사이의 숫자여야 합니다.")
    try:
        threshold = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError("ROI 품질 임계값은 0~1 사이의 숫자여야 합니다.") from error
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("ROI 품질 임계값은 0~1 사이여야 합니다.")
    return threshold


class RoiThresholdConfigStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path if path is not None else DEFAULT_CONFIG).expanduser().resolve()

    def load(self) -> float:
        return validate_roi_threshold(load_config(self.path)["quality"]["min_score"])

    def save(self, value: float) -> float:
        threshold = validate_roi_threshold(value)
        updates = linked_roi_settings(threshold)
        # Validate all existing detector settings before touching the document.
        load_config(self.path)
        document = self.path.read_text(encoding="utf-8-sig")
        root = yaml.compose(document)
        replacements = []
        complete = True
        for section, values in updates.items():
            section_node = next((node for key, node in root.value if key.value == section), None)
            for key, updated in values.items():
                node = (next((node for name, node in section_node.value if name.value == key), None)
                        if section_node is not None else None)
                if node is None:
                    complete = False
                    continue
                if isinstance(updated, list):
                    if not isinstance(node, yaml.SequenceNode) or len(node.value) != len(updated):
                        complete = False
                        continue
                    # Replace individual items, preserving both block and flow list formatting.
                    replacements.extend((item.start_mark.index, item.end_mark.index, repr(value))
                                        for item, value in zip(node.value, updated))
                else:
                    replacements.append((node.start_mark.index, node.end_mark.index, repr(updated)))
        if complete:
            # Apply from the end so source offsets and existing comments stay valid.
            for start, end, text in sorted(replacements, reverse=True):
                document = document[:start] + text + document[end:]
        else:
            data = yaml.safe_load(document)
            for section, values in updates.items():
                data.setdefault(section, {}).update(values)
            document = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=self.path.parent,
                prefix=f".{self.path.name}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                stream.write(document)
                stream.flush()
                os.fsync(stream.fileno())
            load_config(temporary_path)
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return threshold

    def load_for_ui(self) -> tuple[float | None, str]:
        value, message, _, _ = self.settings_for_ui()
        return value, message

    def settings_for_ui(self) -> tuple:
        try:
            config = load_config(self.path)
            threshold = validate_roi_threshold(config["quality"]["min_score"])
        except (OSError, ValueError, yaml.YAMLError) as error:
            return None, f"⚠ ROI 기준 불러오기 실패: {self.path} · {error}", None, "ROI 설정을 불러오지 못했습니다."
        level = roi_level(config)
        label = ROI_LEVEL_LABELS.get(level, "세부 조정")
        return threshold, f"저장된 ROI 검증: **{label}** · 점수 기준 **{threshold:g}** · `{self.path}`", level, roi_settings_summary(config)

    def initial_for_ui(self) -> tuple[float, str]:
        value, message = self.load_for_ui()
        return (value if value is not None else .25), message

    def save_for_ui(self, value: float) -> tuple[float | None, str]:
        try:
            threshold = self.save(value)
        except (OSError, ValueError, yaml.YAMLError) as error:
            return None, f"⚠ ROI 기준 저장 실패: {self.path} · {error}"
        level = roi_level(linked_roi_settings(threshold))
        label = ROI_LEVEL_LABELS.get(level, "세부 조정")
        return threshold, f"✓ ROI **{label}** · 점수 및 연관 검증 기준 적용 및 저장 완료 · ROI를 다시 추출하거나 다시 측정하세요. 기존 추출 결과는 자동으로 재평가하지 않습니다."
