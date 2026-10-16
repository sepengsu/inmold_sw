"""Shared Gradio controls for camera testing and the inspection workbench."""

from collections.abc import Callable

import gradio as gr

from src.roi.threshold_config import RoiThresholdConfigStore, validate_roi_threshold
from src.roi.quality_profiles import (
    ROI_LEVEL_LABELS, ROI_QUALITY_PROFILES, linked_roi_settings, roi_level, roi_settings_summary,
)


BRIGHTNESS_READOUT_JS = """(value) => {
    const percent = Number(value);
    const label = Number.isFinite(percent) ? percent.toLocaleString('ko-KR', {maximumFractionDigits: 2}) : '—';
    return `<div class="brightness-value"><span>설정 밝기</span><strong>${label}%</strong></div>`;
}"""


def brightness_slider(label: str, value: float) -> gr.Slider:
    readout = gr.HTML(
        f'<div class="brightness-value"><span>설정 밝기</span><strong>{value:g}%</strong></div>',
        elem_classes="brightness-readout",
    )
    slider = gr.Slider(0, 100, value=value, step=1, label=label)
    # Frontend-only change also follows config loads and ACK-confirmed all-off updates.
    slider.change(fn=None, inputs=slider, outputs=readout, js=BRIGHTNESS_READOUT_JS,
                  queue=False, api_visibility="private")
    return slider


def roi_threshold_controls(demo: gr.Blocks, store: Callable[[], RoiThresholdConfigStore]) -> None:
    initial, message, selected, details = store().settings_for_ui()
    level = gr.Radio(choices=[(label, name) for name, label in ROI_LEVEL_LABELS.items()],
                     value=selected, label="ROI 검증 강도")
    threshold = gr.Slider(0, 1, value=initial if initial is not None else .25, step=.01,
                          label="ROI 기준 · 연관 검증 자동 조절 (0~1)")
    summary = gr.Markdown(details, elem_classes="roi-config-help")
    gr.Markdown("낮음은 완화, 높음은 엄격한 검증입니다. 단계 또는 ROI 기준을 바꾸면 경계 일치 비율·지지 범위·끊김·오차 허용치도 함께 조절됩니다. 적용 및 저장 후 ROI를 다시 추출하거나 다시 측정하세요.",
                elem_classes="roi-config-help")
    with gr.Row():
        save = gr.Button("ROI 기준 적용 및 저장", size="sm")
        load = gr.Button("ROI 기준 불러오기", size="sm")
    status = gr.Markdown(message, elem_classes="roi-config-status")

    def save_threshold(value: float) -> tuple:
        value, message = store().save_for_ui(value)
        if value is None:
            return gr.skip(), message, gr.skip(), gr.skip()
        config = linked_roi_settings(value)
        return value, message, roi_level(config), roi_settings_summary(config)

    def load_threshold() -> tuple:
        value, message, selected, details = store().settings_for_ui()
        if value is None:
            return gr.skip(), message, gr.skip(), gr.skip()
        return value, message, selected, details

    def select_level(selected: str) -> tuple:
        if selected not in ROI_QUALITY_PROFILES:
            raise ValueError("낮음·중간·높음 중 ROI 검증 강도를 선택하세요.")
        config = ROI_QUALITY_PROFILES[selected]
        return config["quality"]["min_score"], roi_settings_summary(config)

    def preview_threshold(value: float) -> tuple:
        config = linked_roi_settings(validate_roi_threshold(value))
        return roi_level(config), roi_settings_summary(config)

    level.input(select_level, inputs=level, outputs=[threshold, summary], queue=False,
                api_name="roi_level_preview", show_progress="hidden")
    threshold.input(preview_threshold, inputs=threshold, outputs=[level, summary], queue=False,
                    api_name="roi_threshold_preview", show_progress="hidden")
    outputs = [threshold, status, level, summary]
    save.click(save_threshold, inputs=threshold, outputs=outputs,
               api_name="roi_threshold_save", concurrency_id="roi-config", concurrency_limit=1)
    load.click(load_threshold, outputs=outputs,
               api_name="roi_threshold_load", concurrency_id="roi-config", concurrency_limit=1)
    demo.load(load_threshold, outputs=outputs, api_visibility="private", show_progress="hidden",
              concurrency_id="roi-config", concurrency_limit=1)
