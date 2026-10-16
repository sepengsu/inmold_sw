"""Statistics-page presentation for the persistent inspection archive."""

from html import escape
from math import ceil
import sqlite3

import gradio as gr


PAGE_SIZE = 50
TABLE_HEADERS = ["측정 시각", "입력", "P1 (%)", "P2 (%)", "P3 (%)", "평균 (%)", "소요 시간 (초)"]


def summary_html(summary):
    def number(key, suffix="%"):
        value = summary.get(key)
        return "—" if value is None else f"{value:.3f}{suffix}"
    cards = (("측정 수", f"{summary.get('count', 0):,}건"),
             ("평균 함침율", number("average")),
             ("평균 소요 시간", number("elapsed_seconds", "초")))
    html = '<div class="statistics-cards">' + "".join(
        f'<div class="statistics-card"><span>{label}</span><strong>{value}</strong></div>'
        for label, value in cards) + '</div>'
    return html + '<div class="statistics-points">' + " · ".join(
        f'{label} {number(key)}' for label, key in (("P1 평균", "p1"), ("P2 평균", "p2"),
            ("P3 평균", "p3"), ("측정 평균 최솟값", "minimum"), ("최댓값", "maximum"))) + '</div>'


class StatisticsController:
    def __init__(self, store):
        self.store = store

    def query(self, start_date=None, end_date=None, source="all", page=1):
        try:
            summary = self.store().summary(start_date, end_date, source)
            pages = max(1, ceil(summary["count"] / PAGE_SIZE))
            page = min(pages, max(1, int(page)))
            records = self.store().query(start_date, end_date, source,
                limit=PAGE_SIZE, offset=(page - 1) * PAGE_SIZE)
            rows = [[r.measured_at, "카메라" if r.source == "camera" else "파일",
                     round(r.p1, 3), round(r.p2, 3), round(r.p3, 3), round(r.average, 3),
                     None if r.elapsed_seconds is None else round(r.elapsed_seconds, 3)] for r in records]
            message = "조회 조건에 해당하는 측정 이력이 없습니다." if not records else "행을 선택하면 저장된 원본과 결과를 볼 수 있습니다."
            return (summary_html(summary), rows, [r.record_id for r in records], page,
                    f"{page} / {pages} 페이지 · 전체 {summary['count']:,}건", message, gr.update(value=None, visible=False))
        except (OSError, ValueError, sqlite3.Error) as error:
            return summary_html({}), [], [], 1, "1 / 1 페이지", f"조회 실패: {error}", gr.update(value=None, visible=False)

    def export(self, start_date=None, end_date=None, source="all"):
        try:
            path = self.store().export_csv(start_date, end_date, source)
            return gr.update(value=str(path), visible=True), "CSV 생성 완료 · 아래 파일을 눌러 다운로드하세요. 선택한 기간의 전체 기록이 포함됩니다."
        except (OSError, ValueError, sqlite3.Error) as error:
            return gr.update(value=None, visible=False), f"CSV 내보내기 실패: {error}"

    def detail(self, record_ids: list[str], event: gr.SelectData):
        try:
            index = event.index[0] if isinstance(event.index, (list, tuple)) else event.index
            if not event.selected or not 0 <= index < len(record_ids):
                return "측정 항목을 선택하세요.", None, None, None, None, None, {}
            record = self.store().get(record_ids[index])
            original = self.store().read_image(record.original_path)
            result = self.store().read_image(record.result_path)
            crops = []
            for region in record.metadata.get("regions", []):
                x, y, width, height = (region[key] for key in ("x", "y", "width", "height"))
                crops.append(result[y:y + height, x:x + width].copy())
            crops = (crops + [None] * 3)[:3]
            elapsed = "—" if record.elapsed_seconds is None else f"{record.elapsed_seconds:.3f}초"
            text = (f"**{escape(record.measured_at)}** · 평균 **{record.average:.3f}%** · 소요 시간 **{elapsed}**\n\n"
                    f"P1 {record.p1:.3f}% · P2 {record.p2:.3f}% · P3 {record.p3:.3f}%")
            return text, original, result, *crops, record.metadata
        except (OSError, ValueError, KeyError, sqlite3.Error) as error:
            return f"이력 열기 실패: {error}", None, None, None, None, None, {}
