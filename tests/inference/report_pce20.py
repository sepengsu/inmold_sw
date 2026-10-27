r"""Run PCE inference for the 20 verification captures and render an error table.

From the repository root:
    .venv\Scripts\python.exe tests\inference\report_pce20.py
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.PCE.inspection import inspect_frame  # noqa: E402
from src.model import load_checkpoint  # noqa: E402
from src.preprocess import load_config  # noqa: E402


DEFAULT_DATA_ROOT = ROOT / "data" / "검증용 20"
DEFAULT_OUTPUT_DIR = ROOT / "tests" / "inference" / "results"
CASE_LINE = re.compile(r"Case\s+(\d+-\d+)\s+\((정답지 샘플|검증용 샘플)\)\s+([\d.]+)\s*")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def read_answers(path: Path) -> dict[str, tuple[str, float]]:
    """Map each case directory to its supplied P2 impregnation percentage."""
    answers = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        match = CASE_LINE.fullmatch(line.strip())
        if match is None:
            raise ValueError(f"정답지 {line_number}행 형식을 읽지 못했습니다: {line}")
        name = f"case{match[1]}"
        value = float(match[3])
        if name in answers or not 0 <= value <= 100:
            raise ValueError(f"정답지 중복 또는 잘못된 함침율: {name}={value}")
        answers[name] = (match[2], value)
    if len(answers) != 20:
        raise ValueError(f"정답지에 20개 Case가 필요합니다. 현재 {len(answers)}개")
    return answers


def find_cases(data_root: Path, answers: dict) -> list[tuple[str, Path]]:
    folders = {path.name: path for path in data_root.iterdir() if path.is_dir()}
    if folders.keys() != answers.keys():
        missing = sorted(answers.keys() - folders.keys())
        extra = sorted(folders.keys() - answers.keys())
        raise ValueError(f"정답지와 폴더가 다릅니다. 누락={missing}, 추가={extra}")
    cases = []
    for name in sorted(folders, key=lambda value: tuple(map(int, value.removeprefix("case").split("-")))):
        images = [path for path in folders[name].iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        if len(images) != 1:
            raise ValueError(f"{name}에는 입력 이미지가 정확히 1장이어야 합니다. 현재 {len(images)}장")
        cases.append((name, images[0]))
    return cases


def read_rgb(path: Path) -> np.ndarray:
    """Use imdecode because cv2.imread can fail on Korean Windows paths."""
    pixels = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if pixels is None or pixels.dtype != np.uint8:
        raise ValueError(f"8비트 이미지를 읽지 못했습니다: {path}")
    if pixels.ndim == 2:
        return cv2.cvtColor(pixels, cv2.COLOR_GRAY2RGB)
    if pixels.ndim == 3 and pixels.shape[2] == 3:
        return cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
    if pixels.ndim == 3 and pixels.shape[2] == 4:
        return cv2.cvtColor(pixels, cv2.COLOR_BGRA2RGB)
    raise ValueError(f"지원하지 않는 이미지 형식: {path} · shape={pixels.shape}")


def make_row(name: str, group: str, truth: float, scores: tuple[float, float, float],
             image_path: Path) -> dict:
    prediction = float(scores[1])
    difference = prediction - truth
    return {
        "case": name,
        "group": group,
        "image": str(image_path),
        "p2_answer_pct": truth,
        "p2_prediction_pct": prediction,
        "signed_error_pp": difference,
        "absolute_error_pp": abs(difference),
        "relative_error_pct": 100 * abs(difference) / truth if truth else None,
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("요약할 결과가 없습니다.")
    relative = [row["relative_error_pct"] for row in rows if row["relative_error_pct"] is not None]
    return {
        "cases": len(rows),
        "mae_pp": float(np.mean([row["absolute_error_pp"] for row in rows])),
        "mape_pct": float(np.mean(relative)) if relative else None,
        "zero_answer_cases": len(rows) - len(relative),
    }


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    korean = Path("C:/Windows/Fonts") / ("malgunbd.ttf" if bold else "malgun.ttf")
    return ImageFont.truetype(str(korean if korean.exists() else "DejaVuSans.ttf"), size)


def render_table(rows: list[dict], summary: dict, output: Path, checkpoint_name: str,
                 positive_classes: tuple[str, ...]) -> None:
    columns = [("Case", 150), ("구분", 135), ("P2 정답 (%)", 175),
               ("P2 예측 (%)", 175), ("차이 (%p)", 165), ("오차율 (%)", 165)]
    margin, top, header_height, row_height, footer_height = 28, 165, 58, 48, 140
    table_width = sum(width for _, width in columns)
    image = Image.new("RGB", (table_width + 2 * margin,
                              top + header_height + row_height * len(rows) + footer_height), "#F5F7FB")
    draw = ImageDraw.Draw(image)
    title_font, body_font, cell_font = _font(32, True), _font(20), _font(20)
    header_font, summary_font = _font(19, True), _font(22, True)
    draw.text((margin, 20), "PCE 검증용 20 · P2 함침율 오차표", font=title_font, fill="#172B45")
    draw.text((margin, 69), f"모델: {checkpoint_name}  |  검출 대상: {' + '.join(positive_classes)}",
              font=body_font, fill="#334B66")
    draw.text((margin, 103), "차이 = P2 예측 - P2 정답 (%p)     오차율 = |차이| / P2 정답 × 100",
              font=body_font, fill="#334B66")
    draw.text((margin, 134), "정답지는 P2 전용 · P1/P3는 비교와 오차 계산에서 제외",
              font=_font(18), fill="#64748B")

    x = margin
    for title, width in columns:
        draw.rectangle((x, top, x + width, top + header_height), fill="#294A69")
        draw.text((x + width / 2, top + header_height / 2), title, anchor="mm",
                  font=header_font, fill="white")
        x += width

    for index, row in enumerate(rows):
        y = top + header_height + index * row_height
        draw.rectangle((margin, y, margin + table_width, y + row_height),
                       fill="#FFFFFF" if index % 2 == 0 else "#EAF0F6")
        values = [row["case"], "정답지" if row["group"] == "정답지 샘플" else "검증용",
                  f'{row["p2_answer_pct"]:.2f}', f'{row["p2_prediction_pct"]:.2f}',
                  f'{row["signed_error_pp"]:+.2f}',
                  "—" if row["relative_error_pct"] is None else f'{row["relative_error_pct"]:.2f}']
        x = margin
        for column, ((_, width), value) in enumerate(zip(columns, values, strict=True)):
            color = "#A23D28" if column == 5 and row["relative_error_pct"] is not None \
                    and row["relative_error_pct"] >= 5 else "#17324D"
            draw.text((x + width / 2, y + row_height / 2), value, anchor="mm",
                      font=cell_font, fill=color)
            x += width

    footer = top + header_height + row_height * len(rows)
    all_cases, verification = summary["all"], summary["verification"]
    def error_label(value: float | None) -> str:
        return "—" if value is None else f"{value:.2f}%"

    draw.text((margin, footer + 18),
              f'전체 {all_cases["cases"]}건  |  MAE {all_cases["mae_pp"]:.2f}%p  |  평균 오차율 {error_label(all_cases["mape_pct"])}',
              font=summary_font, fill="#125981")
    draw.text((margin, footer + 55),
              f'검증용 {verification["cases"]}건  |  MAE {verification["mae_pp"]:.2f}%p  |  평균 오차율 {error_label(verification["mape_pct"])}',
              font=body_font, fill="#334B66")
    draw.text((margin, footer + 94), "P2만 비교 · 반올림 전 값으로 계산 · 표시는 소수 둘째 자리",
              font=_font(17), fill="#64748B")
    image.save(output)


def run(data_root: Path, checkpoint_path: Path, config_path: Path, output_dir: Path,
        device: str = "cpu") -> dict:
    answers = read_answers(data_root / "정답지.txt")
    cases = find_cases(data_root, answers)
    model, model_info = load_checkpoint(checkpoint_path, device=device)
    config = load_config(config_path)
    rows = []
    for name, image_path in cases:
        result = inspect_frame(read_rgb(image_path), model, model_info["threshold"], config)
        group, answer = answers[name]
        row = make_row(name, group, answer, result.scores, image_path)
        rows.append(row)
        print(f'{name}: P2 정답 {answer:.2f}% · P2 예측 {row["p2_prediction_pct"]:.2f}% '
              f'· 오차율 {row["relative_error_pct"]:.2f}%', flush=True)
    summary = {
        "all": summarize(rows),
        "answer_sheet": summarize([row for row in rows if row["group"] == "정답지 샘플"]),
        "verification": summarize([row for row in rows if row["group"] == "검증용 샘플"]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "pce20_error_table.png"
    render_table(rows, summary, png_path, checkpoint_path.name, model_info["positive_classes"])
    csv_path = output_dir / "pce20_error_table.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root.resolve()),
        "answer_sheet": str((data_root / "정답지.txt").resolve()),
        "answer_sheet_sha256": hashlib.sha256((data_root / "정답지.txt").read_bytes()).hexdigest(),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "architecture": model_info["architecture"],
        "positive_classes": model_info["positive_classes"],
        "pixel_threshold": model_info["threshold"],
        "comparison_point": "P2",
        "metric": "abs(predicted P2 - P2 answer) / P2 answer * 100",
        "summary": summary,
        "rows": rows,
    }
    json_path = output_dir / "pce20_error_table.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f'저장 완료: {png_path}\nCSV: {csv_path}\nJSON: {json_path}')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "model" / "best_model.pt")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "preprocess.yaml")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run(args.data_root, args.checkpoint, args.config, args.output_dir, args.device)


if __name__ == "__main__":
    main()
