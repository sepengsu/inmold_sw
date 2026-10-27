"""Persistent inspection results, native PNGs, date queries, and streamed CSV export."""

from contextlib import contextmanager
import csv
from dataclasses import dataclass, field, replace
from datetime import date, datetime, time, timedelta, timezone
import json
from math import isfinite
import os
from pathlib import Path
import shutil
import sqlite3
from uuid import uuid4

import cv2
import numpy as np


LOCAL_TIMEZONE = timezone(timedelta(hours=9), "Asia/Seoul")
DEFAULT_HISTORY_ROOT = Path(__file__).resolve().parents[1] / "data" / "history"


@dataclass(frozen=True, slots=True)
class InspectionRecord:
    p1: float
    p2: float
    p3: float
    measured_at: str = ""
    source: str = "camera"
    elapsed_seconds: float | None = None
    record_id: str = ""
    original_path: str = ""
    result_path: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def average(self) -> float:
        return (self.p1 + self.p2 + self.p3) / 3


class InspectionHistoryStore:
    def __init__(self, root: str | Path = DEFAULT_HISTORY_ROOT):
        self.root = Path(root).expanduser().resolve()
        self.db_path = self.root / "history.sqlite3"

    @contextmanager
    def _connection(self):
        self.root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            with connection:
                connection.execute("""CREATE TABLE IF NOT EXISTS inspections (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT NOT NULL UNIQUE,
                    measured_at TEXT NOT NULL,
                    measured_utc_us INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    p1 REAL NOT NULL, p2 REAL NOT NULL, p3 REAL NOT NULL,
                    average REAL NOT NULL, elapsed_seconds REAL,
                    original_path TEXT NOT NULL, result_path TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )""")
                connection.execute("CREATE INDEX IF NOT EXISTS inspections_time ON inspections(measured_utc_us DESC, sequence DESC)")
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _record(row) -> InspectionRecord:
        return InspectionRecord(**{name: row[name] for name in (
            "p1", "p2", "p3", "measured_at", "source", "elapsed_seconds", "record_id",
            "original_path", "result_path")}, metadata=json.loads(row["metadata_json"]))

    @staticmethod
    def _where(start_date=None, end_date=None, source="all"):
        clauses, parameters = [], []
        days = []
        for value, comparison, increment in ((start_date, ">=", 0), (end_date, "<", 1)):
            if not value:
                days.append(None)
                continue
            try:
                day = date.fromisoformat(str(value).split("T")[0].split(" ")[0])
            except ValueError as error:
                raise ValueError("날짜는 YYYY-MM-DD 형식으로 입력하세요.") from error
            days.append(day)
            boundary = datetime.combine(day + timedelta(days=increment), time(), LOCAL_TIMEZONE)
            clauses.append(f"measured_utc_us {comparison} ?")
            parameters.append(int(boundary.timestamp() * 1_000_000))
        if all(days) and days[0] > days[1]:
            raise ValueError("시작일은 종료일보다 늦을 수 없습니다.")
        if source not in ("all", "camera", "file"):
            raise ValueError("검사 입력 종류를 확인하세요.")
        if source != "all":
            clauses.append("source = ?")
            parameters.append(source)
        return (" WHERE " + " AND ".join(clauses) if clauses else ""), parameters

    def save(self, record: InspectionRecord, original_rgb: np.ndarray,
             result_rgb: np.ndarray) -> InspectionRecord:
        for value in (record.p1, record.p2, record.p3):
            if not isfinite(value) or not 0 <= value <= 100:
                raise ValueError("함침율은 0~100 범위여야 합니다.")
        if record.elapsed_seconds is not None and (
                not isfinite(record.elapsed_seconds) or record.elapsed_seconds < 0):
            raise ValueError("소요 시간은 0 이상의 유한한 값이어야 합니다.")
        if record.source not in ("camera", "file"):
            raise ValueError("검사 입력 종류를 확인하세요.")
        measured = datetime.fromisoformat(record.measured_at)
        if measured.tzinfo is None:
            raise ValueError("측정 시각에는 시간대가 필요합니다.")
        for pixels in (original_rgb, result_rgb):
            if (not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8
                    or pixels.ndim != 3 or pixels.shape[2] != 3 or not pixels.size):
                raise ValueError("저장 영상은 uint8 RGB 배열이어야 합니다.")
        if original_rgb.shape != result_rgb.shape:
            raise ValueError("원본 영상과 결과 영상의 크기가 다릅니다.")
        metadata_json = json.dumps(record.metadata, ensure_ascii=False, allow_nan=False)
        record_id = uuid4().hex
        folder = self.root / "images" / measured.astimezone(LOCAL_TIMEZONE).strftime("%Y-%m-%d") / record_id
        folder.mkdir(parents=True, exist_ok=False)
        created = []
        saved = replace(record, record_id=record_id,
                        original_path=(folder / "original.png").relative_to(self.root).as_posix(),
                        result_path=(folder / "result.png").relative_to(self.root).as_posix())
        try:
            for name, pixels in (("original.png", original_rgb), ("result.png", result_rgb)):
                ok, encoded = cv2.imencode(".png", cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
                if not ok:
                    raise OSError("검사 영상 PNG 인코딩에 실패했습니다.")
                path = folder / name
                with path.open("xb") as stream:
                    created.append(path)
                    stream.write(encoded.tobytes())
                    stream.flush()
                    os.fsync(stream.fileno())
            # Publish the record only after both complete images exist.
            with self._connection() as connection:
                connection.execute("""INSERT INTO inspections
                    (record_id, measured_at, measured_utc_us, source, p1, p2, p3,
                     average, elapsed_seconds, original_path, result_path, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                    saved.record_id, saved.measured_at, int(measured.timestamp() * 1_000_000),
                    saved.source, float(saved.p1), float(saved.p2), float(saved.p3),
                    float(saved.average), saved.elapsed_seconds, saved.original_path,
                    saved.result_path, metadata_json))
        except Exception:
            for path in created:
                path.unlink(missing_ok=True)
            folder.rmdir()
            raise
        return saved

    def query(self, start_date=None, end_date=None, source="all", *, limit=50, offset=0):
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError("조회 범위를 확인하세요.")
        where, parameters = self._where(start_date, end_date, source)
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM inspections" + where
                + " ORDER BY measured_utc_us DESC, sequence DESC LIMIT ? OFFSET ?",
                [*parameters, limit, offset]).fetchall()
        return [self._record(row) for row in rows]

    def summary(self, start_date=None, end_date=None, source="all"):
        where, parameters = self._where(start_date, end_date, source)
        with self._connection() as connection:
            return dict(connection.execute("""SELECT COUNT(*) AS count,
                AVG(p1) AS p1, AVG(p2) AS p2, AVG(p3) AS p3,
                AVG(average) AS average, MIN(average) AS minimum, MAX(average) AS maximum,
                AVG(elapsed_seconds) AS elapsed_seconds FROM inspections""" + where, parameters).fetchone())

    def get(self, record_id: str):
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM inspections WHERE record_id = ?", (record_id,)).fetchone()
        if row is None:
            raise ValueError("저장된 측정 이력을 찾을 수 없습니다.")
        return self._record(row)

    def read_image(self, relative_path: str) -> np.ndarray:
        path = (self.root / relative_path).resolve()
        if not relative_path or not path.is_relative_to(self.root / "images"):
            raise ValueError("저장 영상 경로가 올바르지 않습니다.")
        pixels = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if pixels is None:
            raise OSError("저장 영상을 읽을 수 없습니다.")
        return cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)

    def clear_all(self) -> int:
        """Delete all measurement rows and app-owned image/export folders."""
        owned_folders = []
        for name in ("images", "exports"):
            folder = self.root / name
            if folder.is_symlink() or folder.resolve().parent != self.root or folder.name != name:
                raise ValueError(f"이력 폴더 경로가 올바르지 않습니다: {folder}")
            owned_folders.append(folder)
        with self._connection() as connection:
            count = connection.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
            connection.execute("DELETE FROM inspections")
            connection.execute("DELETE FROM sqlite_sequence WHERE name = 'inspections'")
        for folder in owned_folders:
            if folder.exists():
                shutil.rmtree(folder)
        return count

    def export_csv(self, start_date=None, end_date=None, source="all") -> Path:
        where, parameters = self._where(start_date, end_date, source)
        folder = self.root / "exports"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"inspection_{datetime.now(LOCAL_TIMEZONE):%Y%m%d_%H%M%S}_{uuid4().hex[:8]}.csv"
        try:
            with self._connection() as connection, path.open("x", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["측정 ID", "측정 시각", "입력", "P1 함침율 (%)", "P2 함침율 (%)",
                    "P3 함침율 (%)", "평균 함침율 (%)", "소요 시간 (초)", "원본 이미지", "결과 이미지",
                    "ROI", "P1·P2·P3 좌표", "모델 정보", "촬영 조건"])
                cursor = connection.execute("SELECT * FROM inspections" + where
                    + " ORDER BY measured_utc_us DESC, sequence DESC", parameters)
                for row in cursor:
                    meta = json.loads(row["metadata_json"])
                    writer.writerow([row[name] for name in ("record_id", "measured_at", "source", "p1", "p2", "p3",
                        "average", "elapsed_seconds", "original_path", "result_path")]
                        + [json.dumps(meta.get(name), ensure_ascii=False) for name in ("roi", "regions", "model", "conditions")])
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return path
