"""Archive durability, native images, time-zone filtering, and full CSV export."""

import csv
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import gradio as gr
import numpy as np

from src.inspection_history import InspectionHistoryStore, InspectionRecord
from src.gui.statistics import StatisticsController


class InspectionHistoryTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name) / "검사 이력"
        self.store = InspectionHistoryStore(self.root)
        self.original = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)
        self.result = 255 - self.original
        self.record = InspectionRecord(90., 80., 70., "2026-10-16T00:00:00+09:00", "file", 1.25,
            metadata={"roi": {"x": 0, "y": 0, "width": 9, "height": 8},
                      "regions": [{"x": x, "y": 2, "width": 2, "height": 4} for x in (0, 3, 6)],
                      "model": {"sha256": "test"}, "conditions": {"노출": None}})

    def test_archive_survives_new_store_and_restores_native_rgb_and_crops(self):
        saved = self.store.save(self.record, self.original, self.result)
        reopened = InspectionHistoryStore(self.root)
        restored = reopened.query(limit=5)[0]
        self.assertEqual(restored, saved)
        np.testing.assert_array_equal(reopened.read_image(restored.original_path), self.original)
        np.testing.assert_array_equal(reopened.read_image(restored.result_path), self.result)
        controller = StatisticsController(lambda: reopened)
        event = gr.SelectData(None, {"index": [0, 2], "value": 90., "selected": True})
        details = controller.detail([saved.record_id], event)
        np.testing.assert_array_equal(details[3], self.result[2:6, 0:2])
        np.testing.assert_array_equal(details[5], self.result[2:6, 6:8])
        self.assertEqual(details[6], self.record.metadata)

    def test_korean_day_filter_includes_last_second_and_handles_utc_timestamps(self):
        timestamps = ("2026-10-15T14:59:59+00:00", "2026-10-15T15:00:00+00:00",
                      "2026-10-16T14:59:59.999999+00:00", "2026-10-16T15:00:00+00:00")
        for timestamp in timestamps:
            self.store.save(replace(self.record, measured_at=timestamp), self.original, self.result)
        rows = self.store.query("2026-10-16", "2026-10-16")
        self.assertEqual([r.measured_at for r in rows], list(reversed(timestamps[1:3])))
        self.assertEqual(self.store.summary("2026-10-16", "2026-10-16")["count"], 2)
        self.assertEqual(self.store.query(source="camera"), [])
        with self.assertRaises(ValueError):
            self.store.query("2026-10-17", "2026-10-16")

    def test_failed_database_insert_leaves_no_partial_images_or_history(self):
        with patch("src.inspection_history.sqlite3.connect", side_effect=sqlite3.OperationalError("write failed")):
            with self.assertRaisesRegex(sqlite3.Error, "write failed"):
                self.store.save(self.record, self.original, self.result)
        self.assertEqual(list(self.root.rglob("*.png")), [])
        self.assertEqual(self.store.summary()["count"], 0)

    def test_csv_exports_all_filtered_records_and_statistics_are_not_page_limited(self):
        base = datetime.fromisoformat(self.record.measured_at)
        for index in range(55):
            self.store.save(replace(self.record, measured_at=(base + timedelta(seconds=index)).isoformat(),
                                    p1=float(index)), self.original, self.result)
        controller = StatisticsController(lambda: self.store)
        first = controller.query("2026-10-16", "2026-10-16")
        second = controller.query("2026-10-16", "2026-10-16", page=2)
        self.assertEqual((len(first[1]), len(second[1])), (50, 5))
        self.assertIn("55건", first[0])
        self.assertAlmostEqual(self.store.summary()["p1"], 27.)
        download, message = controller.export("2026-10-16", "2026-10-16")
        path = download["value"]
        self.assertIn("CSV 생성 완료", message)
        self.assertTrue(Path(path).read_bytes().startswith(b"\xef\xbb\xbf"))
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 55)
        self.assertEqual(float(rows[0]["P1 함침율 (%)"]), 54.)
        self.assertEqual(rows[-1]["소요 시간 (초)"], "1.25")
        self.assertNotIn("판정", rows[0])
        invalid = controller.query("bad date")
        self.assertIn("조회 실패", invalid[5])
        self.assertFalse(invalid[6]["visible"])
        self.assertIsNone(invalid[6]["value"])
        self.assertFalse(controller.export("bad date")[0]["visible"])

    def test_identical_timestamps_use_unique_ids_and_stable_newest_order(self):
        saved = [self.store.save(replace(self.record, p1=float(index)), self.original, self.result)
                 for index in range(3)]
        self.assertEqual(len({r.record_id for r in saved}), 3)
        self.assertEqual([r.record_id for r in self.store.query()], [r.record_id for r in reversed(saved)])
        with self.assertRaises(ValueError):
            self.store.read_image("../other.png")


if __name__ == "__main__":
    unittest.main()
