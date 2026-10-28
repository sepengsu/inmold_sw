"""Archive durability, native images, time-zone filtering, and full CSV export."""

import csv
import cv2
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.inspection_history import InspectionHistoryStore, InspectionRecord
from src.web.service import Workbench


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
        workbench = Workbench.__new__(Workbench)
        workbench.history = reopened
        details = workbench.detail(saved.record_id)
        for key, expected in (("p1", self.result[2:6, 0:2]),
                              ("p3", self.result[2:6, 6:8])):
            pixels = cv2.cvtColor(cv2.imdecode(np.frombuffer(details["images"][key], np.uint8),
                                              cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            np.testing.assert_array_equal(pixels, expected)
        self.assertEqual(details["record"]["metadata"], self.record.metadata)

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
        first = self.store.query("2026-10-16", "2026-10-16", limit=50)
        second = self.store.query("2026-10-16", "2026-10-16", limit=50, offset=50)
        self.assertEqual((len(first), len(second)), (50, 5))
        self.assertEqual(self.store.summary("2026-10-16", "2026-10-16")["count"], 55)
        self.assertAlmostEqual(self.store.summary()["p1"], 27.)
        path = self.store.export_csv("2026-10-16", "2026-10-16")
        self.assertTrue(Path(path).read_bytes().startswith(b"\xef\xbb\xbf"))
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 55)
        self.assertEqual(float(rows[0]["P1 함침율 (%)"]), 54.)
        self.assertEqual(rows[-1]["소요 시간 (초)"], "1.25")
        self.assertNotIn("판정", rows[0])
        with self.assertRaises(ValueError):
            self.store.query("bad date")
        with self.assertRaises(ValueError):
            self.store.export_csv("bad date")

    def test_identical_timestamps_use_unique_ids_and_stable_newest_order(self):
        saved = [self.store.save(replace(self.record, p1=float(index)), self.original, self.result)
                 for index in range(3)]
        self.assertEqual(len({r.record_id for r in saved}), 3)
        self.assertEqual([r.record_id for r in self.store.query()], [r.record_id for r in reversed(saved)])
        with self.assertRaises(ValueError):
            self.store.read_image("../other.png")

    def test_clear_all_removes_history_images_and_exports_only(self):
        saved = self.store.save(self.record, self.original, self.result)
        export = self.store.export_csv()
        outside = self.root.parent / "keep.txt"
        outside.write_text("keep", encoding="utf-8")
        self.assertTrue((self.root / saved.original_path).exists())
        self.assertTrue(export.exists())

        self.assertEqual(self.store.clear_all(), 1)
        self.assertEqual(self.store.summary()["count"], 0)
        self.assertEqual(self.store.query(), [])
        self.assertFalse((self.root / "images").exists())
        self.assertFalse((self.root / "exports").exists())
        self.assertTrue(outside.exists())
        self.assertEqual(self.store.clear_all(), 0)


if __name__ == "__main__":
    unittest.main()
