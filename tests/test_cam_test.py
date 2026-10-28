"""Verify live-preview freshness and deliberate RS232 command routing."""

from time import monotonic
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import cv2

from src.diagnostics.camera import CamTestController
from src.hardware.camera_control import CameraFrame, CameraStatus
from src.hardware.led_control import LightCommunicationError, LightConnectionStatus, build_brightness_packet
from src.preprocess.auto_roi import AutoRoiResult


class CamTestTests(unittest.TestCase):
    def setUp(self):
        self.controller = CamTestController("24346077")
        self.camera = Mock()
        self.camera.get_status.return_value = CameraStatus(
            running=True, connected=True, description="Test camera", serial_number="24346077",
            frames_received=5, last_frame_at=monotonic(), error=None,
        )
        self.rgb = np.full((3, 4, 3), 100, dtype=np.uint8)
        self.camera.get_latest_frame.return_value = self.rgb.copy()
        self.camera.get_latest_camera_frame.return_value = CameraFrame(
            display_rgb=self.rgb, raw_array=np.full((3, 4), 1600, dtype=np.uint16),
            width=4, height=3, pixel_format="BayerGR12", block_id=5,
            camera_timestamp=100, captured_at_ns=100,
        )
        self.light = Mock(port="COM3", baudrate=9600)
        self.light.get_status.return_value = LightConnectionStatus(True, False, None)
        self.light.send.side_effect = lambda packets: packets
        self.controller.camera = self.camera
        self.controller.light = self.light

    def test_preview_preserves_camera_pixels_without_display_gain(self):
        original_settings = self.controller.settings
        image, status, _, stats = self.controller.preview(show_roi=False)
        np.testing.assert_array_equal(image, self.rgb)
        self.assertTrue((self.rgb == 100).all())
        self.assertIn("READY", status)
        self.assertIn("uint16", stats)
        self.light.send.assert_not_called()
        self.assertIs(self.controller.settings, original_settings)
        self.camera.start.assert_not_called()
        self.camera.stop.assert_not_called()

    def test_stale_camera_does_not_show_previous_image(self):
        self.camera.get_status.return_value = CameraStatus(
            running=True, connected=True, description="Test camera", serial_number="24346077",
            frames_received=5, last_frame_at=monotonic() - 3, error=None,
        )
        image, status, _, _ = self.controller.preview()
        self.assertIsNone(image)
        self.assertIn("수신 대기", status)

    def test_invalid_settings_keep_current_camera_running(self):
        result = self.controller.apply_camera("24346077", -1, 10)
        self.assertIn("설정 오류", result)
        self.camera.stop.assert_not_called()

    def test_live_roi_uses_fresh_camera_pixels_and_keeps_the_extracted_mask(self):
        mask = np.array([[True, True, False, False]] * 3)
        self.controller.roi_result = AutoRoiResult(mask, {"roi_detected": True})
        with patch("src.diagnostics.camera.detect_auto_roi") as detector:
            for value in (80, 170):
                self.camera.get_latest_frame.return_value = np.full_like(self.rgb, value)
                image, info = self.controller.live_roi_preview()
                np.testing.assert_array_equal(image, np.where(mask[:, :, None], value, 0)
                                              .repeat(3, axis=2))
                self.assertIn("실시간", info)
            detector.assert_not_called()
        np.testing.assert_array_equal(self.controller.roi_result.mask, mask)
        self.light.send.assert_not_called()

    def test_live_roi_waits_for_a_mask_and_rejects_stale_or_different_size_frames(self):
        image, info = self.controller.live_roi_preview()
        self.assertIsNone(image)
        self.assertIn("ROI를 추출", info)
        self.controller.roi_result = AutoRoiResult(np.ones((3, 4), bool), {"roi_detected": True})
        fresh = self.camera.get_status.return_value
        for connected, last_frame_at in ((False, monotonic()), (True, monotonic() - 3), (True, None)):
            self.camera.get_status.return_value = CameraStatus(
                running=True, connected=connected, description="Test camera", serial_number="24346077",
                frames_received=5, last_frame_at=last_frame_at, error=None,
            )
            self.assertIsNone(self.controller.live_roi_preview()[0])
        self.camera.get_status.return_value = fresh
        self.camera.get_latest_frame.return_value = np.full((4, 4, 3), 100, np.uint8)
        image, info = self.controller.live_roi_preview()
        self.assertIsNone(image)
        self.assertIn("ROI를 다시 추출", info)

    def test_camera_connection_failure_retries_with_backoff(self):
        self.controller._camera_enabled = True
        self.controller._next_camera_retry = 0
        self.camera.get_status.return_value = CameraStatus(
            running=False, connected=False, description="Test camera", serial_number="24346077",
            frames_received=0, last_frame_at=None, error="device busy",
        )
        self.controller.preview()
        self.controller.preview()
        self.camera.start.assert_called_once()

    def test_manual_camera_disconnect_stays_disconnected(self):
        self.controller._camera_enabled = True
        self.controller.disconnect_camera()
        self.camera.get_status.return_value = CameraStatus(
            running=False, connected=False, description="Test camera", serial_number="24346077",
            frames_received=0, last_frame_at=None, error=None,
        )
        self.controller.preview()
        self.camera.start.assert_not_called()

    def test_port_change_requires_reconnect_before_sending(self):
        result = self.controller.send_light(((1, 10),), "COM4", 9600)
        self.assertIn("먼저 RS232 연결", result)
        self.light.send.assert_not_called()

    def test_all_channels_are_validated_before_transmission(self):
        result = self.controller.send_light(((1, 10), (2, float("nan"))), "COM3", 9600)
        self.assertIn("전송 실패", result)
        self.light.send.assert_not_called()

    def test_single_channel_receipt_requires_actual_ack(self):
        result = self.controller.send_light(((2, 10),), "COM3", 9600)
        self.assertIn("ACK 확인 · CH2", result)
        self.assertEqual(self.light.send.call_args.args[0][0].packet, b"\x022026\x03")
        self.light.send.side_effect = LightCommunicationError("no ACK received")
        failed = self.controller.send_light(((2, 10),), "COM3", 9600)
        self.assertIn("전송 실패", failed)
        self.assertNotIn("ACK 확인", failed)

    def test_start_connects_without_sending_brightness(self):
        self.controller.start()
        self.camera.start.assert_called_once()
        self.light.connect.assert_called_once()
        self.light.send.assert_not_called()

    def test_group_apply_all_expands_four_values_to_six_channels(self):
        result = self.controller.send_light_groups((10, 20, 30, 40), "COM3", 9600)
        sent = self.light.send.call_args.args[0]
        self.assertEqual([(item.channel, item.percent) for item in sent],
                         [(1, 10), (2, 10), (3, 20), (4, 30), (5, 40), (6, 40)])
        self.assertIn("ACK 확인 · CH6", result)
        notes = self.controller.light_receipts()
        self.assertEqual(len(notes), 4)
        self.assertIn("CH2 10%", notes[0])
        self.assertIn("CH6 40%", notes[3])

    def test_invalid_last_group_is_rejected_before_any_transmission(self):
        result = self.controller.send_light_groups((10, 20, 30, float("nan")), "COM3", 9600)
        self.assertIn("전송 실패", result)
        self.light.send.assert_not_called()

    def test_panel_partial_failure_never_claims_both_channels_acknowledged(self):
        self.light.send.side_effect = LightCommunicationError(
            "CH6: no ACK received", acknowledged=(build_brightness_packet(5, 40),),
        )
        result = self.controller.send_light_group(4, 40, "COM3", 9600)
        self.assertIn("전송 실패", result)
        self.assertIn("이번 요청의 ACK 확인: CH5", result)
        notes = self.controller.light_receipts()
        self.assertIn("부분 적용", notes[3])
        self.assertIn("CH5 40%", notes[3])
        self.assertNotIn("CH6 40%", notes[3])

    def test_roi_extraction_does_not_block_the_live_preview(self):
        started, release = Event(), Event()
        detected = AutoRoiResult(np.ones((3, 4), bool), {
            "source_shape": [3, 4], "roi_detected": True, "quality_status": "PASS",
            "quality_score": .8, "needs_review_reason": "",
            "top_y": [0] * 4, "bottom_y_exclusive": [3] * 4,
        })

        def slow_detector(*args, **kwargs):
            started.set()
            if not release.wait(3):
                raise RuntimeError("test worker did not release detector")
            return detected

        with patch("src.diagnostics.camera.detect_auto_roi", side_effect=slow_detector), ThreadPoolExecutor(2) as pool:
            extraction = pool.submit(self.controller.extract_roi)
            try:
                self.assertTrue(started.wait(1))
                preview = pool.submit(self.controller.preview).result(timeout=1)
                self.assertEqual(preview[0].shape, (3, 4, 3))
            finally:
                release.set()
            _, message = extraction.result(timeout=2)
        self.assertIn("PASS", message)
        self.light.send.assert_not_called()

    def test_roi_pixels_are_not_amplified(self):
        mask = np.array([[True, True, False, False]] * 3)
        detected = AutoRoiResult(mask, {
            "source_shape": [3, 4], "roi_detected": True, "quality_status": "PASS",
            "quality_score": .8, "needs_review_reason": "",
            "top_y": [0] * 4, "bottom_y_exclusive": [3] * 4,
        })
        with patch("src.diagnostics.camera.detect_auto_roi", return_value=detected):
            image, _ = self.controller.extract_roi()
        np.testing.assert_array_equal(image, np.where(mask[:, :, None], self.rgb, 0))

    def test_save_image_preserves_color_and_saves_matching_single_channel_mono(self):
        with tempfile.TemporaryDirectory() as folder:
            self.controller.capture_dir = Path(folder) / "카메라 저장"
            rgb = np.tile(np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]],
                                   dtype=np.uint8), (3, 1, 1))
            self.camera.get_latest_frame.return_value = rgb
            first, message = self.controller.save_image()
            second, _ = self.controller.save_image()
            color_path, mono_path = map(Path, first)
            self.assertIsNotNone(first)
            self.assertTrue(color_path.is_file())
            self.assertTrue(mono_path.is_file())
            self.assertEqual(len(first), 2)
            self.assertTrue(set(first).isdisjoint(second))
            self.assertEqual(color_path.name.removesuffix("_Color.png"),
                             mono_path.name.removesuffix("_Mono.png"))
            self.assertIn("4 × 3px", message)
            self.assertIn("2장 저장 완료", message)
            color = cv2.imdecode(np.frombuffer(color_path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
            mono = cv2.imdecode(np.frombuffer(mono_path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(cv2.cvtColor(color, cv2.COLOR_BGR2RGB), rgb)
            np.testing.assert_array_equal(mono, np.tile(np.array([76, 150, 29, 255], np.uint8), (3, 1)))
            self.assertEqual(mono.dtype, np.uint8)
            self.assertEqual(self.camera.get_latest_frame.call_count, 2)
        self.light.send.assert_not_called()
        self.camera.start.assert_not_called()
        self.camera.set_exposure_us.assert_not_called()

    def test_save_image_refuses_disconnected_or_stale_frames_and_clears_download(self):
        with tempfile.TemporaryDirectory() as folder:
            self.controller.capture_dir = Path(folder) / "captures"
            for connected, last_frame_at in ((False, monotonic()), (True, monotonic() - 3), (True, None)):
                self.camera.get_status.return_value = CameraStatus(
                    running=True, connected=connected, description="Test camera", serial_number="24346077",
                    frames_received=5, last_frame_at=last_frame_at, error=None,
                )
                download, message = self.controller.save_image()
                self.assertIsNone(download)
                self.assertIn("저장 실패", message)
            self.assertFalse(self.controller.capture_dir.exists())

    def test_named_image_uses_entered_name_and_never_overwrites_previous_capture(self):
        with tempfile.TemporaryDirectory() as folder:
            self.controller.capture_dir = Path(folder) / "이미지 저장"
            first, _ = self.controller.save_image("Case01_시편01")
            paths = list(map(Path, first))
            original_bytes = [path.read_bytes() for path in paths]
            self.camera.get_latest_frame.return_value[:] = 0
            second, _ = self.controller.save_image("Case01_시편01.png")
            third, _ = self.controller.save_image("Case01_시편01.PNG")
            self.assertEqual([path.name for path in paths],
                             ["Case01_시편01_Color.png", "Case01_시편01_Mono.png"])
            self.assertEqual([Path(path).name for path in second],
                             ["Case01_시편01_1_Color.png", "Case01_시편01_1_Mono.png"])
            self.assertEqual([Path(path).name for path in third],
                             ["Case01_시편01_2_Color.png", "Case01_시편01_2_Mono.png"])
            self.assertEqual([path.read_bytes() for path in paths], original_bytes)
            for path in second:
                saved = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
                self.assertTrue((saved == 0).all())

    def test_existing_mono_file_reserves_the_name_for_both_images(self):
        with tempfile.TemporaryDirectory() as folder:
            self.controller.capture_dir = Path(folder)
            first, _ = self.controller.save_image("시편")
            color_path, mono_path = map(Path, first)
            original_mono = mono_path.read_bytes()
            color_path.unlink()
            second, _ = self.controller.save_image("시편")
            self.assertEqual([Path(path).name for path in second],
                             ["시편_1_Color.png", "시편_1_Mono.png"])
            self.assertFalse(color_path.exists())
            self.assertEqual(mono_path.read_bytes(), original_mono)
            self.assertTrue(all(Path(path).is_file() for path in second))

    def test_named_image_rejects_paths_and_invalid_windows_names(self):
        with tempfile.TemporaryDirectory() as folder:
            self.controller.capture_dir = Path(folder) / "captures"
            for name in ("../escape", "..\\escape", "C:test", "folder/name", "bad?name",
                         "CON.png", "NUL", "LPT1.png", "line\nbreak", "case.", ".png", None):
                with self.subTest(name=name):
                    download, message = self.controller.save_image(name)
                    self.assertIsNone(download)
                    self.assertIn("저장 실패", message)
            self.assertFalse(self.controller.capture_dir.exists())

    def test_save_failure_removes_partial_image_and_does_not_offer_old_file(self):
        for completed_files in (0, 1):
            with self.subTest(completed_files=completed_files), tempfile.TemporaryDirectory() as folder:
                self.controller.capture_dir = Path(folder)
                with patch("src.diagnostics.camera.os.fsync", side_effect=[None] * completed_files + [OSError("disk full")]):
                    download, message = self.controller.save_image()
                self.assertIsNone(download)
                self.assertIn("disk full", message)
                self.assertEqual(list(Path(folder).rglob("*.png")), [])

    def test_mono_encoding_failure_does_not_save_color_alone(self):
        encoder = cv2.imencode

        def fail_mono(extension, image):
            return (False, None) if image.ndim == 2 else encoder(extension, image)

        with tempfile.TemporaryDirectory() as folder:
            self.controller.capture_dir = Path(folder) / "captures"
            with patch("src.diagnostics.camera.cv2.imencode", side_effect=fail_mono):
                download, message = self.controller.save_image("시편")
            self.assertIsNone(download)
            self.assertIn("Mono PNG", message)
            self.assertFalse(self.controller.capture_dir.exists())

    def test_image_save_does_not_block_preview_or_change_snapshot_pixels(self):
        started, release = Event(), Event()
        original = self.camera.get_latest_frame.return_value.copy()
        encoder = cv2.imencode

        def slow_encoder(*args, **kwargs):
            started.set()
            if not release.wait(3):
                raise RuntimeError("test worker did not release encoder")
            return encoder(*args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            self.controller.capture_dir = Path(folder)
            with patch("src.diagnostics.camera.cv2.imencode", side_effect=slow_encoder), ThreadPoolExecutor(2) as pool:
                saving = pool.submit(self.controller.save_image)
                try:
                    self.assertTrue(started.wait(1))
                    self.camera.get_latest_frame.return_value[:] = 0
                    preview = pool.submit(self.controller.preview).result(timeout=1)
                    self.assertEqual(preview[0].shape, (3, 4, 3))
                finally:
                    release.set()
                download, _ = saving.result(timeout=2)
            color_path, mono_path = map(Path, download)
            color = cv2.imdecode(np.frombuffer(color_path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
            mono = cv2.imdecode(np.frombuffer(mono_path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(cv2.cvtColor(color, cv2.COLOR_BGR2RGB), original)
            np.testing.assert_array_equal(mono, np.full((3, 4), 100, np.uint8))

    def test_shutdown_releases_both_devices(self):
        self.camera.stop.side_effect = RuntimeError("camera cleanup failed")
        with self.assertRaises(RuntimeError):
            self.controller.close()
        self.light.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
