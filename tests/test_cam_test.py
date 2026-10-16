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

from Cam_test import CamTestController, build_ui
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

    def test_group_buttons_route_to_same_physical_fixtures_as_main_ui(self):
        with patch("Cam_test.list_ports.comports", return_value=[]):
            demo = build_ui(self.controller)
        callbacks = {fn.api_name: fn.fn for fn in demo.fns.values()}
        expected = {
            1: (b"\x021026\x03", b"\x022026\x03"),
            2: (b"\x023026\x03",),
            3: (b"\x024026\x03",),
            4: (b"\x025026\x03", b"\x026026\x03"),
        }
        for group, packets in expected.items():
            with self.subTest(group=group):
                self.light.send.reset_mock()
                result = callbacks[f"light_group_{group}"](10, "COM3", 9600)
                self.light.send.assert_called_once()
                self.assertEqual(tuple(item.packet for item in self.light.send.call_args.args[0]), packets)
                self.assertIn("ACK 확인", result[0])
                self.assertIn("최근 ACK 확인", result[group])
                for index, note in enumerate(result[1:], 1):
                    if index != group:
                        self.assertEqual(note, {"__type__": "update"})
        self.camera.start.assert_not_called()

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

    def test_all_off_writes_six_channels_and_resets_four_group_sliders(self):
        with patch("Cam_test.list_ports.comports", return_value=[]):
            demo = build_ui(self.controller)
        callback = next(fn.fn for fn in demo.fns.values() if fn.api_name == "light_off")
        result = callback("COM3", 9600)
        self.assertEqual(result[-4:], (0, 0, 0, 0))
        self.assertEqual([(item.channel, item.percent) for item in self.light.send.call_args.args[0]],
                         [(channel, 0) for channel in range(1, 7)])
        self.assertIn("CH6", result[0])

    def test_partial_off_only_resets_fully_acknowledged_fixtures(self):
        self.light.send.side_effect = LightCommunicationError(
            "CH6: no ACK received", acknowledged=tuple(build_brightness_packet(channel, 0) for channel in range(1, 6)),
        )
        result = self.controller.turn_off_lights("COM3", 9600)
        self.assertEqual(result[-4:-1], (0, 0, 0))
        self.assertEqual(result[-1], {"__type__": "update"})
        self.assertIn("부분 적용", result[4])
        self.assertNotIn("CH6 0%", result[4])

    def test_building_light_group_ui_never_transmits(self):
        with patch("Cam_test.list_ports.comports", return_value=[]):
            build_ui(self.controller)
        self.light.send.assert_not_called()
        self.light.connect.assert_not_called()

    def test_exposure_replaces_display_gain_in_ui_and_preview_api(self):
        with patch("Cam_test.list_ports.comports", return_value=[]):
            demo = build_ui(self.controller)
        components = demo.config["components"]
        labels = [component["props"].get("label", "") for component in components]
        self.assertIn("카메라 노출 시간 (µs)", labels)
        self.assertFalse(any("밝기 배율" in label for label in labels))
        callbacks = {fn.api_name: fn for fn in demo.fns.values()}
        self.assertEqual(len(callbacks["preview"].inputs), 1)
        self.assertEqual(len(callbacks["extract_roi"].inputs), 0)
        self.assertEqual(callbacks["exposure_apply"].fn, self.controller.apply_exposure)
        self.assertEqual(callbacks["exposure_load"].fn, self.controller.load_exposure)
        self.camera.set_exposure_us.assert_not_called()

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

        with patch("Cam_test.detect_auto_roi", side_effect=slow_detector), ThreadPoolExecutor(2) as pool:
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
        with patch("Cam_test.detect_auto_roi", return_value=detected):
            image, _ = self.controller.extract_roi()
        np.testing.assert_array_equal(image, np.where(mask[:, :, None], self.rgb, 0))

    def test_save_image_preserves_native_rgb_and_uses_unique_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            self.controller.capture_dir = Path(folder) / "카메라 저장"
            rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
            self.camera.get_latest_frame.return_value = rgb
            first, message = self.controller.save_image()
            second, _ = self.controller.save_image()
            path = Path(first["value"])
            self.assertTrue(first["visible"])
            self.assertTrue(path.is_file())
            self.assertNotEqual(first["value"], second["value"])
            self.assertIn("4 × 3px", message)
            self.assertIn("저장 완료", message)
            saved = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(cv2.cvtColor(saved, cv2.COLOR_BGR2RGB), rgb)
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
                self.assertIsNone(download["value"])
                self.assertFalse(download["visible"])
                self.assertIn("저장 실패", message)
            self.assertFalse(self.controller.capture_dir.exists())

    def test_named_image_uses_entered_name_and_never_overwrites_previous_capture(self):
        with tempfile.TemporaryDirectory() as folder:
            self.controller.capture_dir = Path(folder) / "이미지 저장"
            first, _ = self.controller.save_image("Case01_시편01")
            path = Path(first["value"])
            original_bytes = path.read_bytes()
            self.camera.get_latest_frame.return_value[:] = 0
            second, _ = self.controller.save_image("Case01_시편01.png")
            third, _ = self.controller.save_image("Case01_시편01.PNG")
            self.assertEqual(path.name, "Case01_시편01.png")
            self.assertEqual(Path(second["value"]).name, "Case01_시편01_1.png")
            self.assertEqual(Path(third["value"]).name, "Case01_시편01_2.png")
            self.assertEqual(path.read_bytes(), original_bytes)
            saved = cv2.imdecode(np.frombuffer(Path(second["value"]).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            self.assertTrue((saved == 0).all())

    def test_named_image_rejects_paths_and_invalid_windows_names(self):
        with tempfile.TemporaryDirectory() as folder:
            self.controller.capture_dir = Path(folder) / "captures"
            for name in ("../escape", "..\\escape", "C:test", "folder/name", "bad?name",
                         "CON.png", "NUL", "LPT1.png", "line\nbreak", "case.", ".png", None):
                with self.subTest(name=name):
                    download, message = self.controller.save_image(name)
                    self.assertFalse(download["visible"])
                    self.assertIsNone(download["value"])
                    self.assertIn("저장 실패", message)
            self.assertFalse(self.controller.capture_dir.exists())

    def test_save_failure_removes_partial_image_and_does_not_offer_old_file(self):
        with tempfile.TemporaryDirectory() as folder:
            self.controller.capture_dir = Path(folder)
            with patch("Cam_test.os.fsync", side_effect=OSError("disk full")):
                download, message = self.controller.save_image()
            self.assertIsNone(download["value"])
            self.assertFalse(download["visible"])
            self.assertIn("disk full", message)
            self.assertEqual(list(Path(folder).rglob("*.png")), [])

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
            with patch("Cam_test.cv2.imencode", side_effect=slow_encoder), ThreadPoolExecutor(2) as pool:
                saving = pool.submit(self.controller.save_image)
                try:
                    self.assertTrue(started.wait(1))
                    self.camera.get_latest_frame.return_value[:] = 0
                    preview = pool.submit(self.controller.preview).result(timeout=1)
                    self.assertEqual(preview[0].shape, (3, 4, 3))
                finally:
                    release.set()
                download, _ = saving.result(timeout=2)
            saved = cv2.imdecode(np.frombuffer(Path(download["value"]).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(cv2.cvtColor(saved, cv2.COLOR_BGR2RGB), original)

    def test_save_button_is_connected_to_snapshot_callback(self):
        with patch("Cam_test.list_ports.comports", return_value=[]):
            demo = build_ui(self.controller)
        callbacks = {fn.api_name: fn for fn in demo.fns.values()}
        self.assertEqual(callbacks["save_image"].fn, self.controller.save_image)
        self.assertEqual(len(callbacks["save_image"].inputs), 1)
        self.assertEqual(callbacks["save_image"].inputs[0].label, "이미지 이름")
        self.assertEqual(len(callbacks["save_image"].outputs), 2)

    def test_shutdown_releases_both_devices(self):
        self.camera.stop.side_effect = RuntimeError("camera cleanup failed")
        with self.assertRaises(RuntimeError):
            self.controller.close()
        self.light.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
