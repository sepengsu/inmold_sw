"""Shared persisted exposure and camera application, without opening devices."""

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import Mock, patch

import yaml

from Cam_test import CamTestController
from src.hardware.camera_config import DEFAULT_CAMERA_CONFIG, load_camera_settings, save_camera_exposure
from src.hardware.camera_control import BaslerCameraSettings, CameraControl, CameraError, CameraStatus


class CameraConfigTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "camera.yaml"
        self.path.write_text("exposure_us: 10000\n", encoding="utf-8")

    def test_project_default_is_ten_milliseconds(self):
        settings = load_camera_settings()
        self.assertTrue(DEFAULT_CAMERA_CONFIG.is_absolute())
        self.assertEqual(settings.exposure_us, 10000)
        self.assertEqual(settings, BaslerCameraSettings())

    def test_custom_config_changes_only_exposure(self):
        self.path.write_text("exposure_us: 25000\n", encoding="utf-8")
        settings = load_camera_settings(self.path)
        self.assertEqual(settings, replace(BaslerCameraSettings(), exposure_us=25000))

    def test_invalid_config_is_rejected_without_rewriting_file(self):
        documents = ["", "[]", "[invalid", "exposure: 10000\n", "exposure_us: 10000\nunknown: 1\n"]
        for value in (0, -1, float("nan"), float("inf"), True, "10000", None, 10 ** 400):
            documents.append(yaml.safe_dump({"exposure_us": value}))
        for text in documents:
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_camera_settings(self.path)
                self.assertEqual(self.path.read_text(encoding="utf-8"), text)

    def test_missing_file_is_reported_without_creating_it(self):
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, "camera.yaml"):
            load_camera_settings(self.path)
        self.assertFalse(self.path.exists())

    def test_camera_test_initial_settings_come_from_config(self):
        self.path.write_text("exposure_us: 16000\n", encoding="utf-8")
        with patch("Cam_test.CameraControl") as camera:
            controller = CamTestController("24346077", camera_config=self.path)
        self.assertEqual(controller.settings.exposure_us, 16000)
        camera.assert_called_once_with("24346077", settings=controller.settings)
        camera.return_value.start.assert_not_called()

    def test_explicit_camera_settings_take_precedence(self):
        settings = BaslerCameraSettings(exposure_us=7000)
        with patch("Cam_test.CameraControl"):
            controller = CamTestController(settings=settings, camera_config=self.path)
        self.assertIs(controller.settings, settings)
        self.assertEqual(load_camera_settings(self.path).exposure_us, 10000)

    def test_exposure_application_persists_and_reloads_for_next_start(self):
        controller = CamTestController("24346077", camera_config=self.path)
        with patch.object(controller.camera, "set_exposure_us", return_value=12000) as apply, \
                patch.object(controller.camera, "start") as start, \
                patch.object(controller.camera, "stop") as stop:
            value, _, message = controller.apply_exposure(12000)
        self.assertEqual(value, 12000)
        self.assertIn("저장 완료", message)
        apply.assert_called_once_with(12000)
        self.assertEqual(controller.settings.exposure_us, 12000)
        self.assertEqual(load_camera_settings(self.path).exposure_us, 12000)
        self.assertEqual(CamTestController(camera_config=self.path).settings.exposure_us, 12000)
        start.assert_not_called()
        stop.assert_not_called()

    def test_camera_connection_uses_persisted_exposure(self):
        save_camera_exposure(12000, self.path)
        controller = CamTestController("24346077", camera_config=self.path)
        replacement_camera = Mock()
        replacement_camera.get_status.return_value = CameraStatus(
            running=True, connected=False, description="Test camera", serial_number="24346077",
            frames_received=0, last_frame_at=None, error=None,
        )
        with patch.object(controller.camera, "stop"), patch("Cam_test.CameraControl", return_value=replacement_camera):
            result = controller.apply_camera("24346077", 0, 10)
        self.assertEqual(controller.settings.exposure_us, 12000)
        self.assertIn("12000 µs", result)
        self.assertEqual(load_camera_settings(self.path).exposure_us, 12000)
        replacement_camera.start.assert_called_once()

    def test_camera_adjusted_exposure_is_used_in_ui_and_config(self):
        controller = CamTestController(camera_config=self.path)
        with patch.object(controller.camera, "set_exposure_us", return_value=15000):
            value, _, message = controller.apply_exposure(20000)
        self.assertEqual(value, 15000)
        self.assertEqual(controller.settings.exposure_us, 15000)
        self.assertEqual(load_camera_settings(self.path).exposure_us, 15000)
        self.assertIn("요청값 20000", message)

    def test_invalid_exposure_never_touches_camera_or_saved_file(self):
        controller = CamTestController(camera_config=self.path)
        original = self.path.read_bytes()
        with patch.object(controller.camera, "set_exposure_us") as apply:
            for value in (None, True, "10000", 0, -1, float("nan"), float("inf"), 10 ** 400):
                with self.subTest(value=value):
                    result = controller.apply_exposure(value)
                    self.assertEqual(result[0], {"__type__": "update"})
                    self.assertIn("적용 실패", result[2])
        apply.assert_not_called()
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(controller.settings.exposure_us, 10000)

    def test_device_failure_keeps_saved_and_controller_settings(self):
        controller = CamTestController(camera_config=self.path)
        original = self.path.read_bytes()
        with patch.object(controller.camera, "set_exposure_us", side_effect=CameraError("device rejected exposure")):
            value, _, message = controller.apply_exposure(12000)
        self.assertEqual(value, {"__type__": "update"})
        self.assertIn("적용 실패", message)
        self.assertEqual(controller.settings.exposure_us, 10000)
        self.assertEqual(self.path.read_bytes(), original)

    def test_atomic_save_failure_preserves_file_and_reports_applied_value(self):
        controller = CamTestController(camera_config=self.path)
        original = self.path.read_bytes()
        with patch.object(controller.camera, "set_exposure_us", return_value=12000), \
                patch("src.hardware.camera_config.os.replace", side_effect=PermissionError("file busy")):
            value, _, message = controller.apply_exposure(12000)
        self.assertEqual(value, 12000)
        self.assertEqual(controller.settings.exposure_us, 12000)
        self.assertIn("저장 실패", message)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_load_updates_only_input_value_and_reports_invalid_file(self):
        controller = CamTestController(camera_config=self.path)
        save_camera_exposure(18000.5, self.path)
        with patch.object(controller.camera, "set_exposure_us") as apply:
            value, message = controller.load_exposure()
            self.assertEqual(value, 18000.5)
            self.assertIn("불러오기 완료", message)
            self.assertEqual(controller.settings.exposure_us, 10000)
            self.path.write_text("exposure_us: invalid\n", encoding="utf-8")
            value, message = controller.load_exposure()
            self.assertEqual(value, {"__type__": "update"})
            self.assertIn("불러오기 실패", message)
        apply.assert_not_called()

    def test_save_validation_precedes_file_changes(self):
        original = self.path.read_bytes()
        for value in (True, "10000", None, 0, -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    save_camera_exposure(value, self.path)
        self.assertEqual(self.path.read_bytes(), original)

    def test_disconnected_camera_retains_exposure_without_starting_device(self):
        camera = CameraControl()
        with patch.object(camera, "start") as start, patch.object(camera, "_require_camera") as device:
            self.assertEqual(camera.set_exposure_us(12500), 12500)
        self.assertEqual(camera.settings.exposure_us, 12500)
        start.assert_not_called()
        device.assert_not_called()

    def test_connected_camera_updates_sensor_without_reconnecting(self):
        camera = CameraControl()
        camera._connected = True
        device = Mock()
        with patch.object(camera, "_require_camera", return_value=device), \
                patch.object(camera, "_set_enum") as set_enum, \
                patch.object(camera, "_set_number", return_value=12500) as set_number, \
                patch.object(camera, "start") as start, patch.object(camera, "stop") as stop:
            self.assertEqual(camera.set_exposure_us(12500.4), 12500)
        self.assertEqual(camera.settings.exposure_us, 12500)
        set_enum.assert_called_once_with(device, "ExposureAuto", "Off", required=False)
        set_number.assert_called_once_with(device, ("ExposureTime", "ExposureTimeAbs"), 12500.4)
        start.assert_not_called()
        stop.assert_not_called()

    def test_camera_write_error_does_not_update_requested_settings(self):
        camera = CameraControl()
        camera._connected = True
        with patch.object(camera, "_require_camera", side_effect=RuntimeError("device lost")):
            with self.assertRaises(CameraError):
                camera.set_exposure_us(12500)
        self.assertEqual(camera.settings.exposure_us, 10000)

    def test_number_setter_returns_camera_readback_after_clamping(self):
        camera = CameraControl()
        node = Mock(Min=100, Max=20000, Value=12000.25)
        node.GetNode.return_value.GetPrincipalInterfaceType.return_value = "float"
        genicam = Mock(intfIInteger="integer")
        with patch("src.hardware.camera_control._load_pylon", return_value=(Mock(), genicam)), \
                patch.object(camera, "_get_node", return_value=node):
            value = camera._set_number(Mock(), ("ExposureTime",), 50000)
        node.SetValue.assert_called_once_with(20000)
        self.assertEqual(value, 12000.25)

    def test_device_close_waits_for_exposure_write_to_finish(self):
        camera = CameraControl()
        camera._connected = True
        device = Mock()
        camera._camera = device
        started, release = Event(), Event()

        def set_number(*args, **kwargs):
            started.set()
            if not release.wait(2):
                raise RuntimeError("exposure writer was not released")
            return 12000

        with patch.object(camera, "_require_camera", return_value=device), \
                patch.object(camera, "_set_enum"), \
                patch.object(camera, "_set_number", side_effect=set_number), ThreadPoolExecutor(2) as pool:
            exposure = pool.submit(camera.set_exposure_us, 12000)
            try:
                self.assertTrue(started.wait(1))
                close = pool.submit(camera._close_camera)
                device.Close.assert_not_called()
            finally:
                release.set()
            self.assertEqual(exposure.result(timeout=2), 12000)
            close.result(timeout=2)
        device.Close.assert_called_once()

    def test_loaded_exposure_reaches_camera_with_auto_exposure_off(self):
        camera = CameraControl(settings=load_camera_settings(self.path))
        device = Mock()
        with patch.object(camera, "_require_camera", return_value=device), \
                patch.object(camera, "_set_enum") as set_enum, \
                patch.object(camera, "_set_number") as set_number, \
                patch.object(camera, "_set_bool"), patch.object(camera, "_set_node_to_max"):
            camera._apply_settings()
        set_enum.assert_any_call(device, "ExposureAuto", "Off", required=False)
        set_number.assert_any_call(device, ("ExposureTime", "ExposureTimeAbs"), 10000)

    def test_main_ui_passes_config_exposure_to_camera_on_startup(self):
        from src.gui import app

        self.path.write_text("exposure_us: 18000\n", encoding="utf-8")
        camera = Mock()
        with patch.multiple(app, camera_control=Mock(), inspection_engine=None, model_error=None, camera_enabled=False), \
                patch.object(app, "CameraControl", return_value=camera) as factory, \
                patch.object(app, "connect_light"), patch.object(app, "disconnect_light"), \
                patch.object(app.demo, "launch") as launch, \
                patch("src.gui.inspection.InspectionEngine"):
            app.launchApp(camera_serial="24346077", camera_config_path=self.path, inbrowser=False)
        self.assertEqual(factory.call_args.kwargs["settings"].exposure_us, 18000)
        self.assertEqual(factory.call_args.kwargs["serial_number"], "24346077")
        camera.start.assert_called_once()
        camera.stop.assert_called_once()
        launch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
