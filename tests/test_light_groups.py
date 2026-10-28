"""Physical fixture routing through the FastAPI workbench service."""

from threading import RLock
import unittest
from unittest.mock import Mock, patch

from src.hardware import led_control as lights
from src.web.service import Workbench


class LightGroupTests(unittest.TestCase):
    def setUp(self):
        self.port = Mock(is_open=False)
        self.port.open.side_effect = lambda: setattr(self.port, "is_open", True)
        self.port.close.side_effect = lambda: setattr(self.port, "is_open", False)
        self.port.read.return_value = lights.ACK
        factory = patch.object(lights.serial, "Serial", return_value=self.port)
        factory.start()
        self.addCleanup(factory.stop)
        self.workbench = Workbench.__new__(Workbench)
        self.workbench.lock = RLock()
        self.workbench.light = lights.LightController("COM3", 9600)
        self.addCleanup(self.workbench.light.disconnect)

    def test_physical_assignment_matches_light_document(self):
        self.assertEqual([group.channels for group in lights.LIGHT_GROUPS],
                         [(1, 2), (3,), (4,), (5, 6)])

    def test_group_buttons_only_send_assigned_channels(self):
        expected = {1: (1, 2), 2: (3,), 3: (4,), 4: (5, 6)}
        for group, channels in expected.items():
            with self.subTest(group=group):
                self.port.write.reset_mock()
                result = self.workbench.apply_lighting([10, 20, 30, 40], group)
                self.assertTrue(result["success"])
                self.assertEqual([call.args[0][1] - 48 for call in self.port.write.call_args_list],
                                 list(channels))
                for channel in channels:
                    self.assertIn(f"CH{channel}", result["receipts"][str(group)])

    def test_apply_all_maps_four_values_to_six_outputs(self):
        result = self.workbench.apply_lighting([10, 20, 30, 40])
        self.assertTrue(result["success"])
        self.assertEqual([call.args[0] for call in self.port.write.call_args_list],
                         [b"\x021026\x03", b"\x022026\x03", b"\x023051\x03",
                          b"\x024076\x03", b"\x025102\x03", b"\x026102\x03"])
        self.assertIn("CH2 10%", result["receipts"]["1"])
        self.assertIn("CH6 40%", result["receipts"]["4"])

    def test_invalid_last_fixture_sends_nothing(self):
        result = self.workbench.apply_lighting([10, 20, 30, float("nan")])
        self.assertFalse(result["success"])
        self.port.write.assert_not_called()

    def test_partial_panel_ack_is_reported(self):
        self.port.read.side_effect = [lights.ACK, b""]
        result = self.workbench.apply_lighting([10, 20, 30, 40], 4)
        self.assertFalse(result["success"])
        self.assertIn("CH5", result["message"])
        self.assertIn("부분 적용", result["receipts"]["4"])
        self.assertNotIn("CH6 40%", result["receipts"]["4"])
        self.assertEqual(self.port.write.call_count, 2)

    def test_first_panel_failure_stops_remaining_channels(self):
        self.port.read.side_effect = [lights.ACK, b""]
        result = self.workbench.apply_lighting([10, 20, 30, 40])
        self.assertFalse(result["success"])
        self.assertEqual(self.port.write.call_count, 2)
        self.assertIn("부분 적용", result["receipts"]["1"])
        self.assertEqual(result["receipts"]["2"], "최근 ACK: 미확인")


if __name__ == "__main__":
    unittest.main()
