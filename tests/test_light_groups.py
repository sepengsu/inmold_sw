"""Verify the four physical fixtures route to their assigned six channels."""

import unittest
from unittest.mock import Mock, patch

from src.hardware import led_control as lights


class LightGroupTests(unittest.TestCase):
    def setUp(self):
        from src.gui import app

        self.app = app
        self.port = Mock(is_open=False)
        self.port.open.side_effect = lambda: setattr(self.port, "is_open", True)
        self.port.close.side_effect = lambda: setattr(self.port, "is_open", False)
        self.port.read.return_value = lights.ACK
        factory = patch.object(lights.serial, "Serial", return_value=self.port)
        self.factory = factory.start()
        self.addCleanup(factory.stop)
        self.controller = lights.LightController("COM3", 9600)
        controller_patch = patch.object(lights, "_controller", self.controller)
        controller_patch.start()
        self.addCleanup(controller_patch.stop)
        self.addCleanup(self.controller.disconnect)

    def test_physical_assignment_matches_light_document(self):
        self.assertEqual(
            [(group.name, group.channels) for group in lights.LIGHT_GROUPS],
            [("왼쪽 아래 면조명", (1, 2)), ("왼쪽 위 바 조명", (3,)),
             ("오른쪽 위 바 조명", (4,)), ("오른쪽 위 면조명", (5, 6))],
        )

    def test_each_main_ui_button_only_writes_its_fixture_channels(self):
        callbacks = {fn.api_name: fn.fn for fn in self.app.demo.fns.values()}
        expected = {
            1: (b"\x021026\x03", b"\x022026\x03"),
            2: (b"\x023051\x03",),
            3: (b"\x024076\x03",),
            4: (b"\x025102\x03", b"\x026102\x03"),
        }
        for group, packets in expected.items():
            with self.subTest(group=group):
                self.port.write.reset_mock()
                self.port.read.reset_mock()
                output = callbacks[f"lighting_group_{group}"](10, 20, 30, 40)
                self.assertEqual(tuple(call.args[0] for call in self.port.write.call_args_list), packets)
                self.assertEqual(self.port.read.call_count, len(packets))
                self.assertIn(lights.LIGHT_GROUPS[group - 1].name, output[2])
                self.assertIn("적용 완료", output[2])
                for channel in lights.LIGHT_GROUPS[group - 1].channels:
                    self.assertIn(f"CH{channel}", output[3])

    def test_apply_all_maps_four_values_to_all_six_outputs(self):
        output = self.app.applyLighting(10, 20, 30, 40)
        self.assertEqual(
            [call.args[0] for call in self.port.write.call_args_list],
            [b"\x021026\x03", b"\x022026\x03", b"\x023051\x03",
             b"\x024076\x03", b"\x025102\x03", b"\x026102\x03"],
        )
        self.assertEqual(self.port.read.call_count, 6)
        self.assertFalse(output[2]["visible"])
        self.assertIn("CH1~CH6 ACK 확인", output[3])
        self.assertEqual(len(output[4:]), 4)
        self.assertIn("CH2 10%", output[4])
        self.assertIn("CH6 40%", output[7])

    def test_invalid_last_fixture_is_rejected_before_any_channel_changes(self):
        output = self.app.applyLighting(10, 20, 30, float("nan"))
        self.factory.assert_not_called()
        self.port.write.assert_not_called()
        self.assertTrue(output[2]["visible"])
        self.assertIn("전송 실패", output[3])

    def test_panel_requires_both_acknowledgements(self):
        self.port.read.side_effect = [lights.ACK, b""]
        output = self.app.applyLightGroup(4, 10, 20, 30, 40)
        self.assertIn("오른쪽 위 면조명", output[2])
        self.assertIn("CH6: no ACK", output[2])
        self.assertIn("이번 요청의 ACK 확인: CH5", output[2])
        self.assertNotIn("적용 완료", output[2])
        self.assertIn("부분 적용", output[3])
        self.assertIn("CH5 40%", output[3])
        self.assertNotIn("CH6 40%", output[3])
        self.assertFalse(self.controller.get_status().connected)

    def test_all_failure_keeps_modal_open_and_marks_partial_panel(self):
        self.port.read.side_effect = [lights.ACK] * 5 + [b""]
        output = self.app.applyLighting(10, 20, 30, 40)
        self.assertTrue(output[2]["visible"])
        self.assertIn("CH6: no ACK", output[3])
        self.assertIn("CH1, CH2, CH3, CH4, CH5", output[3])
        self.assertIn("최근 ACK 확인", output[4])
        self.assertIn("부분 적용", output[7])
        self.assertNotIn("CH6 40%", output[7])

    def test_first_panel_failure_never_writes_other_fixtures(self):
        self.port.read.side_effect = [lights.ACK, b""]
        output = self.app.applyLighting(10, 20, 30, 40)
        self.assertEqual(self.port.write.call_count, 2)
        self.assertIn("부분 적용", output[4])
        self.assertEqual(output[5:], ({"__type__": "update"},) * 3)

    def test_open_close_and_display_changes_never_send_brightness(self):
        self.assertTrue(self.app.showLighting()["visible"])
        self.assertFalse(self.app.hideLighting()["visible"])
        average, label = self.app.calculateLighting(90, 20, 30, 40)
        self.assertEqual(average, 45)
        self.assertIn("4개 조명", label)
        self.factory.assert_not_called()
        self.port.write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
