"""Exercise explicit serial actions and partial ACK failures without hardware."""

import unittest
from unittest.mock import Mock, patch

import serial

from src.diagnostics.serial import RS232TestController
from src.hardware.led_control import ACK, CHANNEL_COUNT, build_brightness_packet


class RS232TestTests(unittest.TestCase):
    def setUp(self):
        self.port = Mock(is_open=False)
        self.port.open.side_effect = lambda: setattr(self.port, "is_open", True)
        self.port.close.side_effect = lambda: setattr(self.port, "is_open", False)
        self.port.read.return_value = ACK
        factory = patch("src.hardware.led_control.serial.Serial", return_value=self.port)
        self.factory = factory.start()
        self.addCleanup(factory.stop)
        self.controller = RS232TestController("COM3", 9600)
        self.addCleanup(self.controller.close)

    def connect(self):
        self.controller.connect("COM3", 9600)

    def test_start_build_connect_and_close_do_not_change_brightness(self):
        self.factory.assert_not_called()
        self.controller.snapshot()
        self.connect()
        self.controller.close()
        self.port.write.assert_not_called()
        self.port.read.assert_not_called()

    def test_disconnected_apply_requires_deliberate_connection(self):
        self.assertEqual(self.controller.apply(((1, 50),), "COM3", 9600), ())
        self.factory.assert_not_called()
        self.assertIn("먼저 RS232 연결", self.controller.message)

    def test_port_change_cannot_send_to_old_device(self):
        self.connect()
        self.assertEqual(self.controller.apply(((1, 50),), "COM4", 9600), ())
        self.port.write.assert_not_called()
        self.assertIn("설정이 바뀌었습니다", self.controller.message)

    def test_entire_batch_validated_before_any_write(self):
        self.connect()
        for invalid in (-1, 101, float("nan"), float("inf"), None):
            with self.subTest(invalid=invalid):
                self.assertEqual(self.controller.apply(((1, 10), (2, invalid)), "COM3", 9600), ())
        self.port.write.assert_not_called()

    def test_invalid_connection_settings_preserve_current_connection(self):
        self.connect()
        for port, baud in (("", 9600), ("COM4", 0), ("COM4", 9600.5), ("COM4", float("nan"))):
            with self.subTest(port=port, baud=baud):
                self.controller.connect(port, baud)
                self.assertIn("연결 실패", self.controller.message)
                self.assertEqual(self.controller.light.port, "COM3")
                self.assertTrue(self.controller.light.get_status().connected)
        self.port.close.assert_not_called()

    def test_single_channel_packet_ack_and_history(self):
        self.connect()
        self.assertEqual(self.controller.apply(((3, 50),), "COM3", 9600), (3,))
        self.port.write.assert_called_once_with(b"\x023128\x03")
        self.assertEqual(set(self.controller.last_ack), {3})
        status, message, notes, history, connected = self.controller.snapshot()
        self.assertTrue(connected)
        self.assertIn("ACK 확인", status)
        self.assertIn("CH3 50%", message)
        self.assertIn("50%", notes[2])
        self.assertIn("미확인", notes[0])
        self.assertEqual(history[0][1:], ["CH3", "50% / 128", "02 33 31 32 38 03", "02 06 03", "ACK 확인"])

    def test_partial_batch_keeps_success_and_stops_after_failed_ack(self):
        self.connect()
        self.port.read.side_effect = [ACK, b"\x15"]
        accepted = self.controller.apply(((1, 10), (2, 20), (3, 30), (4, 40)), "COM3", 9600)
        self.assertEqual(accepted, (1,))
        self.assertEqual(self.port.write.call_count, 2)
        self.assertEqual(set(self.controller.last_ack), {1})
        self.assertFalse(self.controller.light.get_status().connected)
        self.assertIn("CH2 전송 실패", self.controller.message)
        self.assertIn("이후 채널은 전송하지 않았습니다", self.controller.message)
        self.assertEqual(self.controller.history[0][4], "15")
        self.assertEqual(self.controller.history[1][-1], "ACK 확인")
        self.controller.apply(((3, 30),), "COM3", 9600)
        self.assertEqual(self.port.write.call_count, 2)

    def test_open_failure_is_visible_and_does_not_write(self):
        self.port.open.side_effect = serial.SerialException("port busy")
        self.connect()
        self.assertIn("port busy", self.controller.message)
        self.assertFalse(self.controller.snapshot()[-1])
        self.port.write.assert_not_called()

    def test_disconnect_clears_ack_without_sending_off(self):
        self.connect()
        self.controller.apply(((1, 10),), "COM3", 9600)
        self.port.write.reset_mock()
        self.controller.disconnect()
        self.assertEqual(self.controller.last_ack, {})
        self.port.write.assert_not_called()
        self.controller.apply(((1, 0),), "COM3", 9600)
        self.port.write.assert_not_called()

    def test_reconnection_starts_new_ack_session(self):
        self.connect()
        self.controller.apply(((1, 10),), "COM3", 9600)
        self.controller.light.disconnect()
        self.connect()
        self.assertEqual(self.controller.last_ack, {})
        self.assertIn("명령 대기", self.controller.snapshot()[0])

    def test_ch5_failure_stops_before_ch6(self):
        self.connect()
        self.port.read.side_effect = [ACK, ACK, ACK, ACK, b"\x15"]
        accepted = self.controller.apply(tuple((channel, 10) for channel in range(1, 7)), "COM3", 9600)
        self.assertEqual(accepted, (1, 2, 3, 4))
        self.assertEqual(self.port.write.call_count, 5)
        self.assertNotIn(5, self.controller.last_ack)
        self.assertNotIn(6, self.controller.last_ack)
        self.assertIn("CH5 전송 실패", self.controller.message)

    def test_seventh_channel_rejected_before_any_write(self):
        self.connect()
        accepted = self.controller.apply(((6, 10), (7, 10)), "COM3", 9600)
        self.assertEqual(accepted, ())
        self.port.write.assert_not_called()

    def test_history_is_bounded_and_clearing_it_does_not_send(self):
        self.connect()
        for _ in range(105):
            self.controller.apply(((1, 10),), "COM3", 9600)
        self.assertEqual(len(self.controller.history), 100)
        self.port.write.reset_mock()
        self.controller.clear_history()
        self.assertEqual(self.controller.snapshot()[3], [])
        self.assertIn(1, self.controller.last_ack)
        self.port.write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
