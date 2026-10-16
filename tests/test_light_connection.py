"""RS232 standby must never send brightness commands."""

import unittest
from unittest.mock import Mock, patch

import serial

from src.hardware.led_control import ACK, LightCommunicationError, LightController, build_brightness_packet


class LightConnectionTests(unittest.TestCase):
    def setUp(self):
        self.port = Mock(is_open=False)
        self.port.open.side_effect = lambda: setattr(self.port, "is_open", True)
        self.port.close.side_effect = lambda: setattr(self.port, "is_open", False)
        self.port.read.return_value = ACK
        self.factory = patch("src.hardware.led_control.serial.Serial", return_value=self.port)
        self.serial_factory = self.factory.start()
        self.addCleanup(self.factory.stop)
        self.controller = LightController("COM3", 9600)
        self.addCleanup(self.controller.disconnect)

    def test_connect_keeps_port_open_without_sending_or_reading(self):
        self.controller.connect()
        self.controller.connect()
        self.assertTrue(self.controller.get_status().connected)
        self.assertFalse(self.controller.get_status().ack_verified)
        self.assertFalse(self.port.dtr)
        self.assertFalse(self.port.rts)
        self.serial_factory.assert_called_once()
        self.port.open.assert_called_once()
        self.port.write.assert_not_called()
        self.port.read.assert_not_called()
        self.port.reset_input_buffer.assert_not_called()

    def test_explicit_command_reuses_connection_and_requires_ack(self):
        self.controller.connect()
        command = build_brightness_packet(1, 54)
        self.assertEqual(self.controller.send((command,)), (command,))
        self.serial_factory.assert_called_once()
        self.port.write.assert_called_once_with(command.packet)
        self.port.read.assert_called_once_with(len(ACK))
        self.assertTrue(self.controller.get_status().ack_verified)
        self.assertTrue(self.controller.get_status().connected)

    def test_missing_ack_closes_connection_and_reports_error(self):
        self.port.read.return_value = b""
        with self.assertRaisesRegex(LightCommunicationError, "no ACK received"):
            self.controller.send((build_brightness_packet(1, 54),))
        status = self.controller.get_status()
        self.assertFalse(status.connected)
        self.assertFalse(status.ack_verified)
        self.assertIn("no ACK", status.error)

    def test_failed_open_does_not_send_commands(self):
        self.port.open.side_effect = serial.SerialException("port busy")
        with self.assertRaisesRegex(LightCommunicationError, "port busy"):
            self.controller.connect()
        self.assertFalse(self.controller.get_status().connected)
        self.port.write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
