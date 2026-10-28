"""TCP port monitoring, JSON framing, failure isolation, and no offline replay."""

from dataclasses import replace
import json
import socket
import socketserver
from threading import Event, Thread
from time import monotonic, sleep
import unittest
from unittest.mock import patch

from src.inspection_history import InspectionRecord
from src.result_transfer import ResultSender


class Receiver(socketserver.TCPServer):
    allow_reuse_address = True

    def __init__(self, port=0):
        class Handler(socketserver.StreamRequestHandler):
            def handle(handler):
                handler.connection.settimeout(2)
                line = handler.rfile.readline()
                if line:  # Port checks deliberately connect without sending JSON.
                    handler.server.messages.append((handler.client_address, line))

        super().__init__(("127.0.0.1", port), Handler)
        self.messages = []
        self.thread = Thread(target=self.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()

    def close(self):
        self.shutdown()
        self.server_close()
        self.thread.join(timeout=2)


class ResultTransferTests(unittest.TestCase):
    def setUp(self):
        self.record = InspectionRecord(90.125, 80.25, 70.375,
            "2026-10-28T12:34:56.123+09:00", "camera", 1.23456789)

    def wait_for(self, predicate):
        deadline = monotonic() + 3
        while monotonic() < deadline:
            if predicate():
                return
            sleep(0.005)
        self.fail("TCP worker did not reach the expected state")

    def sender(self, port, **kwargs):
        sender = ResultSender("127.0.0.1", port, local_host="127.0.0.1",
                              timeout=0.1, check_interval=0.04, **kwargs)
        self.addCleanup(sender.stop)
        sender.start()
        return sender

    def receiver(self, port=0):
        receiver = Receiver(port)
        self.addCleanup(receiver.close)
        return receiver

    def test_open_port_enables_and_sends_one_exact_json_line_per_measurement(self):
        receiver = self.receiver()
        sender = self.sender(receiver.server_address[1])
        self.wait_for(lambda: sender.status()["enabled"])
        records = [replace(self.record, p1=self.record.p1 + i) for i in range(3)]
        for record in records:
            self.assertEqual(sender.submit(record)["state"], "queued")
        self.wait_for(lambda: len(receiver.messages) == 3)
        for index, ((address, line), record) in enumerate(zip(receiver.messages, records)):
            self.assertEqual(address[0], "127.0.0.1")
            self.assertTrue(line.endswith(b"\n"))
            self.assertEqual(json.loads(line.decode("utf-8")), {
                "measured_at": record.measured_at, "p1": 90.13 + index,
                "p2": 80.25, "p3": 70.38, "average": (80.25, 80.58, 80.92)[index],
                "elapsed_seconds": 1.23,
            })
        self.wait_for(lambda: sender.status()["last_sent_at"] is not None)
        self.assertEqual(sender.status()["last_result"], "소켓 전송 완료")
        self.assertEqual(sender.status()["last_sent_measurement"], self.record.measured_at)

    def test_wire_values_round_half_up_to_two_places_without_changing_records(self):
        receiver = self.receiver()
        sender = self.sender(receiver.server_address[1])
        self.wait_for(lambda: sender.status()["enabled"])
        cases = [
            ((95.41189931350115, 91.99864191942055, 88.98865784499054, 0.31234159995801747),
             (95.41, 92.0, 88.99, 92.13, 0.31)),
            ((1.005, 2.675, 90.125, 1.005), (1.01, 2.68, 90.13, 31.27, 1.01)),
            ((0.0, 100.0, 50.0, None), (0.0, 100.0, 50.0, 50.0, None)),
        ]
        for index, (values, expected) in enumerate(cases):
            with self.subTest(values=values):
                record = replace(self.record, p1=values[0], p2=values[1], p3=values[2],
                                 elapsed_seconds=values[3])
                sender.submit(record)
                self.wait_for(lambda: len(receiver.messages) == index + 1)
                decoded = json.loads(receiver.messages[index][1])
                self.assertEqual(decoded["measured_at"], record.measured_at)
                self.assertEqual(tuple(decoded[name] for name in (
                    "p1", "p2", "p3", "average", "elapsed_seconds")), expected)
                self.assertEqual((record.p1, record.p2, record.p3, record.elapsed_seconds), values)

    def test_closed_port_skips_and_reopen_sends_only_new_measurements(self):
        # Reserve a port without listening so connection attempts are refused.
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
            sender = self.sender(port)
            self.wait_for(lambda: sender.status()["error"] is not None)
            self.assertFalse(sender.status()["enabled"])
            self.assertEqual(sender.submit(self.record)["state"], "skipped")
        receiver = self.receiver(port)
        self.wait_for(lambda: sender.status()["enabled"])
        self.assertIsNone(sender.status()["error"])
        new_record = replace(self.record, measured_at="2026-10-28T12:35:00+09:00")
        sender.submit(new_record)
        self.wait_for(lambda: len(receiver.messages) == 1)
        self.assertEqual(json.loads(receiver.messages[0][1])["measured_at"], new_record.measured_at)
        receiver.close()
        self.wait_for(lambda: not sender.status()["enabled"])
        self.assertEqual(sender.submit(self.record)["state"], "skipped")

    def test_send_timeout_does_not_block_measurement_and_discards_pending_without_retry(self):
        entered, release = Event(), Event()

        class Connection:
            fail = True

            def __enter__(connection):
                return connection

            def __exit__(connection, *_args):
                pass

            def sendall(connection, _data):
                entered.set()
                release.wait(timeout=2)
                if connection.fail:
                    raise TimeoutError("receiver stopped reading")

        connection = Connection()
        sender = ResultSender(timeout=0.1, check_interval=0.04, max_pending=1)
        self.addCleanup(sender.stop)
        self.addCleanup(release.set)
        with patch.object(sender, "_connect", return_value=connection):
            sender.start()
            self.wait_for(lambda: sender.status()["enabled"])
            self.assertEqual(sender.submit(self.record)["state"], "queued")
            self.assertTrue(entered.wait(timeout=2))
            # While the worker is blocked, status and the next submission still work.
            self.assertTrue(sender.status()["enabled"])
            self.assertEqual(sender.submit(self.record)["state"], "queued")
            self.assertEqual(sender.submit(self.record)["state"], "skipped")
            release.set()
            self.wait_for(lambda: "재전송 안 함" in sender.status()["last_result"])
            connection.fail = False
            self.wait_for(lambda: sender.status()["enabled"])
            self.assertIsNone(sender.status()["last_sent_at"])
            sender.stop()
        self.assertFalse(sender.status()["enabled"])
        self.assertEqual(sender.submit(self.record)["state"], "skipped")

    def test_dedicated_lan_source_address_and_short_timeout_are_used(self):
        sender = ResultSender()
        with patch("src.result_transfer.socket.create_connection") as connect:
            sender._connect()
        connect.assert_called_once_with(("192.168.50.2", 9997), timeout=0.5,
                                        source_address=("192.168.50.1", 0))

    def test_missing_local_interface_stays_disabled(self):
        sender = ResultSender(timeout=0.1, check_interval=0.04)
        self.addCleanup(sender.stop)
        with patch.object(sender, "_connect", side_effect=OSError("local address unavailable")):
            sender.start()
            self.wait_for(lambda: sender.status()["error"] is not None)
            self.assertFalse(sender.status()["enabled"])
            self.assertEqual(sender.submit(self.record)["state"], "skipped")
            sender.stop()


if __name__ == "__main__":
    unittest.main()
