"""Best-effort, in-memory delivery of completed measurements as UTF-8 JSON lines."""

from collections import deque
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import json
import socket
from threading import Condition, Thread
from time import monotonic

from src.inspection_history import InspectionRecord, LOCAL_TIMEZONE


LOCAL_HOST = "192.168.50.1"
RECEIVER_HOST = "192.168.50.2"
RECEIVER_PORT = 9997


def _round_for_transfer(value: float | None) -> float | None:
    if value is None:
        return None
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


class ResultSender:
    def __init__(self, host: str = RECEIVER_HOST, port: int = RECEIVER_PORT, *,
                 local_host: str = LOCAL_HOST, timeout: float = 0.5,
                 check_interval: float = 3.0, max_pending: int = 32):
        if timeout <= 0 or check_interval <= 0 or max_pending < 1:
            raise ValueError("전송 제한 시간·확인 주기·대기 수는 양수여야 합니다.")
        self.host, self.port, self.local_host = host, port, local_host
        self.timeout, self.check_interval = timeout, check_interval
        self.max_pending = max_pending
        self._condition = Condition()
        self._pending = deque()
        self._thread = None
        self._stopping = True
        self._enabled = False
        self._error = None
        self._last_checked_at = None
        self._last_sent_at = None
        self._last_sent_measurement = None
        self._last_result = "전송한 측정 없음"

    def start(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stopping = False
            self._enabled = False
            self._thread = Thread(target=self._run, name="result-tcp-sender", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._enabled = False
            self._pending.clear()
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=self.timeout * 2 + 1)

    def status(self) -> dict:
        with self._condition:
            return {"enabled": self._enabled, "host": self.host, "port": self.port,
                    "local_host": self.local_host, "error": self._error,
                    "last_checked_at": self._last_checked_at,
                    "last_sent_at": self._last_sent_at,
                    "last_sent_measurement": self._last_sent_measurement,
                    "last_result": self._last_result}

    def submit(self, record: InspectionRecord) -> dict:
        """Queue only while enabled; never touch disk or retry an old measurement."""
        with self._condition:
            if self._stopping or not self._enabled:
                return {"state": "skipped", "message": "TCP 자동 전송 비활성 · 이번 결과 전송 생략"}
            if len(self._pending) >= self.max_pending:
                return {"state": "skipped", "message": "TCP 전송 대기 한도 초과 · 이번 결과 전송 생략"}
            try:
                data = json.dumps({
                    "measured_at": record.measured_at,
                    **{name: _round_for_transfer(getattr(record, name))
                       for name in ("p1", "p2", "p3", "average", "elapsed_seconds")},
                }, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n"
            except (ValueError, TypeError, InvalidOperation) as error:
                return {"state": "skipped", "message": f"TCP 전송 데이터 오류: {error}"}
            self._pending.append((record.measured_at, data))
            self._condition.notify()
            return {"state": "queued", "message": "TCP 전송 요청됨 · 처리 결과는 SYSTEM STATUS에서 확인"}

    def _connect(self):
        # Bind to the dedicated LAN interface rather than another camera/Wi-Fi NIC.
        return socket.create_connection((self.host, self.port), timeout=self.timeout,
                                        source_address=(self.local_host, 0))

    def _run(self) -> None:
        next_check = 0.0
        while True:
            with self._condition:
                if self._stopping:
                    return
                # Check the port even if measurements arrive continuously.
                if monotonic() >= next_check:
                    item = None
                elif self._pending:
                    item = self._pending.popleft()
                else:
                    self._condition.wait(timeout=max(0, next_check - monotonic()))
                    continue
            try:
                # Each connection carries zero (probe) or one (measurement) JSON line.
                with self._connect() as connection:
                    if item is not None:
                        connection.sendall(item[1])
            except OSError as error:
                with self._condition:
                    self._enabled = False
                    self._error = str(error)
                    if item is not None or self._pending:
                        self._last_result = "소켓 전송 실패 · 대기 결과 폐기 · 재전송 안 함"
                    self._pending.clear()
            else:
                with self._condition:
                    self._enabled = not self._stopping
                    self._error = None
                    if item is not None:
                        self._last_sent_at = datetime.now(LOCAL_TIMEZONE).isoformat(timespec="milliseconds")
                        self._last_sent_measurement = item[0]
                        self._last_result = "소켓 전송 완료"
            with self._condition:
                self._last_checked_at = datetime.now(LOCAL_TIMEZONE).isoformat(timespec="milliseconds")
            next_check = monotonic() + self.check_interval
