"""Explicit six-channel RS232 diagnostics with ACK history."""

from __future__ import annotations

from datetime import datetime
from math import isfinite
from threading import RLock

from src.hardware.led_control import (
    ACK, BAUD_RATE, CHANNEL_COUNT, COM_PORT, LightCommunicationError,
    LightController, LightTransmission, build_brightness_packet,
)


class RS232TestController:
    def __init__(self, port: str = COM_PORT, baudrate: int = BAUD_RATE,
                 *, light: LightController | None = None, on_light_replace=None):
        self.light = light or LightController(port, baudrate)
        self.on_light_replace = on_light_replace
        self.lock = RLock()
        self.last_ack: dict[int, LightTransmission] = {}
        self.history: list[list[str]] = []
        self.message = "밝기 명령 대기"

    def close(self) -> None:
        self.disconnect()

    @staticmethod
    def _settings(port: str, baudrate: int) -> tuple[str, int]:
        if not isinstance(port, str) or not port.strip() or type(baudrate) is not int or baudrate <= 0:
            raise ValueError("포트와 전송 속도를 확인하세요.")
        return port.strip(), baudrate

    def connect(self, port: str, baudrate: int) -> str:
        with self.lock:
            try:
                port, baudrate = self._settings(port, baudrate)
            except ValueError as error:
                self.message = f"연결 실패: {error}"
                return self.message
            current = self.light.get_status().connected
            if (port, baudrate) != (self.light.port, self.light.baudrate):
                self.light.disconnect()
                self.light = LightController(port, baudrate)
                if self.on_light_replace is not None:
                    self.on_light_replace(self.light)
                current = False
            try:
                self.light.connect()
            except LightCommunicationError as error:
                self.message = f"연결 실패: {error}"
                return self.message
            if not current:
                self.last_ack.clear()
            self.message = f"{port} 연결 완료 · 밝기 명령 대기"
            return self.message

    def disconnect(self) -> str:
        with self.lock:
            self.light.disconnect()
            self.last_ack.clear()
            self.message = "RS232 연결 해제 · 밝기 변경 명령은 전송하지 않았습니다."
            return self.message

    def _log(self, transmission: LightTransmission, received: str, state: str) -> None:
        self.history.insert(0, [datetime.now().isoformat(timespec="seconds"),
                                f"CH{transmission.channel}",
                                f"{transmission.percent:g}% / {transmission.value}",
                                transmission.packet.hex(" ").upper(), received, state])
        del self.history[100:]

    def apply(self, channels: tuple[tuple[int, float], ...], port: str, baudrate: int) -> tuple[int, ...]:
        with self.lock:
            try:
                port, baudrate = self._settings(port, baudrate)
                packets = []
                for channel, value in channels:
                    numeric = float(value)
                    if not isfinite(numeric):
                        raise ValueError(f"CH{channel}: 밝기는 0~100 사이여야 합니다.")
                    packets.append(build_brightness_packet(channel, numeric))
                if (port, baudrate) != (self.light.port, self.light.baudrate):
                    raise ValueError("포트 설정이 바뀌었습니다. 먼저 RS232 연결을 누르세요.")
                if not self.light.get_status().connected:
                    raise ValueError("먼저 RS232 연결을 누르세요.")
            except (TypeError, ValueError) as error:
                self.message = f"전송 실패: {error}"
                return ()
            try:
                acknowledged = self.light.send(tuple(packets))
            except LightCommunicationError as error:
                acknowledged = error.acknowledged
                for sent in acknowledged:
                    self.last_ack[sent.channel] = sent
                    self._log(sent, ACK.hex(" ").upper(), "ACK 확인")
                if len(acknowledged) < len(packets):
                    failed = packets[len(acknowledged)]
                    received = str(error).split("RX=")[-1].split(")")[0] if "RX=" in str(error) else "none"
                    self._log(failed, received, "전송 실패")
                self.message = f"CH{packets[len(acknowledged)].channel} 전송 실패: {error}" if len(acknowledged) < len(packets) else f"전송 실패: {error}"
                if acknowledged:
                    self.message += " · 이번 요청의 ACK 확인: " + ", ".join(f"CH{s.channel}" for s in acknowledged)
                    self.message += ". 이후 채널은 전송하지 않았습니다."
                return tuple(sent.channel for sent in acknowledged)
            for sent in acknowledged:
                self.last_ack[sent.channel] = sent
                self._log(sent, ACK.hex(" ").upper(), "ACK 확인")
            self.message = " · ".join(f"ACK 확인 · CH{sent.channel} {sent.percent:g}% ({sent.value}/255)" for sent in acknowledged)
            return tuple(sent.channel for sent in acknowledged)

    def turn_off(self, port: str, baudrate: int) -> tuple[int, ...]:
        return self.apply(tuple((channel, 0) for channel in range(1, CHANNEL_COUNT + 1)), port, baudrate)

    def clear_history(self) -> None:
        with self.lock:
            self.history.clear()

    def snapshot(self) -> tuple:
        with self.lock:
            connection = self.light.get_status()
            status = (f"{self.light.port} · {self.light.baudrate} bps · "
                      + ("RS232 OPEN · ACK 확인" if connection.ack_verified else
                         "RS232 OPEN · 명령 대기" if connection.connected else
                         connection.error or "연결 해제"))
            notes = [f"CH{channel} {self.last_ack[channel].percent:g}% · 최근 ACK 확인"
                     if channel in self.last_ack else "최근 ACK: 미확인"
                     for channel in range(1, CHANNEL_COUNT + 1)]
            return status, self.message, notes, list(self.history), connection.connected
