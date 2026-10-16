"""LS1506~12A-R(CE) six-channel LED-control module."""

from dataclasses import dataclass
from os import getenv
from threading import Lock

import serial


COM_PORT = getenv("LIGHT_SERIAL_PORT", "COM3")
BAUD_RATE = int(getenv("LIGHT_SERIAL_BAUDRATE", "9600"))
TIMEOUT_SECONDS = float(getenv("LIGHT_SERIAL_TIMEOUT", "1"))
OUTPUT_MAX = 255
CHANNEL_COUNT = 6
ACK = bytes((0x02, 0x06, 0x03))


class LightCommunicationError(RuntimeError):
    """Raised when the controller cannot accept a lighting command."""

    def __init__(self, message: str, *, acknowledged: tuple["LightTransmission", ...] = ()) -> None:
        super().__init__(message)
        self.acknowledged = acknowledged


@dataclass(frozen=True, slots=True)
class LightTransmission:
    channel: int
    percent: float
    value: int
    packet: bytes


@dataclass(frozen=True, slots=True)
class LightGroup:
    name: str
    channels: tuple[int, ...]
    config_key: str

    @property
    def channel_label(self) -> str:
        return " + ".join(f"CH{channel}" for channel in self.channels)


# Physical fixture assignment: keep the two panel-light outputs together.
LIGHT_GROUPS = (
    LightGroup("왼쪽 아래 면조명", (1, 2), "left_lower_panel"),
    LightGroup("왼쪽 위 바 조명", (3,), "left_upper_bar"),
    LightGroup("오른쪽 위 바 조명", (4,), "right_upper_bar"),
    LightGroup("오른쪽 위 면조명", (5, 6), "right_upper_panel"),
)


@dataclass(frozen=True, slots=True)
class LightConnectionStatus:
    connected: bool
    ack_verified: bool
    error: str | None


class LightController:
    """Keep RS232 open without changing brightness until a command is sent."""

    def __init__(self, port: str = COM_PORT, baudrate: int = BAUD_RATE) -> None:
        self.port = port
        self.baudrate = baudrate
        self._serial: serial.Serial | None = None
        self._lock = Lock()
        self._ack_verified = False
        self._error: str | None = None

    def _connect(self) -> None:
        if self._serial is not None and self._serial.is_open:
            return
        controller = serial.Serial(
            port=None, baudrate=self.baudrate, bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
            timeout=TIMEOUT_SECONDS, write_timeout=TIMEOUT_SECONDS,
        )
        controller.dtr = False
        controller.rts = False
        controller.port = self.port
        self._ack_verified = False
        try:
            controller.open()
        except (serial.SerialException, OSError) as error:
            controller.close()
            self._error = f"Unable to open {self.port} at {self.baudrate} bps: {error}"
            raise LightCommunicationError(self._error) from error
        self._serial = controller
        self._error = None

    def connect(self) -> None:
        with self._lock:
            self._connect()

    def disconnect(self) -> None:
        with self._lock:
            if self._serial is not None:
                self._serial.close()
                self._serial = None
            self._ack_verified = False

    def get_status(self) -> LightConnectionStatus:
        with self._lock:
            return LightConnectionStatus(
                connected=self._serial is not None and self._serial.is_open,
                ack_verified=self._ack_verified, error=self._error,
            )

    def send(self, transmissions: tuple[LightTransmission, ...]) -> tuple[LightTransmission, ...]:
        with self._lock:
            self._connect()
            controller = self._serial
            assert controller is not None
            acknowledged = []
            try:
                for transmission in transmissions:
                    controller.reset_input_buffer()
                    controller.write(transmission.packet)
                    controller.flush()
                    response = controller.read(len(ACK))
                    if response != ACK:
                        received = response.hex(" ").upper() or "none"
                        raise LightCommunicationError(
                            f"CH{transmission.channel}: no ACK received (RX={received})"
                        )
                    acknowledged.append(transmission)
                self._ack_verified = bool(transmissions)
                self._error = None
            except (serial.SerialException, OSError, LightCommunicationError) as error:
                self._ack_verified = False
                self._error = str(error)
                controller.close()
                self._serial = None
                raise LightCommunicationError(self._error, acknowledged=tuple(acknowledged)) from error
        return transmissions


_controller = LightController()


def connect_light() -> None:
    _controller.connect()


def disconnect_light() -> None:
    _controller.disconnect()


def get_light_status() -> LightConnectionStatus:
    return _controller.get_status()


def build_brightness_packet(channel: int, percent: float) -> LightTransmission:
    """Convert a UI percentage to one controller packet."""
    if not 1 <= channel <= CHANNEL_COUNT:
        raise ValueError(f"Unsupported channel: {channel}")
    if not 0 <= percent <= 100:
        raise ValueError(f"CH{channel}: brightness must be between 0 and 100")

    value = round(percent * OUTPUT_MAX / 100)
    packet = b"\x02" + str(channel).encode("ascii") + f"{value:03d}".encode("ascii") + b"\x03"
    return LightTransmission(channel=channel, percent=percent, value=value, packet=packet)


def send_transmissions(
    transmissions: tuple[LightTransmission, ...],
) -> tuple[LightTransmission, ...]:
    """Send packets in order, requiring one ACK for every channel."""

    return _controller.send(transmissions)


def set_channel_brightness(channel: int, percent: float) -> LightTransmission:
    """Set one channel and return only after its ACK is received."""
    transmission = build_brightness_packet(channel, percent)
    return send_transmissions((transmission,))[0]


def set_light_group_brightness(group: int, percent: float) -> tuple[LightTransmission, ...]:
    """Set every channel belonging to one physical fixture, requiring each ACK."""
    if not 1 <= group <= len(LIGHT_GROUPS):
        raise ValueError(f"Unsupported lighting group: {group}")
    packets = tuple(build_brightness_packet(channel, percent) for channel in LIGHT_GROUPS[group - 1].channels)
    return send_transmissions(packets)


def set_light_groups_brightness(
    left_lower_panel: float, left_upper_bar: float, right_upper_bar: float, right_upper_panel: float,
) -> tuple[LightTransmission, ...]:
    """Map four fixture settings to CH1..CH6, validating all before any write."""
    values = (left_lower_panel, left_upper_bar, right_upper_bar, right_upper_panel)
    packets = tuple(
        build_brightness_packet(channel, percent)
        for group, percent in zip(LIGHT_GROUPS, values, strict=True)
        for channel in group.channels
    )
    return send_transmissions(packets)


def set_four_brightness(
    light_1: float,
    light_2: float,
    light_3: float,
    light_4: float,
) -> tuple[LightTransmission, ...]:
    """Set CH1~CH4 and return their verified transmissions."""
    return send_transmissions(
        tuple(
            build_brightness_packet(channel, percent)
            for channel, percent in enumerate((light_1, light_2, light_3, light_4), start=1)
        )
    )


if __name__ == "__main__":
    # Example: CH1=54%, CH2=30%, CH3=80%, CH4=100%
    for sent in set_four_brightness(54, 30, 80, 100):
        print(f"CH{sent.channel}: {sent.percent}% => {sent.value}, TX={sent.packet.hex(' ').upper()}")
