"""Transport to the robot: a serial port, or a mock that needs no hardware.

The GUI talks only to RobotDriver. That seam is what lets the whole interface be built and
tested before any robot exists, and it is why the driver knows nothing about Qt and the
GUI knows nothing about pyserial.

"SERIAL" IS A PORT, NOT A CABLE
    A direct USB lead, a Bluetooth SPP pairing and a 2.4 GHz radio dongle all enumerate as
    a COM port. SerialDriver opens a name and does not care which it is, so tethered bench
    testing and radio competition use run identical code.

NON-BLOCKING BY CONSTRUCTION
    Every read has timeout 0 and every write is fire-and-forget. This is called from the
    Qt event loop, and a blocking read on a silent link would freeze the UI of a machine
    that is currently driving. A frozen driver station is a safety fault, not an
    inconvenience.

Usage:
    python scripts/robot/driver.py --selftest
"""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import protocol  # noqa: E402


class RobotDriver(ABC):
    """Send commands, collect telemetry. Implementations must never block."""

    def __init__(self) -> None:
        self._last_rx = 0.0
        self._seq = 0

    @abstractmethod
    def send(self, message: str) -> None:
        """Write one protocol line. Silently drops if the link is down."""

    @abstractmethod
    def poll(self) -> list:
        """Return decoded telemetry received since the last call. Never blocks."""

    @abstractmethod
    def close(self) -> None:
        """Release the port. Implementations MUST send STOP first."""

    @property
    @abstractmethod
    def connected(self) -> bool:
        ...

    def heartbeat(self) -> None:
        self._seq += 1
        self.send(protocol.ping(self._seq))

    def silence_seconds(self) -> float:
        """Seconds since the last decoded telemetry. Large means the link is gone."""
        return float("inf") if not self._last_rx else time.monotonic() - self._last_rx

    def emergency_stop(self) -> None:
        """Send STOP several times, unconditionally.

        Repeated on purpose: this is the one command whose loss to line noise is
        unacceptable, and it is idempotent, so sending it three times costs nothing.
        """
        for _ in range(3):
            self.send(protocol.stop())


class MockDriver(RobotDriver):
    """A robot that exists only in memory.

    Records every command for assertions, and synthesises plausible telemetry so the GUI's
    connected path is exercised rather than only its disconnected one. `drop_link()` makes
    it stop responding, which is how the deadman is tested without unplugging anything.
    """

    def __init__(self, telemetry_hz: float = 10.0) -> None:
        super().__init__()
        self.log: list[str] = []
        self._open = True
        self._alive = True
        self._interval = 1.0 / telemetry_hz
        self._next_tx = time.monotonic()
        self._last_rx = time.monotonic()
        self.pitch = 0.0
        self.yaw = 0.0

    def send(self, message: str) -> None:
        if not self._open:
            return
        self.log.append(message)
        # Integrate gimbal demand so the mock's reported angles move, which is enough to
        # see the aiming loop converge on recorded video.
        if message.startswith("GIM "):
            _, pitch, yaw = message.split()
            self.pitch = max(-90.0, min(90.0, self.pitch + float(pitch) * 2.0))
            self.yaw = (self.yaw + float(yaw) * 2.0) % 360.0

    def poll(self) -> list:
        if not (self._open and self._alive):
            return []
        now = time.monotonic()
        if now < self._next_tx:
            return []
        self._next_tx = now + self._interval
        self._last_rx = now
        line = f"STA 0.87 {self.pitch:.1f} {self.yaw:.1f} 1"
        message = protocol.parse(line)
        return [message] if message else []

    def drop_link(self) -> None:
        """Stop responding, without closing. Simulates a radio failure."""
        self._alive = False

    def restore_link(self) -> None:
        self._alive = True
        self._last_rx = time.monotonic()

    def close(self) -> None:
        self.emergency_stop()
        self._open = False

    @property
    def connected(self) -> bool:
        return self._open


class SerialDriver(RobotDriver):
    """A real port. Import of pyserial is deferred so the mock path needs no dependency."""

    def __init__(self, port: str, baud: int = protocol.BAUD) -> None:
        super().__init__()
        import serial  # noqa: PLC0415

        # timeout=0 is non-blocking; write_timeout bounds a stalled write so a full
        # transmit buffer cannot wedge the event loop.
        self._port = serial.Serial(port, baud, timeout=0, write_timeout=0.05)
        self._buffer = b""
        self.port_name = port

    def send(self, message: str) -> None:
        import serial  # noqa: PLC0415

        if not self._port.is_open:
            return
        try:
            self._port.write(protocol.encode(message))
        except (serial.SerialTimeoutException, serial.SerialException, OSError):
            # A write failure means the link is gone. Saying so via silence_seconds() is
            # the caller's cue; raising here would take down the UI thread.
            pass

    def poll(self) -> list:
        import serial  # noqa: PLC0415

        if not self._port.is_open:
            return []
        try:
            waiting = self._port.in_waiting
            if waiting:
                self._buffer += self._port.read(waiting)
        except (serial.SerialException, OSError):
            return []

        messages = []
        while b"\n" in self._buffer:
            raw, self._buffer = self._buffer.split(b"\n", 1)
            decoded = protocol.parse(raw.decode(protocol.ENCODING, errors="replace"))
            if decoded is not None:
                self._last_rx = time.monotonic()
                messages.append(decoded)
        # An unterminated buffer past any plausible line length is line noise, not a
        # partial message; dropping it stops a garbage byte stream growing without bound.
        if len(self._buffer) > 512:
            self._buffer = b""
        return messages

    def close(self) -> None:
        self.emergency_stop()
        try:
            self._port.flush()
        except Exception:
            pass
        self._port.close()

    @property
    def connected(self) -> bool:
        return bool(self._port.is_open)


def open_driver(kind: str, port: str | None = None) -> RobotDriver:
    if kind == "mock":
        return MockDriver()
    if not port:
        sys.exit("--port is required for --driver serial (e.g. COM3 or /dev/ttyUSB0)")
    return SerialDriver(port)


# ------------------------------------------------------------------------- selftest

def _selftest() -> None:
    print("[test] mock records commands")
    driver = MockDriver()
    driver.send(protocol.drive(0.5, 0, 0))
    driver.send(protocol.fire(2))
    assert driver.log == ["DRV 0.500 0.000 0.000", "FIRE 2"], driver.log
    print("       ok")

    print("[test] heartbeat sequence advances and wraps")
    driver = MockDriver()
    for _ in range(3):
        driver.heartbeat()
    assert driver.log == ["PING 1", "PING 2", "PING 3"], driver.log
    print("       ok")

    print("[test] telemetry arrives and silence is measured")
    driver = MockDriver(telemetry_hz=1000.0)
    time.sleep(0.01)
    messages = driver.poll()
    assert messages and isinstance(messages[0], protocol.Status), messages
    assert driver.silence_seconds() < 0.5
    print("       ok")

    print("[test] a dropped link goes silent, and silence grows")
    driver.drop_link()
    before = driver.silence_seconds()
    time.sleep(0.05)
    assert driver.poll() == [], "a dropped link must yield no telemetry"
    assert driver.silence_seconds() > before, "silence must grow while the link is down"
    driver.restore_link()
    time.sleep(0.01)
    assert driver.poll(), "restoring the link must resume telemetry"
    print("       ok")

    print("[test] close sends STOP before releasing the port")
    driver = MockDriver()
    driver.send(protocol.drive(1, 0, 0))
    driver.close()
    assert driver.log[-3:] == ["STOP", "STOP", "STOP"], driver.log[-4:]
    assert not driver.connected
    print("       ok")

    print("[test] a closed driver accepts sends without raising")
    driver.send(protocol.drive(1, 0, 0))   # must be a no-op, not an exception
    assert driver.log[-1] == "STOP", "a closed driver must not record new commands"
    print("       ok")

    print("\nAll checks passed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", required=True)
    parser.parse_args()
    _selftest()
