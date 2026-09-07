"""Wire format between the driver-station GUI and the robot's microcontroller.

ASCII, newline-framed, one message per line. Deliberately human-readable: the whole
protocol can be exercised from a serial monitor with a keyboard, which is what you want
at 2am when the robot will not move and you need to know whether the fault is the GUI,
the link, or the firmware.

WHY NORMALISED UNITS
    Drive and turret values are -1.0 to 1.0, not m/s or deg/s. The GUI does not know the
    robot's gearing, wheel diameter or gimbal limits, and should not - firmware maps
    normalised demand onto whatever the hardware can do. That also means a firmware change
    to a faster motor does not silently change what the GUI's controls mean.

EVERY VALUE IS CLAMPED ON ENCODE
    A NaN or a 500 reaching a motor controller is a runaway. Clamping here means the
    firmware can trust its input range, and a bug in the aiming controller becomes a
    saturated command rather than an unbounded one.

SEE ALSO
    docs/ROBOT_PROTOCOL.md - the same format written for whoever implements the firmware,
    including the heartbeat requirement this module cannot enforce on its own.

Usage:
    python scripts/robot/protocol.py --selftest
"""

from __future__ import annotations

import math
from dataclasses import dataclass

BAUD = 115200
ENCODING = "ascii"
TERMINATOR = "\n"

# The GUI sends PING at this rate; firmware must stop all motion if it hears nothing for
# TIMEOUT_MS. Defined here so the GUI and the protocol document cannot disagree.
HEARTBEAT_HZ = 10.0
TIMEOUT_MS = 300


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    """Coerce to [low, high], mapping NaN to 0 rather than passing it on."""
    number = float(value)
    if math.isnan(number):
        return 0.0
    return max(low, min(high, number))


def _fmt(value: float) -> str:
    # Three decimals: finer than any motor controller resolves, and keeps a DRV line
    # under 24 bytes so a 10 Hz heartbeat plus drive traffic is nowhere near 115200 baud.
    return f"{_clamp(value):.3f}"


def drive(vx: float, vy: float = 0.0, omega: float = 0.0) -> str:
    """Chassis demand. vx forward, vy strafe, omega yaw rate, each -1..1."""
    return f"DRV {_fmt(vx)} {_fmt(vy)} {_fmt(omega)}"


def gimbal(d_pitch: float, d_yaw: float) -> str:
    """Turret RATE demand, -1..1. Rate not position: the GUI has no encoder feedback."""
    return f"GIM {_fmt(d_pitch)} {_fmt(d_yaw)}"


def fire(count: int = 1) -> str:
    """Launch `count` projectiles. Clamped to 1..10; firmware enforces its own cadence."""
    return f"FIRE {int(max(1, min(10, count)))}"


def intake(on: bool) -> str:
    """Ball collection on or off."""
    return f"INTAKE {1 if on else 0}"


def stop() -> str:
    """All motors to zero. The one command that must never be rate-limited or queued."""
    return "STOP"


def ping(seq: int) -> str:
    """Heartbeat. Firmware stops if these stop arriving - see TIMEOUT_MS."""
    return f"PING {int(seq) & 0xFFFF}"


@dataclass(frozen=True)
class Status:
    """A decoded STA line."""

    battery: float          # 0..1
    pitch: float            # degrees, robot's own frame
    yaw: float              # degrees
    flags: int              # bit field; bit 0 = armed, bit 1 = intake, bit 2 = fault


@dataclass(frozen=True)
class Ack:
    seq: int


@dataclass(frozen=True)
class Error:
    code: int
    text: str


def parse(line: str):
    """One telemetry line -> Status | Ack | Error | None.

    Returns None for anything unrecognised or malformed rather than raising. A serial
    link picks up line noise, and a half-received line during startup is normal; the
    caller's job is to ignore those, not to crash on them. Silence is the correct
    response to a corrupt frame - the next one arrives in 100 ms.
    """
    parts = line.strip().split()
    if not parts:
        return None
    try:
        if parts[0] == "STA" and len(parts) == 5:
            return Status(battery=_clamp(float(parts[1]), 0.0, 1.0),
                          pitch=float(parts[2]), yaw=float(parts[3]),
                          flags=int(parts[4]))
        if parts[0] == "ACK" and len(parts) == 2:
            return Ack(seq=int(parts[1]))
        if parts[0] == "ERR" and len(parts) >= 2:
            return Error(code=int(parts[1]), text=" ".join(parts[2:]))
    except ValueError:
        return None      # a field that should have been numeric was not
    return None


def encode(message: str) -> bytes:
    return (message + TERMINATOR).encode(ENCODING)


# ------------------------------------------------------------------------- selftest

def _selftest() -> None:
    print("[test] commands encode within range")
    assert drive(0.5, -0.25, 1.0) == "DRV 0.500 -0.250 1.000"
    assert drive(99, -99, 0) == "DRV 1.000 -1.000 0.000", "must clamp"
    assert drive(float("nan"), 0, 0) == "DRV 0.000 0.000 0.000", "NaN must become 0"
    assert gimbal(-2.0, 0.1) == "GIM -1.000 0.100"
    assert fire(0) == "FIRE 1" and fire(99) == "FIRE 10", "fire count must clamp"
    assert intake(True) == "INTAKE 1" and intake(False) == "INTAKE 0"
    assert stop() == "STOP"
    assert ping(70000) == f"PING {70000 & 0xFFFF}", "seq must wrap"
    print("       ok")

    print("[test] telemetry round-trips")
    status = parse("STA 0.87 -3.5 42.0 3")
    assert isinstance(status, Status), status
    assert status.battery == 0.87 and status.pitch == -3.5 and status.flags == 3
    assert parse("ACK 17") == Ack(17)
    assert parse("ERR 4 motor stalled") == Error(4, "motor stalled")
    print("       ok")

    print("[test] malformed input returns None, never raises")
    for bad in ("", "   ", "STA", "STA 1 2 3", "STA a b c d", "ACK", "ACK x",
                "\x00\xff garbage", "DRV 1 2 3", "STA 1 2 3 4 5"):
        assert parse(bad) is None, f"{bad!r} should not parse"
    print("       ok")

    print("[test] a full line is framed with exactly one newline")
    assert encode(stop()) == b"STOP\n"
    assert encode(drive(1, 0, 0)).count(b"\n") == 1
    print("       ok")

    print("\nAll checks passed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", required=True,
                        help="The only mode; this module is a library.")
    parser.parse_args()
    _selftest()
