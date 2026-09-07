"""Headless integration checks for the driver station. No robot, no display, no camera.

Covers the behaviour that unit tests on the individual modules cannot: that the GUI's
wiring actually produces the right commands, that the safety interlocks latch, and that
inference is not on the UI thread.

Runs under Qt's offscreen platform, so it works over SSH and in CI.

Usage:
    python scripts/robot/test_station.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "robot"))

import numpy as np  # noqa: E402
from PySide6 import QtCore, QtWidgets  # noqa: E402

import protocol  # noqa: E402
from driver import MockDriver  # noqa: E402
from gui import RobotWindow  # noqa: E402

FRAME = (1920, 1080)


def _window(app):
    """A window on a mock robot and a synthetic video source that never opens.

    The worker's video source is irrelevant here - every check drives _on_frame directly
    with a synthetic frame, which is what makes these deterministic.
    """
    driver = MockDriver(telemetry_hz=1000.0)
    window = RobotWindow(driver, "0", {"device": "cpu"})
    window.worker.stop()          # no real capture in these checks
    window.worker.wait(2000)
    return window, driver


def _link_up(window) -> None:
    window._tick_heartbeat()
    assert window.link_ok, "mock telemetry should bring the link up"


def check_drive_commands(app) -> None:
    print("[test] held keys become drive commands")
    window, driver = _window(app)
    _link_up(window)
    driver.log.clear()

    window._keys.add(QtCore.Qt.Key_W)
    window._tick_control()
    drive = [line for line in driver.log if line.startswith("DRV")]
    assert drive, driver.log
    vx = float(drive[-1].split()[1])
    assert vx > 0, f"W should drive forward, got {vx}"

    window._keys.clear()
    window._keys.add(QtCore.Qt.Key_S)
    driver.log.clear()
    window._tick_control()
    vx = float([l for l in driver.log if l.startswith("DRV")][-1].split()[1])
    assert vx < 0, f"S should reverse, got {vx}"

    window._keys.clear()
    window._keys.add(QtCore.Qt.Key_A)
    driver.log.clear()
    window._tick_control()
    vy = float([l for l in driver.log if l.startswith("DRV")][-1].split()[2])
    assert vy < 0, f"A should strafe left, got {vy}"
    window.close()
    print("       ok")


def check_turret_and_fire(app) -> None:
    print("[test] firing is gated behind ARM, turret keys move the gimbal")
    window, driver = _window(app)
    _link_up(window)

    driver.log.clear()
    window.fire()
    assert not any(l.startswith("FIRE") for l in driver.log), \
        "an unarmed robot must not fire"

    window.btn_arm.setChecked(True)
    driver.log.clear()
    window.fire()
    assert any(l.startswith("FIRE") for l in driver.log), "armed, it should fire"

    window._keys.add(QtCore.Qt.Key_Left)
    driver.log.clear()
    window._tick_control()
    yaw = float([l for l in driver.log if l.startswith("GIM")][-1].split()[2])
    assert yaw < 0, f"left arrow should yaw negative, got {yaw}"
    window.close()
    print("       ok")


def check_deadman(app) -> None:
    print("[test] losing the link latches the station safe")
    window, driver = _window(app)
    _link_up(window)
    window.btn_arm.setChecked(True)
    window.btn_track.setChecked(True)
    assert window.armed and window.auto_track

    driver.drop_link()
    # Silence must exceed LINK_TIMEOUT before the station reacts.
    time.sleep(1.05)
    window._tick_heartbeat()

    assert not window.link_ok, "link should read lost"
    assert not window.armed, "link loss must disarm"
    assert not window.auto_track, "link loss must stop auto-track"

    driver.log.clear()
    window._keys.add(QtCore.Qt.Key_W)
    window._tick_control()
    assert not any(l.startswith("DRV") for l in driver.log), \
        "a lost link must not send drive commands"

    driver.log.clear()
    window.fire()
    assert not any(l.startswith("FIRE") for l in driver.log), \
        "a lost link must not fire"
    window.close()
    print("       ok")


def check_estop_latches(app) -> None:
    print("[test] e-stop latches until an explicit re-arm")
    window, driver = _window(app)
    _link_up(window)
    window.btn_arm.setChecked(True)

    driver.log.clear()
    window.emergency_stop()
    assert "STOP" in driver.log, driver.log
    assert window.estopped and not window.armed

    driver.log.clear()
    window._keys.add(QtCore.Qt.Key_W)
    window._tick_control()
    assert not any(l.startswith("DRV") for l in driver.log), \
        "an e-stopped station must ignore drive input"

    window.btn_arm.setChecked(True)      # the deliberate operator action
    assert not window.estopped and window.armed, "re-arm should clear the latch"
    window._tick_control()
    assert any(l.startswith("DRV") for l in driver.log), "control returns after re-arm"
    window.close()
    print("       ok")


def check_aiming_closes_the_loop(app) -> None:
    print("[test] auto-track aims at the target, with the right sign")
    window, _ = _window(app)
    _link_up(window)
    window.btn_track.setChecked(True)
    frame = np.zeros((FRAME[1], FRAME[0], 3), dtype=np.uint8)

    window._on_frame(frame, [(1, (100.0, 520.0, 40.0, 40.0))], 12.0)
    _, yaw_left = window._aim_demand
    window.aim.reset()
    window._on_frame(frame, [(2, (1780.0, 520.0, 40.0, 40.0))], 12.0)
    _, yaw_right = window._aim_demand
    assert yaw_left < 0 < yaw_right, (yaw_left, yaw_right)

    window.aim.reset()
    window._on_frame(frame, [(3, (950.0, 60.0, 40.0, 40.0))], 12.0)
    pitch_up, _ = window._aim_demand
    assert pitch_up > 0, f"a high target should pitch up, got {pitch_up}"

    # And the demand must actually reach the wire.
    window._keys.clear()
    driver = window.driver
    driver.log.clear()
    window._tick_control()
    gim = [l for l in driver.log if l.startswith("GIM")][-1]
    assert float(gim.split()[1]) > 0, f"aim demand should be sent, got {gim}"

    window.btn_track.setChecked(False)
    assert window._aim_demand == (0.0, 0.0), "disabling auto-track must zero the demand"
    window.close()
    print(f"       ok  (left {yaw_left:+.3f}, right {yaw_right:+.3f}, "
          f"up {pitch_up:+.3f})")


def check_autotrack_never_fires(app) -> None:
    print("[test] auto-track never fires on its own")
    window, driver = _window(app)
    _link_up(window)
    window.btn_arm.setChecked(True)
    window.btn_track.setChecked(True)
    frame = np.zeros((FRAME[1], FRAME[0], 3), dtype=np.uint8)

    driver.log.clear()
    for _ in range(20):
        window._on_frame(frame, [(1, (100.0, 520.0, 40.0, 40.0))], 12.0)
        window._tick_control()
    assert not any(l.startswith("FIRE") for l in driver.log), \
        "aiming must never pull the trigger"
    window.close()
    print("       ok")


def check_inference_off_ui_thread(app) -> None:
    print("[test] inference runs on a worker thread, not the UI thread")
    driver = MockDriver()
    window = RobotWindow(driver, "0", {"device": "cpu"})
    assert isinstance(window.worker, QtCore.QThread)
    assert window.worker.thread() is not window.worker, \
        "the worker's run() must execute off the thread that owns the widget"
    # The heavy object is built inside run(), so constructing the window must not have
    # loaded any model weights.
    assert window.worker._tracker is None, \
        "model weights must not load on the UI thread"
    window.worker.stop()
    window.worker.wait(2000)
    window.close()
    print("       ok")


def check_mode_banner(app) -> None:
    print("[test] the banner cannot call a simulation live")
    driver = MockDriver()

    # mock robot + recorded clip
    window = RobotWindow(driver, "data/tracking/arc02/img1/%06d.jpg", {"device": "cpu"})
    text, live = window._mode()
    assert not live and "recorded clip" in text and "mock" in text, (text, live)
    assert "SIMULATION" in window.banner.text()
    assert "SIMULATION" in window.windowTitle()
    window.worker.stop(); window.worker.wait(2000); window.close()

    # mock robot + real camera is STILL simulation - the robot half is fake
    window = RobotWindow(driver, "0", {"device": "cpu"})
    _, live = window._mode()
    assert not live, "a mock robot is never live, whatever the camera is"
    window.worker.stop(); window.worker.wait(2000); window.close()

    # a real port + a recorded clip is also simulation - the video half is fake
    class FakeSerial(MockDriver):
        @property
        def describe(self):
            return "COM3", True

    window = RobotWindow(FakeSerial(), "data/tracking/arc02/img1/%06d.jpg",
                         {"device": "cpu"})
    _, live = window._mode()
    assert not live, "recorded video is never live, whatever the robot is"
    window.worker.stop(); window.worker.wait(2000); window.close()

    # both real
    window = RobotWindow(FakeSerial(), "0", {"device": "cpu"})
    text, live = window._mode()
    assert live and "COM3" in text and "device 0" in text, (text, live)
    assert "LIVE" in window.banner.text() and "SIMULATION" not in window.banner.text()
    window.worker.stop(); window.worker.wait(2000); window.close()
    print("       ok")


def main() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    for check in (check_drive_commands, check_turret_and_fire, check_deadman,
                  check_estop_latches, check_aiming_closes_the_loop,
                  check_autotrack_never_fires, check_inference_off_ui_thread,
                  check_mode_banner):
        check(app)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
