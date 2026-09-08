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
from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

import gui  # noqa: E402
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

    window._keys.add(QtCore.Qt.Key_J)
    driver.log.clear()
    window._tick_control()
    yaw = float([l for l in driver.log if l.startswith("GIM")][-1].split()[2])
    assert yaw < 0, f"J should yaw the turret negative, got {yaw}"
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


def check_tuning_controls(app) -> None:
    print("[test] tuning controls actually change behaviour, not just the display")
    window, driver = _window(app)
    _link_up(window)

    # Drive speed must reach the wire.
    window.tuning_widgets["drive speed"].setValue(0.20)
    window._keys.add(QtCore.Qt.Key_W)
    driver.log.clear()
    window._tick_control()
    slow = float([l for l in driver.log if l.startswith("DRV")][-1].split()[1])
    window.tuning_widgets["drive speed"].setValue(0.90)
    driver.log.clear()
    window._tick_control()
    fast = float([l for l in driver.log if l.startswith("DRV")][-1].split()[1])
    assert fast > slow, f"raising drive speed must raise demand: {slow} -> {fast}"
    window._keys.clear()

    # Turret speed likewise.
    window.tuning_widgets["turret speed"].setValue(0.15)
    window._keys.add(QtCore.Qt.Key_L)
    driver.log.clear()
    window._tick_control()
    slow_yaw = float([l for l in driver.log if l.startswith("GIM")][-1].split()[2])
    window.tuning_widgets["turret speed"].setValue(0.80)
    driver.log.clear()
    window._tick_control()
    fast_yaw = float([l for l in driver.log if l.startswith("GIM")][-1].split()[2])
    assert fast_yaw > slow_yaw, (slow_yaw, fast_yaw)
    window._keys.clear()

    # Aim gain must reach the controller, and the spinbox must be wired to the same
    # AimConfig instance the controller uses - a copy would look right and do nothing.
    window.tuning_widgets["aim gain yaw"].setValue(2.5)
    assert window.aim.config.gain_yaw == 2.5, window.aim.config.gain_yaw
    window.tuning_widgets["aim deadband"].setValue(0.0)
    window.tuning_widgets["aim max rate"].setValue(1.0)
    frame = np.zeros((FRAME[1], FRAME[0], 3), dtype=np.uint8)
    window.btn_track.setChecked(True)
    window.aim.reset()
    window._on_frame(frame, [(1, (1500.0, 520.0, 40.0, 40.0))], 12.0)
    _, high_gain = window._aim_demand

    window.tuning_widgets["aim gain yaw"].setValue(0.5)
    window.aim.reset()
    window._on_frame(frame, [(1, (1500.0, 520.0, 40.0, 40.0))], 12.0)
    _, low_gain = window._aim_demand
    assert high_gain > low_gain > 0, (high_gain, low_gain)
    window.close()
    print(f"       ok  (drive {slow}->{fast}, aim gain {low_gain:.3f}->{high_gain:.3f})")


def check_team_key_mapping(app) -> None:
    print("[test] chassis mapping matches the team's agreed scheme")
    window, driver = _window(app)
    _link_up(window)

    # Arrows rotate the CHASSIS, not the turret - the team's map, and the opposite of
    # this GUI's first version. Getting it backwards would spin the robot when the
    # driver meant to aim.
    for key, sign, what in ((QtCore.Qt.Key_Right, +1, "right arrow"),
                            (QtCore.Qt.Key_Left, -1, "left arrow")):
        window._keys.clear()
        window._keys.add(key)
        driver.log.clear()
        window._tick_control()
        drv = [l for l in driver.log if l.startswith("DRV")][-1]
        gim = [l for l in driver.log if l.startswith("GIM")][-1]
        omega = float(drv.split()[3])
        assert omega * sign > 0, f"{what} should rotate the chassis, got {drv}"
        assert gim == "GIM 0.000 0.000", f"{what} must not move the turret: {gim}"

    # Strafe, which a mecanum base has and a differential one does not.
    window._keys.clear()
    window._keys.add(QtCore.Qt.Key_D)
    driver.log.clear()
    window._tick_control()
    vy = float([l for l in driver.log if l.startswith("DRV")][-1].split()[2])
    assert vy > 0, f"D should strafe right, got {vy}"

    # Diagonals: W+A must produce translation on BOTH axes in one command. A
    # one-function-per-key design cannot express this, and the mecanum kinematics
    # table's curved and arcing paths depend on it.
    window._keys.clear()
    window._keys.update({QtCore.Qt.Key_W, QtCore.Qt.Key_A})
    driver.log.clear()
    window._tick_control()
    parts = [l for l in driver.log if l.startswith("DRV")][-1].split()
    vx, vy = float(parts[1]), float(parts[2])
    assert vx > 0 and vy < 0, f"W+A should be forward-left in one demand, got {vx},{vy}"

    # And translation plus rotation together - the "curved trajectory" case.
    window._keys.add(QtCore.Qt.Key_Right)
    driver.log.clear()
    window._tick_control()
    parts = [l for l in driver.log if l.startswith("DRV")][-1].split()
    assert float(parts[1]) > 0 and float(parts[3]) > 0,         f"translate+rotate must be simultaneous, got {parts}"

    # +/- step the speed, as the team specified.
    window._keys.clear()
    before = window.drive_speed
    window.keyPressEvent(QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress, QtCore.Qt.Key_Minus, QtCore.Qt.NoModifier))
    assert window.drive_speed < before, "minus should slow down"
    window.keyPressEvent(QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress, QtCore.Qt.Key_Plus, QtCore.Qt.NoModifier))
    window.keyPressEvent(QtGui.QKeyEvent(
        QtCore.QEvent.KeyPress, QtCore.Qt.Key_Plus, QtCore.Qt.NoModifier))
    assert window.drive_speed > before, "plus should speed up"
    # The spin box must agree with what is actually sent.
    assert abs(window.tuning_widgets["drive speed"].value() - window.drive_speed) < 1e-9
    window.close()
    print("       ok")


def check_legend_matches_bindings(app) -> None:
    print("[test] the on-screen legend names every key the control path reads")
    import inspect
    import re

    # This caught a real defect: the mapping moved to the team's scheme while the label
    # still advertised "QE rotate · arrows turret", so the panel instructed an operator to
    # rotate with keys that no longer did anything and to aim with keys that by then span
    # the chassis. Behaviour tests all passed - none of them read the label.
    source = (inspect.getsource(gui.RobotWindow._tick_control)
              + inspect.getsource(gui.RobotWindow.keyPressEvent))
    read_by_control = set(re.findall(r"Qt\.Key_(\w+)", source))
    named_in_legend = {key for keys in gui.LEGEND_KEYS.values() for key in keys}

    unnamed = read_by_control - named_in_legend
    assert not unnamed, (
        f"the control path reads {sorted(unnamed)} but the legend never mentions them - "
        f"add them to LEGEND_KEYS and LEGEND"
    )

    for token in gui.LEGEND_KEYS:
        assert token in gui.LEGEND, f"LEGEND_KEYS lists {token!r}, LEGEND does not show it"

    # And the stale mapping must not come back.
    assert "QE rotate" not in gui.LEGEND, "the legend advertises the superseded mapping"
    assert "arrows turret" not in gui.LEGEND, "arrows rotate the chassis, not the turret"

    window, _ = _window(app)
    assert window.help_label.text() == gui.LEGEND, "the panel must show LEGEND itself"
    window.close()
    print(f"       ok  ({len(read_by_control)} keys bound, all named)")


def main() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    for check in (check_drive_commands, check_turret_and_fire, check_deadman,
                  check_estop_latches, check_aiming_closes_the_loop,
                  check_autotrack_never_fires, check_inference_off_ui_thread,
                  check_mode_banner, check_tuning_controls, check_team_key_mapping,
                  check_legend_matches_bindings):
        check(app)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
