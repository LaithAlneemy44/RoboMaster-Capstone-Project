"""The driver-station window: video, controls, telemetry, and the safety interlocks.

Qt owns the UI thread and nothing else may block it. Inference costs 60-400 ms per frame,
so it runs on a worker thread and hands finished frames back by signal. A UI that stalls
while the robot is driving is a safety fault, not a responsiveness complaint.

SAFETY IS STRUCTURAL HERE, NOT A FEATURE

    Deadman        PING at 10 Hz. The firmware must stop if it hears nothing for 300 ms.
                   The GUI cannot enforce that alone - a frozen GUI sends nothing, which
                   is exactly the case the firmware timeout exists to catch. See
                   docs/ROBOT_PROTOCOL.md.
    Link loss      Telemetry silence past LINK_TIMEOUT greys the window and latches
                   disarmed. A GUI that looks live while disconnected is worse than one
                   that crashes.
    E-stop         Esc, or the big button. Latches. Re-arming is deliberate and explicit.
    Firing         Gated behind ARM. Auto-track aims; it never fires. Auto-aim plus
                   auto-fire is not a state anyone should reach by accident.

CONTROLS
    W A S D          drive forward / strafe / back
    Q E              rotate chassis
    arrow keys       turret pitch and yaw
    Space            fire (requires ARM)
    F                intake on/off
    T                auto-track on/off
    Esc              emergency stop
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

import aiming  # noqa: E402
import protocol  # noqa: E402
import vision  # noqa: E402

CONTROL_HZ = 30           # drive/turret command rate
LINK_TIMEOUT = 1.0        # seconds of telemetry silence before the link counts as lost
DRIVE_SPEED = 0.6         # normalised demand for a held key; conservative by default
TURRET_SPEED = 0.5


class VisionWorker(QtCore.QThread):
    """Reads frames and, when enabled, runs the tracker. Never touches widgets."""

    frame_ready = QtCore.Signal(object, list, float)
    failed = QtCore.Signal(str)

    def __init__(self, source, tracker_kwargs: dict) -> None:
        super().__init__()
        self._source_spec = source
        self._tracker_kwargs = tracker_kwargs
        self._running = True
        self._tracking = False
        self._tracker = None

    def set_tracking(self, enabled: bool) -> None:
        self._tracking = enabled

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        try:
            source = vision.FrameSource(self._source_spec)
        except RuntimeError as exc:
            self.failed.emit(str(exc))
            return

        while self._running:
            frame = source.read()
            if frame is None:
                # A file or clip ran out. Loop it, so a recorded source can be used for
                # extended testing without babysitting.
                try:
                    source.close()
                    source = vision.FrameSource(self._source_spec)
                    frame = source.read()
                except RuntimeError as exc:
                    self.failed.emit(str(exc))
                    return
                if frame is None:
                    self.failed.emit("Video source produced no frames.")
                    return

            tracks, fps = [], 0.0
            if self._tracking:
                if self._tracker is None:
                    # Built here, not in __init__: loading weights takes seconds and
                    # would otherwise block whichever thread constructed the worker.
                    self._tracker = vision.AutoTracker(**self._tracker_kwargs)
                tracks = self._tracker.process(frame)
                fps = self._tracker.measured_fps()
            self.frame_ready.emit(frame, tracks, fps)
        source.close()


class RobotWindow(QtWidgets.QMainWindow):
    def __init__(self, driver, source, tracker_kwargs: dict) -> None:
        super().__init__()
        self.driver = driver
        self.aim = aiming.AimController()
        self.setWindowTitle("Robot driver station")

        self.armed = False
        self.estopped = False
        self.intake_on = False
        self.auto_track = False
        self.link_ok = False
        self._keys: set[int] = set()
        self._last_status = None
        self._aim_demand = (0.0, 0.0)
        self._target_id = None

        self._build_ui()

        self.worker = VisionWorker(source, tracker_kwargs)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.failed.connect(self._on_vision_failed)
        self.worker.start()

        self._control_timer = QtCore.QTimer(self)
        self._control_timer.timeout.connect(self._tick_control)
        self._control_timer.start(int(1000 / CONTROL_HZ))

        self._heartbeat_timer = QtCore.QTimer(self)
        self._heartbeat_timer.timeout.connect(self._tick_heartbeat)
        self._heartbeat_timer.start(int(1000 / protocol.HEARTBEAT_HZ))

    # ------------------------------------------------------------------ construction

    def _build_ui(self) -> None:
        self.video = QtWidgets.QLabel("waiting for video…")
        self.video.setMinimumSize(960, 540)
        self.video.setAlignment(QtCore.Qt.AlignCenter)
        self.video.setStyleSheet("background:#111; color:#888;")

        self.btn_estop = QtWidgets.QPushButton("EMERGENCY STOP  (Esc)")
        self.btn_estop.setMinimumHeight(64)
        self.btn_estop.setStyleSheet(
            "background:#b00020; color:white; font-size:18px; font-weight:bold;")
        self.btn_estop.clicked.connect(self.emergency_stop)

        self.btn_arm = QtWidgets.QPushButton("ARM")
        self.btn_arm.setCheckable(True)
        self.btn_arm.setMinimumHeight(40)
        self.btn_arm.toggled.connect(self._on_arm)

        self.btn_track = QtWidgets.QPushButton("AUTO-TRACK  (T)")
        self.btn_track.setCheckable(True)
        self.btn_track.setMinimumHeight(40)
        self.btn_track.toggled.connect(self._on_auto_track)

        self.btn_intake = QtWidgets.QPushButton("INTAKE  (F)")
        self.btn_intake.setCheckable(True)
        self.btn_intake.setMinimumHeight(40)
        self.btn_intake.toggled.connect(self._on_intake)

        self.btn_fire = QtWidgets.QPushButton("FIRE  (Space)")
        self.btn_fire.setMinimumHeight(40)
        self.btn_fire.clicked.connect(self.fire)

        self.lbl_link = QtWidgets.QLabel("link: —")
        self.lbl_batt = QtWidgets.QLabel("battery: —")
        self.lbl_gimbal = QtWidgets.QLabel("turret: —")
        self.lbl_fps = QtWidgets.QLabel("tracker: off")
        self.lbl_target = QtWidgets.QLabel("target: none")
        for label in (self.lbl_link, self.lbl_batt, self.lbl_gimbal,
                      self.lbl_fps, self.lbl_target):
            label.setStyleSheet("font-family: monospace;")

        side = QtWidgets.QVBoxLayout()
        side.addWidget(self.btn_estop)
        side.addSpacing(12)
        for widget in (self.btn_arm, self.btn_track, self.btn_intake, self.btn_fire):
            side.addWidget(widget)
        side.addSpacing(12)
        box = QtWidgets.QGroupBox("telemetry")
        inner = QtWidgets.QVBoxLayout()
        for label in (self.lbl_link, self.lbl_batt, self.lbl_gimbal,
                      self.lbl_fps, self.lbl_target):
            inner.addWidget(label)
        box.setLayout(inner)
        side.addWidget(box)
        side.addStretch(1)
        side.addWidget(QtWidgets.QLabel(
            "WASD drive · QE rotate · arrows turret\nSpace fire · F intake · T track"))

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.video, stretch=1)
        panel = QtWidgets.QWidget()
        panel.setLayout(side)
        panel.setFixedWidth(280)
        layout.addWidget(panel)

        central = QtWidgets.QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    # ------------------------------------------------------------------ vision

    def _on_vision_failed(self, message: str) -> None:
        self.video.setText(f"video failed\n\n{message}")

    def _on_frame(self, frame, tracks, fps) -> None:
        if self.auto_track:
            height, width = frame.shape[:2]
            pitch, yaw, target = self.aim.update(tracks, (width, height))
            self._aim_demand = (pitch, yaw)
            self._target_id = target
        else:
            self._aim_demand = (0.0, 0.0)
            self._target_id = None

        self.lbl_fps.setText(f"tracker: {fps:5.1f} FPS" if self.auto_track
                             else "tracker: off")
        self.lbl_target.setText(f"target: {self._target_id}"
                                if self._target_id is not None else "target: none")
        self.video.setPixmap(self._render(frame, tracks))

    def _render(self, frame, tracks) -> QtGui.QPixmap:
        import cv2  # noqa: PLC0415

        canvas = frame.copy()
        height, width = canvas.shape[:2]

        # Crosshair, so the operator can see where the turret is pointing relative to
        # the aim controller's notion of centre.
        cx, cy = width // 2, height // 2
        cv2.line(canvas, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 1)
        cv2.line(canvas, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 1)

        for track_id, (x, y, w, h) in tracks:
            locked = track_id == self._target_id
            colour = (0, 0, 255) if locked else (0, 200, 0)
            thickness = 3 if locked else 1
            p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
            cv2.rectangle(canvas, p1, p2, colour, thickness)
            cv2.putText(canvas, str(track_id), (int(x), max(12, int(y) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        if not self.link_ok:
            # Unmissable, because a stale-looking picture on a live robot is dangerous.
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
            cv2.putText(canvas, "LINK LOST", (cx - 210, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4)

        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        image = QtGui.QImage(canvas.data, width, height, 3 * width,
                             QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(image).scaled(
            self.video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

    # ------------------------------------------------------------------ control loop

    def _tick_heartbeat(self) -> None:
        self.driver.heartbeat()
        for message in self.driver.poll():
            if isinstance(message, protocol.Status):
                self._last_status = message
        self._refresh_link()

    def _refresh_link(self) -> None:
        was_ok = self.link_ok
        self.link_ok = self.driver.silence_seconds() < LINK_TIMEOUT
        if was_ok and not self.link_ok:
            # Losing the link disarms. Re-arming is an explicit operator action, never
            # something that happens on its own when the radio comes back.
            self._latch_safe("link lost")
        self.lbl_link.setText(f"link: {'OK' if self.link_ok else 'LOST'}")
        self.lbl_link.setStyleSheet(
            "font-family: monospace; color: %s;" % ("#0a0" if self.link_ok else "#b00"))
        if self._last_status is not None:
            status = self._last_status
            self.lbl_batt.setText(f"battery: {status.battery * 100:4.0f}%")
            self.lbl_gimbal.setText(
                f"turret: p{status.pitch:+6.1f} y{status.yaw:6.1f}")

    def _tick_control(self) -> None:
        if self.estopped or not self.link_ok:
            return

        keys = self._keys
        forward = (QtCore.Qt.Key_W in keys) - (QtCore.Qt.Key_S in keys)
        strafe = (QtCore.Qt.Key_D in keys) - (QtCore.Qt.Key_A in keys)
        rotate = (QtCore.Qt.Key_E in keys) - (QtCore.Qt.Key_Q in keys)
        self.driver.send(protocol.drive(forward * DRIVE_SPEED,
                                        strafe * DRIVE_SPEED,
                                        rotate * DRIVE_SPEED))

        if self.auto_track:
            pitch, yaw = self._aim_demand
        else:
            pitch = ((QtCore.Qt.Key_Up in keys) - (QtCore.Qt.Key_Down in keys)) \
                * TURRET_SPEED
            yaw = ((QtCore.Qt.Key_Right in keys) - (QtCore.Qt.Key_Left in keys)) \
                * TURRET_SPEED
        self.driver.send(protocol.gimbal(pitch, yaw))

    # ------------------------------------------------------------------ actions

    def emergency_stop(self) -> None:
        self.estopped = True
        self.driver.emergency_stop()
        self._latch_safe("emergency stop")

    def _latch_safe(self, reason: str) -> None:
        """Drop into a state that needs an explicit operator action to leave."""
        self.armed = False
        self.auto_track = False
        self.intake_on = False
        self.aim.reset()
        for button in (self.btn_arm, self.btn_track, self.btn_intake):
            button.blockSignals(True)
            button.setChecked(False)
            button.blockSignals(False)
        self.worker.set_tracking(False)
        self.statusBar().showMessage(f"SAFE: {reason} — re-arm to continue")

    def _on_arm(self, checked: bool) -> None:
        if checked and self.estopped:
            # Re-arming after an e-stop clears it, which is the one place the latch is
            # released, and it takes a deliberate click to get here.
            self.estopped = False
            self.statusBar().showMessage("re-armed")
        self.armed = checked
        self.btn_arm.setText("ARMED" if checked else "ARM")

    def _on_auto_track(self, checked: bool) -> None:
        self.auto_track = checked
        self.worker.set_tracking(checked)
        if not checked:
            self.aim.reset()
            self._aim_demand = (0.0, 0.0)
            self._target_id = None

    def _on_intake(self, checked: bool) -> None:
        self.intake_on = checked
        self.driver.send(protocol.intake(checked))

    def fire(self) -> None:
        if not self.armed:
            self.statusBar().showMessage("FIRE ignored — not armed")
            return
        if not self.link_ok:
            self.statusBar().showMessage("FIRE ignored — link lost")
            return
        self.driver.send(protocol.fire(1))
        self.statusBar().showMessage("fired")

    # ------------------------------------------------------------------ input

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == QtCore.Qt.Key_Escape:
            self.emergency_stop()
        elif key == QtCore.Qt.Key_Space:
            self.fire()
        elif key == QtCore.Qt.Key_T:
            self.btn_track.toggle()
        elif key == QtCore.Qt.Key_F:
            self.btn_intake.toggle()
        elif not event.isAutoRepeat():
            self._keys.add(key)

    def keyReleaseEvent(self, event) -> None:
        if not event.isAutoRepeat():
            self._keys.discard(event.key())

    def focusOutEvent(self, event) -> None:
        # Losing focus with keys held would leave the robot driving at the last demand.
        self._keys.clear()
        super().focusOutEvent(event)

    def closeEvent(self, event) -> None:
        self.worker.stop()
        self.worker.wait(2000)
        self.driver.close()
        super().closeEvent(event)
