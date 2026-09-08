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
DRIVE_SPEED = 0.6         # starting demand for a held key; conservative, and both are
TURRET_SPEED = 0.5        # adjustable at runtime - see _build_tuning()


class VisionWorker(QtCore.QThread):
    """Reads frames and, when enabled, runs the tracker. Never touches widgets."""

    frame_ready = QtCore.Signal(object, list, float)
    failed = QtCore.Signal(str)
    exposure_state = QtCore.Signal(str)

    def __init__(self, source, tracker_kwargs: dict,
                 lock_exposure: bool = False, exposure_abs=None) -> None:
        super().__init__()
        self._source_spec = source
        self._tracker_kwargs = tracker_kwargs
        self._lock_exposure = lock_exposure
        self._exposure_abs = exposure_abs
        self._running = True
        self._tracking = False
        self._tracker = None

    def set_tracking(self, enabled: bool) -> None:
        self._tracking = enabled

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        try:
            source = vision.FrameSource(self._source_spec,
                                        lock_exposure=self._lock_exposure,
                                        exposure_abs=self._exposure_abs)
        except RuntimeError as exc:
            self.failed.emit(str(exc))
            return
        if source.exposure_status:
            self.exposure_state.emit(source.exposure_status)

        while self._running:
            frame = source.read()
            if frame is None:
                # A file or clip ran out. Loop it, so a recorded source can be used for
                # extended testing without babysitting.
                try:
                    source.close()
                    source = vision.FrameSource(self._source_spec,
                                                lock_exposure=self._lock_exposure,
                                                exposure_abs=self._exposure_abs)
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
    def __init__(self, driver, source, tracker_kwargs: dict,
                 lock_exposure: bool = False, exposure_abs=None) -> None:
        super().__init__()
        self.driver = driver
        self._source = source
        self.aim = aiming.AimController()

        self.armed = False
        self.estopped = False
        self.intake_on = False
        self.auto_track = False
        self.link_ok = False
        self._keys: set[int] = set()
        self._last_status = None
        self._aim_demand = (0.0, 0.0)
        self._target_id = None
        # Live state, not module constants: a driver wants a slow precision mode to line
        # up a shot and full speed to cross the field, and switching between them should
        # not need a code edit and a restart.
        self.drive_speed = DRIVE_SPEED
        self.turret_speed = TURRET_SPEED

        self._build_ui()
        text, live = self._mode()
        self.setWindowTitle(
            f"Robot driver station  —  {'LIVE' if live else 'SIMULATION'}  —  {text}")

        self.worker = VisionWorker(source, tracker_kwargs,
                                   lock_exposure=lock_exposure,
                                   exposure_abs=exposure_abs)
        self.worker.exposure_state.connect(self._on_exposure_state)
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

    def _mode(self) -> tuple[str, bool]:
        """(banner text, everything is live).

        Two independent axes - the robot link and the video source - and the station is
        only LIVE when both are. Anything else is simulation and says so, because the
        one unacceptable failure here is believing a recorded clip and a mock robot are
        the real thing.
        """
        robot_label, robot_real = self.driver.describe
        spec = str(self._source)
        video_real = spec.isdigit() or "://" in spec
        video_label = (f"device {spec}" if spec.isdigit() else
                       "stream" if "://" in spec else "recorded clip")
        live = robot_real and video_real
        return f"robot: {robot_label}  ·  video: {video_label}", live

    def _spin(self, low, high, step, value, setter):
        box = QtWidgets.QDoubleSpinBox()
        box.setRange(low, high)
        box.setSingleStep(step)
        box.setDecimals(3)
        box.setValue(value)
        box.valueChanged.connect(setter)
        return box

    def _build_tuning(self) -> QtWidgets.QGroupBox:
        """Speeds and aim gains, adjustable while running.

        The aim gains cannot be tuned without the physical gimbal - there is no way to
        learn from recorded video how hard to push a real motor - so they must be
        reachable during a bench session rather than requiring a code edit and a restart
        between every attempt. The approved plan called for exactly this.
        """
        config = self.aim.config

        def set_gain_yaw(v):
            config.gain_yaw = v

        def set_gain_pitch(v):
            config.gain_pitch = v

        def set_deadband(v):
            config.deadband = v

        def set_max_rate(v):
            config.max_rate = v

        def set_drive(v):
            self.drive_speed = v

        def set_turret(v):
            self.turret_speed = v

        rows = (
            ("drive speed", self._spin(0.05, 1.0, 0.05, self.drive_speed, set_drive)),
            ("turret speed", self._spin(0.05, 1.0, 0.05, self.turret_speed, set_turret)),
            ("aim gain yaw", self._spin(0.0, 5.0, 0.1, config.gain_yaw, set_gain_yaw)),
            ("aim gain pitch",
             self._spin(0.0, 5.0, 0.1, config.gain_pitch, set_gain_pitch)),
            ("aim deadband", self._spin(0.0, 0.5, 0.01, config.deadband, set_deadband)),
            ("aim max rate", self._spin(0.05, 1.0, 0.05, config.max_rate, set_max_rate)),
        )
        form = QtWidgets.QFormLayout()
        form.setContentsMargins(6, 6, 6, 6)
        form.setSpacing(4)
        self.tuning_widgets = {}
        for label, widget in rows:
            form.addRow(label, widget)
            self.tuning_widgets[label] = widget
        box = QtWidgets.QGroupBox("tuning (aim gains are UNTUNED)")
        box.setLayout(form)
        return box

    def _build_ui(self) -> None:
        text, live = self._mode()
        self.banner = QtWidgets.QLabel(
            ("LIVE  —  " if live else "SIMULATION  —  ") + text)
        self.banner.setAlignment(QtCore.Qt.AlignCenter)
        self.banner.setMinimumHeight(30)
        self.banner.setStyleSheet(
            "background:#1b5e20; color:white; font-weight:bold; font-size:14px;"
            if live else
            "background:#e65100; color:white; font-weight:bold; font-size:14px;")

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
        side.addWidget(self._build_tuning())
        side.addStretch(1)
        side.addWidget(QtWidgets.QLabel(
            "WASD drive · QE rotate · arrows turret\nSpace fire · F intake · T track"))

        columns = QtWidgets.QHBoxLayout()
        columns.addWidget(self.video, stretch=1)
        panel = QtWidgets.QWidget()
        panel.setLayout(side)
        panel.setFixedWidth(300)
        columns.addWidget(panel)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.banner)
        layout.addLayout(columns)

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
        # Left/right arrows rotate the chassis, per the team's agreed mapping. The turret
        # moved to IJKL to free them: splitting the arrow cluster between chassis and
        # turret is the kind of thing that gets pressed wrong under pressure.
        rotate = (QtCore.Qt.Key_Right in keys) - (QtCore.Qt.Key_Left in keys)
        self.driver.send(protocol.drive(forward * self.drive_speed,
                                        strafe * self.drive_speed,
                                        rotate * self.drive_speed))

        if self.auto_track:
            pitch, yaw = self._aim_demand
        else:
            pitch = ((QtCore.Qt.Key_I in keys) - (QtCore.Qt.Key_K in keys)) \
                * self.turret_speed
            yaw = ((QtCore.Qt.Key_L in keys) - (QtCore.Qt.Key_J in keys)) \
                * self.turret_speed
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
        elif key in (QtCore.Qt.Key_Plus, QtCore.Qt.Key_Equal):
            self._nudge_speed(+0.05)
        elif key in (QtCore.Qt.Key_Minus, QtCore.Qt.Key_Underscore):
            self._nudge_speed(-0.05)
        elif not event.isAutoRepeat():
            self._keys.add(key)

    def _on_exposure_state(self, message: str) -> None:
        """Say out loud whether exposure is actually locked.

        A failed lock is not cosmetic: the classical detector gates on absolute
        brightness, so an unlocked camera changes what the detector sees as the turret
        pans. Reported in the status bar and the console rather than swallowed, because
        the failure mode otherwise only shows up as inconsistent detection much later.
        """
        self.statusBar().showMessage(message, 10000)
        print(message)

    def _nudge_speed(self, delta: float) -> None:
        """+/- step the drive speed, as the team's key map specifies.

        Driven through the spin box rather than the attribute, so the displayed value and
        the value actually sent can never disagree - an operator who cannot see the
        current speed is guessing.
        """
        box = self.tuning_widgets["drive speed"]
        box.setValue(max(box.minimum(), min(box.maximum(), box.value() + delta)))
        self.statusBar().showMessage(f"drive speed {box.value():.2f}", 1500)

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
