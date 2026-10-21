"""Link test window: key presses send LED flash commands through the Pico WiFi link.

A stripped-down copy of scripts/robot/gui.py for checking the link end to end with a spare
board (link/mkr_flash_test) before any movement commands exist. No camera, no tracking, no
DRV/GIM protocol: each key sends one line to the LAPTOP Pico, which relays it to the ROBOT
Pico and on to the receiver's Serial1.

    MacBook ──USB──▶ LAPTOP Pico ···WiFi UDP···▶ ROBOT Pico ──UART──▶ receiver LED

Optionally also open the receiver's own USB port to see, in the same window, what it
actually parsed. That closes the loop on the bench. On the robot there is no such cable.

CONTROLS
    1-5      blink that many times
    F        fast flicker
    H        hold on for 1 s
    0, Esc   LED off

Usage:
    .venv/bin/python link/flash_test_gui.py
    .venv/bin/python link/flash_test_gui.py --link /dev/cu.usbmodemA --receiver /dev/cu.usbmodemB
"""

from __future__ import annotations

import argparse
import sys
import time

import serial
from serial.tools import list_ports
from PySide6 import QtCore, QtGui, QtWidgets

BAUD = 115200
POLL_MS = 20

# Qt key -> (pattern code sent on the wire, label). The legend is built from this table so
# the on-screen keys cannot drift from the bindings.
BINDINGS = {
    QtCore.Qt.Key_1: ("1", "1 blink"),
    QtCore.Qt.Key_2: ("2", "2 blinks"),
    QtCore.Qt.Key_3: ("3", "3 blinks"),
    QtCore.Qt.Key_4: ("4", "4 blinks"),
    QtCore.Qt.Key_5: ("5", "5 blinks"),
    QtCore.Qt.Key_F: ("F", "fast flicker"),
    QtCore.Qt.Key_H: ("H", "hold 1 s"),
    QtCore.Qt.Key_0: ("0", "off"),
    QtCore.Qt.Key_Escape: ("0", "off"),
}


def format_command(seq: int, code: str) -> str:
    # Plain ASCII only: LAPTOP's USB serial is also its REPL, and 0x03/0x04 would kill main.py.
    return f"L,{seq},{code}\n"


def port_names() -> list[str]:
    return sorted(p.device for p in list_ports.comports() if "usbmodem" in p.device)


class FlashWindow(QtWidgets.QMainWindow):
    def __init__(self, link_port: str | None, receiver_port: str | None) -> None:
        super().__init__()
        self.link: serial.Serial | None = None
        self.receiver: serial.Serial | None = None
        self._rx_buf = b""
        self.seq = 0
        self.sent: list[str] = []      # every line written, for tests and the log

        self._build_ui()
        self._refresh_ports()
        if link_port:
            self.link_box.setCurrentText(link_port)
            self._open_link()
        if receiver_port:
            self.rx_box.setCurrentText(receiver_port)
            self._open_receiver()
        self._update_title()

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start(POLL_MS)

    # ------------------------------------------------------------------ construction

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        ports = QtWidgets.QGridLayout()
        self.link_box = QtWidgets.QComboBox()
        self.link_box.setEditable(True)
        self.rx_box = QtWidgets.QComboBox()
        self.rx_box.setEditable(True)
        self.btn_link = QtWidgets.QPushButton("Connect")
        self.btn_link.clicked.connect(self._open_link)
        self.btn_rx = QtWidgets.QPushButton("Connect")
        self.btn_rx.clicked.connect(self._open_receiver)
        btn_refresh = QtWidgets.QPushButton("Refresh ports")
        btn_refresh.clicked.connect(self._refresh_ports)
        ports.addWidget(QtWidgets.QLabel("LAPTOP Pico (sends)"), 0, 0)
        ports.addWidget(self.link_box, 0, 1)
        ports.addWidget(self.btn_link, 0, 2)
        ports.addWidget(QtWidgets.QLabel("Receiver USB (optional)"), 1, 0)
        ports.addWidget(self.rx_box, 1, 1)
        ports.addWidget(self.btn_rx, 1, 2)
        ports.addWidget(btn_refresh, 2, 2)
        layout.addLayout(ports)

        buttons = QtWidgets.QHBoxLayout()
        seen = set()
        for key, (code, label) in BINDINGS.items():
            if code in seen:
                continue
            seen.add(code)
            btn = QtWidgets.QPushButton(f"{QtGui.QKeySequence(key).toString()}\n{label}")
            btn.setFocusPolicy(QtCore.Qt.NoFocus)   # keep keys going to the window
            btn.clicked.connect(lambda _=False, c=code: self.send_code(c))
            buttons.addWidget(btn)
        layout.addLayout(buttons)

        self.last_label = QtWidgets.QLabel("nothing sent yet")
        font = self.last_label.font()
        font.setPointSize(font.pointSize() + 6)
        self.last_label.setFont(font)
        layout.addWidget(self.last_label)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFocusPolicy(QtCore.Qt.NoFocus)
        self.log.setMaximumBlockCount(2000)
        layout.addWidget(self.log)

        self.setCentralWidget(central)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.resize(820, 520)

    def _update_title(self) -> None:
        state = self.link.port if self.link else "NOT CONNECTED"
        self.setWindowTitle(f"Link flash test  —  sending to: {state}")

    def _refresh_ports(self) -> None:
        names = port_names()
        for box in (self.link_box, self.rx_box):
            current = box.currentText()
            box.clear()
            box.addItems(names)
            if current:
                box.setCurrentText(current)

    def _open(self, box: QtWidgets.QComboBox, other: serial.Serial | None):
        name = box.currentText().strip()
        if not name:
            self._log("no port selected")
            return None
        if other is not None and other.port == name:
            self._log(f"{name} is already open as the other port")
            return None
        try:
            port = serial.Serial(name, BAUD, timeout=0, write_timeout=0)
        except serial.SerialException as exc:
            self._log(f"could not open {name}: {exc}")
            return None
        self._log(f"opened {name}")
        return port

    def _open_link(self) -> None:
        if self.link:
            self.link.close()
        self.link = self._open(self.link_box, self.receiver)
        self._update_title()

    def _open_receiver(self) -> None:
        if self.receiver:
            self.receiver.close()
        self.receiver = self._open(self.rx_box, self.link)

    # ------------------------------------------------------------------ sending

    def send_code(self, code: str) -> None:
        line = format_command(self.seq, code)
        self.seq += 1
        self.sent.append(line)
        if self.link is None:
            self._log(f"(not connected) would send {line.strip()}")
        else:
            try:
                self.link.write(line.encode("ascii"))
            except serial.SerialException as exc:
                self._log(f"write failed, link closed: {exc}")
                self.link.close()
                self.link = None
                self._update_title()
                return
            self._log(f"sent {line.strip()}")
        self.last_label.setText(f"last sent: {line.strip()}")

    def _poll(self) -> None:
        # Drain both ports so neither buffer fills; only the receiver's output is shown.
        for name in ("link", "receiver"):
            port = getattr(self, name)
            if port is None:
                continue
            try:
                data = port.read(4096)
            except serial.SerialException as exc:
                self._log(f"{name} port lost: {exc}")
                port.close()
                setattr(self, name, None)
                self._update_title()
                continue
            if name == "receiver" and data:
                self._rx_buf += data
                *lines, self._rx_buf = self._rx_buf.split(b"\n")
                for raw in lines:
                    text = raw.decode(errors="replace").strip()
                    if text:
                        self._log(f"    receiver: {text}")

    def _log(self, text: str) -> None:
        self.log.appendPlainText(f"{time.strftime('%H:%M:%S')}  {text}")

    # ------------------------------------------------------------------ events

    def keyPressEvent(self, event) -> None:
        binding = BINDINGS.get(event.key())
        if binding is None or event.isAutoRepeat():
            super().keyPressEvent(event)
            return
        self.send_code(binding[0])

    def closeEvent(self, event) -> None:
        for port in (self.link, self.receiver):
            if port:
                port.close()
        super().closeEvent(event)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--link", help="LAPTOP Pico port, e.g. /dev/cu.usbmodem1101")
    parser.add_argument("--receiver", help="receiver board's USB port, optional")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    window = FlashWindow(args.link, args.receiver)
    window.show()
    window.activateWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
