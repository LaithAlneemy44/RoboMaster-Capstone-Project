"""Driver station for the robot: video, manual control, and automatic tracking.

Runs on the OPERATOR'S laptop, not on the robot. Control goes out over a serial port;
video arrives separately, because a 115200-baud link cannot carry it.

"Serial" here means a port, not a cable: a direct USB lead, a Bluetooth SPP pairing and a
2.4 GHz radio dongle all enumerate the same way, so bench testing and competition use run
identical code.

WHAT AUTO-TRACK IS, AND IS NOT
    The tracker runs HERE, on the laptop, because this is where the frames are. That makes
    this a driver station, not the on-robot deployment the project's CPU benchmarks
    describe - those say what this pipeline would cost on the robot's own board. Auto-track
    aims the turret. It never fires.

Usage:
    # no hardware needed - a recorded clip and a simulated robot
    python scripts/robot_gui.py --driver mock \\
        --source data/tracking/arc02/img1/%06d.jpg

    # real robot, webcam or capture device
    python scripts/robot_gui.py --driver serial --port COM3 --source 0

    # everything except the window, for CI or a headless check
    python scripts/robot_gui.py --driver mock --source 0 --check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "robot"))

DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "fast_960" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--driver", choices=("mock", "serial"), default="mock",
                        help="'mock' needs no hardware and logs every command.")
    parser.add_argument("--port", default=None,
                        help="Serial port, e.g. COM3 or /dev/ttyUSB0.")
    parser.add_argument("--source", default="0",
                        help="Device index, stream URL, or clip such as "
                             "data/tracking/arc02/img1/%%06d.jpg")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                        help="Detector weights. fast_960 by default: the highest armor "
                             "AP measured (0.4537) at 62.6 ms on six cores.")
    parser.add_argument("--family", choices=("yolo", "ssd", "frcnn", "classical"),
                        default="yolo")
    parser.add_argument("--config", default=None, help="Classical detector config name.")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu",
                        help="Detector device. cpu is the deployment case.")
    parser.add_argument("--tracker", choices=("classical", "sort", "vit", "goturn"),
                        default="classical")
    parser.add_argument("--params", choices=("default", "tuned"), default="default",
                        help="Classical tracker parameters. 'default' on purpose: TUNED "
                             "was tuned on 30 fps clips and its max_age of 30 frames "
                             "becomes 2-5 seconds at the 6-16 FPS this pipeline runs at.")
    parser.add_argument("--target", default="armor",
                        help="Class to aim at. armor is the actual aim point; only the "
                             "YOLO family detects it well.")
    parser.add_argument("--check", action="store_true",
                        help="Build everything and exit without showing a window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from driver import open_driver  # noqa: PLC0415

    tracker_kwargs = {
        "weights": args.weights, "family": args.family, "config": args.config,
        "imgsz": args.imgsz, "conf": args.conf, "device": args.device,
        "tracker": args.tracker, "params": args.params,
        "classes": (args.target,),
    }

    if args.family != "classical" and not Path(args.weights).is_file():
        sys.exit(f"Missing detector weights: {args.weights}\n"
                 f"Train them, or pass --weights / --family classical --config strict.")

    robot = open_driver(args.driver, args.port)

    from PySide6 import QtWidgets  # noqa: PLC0415

    from gui import RobotWindow  # noqa: PLC0415

    app = QtWidgets.QApplication(sys.argv)
    window = RobotWindow(robot, args.source, tracker_kwargs)

    if args.check:
        # Construct everything, prove it holds together, and leave without a window.
        # This is what runs where there is no display.
        window.worker.stop()
        window.worker.wait(2000)
        robot.close()
        print("check OK: driver, window and worker constructed and shut down cleanly")
        return

    window.resize(1280, 640)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
