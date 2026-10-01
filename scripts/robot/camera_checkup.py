"""Camera bring-up: measure what the camera actually does, before trusting it.

Camera day should be measurement, not debugging. Every number this prints is one that
CAMERA_HANDOFF.md currently marks "not measured", or one that decides whether the
perception stack can work at all with this lens at this distance.

WHAT IT CHECKS, AND WHY EACH ONE MATTERS

  1. NEGOTIATED FORMAT. A webcam advertises many modes and silently gives you whichever
     one it can sustain. MJPEG means paying a decode per frame - measured at 6.0-16.7 ms
     for 1920x1080 elsewhere in this project, and excluded from every latency figure in
     the tables. YUY2 avoids it. You cannot know which you got without asking.

  2. DELIVERED FRAME RATE. Requesting 30 fps and receiving it are different things,
     especially under poor light where cameras silently extend exposure and halve the
     rate. Every tracker here counts FRAMES, not seconds (see section 5 of the handoff),
     so a camera that quietly drops to 15 fps changes max_age from 0.33 s to 0.67 s.

  3. CAPTURE LATENCY AND JITTER. grab() to retrieve(), plus frame-to-frame interval.
     A *variable* interval is worse for a constant-velocity Kalman filter than a slow
     but steady one, because it breaks the dt assumption differently every frame.

  4. EXPOSURE CONTROL. The classical detector gates on absolute brightness
     (value_min=200 in the tuned config). Auto-exposure hunting as the turret pans slides
     the image under that threshold. Reports whether the lock is reachable here.

  5. ARMOR PLATE PIXEL SIZE - the one that decides everything. The models were trained on
     plates with a median size of 25 x 22 px. Lens field of view and engagement distance
     together determine what you actually get. Halving plate pixels is equivalent to
     halving network input, and that step is measured in this project as armor AP
     0.2943 -> 0.0068. Click two corners of a real plate and compare.

Usage:
    python scripts/robot/camera_checkup.py --device 0
    python scripts/robot/camera_checkup.py --device 0 --width 1280 --height 720
    python scripts/robot/camera_checkup.py --list
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "robot"))

TRAINED_PLATE = (25, 22)   # median armor plate in the ROCO training data, px at 1920x1080


def fourcc_of(cap) -> str:
    import cv2  # noqa: PLC0415

    value = int(cap.get(cv2.CAP_PROP_FOURCC))
    if not value:
        return "unknown"
    text = "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))
    return text if text.isprintable() else f"raw:{value}"


def list_devices(limit: int = 6) -> None:
    import cv2  # noqa: PLC0415

    print("probing device indices (an unavailable index may just be in use)\n")
    for index in range(limit):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"  [{index}] unavailable")
            cap.release()
            continue
        ok, _ = cap.read()
        print(f"  [{index}] {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}  {fourcc_of(cap)}"
              f"  first frame: {'ok' if ok else 'FAILED'}")
        cap.release()


def report_format(cap, requested) -> None:
    import cv2  # noqa: PLC0415

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = fourcc_of(cap)
    print(f"  resolution  : {width}x{height}" +
          (f"   (requested {requested[0]}x{requested[1]})" if requested else ""))
    print(f"  pixel format: {codec}")
    if codec.upper().startswith(("MJPG", "JPEG")):
        print("                MJPEG - you pay a JPEG decode per frame. Every latency")
        print("                figure in CAMERA_HANDOFF excludes decode, so add it back.")
    elif codec.upper() in ("YUY2", "YUYV", "NV12", "I420", "BGR3"):
        print("                uncompressed - no decode cost, higher USB bandwidth.")
    print(f"  driver fps  : {cap.get(cv2.CAP_PROP_FPS):.1f}  (what it claims, not what")
    print( "                it delivers - measured below)")


def measure_throughput(cap, n: int = 120) -> dict:
    """Delivered frame rate, capture latency, and interval jitter."""
    import cv2  # noqa: PLC0415

    for _ in range(10):            # let auto-exposure and the pipeline settle
        cap.read()

    intervals, retrieves = [], []
    last = time.perf_counter()
    failures = 0
    for _ in range(n):
        t0 = time.perf_counter()
        grabbed = cap.grab()
        t1 = time.perf_counter()
        ok, _frame = cap.retrieve() if grabbed else (False, None)
        t2 = time.perf_counter()
        if not ok:
            failures += 1
            continue
        retrieves.append((t2 - t1) * 1e3)
        intervals.append((t2 - last) * 1e3)
        last = t2

    if not intervals:
        return {"failed": True, "failures": failures}
    return {
        "failed": False,
        "failures": failures,
        "fps": 1000.0 / statistics.fmean(intervals),
        "interval_mean": statistics.fmean(intervals),
        "interval_sd": statistics.pstdev(intervals) if len(intervals) > 1 else 0.0,
        "interval_p95": sorted(intervals)[int(0.95 * len(intervals)) - 1],
        "retrieve_mean": statistics.fmean(retrieves),
    }


def report_exposure(device: int) -> None:
    import camera  # noqa: PLC0415

    print(f"  platform    : {sys.platform}")
    if not camera.supported():
        print("  exposure    : no lock path implemented here. macOS uses uvc-util;")
        print("                Linux would use v4l2-ctl. On the deployment Mac, run")
        print("                `python scripts/robot/camera.py -I 0` to list controls.")
        return
    if not camera.tool_available():
        print("  exposure    : uvc-util NOT on PATH - install it, or the camera hunts")
        print("                its own exposure while the turret pans.")
        return
    controls = camera.list_controls(device)
    print(f"  uvc controls: {len(controls)} advertised")
    auto = camera.resolve(camera.AUTO_MODE_CANDIDATES, controls)
    abs_ctrl = camera.resolve(camera.EXPOSURE_ABS_CANDIDATES, controls)
    print(f"  auto-exposure control : {auto or 'NOT FOUND'}")
    print(f"  exposure-time control : {abs_ctrl or 'NOT FOUND'}")
    if auto:
        report = camera.apply_lock(device, None)
        print(f"  lock test   : {camera.describe(report)}")


def measure_plate(cap) -> None:
    """Click two opposite corners of an armor plate; compare against the training median.

    This is the number that decides whether any of the detectors can work at this
    distance with this lens. Everything else on this page is plumbing by comparison.
    """
    import cv2  # noqa: PLC0415

    print("\n--- plate measurement ---")
    print("  Point the camera at an armor plate at REAL engagement distance.")
    print("  Click one corner, then the opposite corner. 'r' resets, 'q' finishes.")

    clicks: list[tuple[int, int]] = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    window = "plate measurement - click two corners, q to finish"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("  capture failed")
            break
        for point in clicks[-2:]:
            cv2.circle(frame, point, 5, (0, 255, 255), 2)
        if len(clicks) >= 2:
            (x1, y1), (x2, y2) = clicks[-2], clicks[-1]
            w, h = abs(x2 - x1), abs(y2 - y1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            ratio_w = w / TRAINED_PLATE[0]
            ratio_h = h / TRAINED_PLATE[1]
            label = f"{w}x{h}px  vs trained {TRAINED_PLATE[0]}x{TRAINED_PLATE[1]}  ({ratio_w:.2f}x)"
            cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
        cv2.imshow(window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            clicks.clear()
        elif key in (ord("q"), 27):
            break

    cv2.destroyWindow(window)

    if len(clicks) >= 2:
        (x1, y1), (x2, y2) = clicks[-2], clicks[-1]
        w, h = abs(x2 - x1), abs(y2 - y1)
        print(f"\n  measured plate : {w} x {h} px")
        print(f"  training median: {TRAINED_PLATE[0]} x {TRAINED_PLATE[1]} px")
        scale = (w / TRAINED_PLATE[0] + h / TRAINED_PLATE[1]) / 2
        print(f"  ratio          : {scale:.2f}x")
        if scale >= 0.9:
            print("  VERDICT: at or above what the models were trained on. Detection")
            print("           should behave as the accuracy tables describe.")
        elif scale >= 0.6:
            print("  VERDICT: smaller than training. Expect armor AP below the table")
            print("           figures. Move closer, narrow the FOV, or raise imgsz.")
        else:
            print("  VERDICT: FAR below training size. Halving plate pixels is")
            print("           equivalent to halving network input, measured in this")
            print("           project as armor AP 0.2943 -> 0.0068. Fix the optics or")
            print("           the engagement distance before trusting any detection.")
    else:
        print("  no measurement taken")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--frames", type=int, default=120,
                        help="Frames to time for the throughput measurement.")
    parser.add_argument("--list", action="store_true", help="Probe indices and exit.")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Skip the plate measurement (needs a display).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import cv2  # noqa: PLC0415

    if args.list:
        list_devices()
        return

    requested = (args.width, args.height) if args.width and args.height else None
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        sys.exit(f"Could not open device {args.device}. Try --list.")
    if requested:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    # A webcam buffers frames; without this the pipeline reads stale ones and every
    # latency figure is optimistic by however deep the queue is.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"\n=== device {args.device} ===")
    report_format(cap, requested)

    print(f"\n=== throughput over {args.frames} frames ===")
    stats = measure_throughput(cap, args.frames)
    if stats["failed"]:
        print(f"  no frames captured ({stats['failures']} failures)")
    else:
        print(f"  delivered fps : {stats['fps']:.1f}")
        print(f"  interval      : {stats['interval_mean']:.1f} ms mean, "
              f"{stats['interval_sd']:.1f} ms sd, {stats['interval_p95']:.1f} ms p95")
        print(f"  retrieve cost : {stats['retrieve_mean']:.2f} ms")
        print(f"  dropped       : {stats['failures']}")
        if stats["interval_sd"] > 0.25 * stats["interval_mean"]:
            print("  WARNING jitter is high relative to the interval. A variable frame")
            print("          interval breaks the trackers' dt = 1 frame assumption")
            print("          differently every frame - a fixed-rate loop that drops")
            print("          frames is kinder than a free-running one (handoff S5).")

    print("\n=== exposure control ===")
    report_exposure(args.device)

    if not args.no_interactive:
        try:
            measure_plate(cap)
        except cv2.error as exc:
            print(f"\n  interactive measurement unavailable (no display?): {exc}")

    cap.release()
    print("\nNext: run the driver station against this camera end to end -")
    print(f"  python scripts/robot_gui.py --driver mock --source {args.device}")


if __name__ == "__main__":
    main()
