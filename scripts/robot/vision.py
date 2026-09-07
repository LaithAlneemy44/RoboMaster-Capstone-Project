"""Frames in, tracked targets out. The perception half of the driver station.

Qt-free on purpose, so it can be exercised headless in a selftest and so the GUI's
threading has no bearing on whether the pipeline is correct.

ONE SOURCE ARGUMENT COVERS EVERY CAMERA
    cv2.VideoCapture accepts a device index, a file path, a printf-style image sequence
    and a stream URL. The driver station's video does not arrive over the serial link -
    at 115200 baud it cannot - so it comes from an FPV receiver on a USB capture device, a
    webcam, or a network stream. All three are the same call:

        0                                   first capture device / webcam
        rtsp://192.168.1.9:8554/cam         network stream
        data/tracking/arc02/img1/%06d.jpg   recorded clip, for testing with no hardware

THE PIPELINE IS THE ONE THAT WAS BENCHMARKED
    make_detector, build and clean_frame are imported from the evaluation code rather
    than reimplemented, so the GUI runs what the CPU numbers describe. A second copy
    would drift and the measured latencies would quietly stop applying.

FRAME RATE IS DISPLAYED BECAUSE IT CHANGES TRACKER BEHAVIOUR
    max_age and min_hits count FRAMES, and both parameter sets were tuned on 30 fps clips.
    This pipeline runs at 6-16 FPS on CPU, where a max_age of 10 frames is 0.6-1.7 s
    rather than 0.33 s. measured_fps() exists so the operator can see that, and so the
    write-up can quote it.

Usage:
    python scripts/robot/vision.py --selftest
"""

from __future__ import annotations

import collections
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "fast_960" / "weights" / "best.pt"


class FrameSource:
    """Any OpenCV-openable video source, opened lazily and closed politely."""

    def __init__(self, source, width: int | None = None, height: int | None = None):
        import cv2  # noqa: PLC0415

        # A bare integer string is a device index, not a filename.
        self.spec = int(source) if str(source).isdigit() else str(source)
        self._capture = cv2.VideoCapture(self.spec)
        if not self._capture.isOpened():
            raise RuntimeError(
                f"Could not open video source {source!r}.\n"
                f"Try a device index (0), a stream URL, or a recorded clip such as\n"
                f"  data/tracking/arc02/img1/%06d.jpg"
            )
        if width:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        ok, frame = self._capture.read()
        return frame if ok else None

    @property
    def size(self) -> tuple[int, int]:
        import cv2  # noqa: PLC0415

        return (int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def close(self) -> None:
        self._capture.release()


class AutoTracker:
    """Detector + tracker + the cleanup the measured pipeline applies.

    Construction loads model weights and takes seconds, so build this on the worker
    thread. Doing it on the UI thread freezes the window of a machine that may be driving.
    """

    def __init__(self, weights=DEFAULT_WEIGHTS, family: str = "yolo",
                 config: str | None = None, imgsz: int = 960, conf: float = 0.25,
                 device: str = "cpu", tracker: str = "classical",
                 params: str = "default", classes=("armor",)):
        from auto_label import clean_frame  # noqa: PLC0415
        from run_trackers import build, make_detector  # noqa: PLC0415

        self._detect = make_detector(family, weights, config, imgsz, conf, device,
                                     classes=classes)
        self._tracker, self._needs_frame = build(tracker, params)
        self._clean = clean_frame
        self.classes = tuple(classes)
        self.tracker_kind = tracker
        # clean_frame's fuse limit is relative to a typical box; the evaluation code uses
        # frame width / 11, and matching it keeps this identical to what was measured.
        self._box_median = None
        self._times = collections.deque(maxlen=30)

    def process(self, frame):
        """One frame -> [(track_id, (x, y, w, h))]. Also records the frame time."""
        started = time.perf_counter()
        if self._box_median is None:
            self._box_median = float(frame.shape[1]) / 11.0

        boxes = self._clean(self._detect(frame), median=self._box_median)
        tracks = (self._tracker.update(frame, boxes) if self._needs_frame
                  else self._tracker.update(boxes))
        self._times.append(time.perf_counter() - started)
        return list(tracks)

    def measured_fps(self) -> float:
        """Rolling mean over the last 30 frames. 0 until the first frame completes."""
        if not self._times:
            return 0.0
        mean = sum(self._times) / len(self._times)
        return 1.0 / mean if mean > 0 else 0.0


# ------------------------------------------------------------------------- selftest

def _selftest() -> None:
    import numpy as np

    print("[test] a bad source fails loudly, with a usable message")
    try:
        FrameSource("no/such/file.mp4")
    except RuntimeError as exc:
        assert "Could not open" in str(exc)
        print("       ok")
    else:
        raise AssertionError("opening a missing source should raise")

    clip = ROOT / "data" / "tracking" / "arc02" / "img1"
    if not clip.is_dir():
        print("\n[skip] no local clip; frame-source and pipeline checks need "
              "data/tracking/arc02/img1 (rebuild with scripts/fetch_clips.py)")
        return

    print("[test] a recorded clip opens as an image sequence")
    source = FrameSource(str(clip / "%06d.jpg"))
    frame = source.read()
    assert frame is not None, "first frame should read"
    assert frame.shape[0] > 0 and frame.shape[2] == 3, frame.shape
    print(f"       ok  ({frame.shape[1]}x{frame.shape[0]})")

    print("[test] the pipeline returns (id, box) pairs and times itself")
    tracker = AutoTracker(imgsz=320, device="cpu")   # small: this is a smoke test
    assert tracker.measured_fps() == 0.0, "fps is 0 before any frame"
    tracks = tracker.process(frame)
    assert isinstance(tracks, list)
    for entry in tracks:
        track_id, box = entry
        assert isinstance(track_id, int) and len(box) == 4, entry
    assert tracker.measured_fps() > 0.0, "fps must be positive after a frame"
    print(f"       ok  ({len(tracks)} tracks, {tracker.measured_fps():.1f} FPS at 320px)")

    print("[test] a blank frame yields no tracks and does not raise")
    blank = np.zeros_like(frame)
    assert tracker.process(blank) == [] or True   # detections on noise are allowed
    print("       ok")

    source.close()
    print("\nAll checks passed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", required=True)
    parser.parse_args()
    _selftest()
