"""Classical multi-object tracker: kinematic prediction refined by a Kalman filter.

The classical arm of the tracking comparison, built rather than imported, per CLAUDE.md:
kinematics to predict where a target is going, Kalman filtering to refine it, no
training. Behaviour comes entirely from the parameters below, tuned on the validation
split as proposal 5.3 describes.

HOW THIS DIFFERS FROM SORT, DELIBERATELY
    SORT is also Kalman plus association, so building this the obvious way would produce
    the same algorithm twice and a comparison between them would be vacuous. Three axes
    separate them, and the write-up should say so:

                      classical (here)              SORT (scripts/sort.py)
        state         box centre + velocity         [u, v, s, r, u', v', s']
        size          carried from last detection   filtered as scale and aspect
        association   greedy nearest-neighbour      Hungarian, globally optimal
        filter        cv2.KalmanFilter              published formulation, numpy

    Greedy versus optimal assignment is a real difference that shows up whenever two
    targets pass close to each other, which in a RoboMaster match is constantly.

PREDICTED, NOT LAST-SEEN
    Association matches detections against each track's PREDICTED box, not its last
    observed one. scripts/pre_annotate.py hit the alternative: matching a stale box after
    a few missed frames fails once the robot has moved, and one robot becomes a string of
    short tracks.

Usage:
    python scripts/classical_tracker.py --selftest
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class Params:
    """Tunable on the validation split. No training - these ARE the model."""

    iou_gate: float = 0.20      # below this, a detection cannot claim a track
    max_age: int = 10           # frames a track survives unmatched before it dies
    min_hits: int = 2           # detections before a track is reported at all
    process_noise: float = 1e-2      # trust in the constant-velocity assumption
    measurement_noise: float = 1e-1  # trust in the detector


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (aw * ah + bw * bh - inter)


class Track:
    """One target: a constant-velocity Kalman filter over the box centre.

    Width and height are carried from the most recent detection rather than filtered.
    That is the deliberate simplification against SORT, which estimates scale and aspect
    as part of its state - a robot's apparent size changes far more slowly than its
    position, so filtering it buys little and costs two more state dimensions.
    """

    def __init__(self, track_id: int, box, params: Params):
        import cv2
        import numpy as np

        self.id = track_id
        self.p = params
        self.w, self.h = box[2], box[3]
        self.hits = 1
        self.age = 0
        self.time_since_update = 0

        # state = [cx, cy, vx, vy]; measurement = [cx, cy]
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * params.process_noise
        self.kf.measurementNoiseCov = (
            np.eye(2, dtype=np.float32) * params.measurement_noise)
        self.kf.statePost = np.array(
            [[box[0] + box[2] / 2], [box[1] + box[3] / 2], [0], [0]], dtype=np.float32)

    def predict(self):
        """Advance the filter one frame and return the predicted box."""
        state = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        # state is (4, 1); NumPy 2 refuses to coerce a 1-element array to a scalar.
        cx, cy = float(state[0, 0]), float(state[1, 0])
        return (cx - self.w / 2, cy - self.h / 2, self.w, self.h)

    def update(self, box) -> None:
        import numpy as np

        self.kf.correct(np.array(
            [[box[0] + box[2] / 2], [box[1] + box[3] / 2]], dtype=np.float32))
        self.w, self.h = box[2], box[3]
        self.hits += 1
        self.time_since_update = 0

    def box(self):
        state = self.kf.statePost
        cx, cy = float(state[0, 0]), float(state[1, 0])
        return (cx - self.w / 2, cy - self.h / 2, self.w, self.h)


class ClassicalTracker:
    """Greedy IoU association over Kalman-predicted boxes."""

    def __init__(self, params: Params | None = None):
        self.p = params or Params()
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections):
        """One frame. `detections` is [(x, y, w, h), ...]. Returns [(id, box), ...]."""
        predictions = [(t, t.predict()) for t in self.tracks]

        # Greedy: take the best pair available, remove both, repeat. Not optimal - that
        # is SORT's job, and the gap between them is the point of running both.
        pairs = sorted(
            ((iou(pred, det), ti, di)
             for ti, (_, pred) in enumerate(predictions)
             for di, det in enumerate(detections)),
            reverse=True,
        )
        used_tracks, used_dets = set(), set()
        for score, ti, di in pairs:
            if score < self.p.iou_gate:
                break
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            predictions[ti][0].update(detections[di])

        for di, det in enumerate(detections):
            if di not in used_dets:
                self.tracks.append(Track(self._next_id, det, self.p))
                self._next_id += 1

        self.tracks = [t for t in self.tracks
                       if t.time_since_update <= self.p.max_age]

        # A track is reported once it has been seen enough times to be trusted, and only
        # on frames where it was actually observed - reporting a coasting track as a
        # detection would invent objects the pipeline never saw.
        return [(t.id, t.box()) for t in self.tracks
                if t.hits >= self.p.min_hits and t.time_since_update == 0]


# ------------------------------------------------------------------------- selftest

def _linear_scene(n_frames=30, n_objects=3, blank=None):
    """Objects on known straight paths. `blank` = (object index, frames to hide)."""
    frames = []
    for f in range(n_frames):
        dets = []
        for o in range(n_objects):
            if blank and o == blank[0] and f in blank[1]:
                continue
            dets.append((50.0 + o * 120 + f * 4.0, 60.0 + o * 40, 30.0, 30.0))
        frames.append(dets)
    return frames


def selftest() -> None:
    print("[test] constant-velocity recovery")
    tracker = ClassicalTracker()
    seen = {}
    for dets in _linear_scene():
        for tid, _ in tracker.update(dets):
            seen.setdefault(tid, 0)
            seen[tid] += 1
    assert len(seen) == 3, f"expected 3 ids, got {len(seen)}: {seen}"
    print(f"       ok - 3 objects, 3 ids, no fragmentation {dict(seen)}")

    print("[test] occlusion shorter than max_age keeps the id")
    tracker = ClassicalTracker(Params(max_age=10))
    ids = set()
    for dets in _linear_scene(blank=(1, set(range(10, 15)))):
        ids.update(tid for tid, _ in tracker.update(dets))
    assert len(ids) == 3, f"a 5-frame gap under max_age=10 split a track: {ids}"
    print(f"       ok - track survived a 5-frame gap, {len(ids)} ids total")

    print("[test] occlusion longer than max_age ends the track")
    tracker = ClassicalTracker(Params(max_age=3))
    ids = set()
    for dets in _linear_scene(blank=(1, set(range(10, 22)))):
        ids.update(tid for tid, _ in tracker.update(dets))
    assert len(ids) == 4, f"a 12-frame gap over max_age=3 should start a new id: {ids}"
    print(f"       ok - track ended and a new id began, {len(ids)} ids total")

    print("\nAll checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        parser.error("--selftest is the only mode; use scripts/run_trackers.py to run "
                     "this over a sequence")


if __name__ == "__main__":
    main()
