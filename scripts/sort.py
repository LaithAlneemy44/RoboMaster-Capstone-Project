"""SORT: Simple Online and Realtime Tracking - Kalman prediction, Hungarian assignment.

Implemented from the published formulation rather than vendored. CLAUDE.md flags SORT
as a dependency-rot risk, and the reference repository is old enough that pinning it
would fight the NumPy and OpenCV versions this project needs (motmetrics already needed
a shim for exactly that). Implementing it also makes every parameter inspectable and
tunable on the validation split, which 5.3 requires.

WHAT MAKES THIS *NOT* THE CLASSICAL TRACKER
    scripts/classical_tracker.py is also Kalman plus association. The two are separated
    deliberately, or the comparison between them would be vacuous:

                      SORT (here)                  classical
        state         [u, v, s, r, u', v', s']     box centre + velocity
        size          filtered as scale + aspect   carried from last detection
        association   Hungarian, globally optimal  greedy nearest-neighbour
        filter        published formulation        cv2.KalmanFilter

    The association difference is the interesting one. Greedy takes the best available
    pair and commits; Hungarian minimises total cost across all pairs at once. They
    diverge exactly when two targets pass close together - which in a RoboMaster match
    is most of the time.

STATE PARAMETERISATION
    SORT tracks scale (area) and aspect rather than width and height, on the argument
    that area changes smoothly as a target approaches while aspect is near-constant.
    Velocity is modelled for centre and scale but NOT aspect, which is why the state has
    seven terms and not eight.

Usage:
    python scripts/sort.py --selftest
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class Params:
    """Tunable on the validation split, matching classical_tracker.Params so the two
    can be tuned on equal terms."""

    iou_gate: float = 0.20
    max_age: int = 10
    min_hits: int = 2


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (aw * ah + bw * bh - inter)


def box_to_z(box):
    """(x, y, w, h) -> SORT's measurement [centre x, centre y, scale, aspect]."""
    import numpy as np

    x, y, w, h = box
    w, h = max(w, 1e-6), max(h, 1e-6)
    return np.array([[x + w / 2], [y + h / 2], [w * h], [w / h]], dtype=np.float64)


def x_to_box(state):
    """SORT state -> (x, y, w, h). Scale is an AREA, so width is its square root."""
    import numpy as np

    cx, cy, s, r = (float(state[i, 0]) for i in range(4))
    s, r = max(s, 1e-6), max(r, 1e-6)
    w = float(np.sqrt(s * r))
    h = s / w if w > 0 else 0.0
    return (cx - w / 2, cy - h / 2, w, h)


class KalmanBoxTracker:
    """One target under SORT's constant-velocity model over centre and scale."""

    count = 0

    def __init__(self, box, params: Params):
        import numpy as np

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.p = params
        self.hits = 1
        self.time_since_update = 0

        # x = [u, v, s, r, u', v', s']
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.zeros((4, 7))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0        # scale and aspect are noisier than position
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0      # velocities start almost entirely unknown
        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01        # aspect drifts slowly
        self.Q[4:, 4:] *= 0.01

        self.x = np.zeros((7, 1))
        self.x[:4] = box_to_z(box)

    def predict(self):
        import numpy as np

        # A negative predicted area is physically meaningless; SORT zeroes the scale
        # velocity rather than letting the box invert.
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        return x_to_box(self.x)

    def update(self, box) -> None:
        import numpy as np

        z = box_to_z(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P
        self.hits += 1
        self.time_since_update = 0

    def box(self):
        return x_to_box(self.x)


class Sort:
    """SORT: predict, associate by Hungarian on IoU, update."""

    def __init__(self, params: Params | None = None):
        self.p = params or Params()
        self.tracks: list[KalmanBoxTracker] = []

    def update(self, detections):
        """One frame. `detections` is [(x, y, w, h), ...]. Returns [(id, box), ...]."""
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        predictions = [t.predict() for t in self.tracks]

        matched_tracks, matched_dets = set(), set()
        if predictions and detections:
            cost = np.zeros((len(predictions), len(detections)))
            for ti, pred in enumerate(predictions):
                for di, det in enumerate(detections):
                    cost[ti, di] = -iou(pred, det)   # maximise IoU = minimise -IoU
            # The globally optimal assignment, which is the whole difference from the
            # classical tracker's greedy pass.
            rows, cols = linear_sum_assignment(cost)
            for ti, di in zip(rows, cols):
                if -cost[ti, di] < self.p.iou_gate:
                    continue   # optimal overall, still too poor to be the same object
                self.tracks[ti].update(detections[di])
                matched_tracks.add(ti)
                matched_dets.add(di)

        for di, det in enumerate(detections):
            if di not in matched_dets:
                self.tracks.append(KalmanBoxTracker(det, self.p))

        self.tracks = [t for t in self.tracks
                       if t.time_since_update <= self.p.max_age]

        return [(t.id, t.box()) for t in self.tracks
                if t.hits >= self.p.min_hits and t.time_since_update == 0]


# ------------------------------------------------------------------------- selftest

def _linear_scene(n_frames=30, n_objects=3, blank=None):
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
    print("[test] box <-> state round trip")
    box = (10.0, 20.0, 30.0, 40.0)
    back = x_to_box(box_to_z(box))
    assert all(abs(a - b) < 1e-6 for a, b in zip(box, back)), back
    print(f"       ok - {box} survives the scale/aspect parameterisation")

    print("[test] constant-velocity recovery")
    KalmanBoxTracker.count = 0
    tracker = Sort()
    seen = {}
    for dets in _linear_scene():
        for tid, _ in tracker.update(dets):
            seen[tid] = seen.get(tid, 0) + 1
    assert len(seen) == 3, f"expected 3 ids, got {len(seen)}: {seen}"
    print(f"       ok - 3 objects, 3 ids, no fragmentation {dict(seen)}")

    print("[test] occlusion shorter than max_age keeps the id")
    KalmanBoxTracker.count = 0
    tracker = Sort(Params(max_age=10))
    ids = set()
    for dets in _linear_scene(blank=(1, set(range(10, 15)))):
        ids.update(tid for tid, _ in tracker.update(dets))
    assert len(ids) == 3, f"a 5-frame gap under max_age=10 split a track: {ids}"
    print(f"       ok - track survived a 5-frame gap, {len(ids)} ids total")

    print("[test] occlusion longer than max_age ends the track")
    KalmanBoxTracker.count = 0
    tracker = Sort(Params(max_age=3))
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
