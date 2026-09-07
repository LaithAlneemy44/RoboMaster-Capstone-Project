"""Turn a tracked box into a turret rate demand.

A proportional controller on pixel error, with a deadband. Deliberately the simplest thing
that can work: without hardware there is no way to tune a derivative or integral term
honestly, and an untuned D term on a real gimbal oscillates. Integral windup on a turret
that can be physically blocked is worse still. P only, gains exposed, defaults
conservative.

SIGN CONVENTIONS - the thing most likely to send the turret the wrong way

    image x grows RIGHT, image y grows DOWN (OpenCV).
    positive yaw   = turn RIGHT      positive pitch = aim UP

    target left of centre  -> negative yaw rate
    target above centre    -> positive pitch rate   (note the inversion against y)

    The y inversion is the one that bites: a target above centre has a NEGATIVE y error
    but needs a POSITIVE pitch demand. Asserted in the selftest rather than trusted.

TARGET SELECTION IS STICKY
    Picking the box nearest the crosshair every frame makes the turret flick between two
    robots straddling the centre. Selection therefore prefers the track id it already
    holds while that track is alive, and only re-picks when it disappears. Identity is
    what the tracker is for; this is where the project spends it.

Usage:
    python scripts/robot/aiming.py --selftest
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AimConfig:
    """Gains are UNTUNED. They cannot be tuned without the physical gimbal."""

    gain_yaw: float = 1.2
    gain_pitch: float = 1.0
    # Fraction of half-frame inside which the turret holds still. Below roughly 2% the
    # turret hunts on detector jitter alone, since box centres move a pixel or two
    # frame to frame even on a stationary target.
    deadband: float = 0.03
    max_rate: float = 0.6      # ceiling on demand, well inside the protocol's 1.0


def _centre(box) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def select_target(tracks, frame_size, held_id=None):
    """Choose which track to aim at. Returns (track_id, box) or (None, None).

    `tracks` is the tracker's output: an iterable of (track_id, (x, y, w, h)), which is
    exactly what ClassicalTracker.update() and Sort.update() return.
    """
    tracks = list(tracks)
    if not tracks:
        return None, None

    # Hysteresis: keep the current target while it exists, so the turret does not swap
    # between two robots that are near-equidistant from the crosshair.
    for track_id, box in tracks:
        if held_id is not None and track_id == held_id:
            return track_id, box

    width, height = frame_size
    cx, cy = width / 2.0, height / 2.0

    def distance(entry):
        x, y = _centre(entry[1])
        return (x - cx) ** 2 + (y - cy) ** 2

    return min(tracks, key=distance)


class AimController:
    """Pixel error in, normalised turret rate out."""

    def __init__(self, config: AimConfig | None = None) -> None:
        self.config = config or AimConfig()
        self.held_id = None

    def reset(self) -> None:
        """Forget the held target. Call when auto-track is switched off."""
        self.held_id = None

    def update(self, tracks, frame_size) -> tuple[float, float, object]:
        """Returns (d_pitch, d_yaw, target_id). Zero rates when there is no target."""
        track_id, box = select_target(tracks, frame_size, self.held_id)
        self.held_id = track_id
        if box is None:
            return 0.0, 0.0, None

        width, height = frame_size
        x, y = _centre(box)
        # Normalise to -1..1 across each half-axis, so gains mean the same thing at any
        # resolution - the pipeline runs anywhere from 320 to 1920 wide.
        error_x = (x - width / 2.0) / (width / 2.0)
        error_y = (y - height / 2.0) / (height / 2.0)

        config = self.config
        yaw = 0.0 if abs(error_x) < config.deadband else config.gain_yaw * error_x
        # Negated: y grows downward, pitch grows upward.
        pitch = 0.0 if abs(error_y) < config.deadband else -config.gain_pitch * error_y

        limit = config.max_rate
        return (max(-limit, min(limit, pitch)),
                max(-limit, min(limit, yaw)),
                track_id)


# ------------------------------------------------------------------------- selftest

def _selftest() -> None:
    size = (1920, 1080)
    controller = AimController()

    print("[test] no targets means no motion")
    assert controller.update([], size) == (0.0, 0.0, None)
    print("       ok")

    print("[test] a centred target sits inside the deadband")
    controller.reset()
    pitch, yaw, _ = controller.update([(1, (950, 520, 20, 20))], size)
    assert pitch == 0.0 and yaw == 0.0, (pitch, yaw)
    print("       ok")

    print("[test] left of centre yaws negative, right positive")
    controller.reset()
    _, yaw_left, _ = controller.update([(1, (100, 520, 40, 40))], size)
    controller.reset()
    _, yaw_right, _ = controller.update([(2, (1700, 520, 40, 40))], size)
    assert yaw_left < 0 < yaw_right, (yaw_left, yaw_right)
    print(f"       ok  (left {yaw_left:+.3f}, right {yaw_right:+.3f})")

    print("[test] above centre pitches UP despite a negative y error")
    controller.reset()
    pitch_up, _, _ = controller.update([(1, (950, 50, 40, 40))], size)
    controller.reset()
    pitch_down, _, _ = controller.update([(2, (950, 1000, 40, 40))], size)
    assert pitch_up > 0 > pitch_down, (pitch_up, pitch_down)
    print(f"       ok  (up {pitch_up:+.3f}, down {pitch_down:+.3f})")

    print("[test] demand is clipped to max_rate")
    controller.reset()
    pitch, yaw, _ = controller.update([(1, (0, 0, 4, 4))], size)
    limit = controller.config.max_rate
    assert abs(yaw) <= limit and abs(pitch) <= limit, (pitch, yaw)
    print(f"       ok  (|yaw| {abs(yaw):.3f} <= {limit})")

    print("[test] the held target is kept while it lives")
    controller.reset()
    far = (1, (1700, 520, 40, 40))
    near = (2, (980, 540, 40, 40))
    _, _, first = controller.update([far], size)
    assert first == 1
    _, _, second = controller.update([near, far], size)
    assert second == 1, "a nearer track must not steal the lock"
    _, _, third = controller.update([near], size)
    assert third == 2, "once the held track is gone, re-pick"
    print("       ok")

    print("[test] reset drops the lock")
    controller.reset()
    assert controller.held_id is None
    _, _, picked = controller.update([near, far], size)
    assert picked == 2, "after a reset, pick the nearest"
    print("       ok")

    print("\nAll checks passed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", required=True)
    parser.parse_args()
    _selftest()
