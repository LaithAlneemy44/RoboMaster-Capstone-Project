"""Find the segments of a match video worth labelling, so nobody scrubs 90 minutes.

Match recordings are mostly not useful as tracking data: menus, crowd shots, replays,
overhead views where robots are a handful of pixels. What tracking labelling wants is
sustained close-in combat - several robots, large in frame, moving.

That is measurable rather than a matter of taste, so this samples the video, runs the
trained detector over the samples, and ranks fixed-length windows by how many robots are
visible and how big they are. Size matters as much as count: the detection sweep showed
armor is undetectable below 640px input, and a robot that is 40px wide is painful to
label and marginal to track.

RUNS ON THE GPU, DELIBERATELY
    CLAUDE.md forbids benchmarking inference on the GPU, and that stands - but this is
    not a benchmark. No timing from this script is ever reported. It is a search over
    ~700 frames per video, and doing it on the CPU would waste an hour per match for
    numbers nobody reads.

Usage:
    python scripts/find_segments.py --video data/tracking/_raw/VRL-xmK0nvw.mp4
    python scripts/find_segments.py --video ... --window 10 --top 5
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "yolo_960" / "weights" / "best.pt"
TARGET_CLASS = "car"

# A robot narrower than this is hard to label accurately and marginal to track.
MIN_USEFUL_SIZE_PX = 60


def sample(video: Path, weights: Path, step: float, imgsz: int, conf: float,
           device: str):
    """Detect on one frame every `step` seconds. Returns [(seconds, count, size)]."""
    import cv2
    from ultralytics import YOLO

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        sys.exit(f"OpenCV could not open {video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps

    model = YOLO(str(weights))
    wanted = [i for i, n in model.names.items() if n == TARGET_CLASS]
    if not wanted:
        sys.exit(f"Model has no {TARGET_CLASS!r} class: {model.names}")
    target_id = wanted[0]

    def boxes_of(frame):
        result = model.predict(frame, imgsz=imgsz, conf=conf, device=device,
                               verbose=False)[0]
        out = []
        for box in result.boxes:
            if int(box.cls[0]) != target_id:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            out.append((x1, y1, x2 - x1, y2 - y1))
        return out

    points = []
    seconds = 0.0
    while seconds < duration:
        capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
        ok, frame = capture.read()
        if not ok:
            break
        # Two CONSECUTIVE frames, not just one. An earlier version scored only robot
        # count and size, and duly picked stationary standoffs - six large robots
        # parked in frame scores perfectly while being worthless tracking data, since
        # every tracker succeeds on it and none is distinguished. Motion has to be
        # measured between adjacent frames, which means decoding both.
        ok2, nxt = capture.read()
        first = boxes_of(frame)
        second = boxes_of(nxt) if ok2 else []

        sizes = [(w * h) ** 0.5 for _, _, w, h in first]
        moved = 0
        for box in first:
            best = max((_iou(box, c), c) for c in second) if second else (0.0, None)
            if best[0] > 0:
                cx = box[0] + box[2] / 2 - (best[1][0] + best[1][2] / 2)
                cy = box[1] + box[3] / 2 - (best[1][1] + best[1][3] / 2)
                if (cx * cx + cy * cy) ** 0.5 > 2.0:
                    moved += 1
        fraction_moving = moved / len(first) if first else 0.0

        points.append((seconds, len(sizes),
                       statistics.median(sizes) if sizes else 0.0, fraction_moving))
        print(f"\r[scan] {seconds:6.0f}s / {duration:.0f}s", end="", flush=True)
        seconds += step
    capture.release()
    print()
    return points, duration, fps


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (aw * ah + bw * bh - inter)


def rank(points, step: float, window: float, top: int):
    """Score every window, then take the best non-overlapping ones.

    Score is count x median size: a window full of distant specks and a window with one
    close robot are both poor, and multiplying penalises each.
    """
    span = max(1, int(round(window / step)))
    scored = []
    for i in range(len(points) - span + 1):
        chunk = points[i:i + span]
        counts = [c for _, c, _, _ in chunk]
        sizes = [s for _, _, s, _ in chunk if s > 0]
        motions = [m for _, _, _, m in chunk]
        if not sizes or min(counts) == 0:
            continue  # a gap with no robots at all breaks a tracking sequence
        mean_count = statistics.fmean(counts)
        med_size = statistics.median(sizes)
        mean_motion = statistics.fmean(motions)
        # Motion is a multiplier, not a bonus: a window where nothing moves scores zero
        # however many large robots are parked in it.
        score = mean_count * med_size * mean_motion
        scored.append((score, chunk[0][0], mean_count, med_size, mean_motion))

    scored.sort(reverse=True)
    chosen = []
    for row in scored:
        if any(abs(row[1] - c[1]) < window for c in chosen):
            continue  # keep picks from overlapping each other
        chosen.append(row)
        if len(chosen) >= top:
            break
    return chosen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--step", type=float, default=2.0, help="Sampling interval (s).")
    parser.add_argument("--window", type=float, default=10.0, help="Clip length (s).")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--device", default="0", help="'0' = GPU. See module docstring.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        sys.exit(f"Missing {args.video}")

    points, duration, fps = sample(args.video, args.weights, args.step, args.imgsz,
                                   args.conf, args.device)
    best = rank(points, args.step, args.window, args.top)
    if not best:
        print("No window had robots visible throughout - try a shorter --window.")
        return

    print(f"\n{args.video.name}  ({duration / 60:.1f} min @ {fps:.0f} fps)")
    print(f"{'rank':>5}{'start':>9}{'end':>8}{'robots':>9}{'size px':>10}{'moving':>9}"
          "  suggested command")
    print("-" * 104)
    stem = args.video.stem
    for i, (_, start, count, size, motion) in enumerate(best, 1):
        small = "  (small)" if size < MIN_USEFUL_SIZE_PX else ""
        print(f"{i:5}{start:8.0f}s{start + args.window:7.0f}s{count:9.1f}"
              f"{size:10.0f}{motion * 100:8.0f}%{small}")
        print(f"      python scripts/fetch_clips.py --url https://youtu.be/{stem} "
              f"--name {stem[:6].lower()}_{int(start)} "
              f"--start {start:.0f} --end {start + args.window:.0f}")


if __name__ == "__main__":
    main()
