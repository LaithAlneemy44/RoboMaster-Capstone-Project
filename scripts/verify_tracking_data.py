"""Decide whether a clip is trackable BEFORE anyone spends weeks labelling it.

This is the check that would have saved the DJI ROCO dataset from being mistaken for
tracking data. ROCO looks perfect on inspection - 2655 images, seven match clips, frame
indices running 0..419 with a gap of exactly 1. It is genuinely consecutive video. But
measuring how far objects move between those consecutive frames:

    class      median IoU   displacement   object size   disp/size
    base           0.919         4.3 px      135.2 px        0.03
    car            0.514        28.5 px      101.9 px        0.28
    watcher        0.141        44.6 px       63.3 px        0.70
    armor          0.041        22.9 px       21.5 px        1.06

base is a fixed arena structure and barely moves, so the camera is steady and that
motion is real. Yet armor travels MORE THAN ITS OWN WIDTH per frame, giving a median
self-IoU of 0.041 - an armor plate does not overlap itself from one frame to the next.
IoU-based association, which is exactly what SORT does, cannot work on that. The frames
were sampled several video frames apart at export, and nothing recovers what was thrown
away.

So the gate is the displacement/size ratio, not frame numbering and not whether the
source happened to be video.

WHERE THE BOXES COME FROM
    Before labelling there are no ground-truth boxes, so by default this runs a trained
    detector over the frames and measures ITS boxes. That is sound for this purpose:
    the question is whether objects move too far between frames, which does not depend
    on the boxes being perfect. Ground truth is used instead when available (--coco for
    ROCO-style data, --mot for an already-labelled clip).

Usage:
    # Gate a candidate clip before labelling it
    python scripts/verify_tracking_data.py --frames data/tracking/clip01/img --class car

    # Reproduce the ROCO verdict (this SHOULD fail)
    python scripts/verify_tracking_data.py --coco data/splits/coco_train.json --class armor
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "yolo_960" / "weights" / "best.pt"

# ROCO armor scored 1.06 and is unusable. Comfortably trackable data sits far below
# this; 0.30 still means a target moves 30% of its own width per frame, which is
# already demanding for IoU association.
DEFAULT_MAX_DISP_RATIO = 0.30

# Fraction of boxes with no overlapping counterpart in the next frame. ROCO armor sits
# around 35%: objects effectively teleport, and no association scheme survives that.
DEFAULT_MAX_LOST_PCT = 20.0

# The opposite failure. A sequence where nothing moves passes every upper bound and is
# useless as a benchmark - all trackers score well and none is distinguished. This is a
# warning rather than a hard failure, because some static objects are normal and
# expected (MOT benchmarks are full of parked cars and standing people).
DEFAULT_MIN_MOVING_PCT = 25.0

MIN_SAMPLES = 50  # below this a per-class verdict is noise


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (aw * ah + bw * bh - inter)


def centre(b):
    return b[0] + b[2] / 2, b[1] + b[3] / 2


def sequence_stats(sequences):
    """Per-class inter-frame motion, pooled across clips.

    `sequences` maps clip name -> [(frame_index, [(class_name, xywh), ...]), ...].
    Only genuinely adjacent frames (index difference of 1) are compared; a gap means
    the pair says nothing about frame-to-frame motion.
    """
    per_class = collections.defaultdict(lambda: ([], [], []))
    unmatched = collections.Counter()
    totals = collections.Counter()

    for frames in sequences.values():
        frames.sort(key=lambda f: f[0])
        for (i1, boxes1), (i2, boxes2) in zip(frames, frames[1:]):
            if i2 - i1 != 1:
                continue
            by_class = collections.defaultdict(list)
            for name, box in boxes2:
                by_class[name].append(box)
            for name, box in boxes1:
                totals[name] += 1
                candidates = by_class.get(name)
                if not candidates:
                    unmatched[name] += 1
                    continue
                best = max(candidates, key=lambda c: iou(box, c))
                overlap = iou(box, best)
                if overlap <= 0.0:
                    # No overlap with ANY same-class box means this object has no
                    # counterpart in the next frame. Recording the distance to whatever
                    # happened to be nearest would invent a displacement - earlier this
                    # produced 900-1400px "motion" that was really one robot being
                    # paired with a different robot across the arena. Count it as
                    # unmatched instead: a high unmatched rate is itself the failure
                    # signal, and it is what ROCO armor actually suffers from.
                    unmatched[name] += 1
                    continue
                ious, disps, sizes = per_class[name]
                ious.append(overlap)
                disps.append(math.dist(centre(box), centre(best)))
                sizes.append(math.sqrt(box[2] * box[3]))
    return per_class, unmatched, totals


def from_coco(path: Path):
    """ROCO-style COCO json to sequences, grouped by the clip in each filename."""
    data = json.loads(path.read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in data["categories"]}
    anns = collections.defaultdict(list)
    for a in data["annotations"]:
        anns[a["image_id"]].append((names[a["category_id"]], a["bbox"]))

    sequences = collections.defaultdict(list)
    for image in data["images"]:
        match = re.match(r"^(.*)_(\d+)_jpg\.rf\.", os.path.basename(image["file_name"]))
        if not match:
            continue
        sequences[match.group(1)].append((int(match.group(2)), anns[image["id"]]))
    return sequences


def from_mot(path: Path, class_name: str):
    """MOT Challenge gt.txt to a single sequence."""
    frames = collections.defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        box = [float(v) for v in parts[2:6]]
        frames[int(parts[0])].append((class_name, box))
    return {path.parent.name: list(frames.items())}


def from_frames(directory: Path, weights: Path, imgsz: int, conf: float, limit: int,
                device: str):
    """Run a trained detector over extracted frames to get boxes to measure."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from ultralytics import YOLO  # noqa: PLC0415

    images = sorted(
        p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not images:
        sys.exit(f"No frames found in {directory}")
    if limit:
        images = images[:limit]

    model = YOLO(str(weights))
    print(f"[detect] {len(images)} frames with {weights.name} at {imgsz}px ...")

    frames = []
    for index, path in enumerate(images):
        result = model.predict(str(path), imgsz=imgsz, conf=conf, device=device,
                               verbose=False)[0]
        boxes = []
        for box in result.boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            boxes.append((model.names[int(box.cls[0])], [x1, y1, x2 - x1, y2 - y1]))
        frames.append((index, boxes))
        print(f"\r[detect] {index + 1}/{len(images)}", end="", flush=True)
    print()
    return {directory.name: frames}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frames", type=Path, help="Directory of extracted frames.")
    source.add_argument("--coco", type=Path, help="COCO json with ground-truth boxes.")
    source.add_argument("--mot", type=Path, help="MOT Challenge gt.txt.")
    parser.add_argument("--class", dest="target", default="car",
                        help="Class the verdict is based on (default: car).")
    parser.add_argument("--max-disp-ratio", type=float, default=DEFAULT_MAX_DISP_RATIO)
    parser.add_argument("--max-lost-pct", type=float, default=DEFAULT_MAX_LOST_PCT)
    parser.add_argument("--min-moving-pct", type=float, default=DEFAULT_MIN_MOVING_PCT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    # GPU by default: this is a geometry measurement, not a latency benchmark, and
    # no timing from it is ever reported. CLAUDE.md forbids the latter, not the former.
    parser.add_argument("--device", default="0", help="'0' = GPU, 'cpu' = CPU.")
    parser.add_argument("--limit", type=int, default=200,
                        help="Max frames to detect over (0 = all).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frames:
        sequences = from_frames(args.frames, args.weights, args.imgsz, args.conf,
                                args.limit, args.device)
    elif args.coco:
        sequences = from_coco(args.coco)
    else:
        sequences = from_mot(args.mot, args.target)

    stats, unmatched, totals = sequence_stats(sequences)
    if not stats:
        sys.exit("No adjacent-frame pairs found - is this a numbered sequence?")

    print(f"\n{'class':10}{'n':>7}{'lost%':>7}{'medIoU':>8}{'p90 disp':>10}"
          f"{'size':>8}{'p90/size':>10}{'moving%':>9}")
    print("-" * 70)
    verdict = None
    for name, (ious, disps, sizes) in sorted(stats.items(), key=lambda x: -len(x[1][0])):
        if len(ious) < MIN_SAMPLES:
            continue
        ordered = sorted(disps)
        # p90, not the median. Most robots in a match are parked - base, guards,
        # disabled units - so the median is ~0 whatever the moving ones are doing, and
        # reports every clip as trivially safe. The demanding case is what decides
        # whether association holds, so the upper tail is what gets judged.
        p90 = ordered[int(0.90 * (len(ordered) - 1))]
        size = statistics.median(sizes)
        ratio = p90 / size if size else float("inf")
        lost = 100.0 * unmatched[name] / totals[name] if totals[name] else 0.0
        moving = 100.0 * sum(1 for d in disps if d > 2.0) / len(disps)
        flag = "  <-- target" if name == args.target else ""
        print(f"{name:10}{len(ious):7}{lost:6.1f}%{statistics.median(ious):8.3f}"
              f"{p90:10.1f}{size:8.0f}{ratio:10.2f}{moving:8.1f}%{flag}")
        if name == args.target:
            verdict = (ratio, lost, moving)

    print("-" * 70)
    if verdict is None:
        sys.exit(f"\nFAIL: class {args.target!r} had under {MIN_SAMPLES} samples - "
                 "cannot judge. Check the class name or detect over more frames.")

    ratio, lost, moving = verdict
    if ratio > args.max_disp_ratio:
        sys.exit(
            f"\nFAIL: {args.target} p90 motion is {ratio:.2f}x its own size per frame "
            f"(limit {args.max_disp_ratio:.2f}).\n"
            "Frames are too far apart for IoU association - SORT would break on this.\n"
            "Re-extract at a higher frame rate before labelling anything."
        )
    if lost > args.max_lost_pct:
        sys.exit(
            f"\nFAIL: {lost:.0f}% of {args.target} boxes have no overlapping "
            f"counterpart in the next frame (limit {args.max_lost_pct:.0f}%).\n"
            "Objects are vanishing between frames - nothing can associate that."
        )
    print(f"\nPASS: {args.target} p90 motion is {ratio:.2f}x its own size per frame, "
          f"{lost:.0f}% lost (limits {args.max_disp_ratio:.2f} / "
          f"{args.max_lost_pct:.0f}%).")

    if moving < args.min_moving_pct:
        # Passing the upper bound is only half the question. A sequence where nothing
        # moves is trivially trackable and will not separate one tracker from another,
        # so it is labelling effort spent on a benchmark that cannot discriminate.
        print(f"\nWARNING: only {moving:.0f}% of {args.target} boxes move more than "
              f"2px per frame (want >={args.min_moving_pct:.0f}%).\n"
              "         This clip is nearly static. Every tracker will score well on "
              "it and\n         the comparison will not discriminate. Prefer a segment "
              "with real motion.")
    if args.coco:
        # ROCO car passes this at 0.28, which must not be read as "ROCO is usable for
        # tracking". This gate measures MOTION only. ROCO is disqualified separately
        # because its annotations carry no track ids linking an object across frames.
        print("Note: this measures motion only. A COCO detection set still has no "
              "track ids,\n      so passing here does not make it usable as tracking "
              "ground truth.")


if __name__ == "__main__":
    main()
