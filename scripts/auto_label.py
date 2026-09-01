"""Generate tracking ground truth automatically, with no human in the loop.

Hand-labelling was tried and abandoned: the seeded tracks needed too much correction
(25 tracks for ~8 robots on arc04, boxes nested inside boxes, one chassis split across
two half-boxes). This does the whole job offline and writes gt/gt.txt directly.

WHAT THIS COSTS - READ BEFORE QUOTING ANY TRACKING ACCURACY NUMBER
    Ground truth produced by IoU + motion association shares its assumptions with SORT
    and the classical tracker, and not with GOTURN and VitTrack, which match on
    appearance. So the tracking ACCURACY comparison is partly circular and tilts toward
    the motion-based trackers - exactly the classical-vs-DL axis under study. The bias
    can be reduced and disclosed; it cannot be removed.

    Unaffected: detection accuracy (ROCO, human ground truth), detection CPU cost, and
    tracking CPU cost - which needs no labels at all and is the project's stated
    contribution. That is why the tracking headline is the CPU comparison and accuracy
    is reported underneath with this caveat attached.

WHY OFFLINE
    Every tracker being evaluated is ONLINE: one pass, in order, no future information.
    This labeller is OFFLINE - it holds the whole clip, and stitches fragments using
    frames that an online tracker had not seen when it made the same decision. That
    asymmetry is deliberate and is what makes the reference better than the things
    scored against it, structurally rather than by tuning.

Usage:
    python scripts/auto_label.py --seq data/tracking/arc04
    python scripts/auto_label.py --all
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

TRACKING = ROOT / "data" / "tracking"
QUALITY_CSV = ROOT / "results" / "label_quality.csv"
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "yolo_960" / "weights" / "best.pt"

# A 3v3 match fields 6 robots plus a base and a sentry, so a clip showing more than
# about a dozen distinct targets is fragmenting rather than seeing a crowd.
PLAUSIBLE_TRACKS = (2, 14)


# Tuned on arc04: 0.7/0.35 left 4.7 boxes per frame against roughly 3.4 robots
# actually in that window; 0.5/0.15 brings it to 4.0. Looser still starts fusing
# robots parked beside each other, which the size envelope only partly prevents.
def clean_frame(boxes, containment_max=0.5, iou_max=0.15, fuse_limit=1.6,
                median=120.0):
    """Resolve multiple detections that describe one robot.

    Two failure modes seen while labelling by hand:

      NESTED   a small box entirely inside a large one - the detector fired on the
               turret as well as the chassis. On arc04 frame 1, #2 lay 96% inside #6.
      SPLIT    two boxes covering opposite halves of one robot. #5 spanned x 327-469
               and #7 x 239-363 of the same chassis, overlapping at only IoU 0.15.

    Nested boxes are dropped. Split boxes are FUSED into their union, but only when the
    union stays within `fuse_limit` of the clip's median box - otherwise two robots
    parked side by side would be welded into one target.
    """
    from label_tracks import _containment, _iou  # noqa: PLC0415

    kept = sorted(boxes, key=lambda b: -(b[2] * b[3]))
    out = []
    for box in kept:
        merged = False
        for i, other in enumerate(out):
            if _containment(box, other) >= containment_max:
                merged = True  # a part of a robot already accounted for
                break
            if _iou(box, other) >= iou_max:
                x0 = min(box[0], other[0])
                y0 = min(box[1], other[1])
                x1 = max(box[0] + box[2], other[0] + other[2])
                y1 = max(box[1] + box[3], other[1] + other[3])
                if ((x1 - x0) * (y1 - y0)) ** 0.5 <= fuse_limit * median:
                    out[i] = (x0, y0, x1 - x0, y1 - y0)
                merged = True
                break
        if not merged:
            out.append(tuple(box))
    return out


# Swept on arc04/arc06. 45/0.30 left ratio 4.1 with tracks breaking every ~50 frames;
# 120/0.80 gives ratio 3.1 and median track 183 of 300 frames. Going further (150/1.20)
# helps arc04 again but stops helping arc06, so this is where the return flattens.
def stitch(tracks, max_gap=120, max_drift=0.80):
    """Join fragments of one robot, using the whole clip at once.

    This is the offline advantage. An online tracker decides at frame N whether the box
    in front of it continues an existing track, knowing only frames up to N. Here both
    ends of every gap are already known, so a track that died at frame 79 can be joined
    to one that appeared at 96 on evidence the online tracker never had.
    """
    from label_tracks import interpolate  # noqa: PLC0415

    spans = {t: (min(kf), max(kf)) for t, kf in tracks.items()}
    parent = {t: t for t in tracks}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges, accepted = [], []
    for a in tracks:
        for b in tracks:
            if a == b:
                continue
            gap = spans[b][0] - spans[a][1]
            if not 0 < gap <= max_gap:
                continue
            end = interpolate(tracks[a], spans[a][1])
            start = interpolate(tracks[b], spans[b][0])
            if not end or not start:
                continue
            size = max(1.0, (end[2] * end[3]) ** 0.5)
            drift = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5 / size
            if drift <= max_drift:
                edges.append((drift, a, b))

    # Best evidence first, and only join spans that do not overlap in time - merging
    # co-alive tracks interleaves their keyframes and makes the box alternate.
    for _, a, b in sorted(edges):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        sa, sb = spans[ra], spans[rb]
        if sa[1] >= sb[0] and sb[1] >= sa[0]:
            continue
        parent[rb] = ra
        spans[ra] = (min(sa[0], sb[0]), max(sa[1], sb[1]))
        accepted.append(_)

    # Surfaced because p99 step cannot see a bad stitch: interpolation smooths the
    # joined gap, so a join between two different robots looks perfectly calm.
    stitch.last_joins = len(accepted)
    stitch.last_max_drift = max(accepted) if accepted else 0.0
    joined: dict[int, dict] = {}
    for t in sorted(tracks):
        root = find(t)
        joined.setdefault(root, {}).update(tracks[t])
    return joined


def drop_short(tracks, min_frames=15):
    """Detector flicker surviving as a track is noise, not a robot."""
    return {t: kf for t, kf in tracks.items()
            if max(kf) - min(kf) + 1 >= min_frames}


def quality(name: str, tracks, per_frame, n_frames: int) -> dict:
    """Measure implausibility. No ground truth exists, but nonsense is still countable."""
    from label_tracks import interpolate  # noqa: PLC0415

    spans = {t: (min(kf), max(kf)) for t, kf in tracks.items()}
    lengths = [hi - lo + 1 for lo, hi in spans.values()]

    disp_ratio, area_cv = [], []
    for t, kf in tracks.items():
        lo, hi = spans[t]
        boxes = [interpolate(kf, f) for f in range(lo, hi + 1)]
        boxes = [b for b in boxes if b]
        for p, q in zip(boxes, boxes[1:]):
            size = max(1.0, (p[2] * p[3]) ** 0.5)
            step = (((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5) / size
            disp_ratio.append(step)
        areas = [b[2] * b[3] for b in boxes]
        if len(areas) > 1 and statistics.fmean(areas) > 0:
            area_cv.append(statistics.pstdev(areas) / statistics.fmean(areas))

    ordered = sorted(disp_ratio)
    p99 = ordered[int(0.99 * (len(ordered) - 1))] if ordered else 0.0
    boxes_per_frame = statistics.fmean([len(b) for b in per_frame]) if per_frame else 0.0

    row = {
        "sequence": name,
        "frames": n_frames,
        "tracks": len(tracks),
        "median_track_len": round(statistics.median(lengths), 1) if lengths else 0,
        "short_tracks": sum(1 for x in lengths if x < 30),
        "p99_step_over_size": round(p99, 3),
        "median_area_cv": round(statistics.median(area_cv), 3) if area_cv else 0.0,
        "boxes_per_frame": round(boxes_per_frame, 1),
        # tracks per robot-on-screen: ~1-2 means one track per robot, >3 is
        # fragmentation. Scales with how crowded the clip is, unlike a fixed count.
        "tracks_per_robot": round(len(tracks) / max(boxes_per_frame, 0.1), 1),
        "stitch_joins": getattr(stitch, "last_joins", 0),
        "stitch_max_drift": round(getattr(stitch, "last_max_drift", 0.0), 2),
    }
    problems = []
    if row["tracks_per_robot"] > 3.0:
        problems.append(f"{row['tracks_per_robot']} tracks per robot on screen - "
                        "fragmentation")
    if p99 > 1.0:
        problems.append(f"p99 step {p99:.2f}x box size - ids may be teleporting")
    if row["median_track_len"] < n_frames * 0.15:
        problems.append("tracks are short relative to the clip")
    row["flags"] = "; ".join(problems) if problems else "ok"
    return row


def label_one(seq: Path, args) -> dict:
    from label_tracks import save_state  # noqa: PLC0415
    from pre_annotate import associate, detect, read_seqinfo, to_keyframes  # noqa: PLC0415

    info = read_seqinfo(seq)
    images = sorted((seq / info["imDir"]).glob("*" + info["imExt"]))
    if not images:
        sys.exit(f"No frames in {seq / info['imDir']}")

    print(f"\n=== {seq.name} ({len(images)} frames) ===")
    raw = detect(images, args.weights, args.imgsz, args.conf, args.device, args.target)

    sizes = [(b[2] * b[3]) ** 0.5 for frame in raw for b in frame]
    median = statistics.median(sizes) if sizes else 120.0
    cleaned = [clean_frame(frame, median=median) for frame in raw]
    before = sum(len(f) for f in raw)
    after = sum(len(f) for f in cleaned)
    print(f"[boxes] {before} detections -> {after} after nesting/split cleanup "
          f"(median box {median:.0f}px)")

    tracks = associate(cleaned, args.iou_thresh, args.max_age)
    print(f"[link]  {len(tracks)} raw tracks")
    tracks = stitch(tracks, args.max_gap, args.max_drift)
    print(f"[stitch] {len(tracks)} after joining fragments offline")
    tracks = drop_short(tracks, args.min_frames)
    print(f"[prune] {len(tracks)} after dropping tracks under {args.min_frames} frames")

    keyed = to_keyframes(tracks, args.keyframe_stride, 1)
    # Renumber from 1 so ids are contiguous after stitching and pruning.
    keyed = {new: kf for new, (_, kf) in enumerate(sorted(keyed.items()), start=1)}
    save_state(seq, keyed, set(), args.target)

    row = quality(seq.name, keyed, cleaned, len(images))
    print(f"[check] tracks {row['tracks']}, median length {row['median_track_len']}, "
          f"p99 step {row['p99_step_over_size']}x box, flags: {row['flags']}")
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--seq", type=Path)
    target.add_argument("--all", action="store_true", help="Every clip in data/tracking.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Higher than pre_annotate: no human filters these.")
    parser.add_argument("--class", dest="target_cls", default="car")
    parser.add_argument("--iou-thresh", type=float, default=0.15)
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--max-gap", type=int, default=120)
    parser.add_argument("--max-drift", type=float, default=0.80)
    parser.add_argument("--min-frames", type=int, default=15)
    parser.add_argument("--keyframe-stride", type=int, default=5)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    args.target = args.target_cls
    return args


def main() -> None:
    args = parse_args()
    seqs = ([p for p in sorted(TRACKING.glob("arc*")) if (p / "seqinfo.ini").is_file()]
            if args.all else [args.seq])

    rows = [label_one(seq, args) for seq in seqs]

    QUALITY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with QUALITY_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'sequence':10}{'tracks':>8}{'medLen':>8}{'p99step':>9}{'areaCV':>8}  flags")
    for r in rows:
        print(f"{r['sequence']:10}{r['tracks']:8}{r['median_track_len']:8}"
              f"{r['p99_step_over_size']:9}{r['median_area_cv']:8}  {r['flags']}")
    print(f"\nwrote {QUALITY_CSV}")
    bad = [r["sequence"] for r in rows if r["flags"] != "ok"]
    if bad:
        print(f"FLAGGED: {', '.join(bad)} - inspect before using as ground truth.")


if __name__ == "__main__":
    main()
