"""Score tracker output against ground truth with MOT Challenge metrics.

The tracking counterpart to scripts/evaluate_detection.py, and deliberately shaped like
it: one row per config in a CSV, bootstrap confidence intervals computed the same way,
so the two results tables mean the same thing when read side by side.

Reports the metrics proposal 5.4 asks for on the tracking side - MOTA, IDF1, ID
switches, MOTP - via motmetrics, imported through scripts/mot_compat.py because
motmetrics 1.4.0 calls np.asfarray, which NumPy 2 removed.

CONFIDENCE INTERVALS
    Resampled over FRAMES, not sequences. With two or three labelled clips a
    sequence-level bootstrap has almost no resolution, so the interval would be
    decorative. MOTA decomposes into per-frame counts - misses, false positives, identity
    switches, ground-truth objects - so resampling frames and recomputing is exact rather
    than an approximation.

    The same caveat applies as for the detection CIs: this measures within-clip variance
    and does NOT estimate how the tracker would do on a different match.

Usage:
    python scripts/eval_tracking.py --seq data/tracking/arc04 \\
        --results results/tracking/arc04/sort_gt.txt --name sort_gt
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
DEFAULT_CSV = ROOT / "results" / "tracking.csv"


def load_mot(path: Path, label: str):
    from mot_compat import mm  # noqa: PLC0415

    if not path.is_file():
        sys.exit(f"Missing {label}: {path}")
    return mm.io.loadtxt(str(path), fmt="mot15-2D")


def per_frame_counts(events):
    """MOTA's ingredients, per frame: misses, false positives, switches, gt objects."""
    counts = {}
    for (frame, _), row in events.iterrows():
        entry = counts.setdefault(frame, [0, 0, 0, 0])
        kind = row["Type"]
        if kind == "MISS":
            entry[0] += 1
        elif kind == "FP":
            entry[1] += 1
        elif kind == "SWITCH":
            entry[2] += 1
        # A ground-truth object is present for MATCH, MISS and SWITCH alike.
        if kind in ("MATCH", "MISS", "SWITCH"):
            entry[3] += 1
    return counts


def bootstrap_mota(counts, n: int, seed: int):
    """Percentile bootstrap over frames, matching evaluate_detection.py's scheme."""
    if n <= 0 or not counts:
        return float("nan"), float("nan")
    import numpy as np

    frames = list(counts)
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        picked = [counts[frames[rng.randrange(len(frames))]] for _ in frames]
        misses = sum(c[0] for c in picked)
        fps = sum(c[1] for c in picked)
        switches = sum(c[2] for c in picked)
        objects = sum(c[3] for c in picked)
        if objects:
            samples.append(1.0 - (misses + fps + switches) / objects)
    if not samples:
        return float("nan"), float("nan")
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seq", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--name", required=True, help="Config name for the row.")
    parser.add_argument("--gt", type=Path, default=None, help="Defaults to seq/gt/gt.txt")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="Distance threshold for a match.")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from mot_compat import mm  # noqa: PLC0415

    gt_path = args.gt or (args.seq / "gt" / "gt.txt")
    gt = load_mot(gt_path, "ground truth")
    res = load_mot(args.results, "results")

    acc = mm.utils.compare_to_groundtruth(gt, res, "iou", distth=args.iou)
    metrics = ["mota", "motp", "idf1", "num_switches", "num_false_positives",
               "num_misses", "num_objects", "num_unique_objects", "mostly_tracked",
               "mostly_lost"]
    summary = mm.metrics.create().compute(acc, metrics=metrics, name=args.name)
    got = {m: summary[m].iloc[0] for m in metrics}

    counts = per_frame_counts(acc.mm_events if hasattr(acc, "mm_events") else acc.events)
    low, high = bootstrap_mota(counts, args.bootstrap, args.seed)

    print(f"\nsequence : {args.seq.name}")
    print(f"results  : {args.results.name}")
    print(f"{'MOTA':>10} {got['mota']:.4f}   95% CI [{low:.4f}, {high:.4f}]")
    print(f"{'MOTP':>10} {got['motp']:.4f}")
    print(f"{'IDF1':>10} {got['idf1']:.4f}")
    print(f"{'ID sw':>10} {int(got['num_switches'])}")
    print(f"{'misses':>10} {int(got['num_misses'])}   "
          f"false pos {int(got['num_false_positives'])}   "
          f"gt objects {int(got['num_objects'])}")
    print(f"{'MT / ML':>10} {int(got['mostly_tracked'])} / {int(got['mostly_lost'])} "
          f"of {int(got['num_unique_objects'])} tracks")
    print("\nCI is within-clip only: it does not estimate across-match variance.")

    if args.no_write:
        return
    row = {
        "name": args.name,
        "sequence": args.seq.name,
        "mota": round(float(got["mota"]), 6),
        "mota_ci_low": round(low, 6),
        "mota_ci_high": round(high, 6),
        "motp": round(float(got["motp"]), 6),
        "idf1": round(float(got["idf1"]), 6),
        "id_switches": int(got["num_switches"]),
        "misses": int(got["num_misses"]),
        "false_positives": int(got["num_false_positives"]),
        "gt_objects": int(got["num_objects"]),
        "mostly_tracked": int(got["mostly_tracked"]),
        "mostly_lost": int(got["mostly_lost"]),
        "iou_thresh": args.iou,
        "bootstrap_n": args.bootstrap,
    }
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    is_new = not args.csv.exists()
    with args.csv.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    print(f"appended to {args.csv}")


if __name__ == "__main__":
    main()
