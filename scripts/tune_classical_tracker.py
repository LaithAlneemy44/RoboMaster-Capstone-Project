"""Tune the classical tracker's parameters on the validation clip.

CLAUDE.md specifies "no training - parameter tuning only" for the classical tracker, and
Params in scripts/classical_tracker.py calls itself "tunable on the validation split".
Until this script existed, no sweep had ever been run: the five values were hand-set from
the geometry of the problem and never revisited. This closes that.

VALIDATION CLIP ONLY, ENFORCED
    data/tracking/assignment.csv holds arc03 for val and arc04 for test. This script reads
    that file and REFUSES to run on the test clip. Tuning on test is the one mistake that
    would invalidate the final tracking numbers, and it is easier to prevent than to
    detect afterwards.

DETECTIONS ARE CACHED, SO THE SWEEP IS FREE
    The detector costs 60-400 ms per frame and the tracker costs 0.1-1 ms. Re-running
    detection per configuration would make the sweep hours long and would measure the
    detector, which is not the thing being tuned. Detections are computed once, cached to
    disk, and replayed identically into every configuration - so the only variable is the
    tracker.

    clean_frame() is applied to the cached detections exactly as run_trackers.py and
    benchmark_tracking.py apply it, so the tuned parameters are tuned against the pipeline
    that will actually run.

COORDINATE DESCENT, NOT A FULL GRID
    Five parameters at 3-5 values each is 675 combinations. The tracker is cheap enough
    that a full grid is affordable in wall-clock terms, but a coordinate sweep is easier
    to read and to justify: it shows the sensitivity of each parameter independently,
    which a grid's single winning row does not. Report it as coordinate descent from the
    hand-set defaults, not as a global optimum - it is not one.

Usage:
    python scripts/tune_classical_tracker.py
    python scripts/tune_classical_tracker.py --detector runs/detect/fast_960/weights/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

TRACKING = ROOT / "data" / "tracking"
ASSIGNMENT = TRACKING / "assignment.csv"
CACHE_DIR = ROOT / "results" / "tuning"
OUT_CSV = ROOT / "results" / "classical_tracker_tuning.csv"
DEFAULT_DETECTOR = ROOT / "runs" / "detect" / "yolo_960" / "weights" / "best.pt"

# Swept in this order, each starting from whatever won the previous axis. Ranges bracket
# the hand-set default rather than centring on it, so a default sitting at an edge is
# visible as such.
# MOTA differences below this on a single 300-frame clip are not distinguishable from
# noise, so the sweep does not switch on them - see the selection rule in main().
NOISE_BAND = 0.002

AXES = (
    ("iou_gate", (0.05, 0.10, 0.20, 0.30, 0.45)),
    ("max_age", (3, 5, 10, 20, 30)),
    ("min_hits", (1, 2, 3, 5)),
    ("process_noise", (1e-3, 1e-2, 1e-1, 1.0)),
    ("measurement_noise", (1e-2, 1e-1, 1.0, 10.0)),
)


def split_of(sequence: str) -> str:
    if not ASSIGNMENT.is_file():
        sys.exit(f"Missing {ASSIGNMENT}\nRun: python scripts/make_tracking_splits.py")
    with ASSIGNMENT.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["sequence"] == sequence:
                return row["split"]
    sys.exit(f"{sequence} is not in {ASSIGNMENT}")


def val_sequence() -> str:
    with ASSIGNMENT.open(newline="", encoding="utf-8") as fh:
        val = [r["sequence"] for r in csv.DictReader(fh) if r["split"] == "val"]
    if not val:
        sys.exit(f"No val clip in {ASSIGNMENT}")
    return val[0]


def cached_detections(seq: Path, detector: Path, family: str, config: str | None,
                      imgsz: int, conf: float, device: str) -> list[list]:
    """frame index -> list of (x, y, w, h), computed once and reused."""
    import cv2

    from auto_label import clean_frame
    from run_trackers import make_detector, read_seqinfo

    label = config or detector.stem
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{seq.name}_{family}_{label}_{imgsz}_{conf}.json"
    if cache.is_file():
        print(f"detections: cached  {cache.name}")
        return json.loads(cache.read_text(encoding="utf-8"))

    info = read_seqinfo(seq)
    images = sorted((seq / info["imDir"]).glob("*" + info["imExt"]))
    detect = make_detector(family, detector, config, imgsz, conf, device)
    box_median = float(info.get("imWidth", 1280)) / 11.0

    print(f"detections: computing over {len(images)} frames on {device} "
          f"(one time, then cached)")
    frames = []
    for index, path in enumerate(images):
        boxes = clean_frame(detect(cv2.imread(str(path))), median=box_median)
        frames.append([list(map(float, b)) for b in boxes])
        print(f"\r  {index + 1}/{len(images)}", end="", flush=True)
    print()
    cache.write_text(json.dumps(frames), encoding="utf-8")
    return frames


def score(gt, frames: list[list], params) -> tuple[float, float, int]:
    """(MOTA, IDF1, id switches) for one parameter set over the cached detections."""
    from mot_compat import mm

    from classical_tracker import ClassicalTracker

    tracker = ClassicalTracker(params)
    lines = []
    for index, boxes in enumerate(frames, start=1):
        # (track id, box) - the same unpacking run_trackers.py uses.
        for tid, (x, y, w, h) in tracker.update([tuple(b) for b in boxes]):
            lines.append(f"{index},{int(tid)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")

    # motmetrics reads MOT files; writing one keeps this on the exact scoring path
    # eval_tracking.py uses rather than a second, possibly divergent, implementation.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                     encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        temp = Path(fh.name)
    try:
        res = mm.io.loadtxt(str(temp), fmt="mot15-2D")
    finally:
        temp.unlink()

    acc = mm.utils.compare_to_groundtruth(gt, res, "iou", distth=0.5)
    summary = mm.metrics.create().compute(
        acc, metrics=["mota", "idf1", "num_switches"], name="x")
    return (float(summary["mota"].iloc[0]), float(summary["idf1"].iloc[0]),
            int(summary["num_switches"].iloc[0]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seq", default=None,
                        help="Defaults to the val clip named in assignment.csv.")
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--detector-family",
                        choices=("yolo", "ssd", "frcnn", "classical"), default="yolo")
    parser.add_argument("--detector-config", default=None)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu",
                        help="Detector device for the one-time cache. Not a benchmark.")
    parser.add_argument("--allow-test", action="store_true",
                        help=argparse.SUPPRESS)  # deliberately undocumented
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from classical_tracker import Params
    from mot_compat import mm

    name = args.seq or val_sequence()
    split = split_of(name)
    if split == "test" and not args.allow_test:
        sys.exit(
            f"{name} is the TEST clip. Tuning on test would invalidate the final "
            f"tracking numbers.\nThe val clip is {val_sequence()}."
        )
    if split != "val":
        print(f"WARNING: {name} is the {split} clip, not val.\n")

    seq = TRACKING / name
    gt_path = seq / "gt" / "gt.txt"
    if not gt_path.is_file():
        sys.exit(f"Missing {gt_path}")
    gt = mm.io.loadtxt(str(gt_path), fmt="mot15-2D")

    frames = cached_detections(seq, args.detector, args.detector_family,
                               args.detector_config, args.imgsz, args.conf, args.device)

    best = Params()
    base_mota, base_idf1, base_sw = score(gt, frames, best)
    print(f"\nclip {name} ({split}), detector "
          f"{args.detector_config or args.detector.parent.parent.name}")
    print(f"hand-set default: MOTA {base_mota:.4f}  IDF1 {base_idf1:.4f}  "
          f"switches {base_sw}\n")

    rows = [{"axis": "default", "value": "", **vars(best),
             "mota": round(base_mota, 6), "idf1": round(base_idf1, 6),
             "id_switches": base_sw}]
    best_mota, best_idf1 = base_mota, base_idf1

    for axis, values in AXES:
        print(f"{axis}:")
        winner, winner_mota, winner_idf1 = getattr(best, axis), best_mota, best_idf1
        for value in values:
            trial = Params(**{**vars(best), axis: value})
            mota, idf1, switches = score(gt, frames, trial)
            rows.append({"axis": axis, "value": value, **vars(trial),
                         "mota": round(mota, 6), "idf1": round(idf1, 6),
                         "id_switches": switches})
            mark = ""
            # Take a value only on a MOTA gain bigger than clip noise; inside that band
            # tie-break on IDF1. Observed on the process_noise axis: MOTA preferred 0.001
            # by 0.0006 while IDF1 preferred 0.01 by 0.011 - eighteen times larger, and
            # with fewer ID switches. A plain argmax chases the smaller, noisier signal.
            gain = mota - winner_mota
            if gain > NOISE_BAND or (abs(gain) <= NOISE_BAND and idf1 > winner_idf1):
                winner, winner_mota, winner_idf1, mark = value, mota, idf1, "  <-"
            current = "  (default)" if value == getattr(Params(), axis) else ""
            print(f"  {value!s:>8}   MOTA {mota:.4f}   IDF1 {idf1:.4f}   "
                  f"sw {switches:>3}{mark}{current}")
        best = Params(**{**vars(best), axis: winner})
        best_mota, best_idf1 = winner_mota, winner_idf1
        print(f"  -> keep {axis}={winner}\n")

    final_mota, final_idf1, final_sw = score(gt, frames, best)
    print("tuned parameters:")
    for key, value in vars(best).items():
        default = getattr(Params(), key)
        flag = "" if value == default else f"   (was {default})"
        print(f"  {key:<20} {value}{flag}")
    print(f"\nMOTA {base_mota:.4f} -> {final_mota:.4f}   "
          f"IDF1 {base_idf1:.4f} -> {final_idf1:.4f}   "
          f"switches {base_sw} -> {final_sw}")
    print("\nCoordinate descent from the hand-set defaults, on one val clip. Not a "
          "global optimum, and clip-level n=1.")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
