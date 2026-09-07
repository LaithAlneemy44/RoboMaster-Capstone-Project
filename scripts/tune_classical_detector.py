"""Sweep the classical detector's parameters on the validation split.

CLAUDE.md specifies "no training - parameter tuning only" for the classical detector, and
Params in scripts/classical_detector.py calls its fields "the axes". The five committed
configs vary only three of the thirteen, and were hand-picked rather than searched. This
is the detector counterpart to scripts/tune_classical_tracker.py, and closes the same gap.

WHAT IS BEING OPTIMISED, AND WHY IT IS NOT mAP
    The classical detector emits ONE class, armor. Its overall mAP is therefore mostly a
    statement about the four classes it never attempts, and improving it would mean
    detecting things the algorithm was never built to find. AP_armor is the honest
    objective, and armor is also the aim point the whole system exists to hit.

COORDINATE DESCENT FROM THE COMMITTED DEFAULT
    Thirteen parameters do not admit a grid. Sweeping one axis at a time from the
    committed `balanced` config shows each parameter's sensitivity independently, which
    is more useful for the write-up than a single winning row, and is cheap: one
    evaluation is ~15 s, so a full sweep is minutes rather than hours.

    Report it as coordinate descent, not a global optimum. It is not one.

TUNED ON VAL, AND VAL IS WHAT GETS REPORTED
    The detection split has no test set - CLAUDE.md: "no test set needed, combined models
    are what's tested" - so there is nowhere else to tune. Any gain found here is
    therefore optimistic by construction, and the write-up must say so. The five existing
    configs have the same exposure: they were also chosen by looking at val.

Usage:
    python scripts/tune_classical_detector.py
    python scripts/tune_classical_detector.py --quick     # fewer images, for a smoke test
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

OUT_CSV = ROOT / "results" / "classical_detector_tuning.csv"

# Swept in this order, each starting from whatever won the previous axis. Ranges bracket
# the committed value rather than centring on it, so a default sitting at an edge shows
# up as one.
AXES = (
    ("value_min", (140, 155, 170, 185, 200, 215)),
    ("sat_min", (20, 30, 40, 55, 70)),
    ("ncc_min", (0.15, 0.20, 0.25, 0.30, 0.40, 0.50)),
    ("min_area", (8, 12, 15, 20, 30)),
    ("min_side", (2, 3, 4)),
    ("max_side", (40, 60, 90)),
    ("nms_iou", (0.25, 0.40, 0.55)),
    ("pad_ratio", (1.0, 1.2, 1.5, 2.0)),
    ("hue_slack", (0, 5, 10)),
    ("close_kernel", (1, 3, 5)),
)

# AP differences below this on 397 images are not worth chasing; the tracker sweep learnt
# the same lesson when a plain argmax preferred a 0.0006 MOTA gain over a 0.011 IDF1 one.
NOISE_BAND = 0.0015


def evaluate(params, targets, name_to_cat, gt_raw, conf: float) -> tuple[float, int]:
    """(AP_armor, detection count) for one parameter set on the val images."""
    import cv2
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    from classical_detector import ClassicalDetector
    from evaluate_detection import apply_ignore_policy

    detector = ClassicalDetector(params)
    detections = []
    for image_id, path in targets:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        for (x, y, w, h), score, cls in detector.detect(frame):
            if score < conf or cls not in name_to_cat:
                continue
            detections.append({"image_id": image_id, "category_id": name_to_cat[cls],
                               "bbox": [x, y, w, h], "score": float(score)})
    if not detections:
        return 0.0, 0

    # The same ignore policy the reported numbers use, so a tuned config can be dropped
    # into evaluate_detection.py and reproduce what this printed.
    scored_gt, kept, _ = apply_ignore_policy(gt_raw, detections, 0.5)
    if not kept:
        return 0.0, len(detections)

    coco_gt = COCO()
    coco_gt.dataset = scored_gt
    coco_gt.createIndex()
    coco_eval = COCOeval(coco_gt, coco_gt.loadRes(kept), "bbox")
    coco_eval.params.catIds = [name_to_cat["armor"]]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return float(coco_eval.stats[0]), len(detections)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--conf", type=float, default=0.0,
                        help="Score floor. 0 keeps the whole PR curve, as mAP wants.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Use only the first N val images.")
    parser.add_argument("--quick", action="store_true",
                        help="60 images and the three main axes - a smoke test.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import json

    from classical_detector import CONFIGS
    from predict_to_coco import DEFAULT_GT, load_targets

    targets, name_to_cat = load_targets(DEFAULT_GT)
    if args.quick:
        args.limit = args.limit or 60
    if args.limit:
        targets = targets[:args.limit]
    gt_raw = json.loads(DEFAULT_GT.read_text(encoding="utf-8"))
    if "armor" not in name_to_cat:
        sys.exit(f"Ground truth has no 'armor' class: {sorted(name_to_cat)}")

    axes = AXES[:3] if args.quick else AXES
    best = CONFIGS["balanced"]
    started = time.perf_counter()
    base_ap, base_n = evaluate(best, targets, name_to_cat, gt_raw, args.conf)
    per = time.perf_counter() - started

    print(f"\nimages   : {len(targets)}   ~{per:.0f}s per evaluation")
    print(f"objective: AP_armor  (the only class this detector emits)")
    print(f"start    : balanced, AP_armor {base_ap:.4f}, {base_n} detections\n")

    rows = [{"axis": "default", "value": "", **dataclasses.asdict(best),
             "ap_armor": round(base_ap, 6), "detections": base_n}]
    best_ap = base_ap

    for axis, values in axes:
        print(f"{axis}:")
        winner, winner_ap = getattr(best, axis), best_ap
        for value in values:
            trial = dataclasses.replace(best, **{axis: value})
            ap, count = evaluate(trial, targets, name_to_cat, gt_raw, args.conf)
            rows.append({"axis": axis, "value": value, **dataclasses.asdict(trial),
                         "ap_armor": round(ap, 6), "detections": count})
            mark = ""
            if ap - winner_ap > NOISE_BAND:
                winner, winner_ap, mark = value, ap, "  <-"
            current = "  (default)" if value == getattr(CONFIGS["balanced"], axis) else ""
            print(f"  {value!s:>7}   AP_armor {ap:.4f}   {count:>7} dets{mark}{current}")
        best = dataclasses.replace(best, **{axis: winner})
        best_ap = winner_ap
        print(f"  -> keep {axis}={winner}\n")

    final_ap, final_n = evaluate(best, targets, name_to_cat, gt_raw, args.conf)
    print("tuned parameters:")
    for field in dataclasses.fields(best):
        value = getattr(best, field.name)
        default = getattr(CONFIGS["balanced"], field.name)
        flag = "" if value == default else f"   (was {default})"
        print(f"  {field.name:<18} {value}{flag}")
    print(f"\nAP_armor {base_ap:.4f} -> {final_ap:.4f}   "
          f"detections {base_n} -> {final_n}")
    print(f"elapsed {(time.perf_counter() - started) / 60:.1f} min")
    print("\nCoordinate descent from `balanced`, on the val split. Not a global optimum, "
          "and tuned on the same images it is reported on - the detection split has no "
          "test set, so any gain here is optimistic.")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
