"""Score detection predictions - ONE evaluator for every model family.

Ultralytics computes mAP its own way and torchvision computes it another way, and
neither matches the other. If YOLO were scored by Ultralytics and MobileNet-SSD by
torchvision, the headline comparison of this project would be between two different
metrics. So every model - YOLO, Fast YOLO, MobileNet-SSD and the classical detector -
is scored here instead, from a COCO-format predictions file, against
data/splits/coco_val.json.

The `ignore` class policy (see IGNORE_CLASS in make_splits.py):
  1. `ignore` ground truth is removed, so an unfound ignore region is not a miss.
  2. `ignore` predictions are dropped, so the class is never scored.
  3. Predictions overlapping an ignore region at IoA >= --ignore-ioa are dropped,
     so firing on an ambiguous robot is not punished as a false positive.
Step 3 is why COCO's own `iscrowd` is not enough: iscrowd only excludes matches
within the SAME category, and an ignore region has to suppress any class.

Confidence intervals come from bootstrapping over val IMAGES. That is the only
honest CI available here: the val split is a single match clip, so clip-level n = 1
and these intervals describe variance within one match, not across matches.

Usage:
    # self-test: score the ground truth against itself, mAP must be 1.0
    python scripts/evaluate_detection.py --gt-as-predictions

    python scripts/evaluate_detection.py --predictions preds.json --name yolo11n_640
    python scripts/evaluate_detection.py --predictions preds.json --name x --bootstrap 0

Appends one row per config to results/detection.csv so runs accumulate uniformly.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import hashlib
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GT = ROOT / "data" / "splits" / "coco_val.json"
DEFAULT_RESULTS = ROOT / "results" / "detection.csv"

IGNORE_CLASS = "ignore"


# --------------------------------------------------------------------------
# ignore-region handling
# --------------------------------------------------------------------------

def _xywh_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return x, y, x + w, y + h


def ioa(pred: list[float], region: list[float]) -> float:
    """Intersection over the PREDICTION's area.

    Deliberately not IoU: a small detection sitting entirely inside a large ignore
    region has low IoU but should still be suppressed. IoA answers the question
    that actually matters - "is this detection inside an ignore region?"
    """
    px1, py1, px2, py2 = _xywh_to_xyxy(pred)
    rx1, ry1, rx2, ry2 = _xywh_to_xyxy(region)
    iw = min(px2, rx2) - max(px1, rx1)
    ih = min(py2, ry2) - max(py1, ry1)
    if iw <= 0 or ih <= 0:
        return 0.0
    area = (px2 - px1) * (py2 - py1)
    return (iw * ih) / area if area > 0 else 0.0


def apply_ignore_policy(
    gt: dict, predictions: list[dict], ioa_thresh: float
) -> tuple[dict, list[dict], dict[str, int]]:
    """Strip `ignore` from the ground truth and suppress predictions it covers."""
    ignore_id = next(
        (c["id"] for c in gt["categories"] if c["name"] == IGNORE_CLASS), None
    )
    if ignore_id is None:
        sys.exit(f"No {IGNORE_CLASS!r} category in the ground truth - cannot proceed.")

    regions: dict[int, list[list[float]]] = {}
    for ann in gt["annotations"]:
        if ann["category_id"] == ignore_id:
            regions.setdefault(ann["image_id"], []).append(ann["bbox"])

    scored = copy.deepcopy(gt)
    scored["annotations"] = [
        a for a in gt["annotations"] if a["category_id"] != ignore_id
    ]
    scored["categories"] = [c for c in gt["categories"] if c["id"] != ignore_id]
    # The export's id-0 entry is a Roboflow dummy supercategory with no annotations.
    scored["categories"] = [
        c for c in scored["categories"] if c["id"] != 0 or c["name"] != "robomaster-car"
    ]

    kept, dropped_class, dropped_region = [], 0, 0
    for det in predictions:
        if det["category_id"] == ignore_id:
            dropped_class += 1
            continue
        if any(
            ioa(det["bbox"], r) >= ioa_thresh for r in regions.get(det["image_id"], ())
        ):
            dropped_region += 1
            continue
        kept.append(det)

    stats = {
        "gt_ignore_removed": len(gt["annotations"]) - len(scored["annotations"]),
        "pred_ignore_class_dropped": dropped_class,
        "pred_in_ignore_region_dropped": dropped_region,
    }
    return scored, kept, stats


# --------------------------------------------------------------------------
# AP accumulation
# --------------------------------------------------------------------------
# pycocotools does the per-image matching (trusted), but its accumulate() cannot
# represent an image sampled twice, which a bootstrap resample requires. So the
# accumulation is reimplemented here over its own evalImgs output. main() asserts
# this reproduces the official mAP exactly before any of it is reported.

def accumulate_ap(
    eval_imgs: list, params, img_positions: list[int]
) -> tuple[float, float, np.ndarray]:
    """Return (mAP@[.5:.95], mAP@.5, per-class AP@[.5:.95]) over sampled images.

    `img_positions` indexes into params.imgIds and may repeat entries.
    """
    n_cat = len(params.catIds)
    n_img = len(params.imgIds)
    n_area = len(params.areaRng)
    n_iou = len(params.iouThrs)
    rec_thrs = params.recThrs
    max_det = params.maxDets[-1]

    per_class = np.full((n_cat, n_iou), np.nan)

    for k in range(n_cat):
        # evalImgs is flat and ordered [catIdx][areaIdx][imgIdx]; area index 0 is
        # 'all', which is the only range this project reports.
        base = k * n_area * n_img
        entries = [eval_imgs[base + i] for i in img_positions]
        entries = [e for e in entries if e is not None]
        if not entries:
            continue

        n_gt = sum(int(np.count_nonzero(e["gtIgnore"] == 0)) for e in entries)
        if n_gt == 0:
            continue

        scores = np.concatenate([e["dtScores"][:max_det] for e in entries])
        if scores.size == 0:
            per_class[k] = 0.0
            continue
        order = np.argsort(-scores, kind="mergesort")

        matches = np.concatenate(
            [e["dtMatches"][:, :max_det] for e in entries], axis=1
        )[:, order]
        ignored = np.concatenate(
            [e["dtIgnore"][:, :max_det] for e in entries], axis=1
        )[:, order]

        tps = np.logical_and(matches, np.logical_not(ignored))
        fps = np.logical_and(np.logical_not(matches), np.logical_not(ignored))
        tp_sum = np.cumsum(tps, axis=1).astype(float)
        fp_sum = np.cumsum(fps, axis=1).astype(float)

        for t in range(n_iou):
            tp, fp = tp_sum[t], fp_sum[t]
            recall = tp / n_gt
            precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

            # Monotonic envelope, then sample at the 101 standard recall points.
            precision = np.maximum.accumulate(precision[::-1])[::-1]
            idx = np.searchsorted(recall, rec_thrs, side="left")
            sampled = np.zeros(len(rec_thrs))
            valid = idx < len(precision)
            sampled[valid] = precision[idx[valid]]
            per_class[k, t] = sampled.mean()

    with np.errstate(invalid="ignore"):
        overall = float(np.nanmean(per_class)) if not np.all(np.isnan(per_class)) else 0.0
        at50 = float(np.nanmean(per_class[:, 0])) if n_iou else 0.0
        by_class = np.nanmean(per_class, axis=1)
    return overall, at50, by_class


def best_f1(coco_eval, n_cat: int):
    """Best F1 on the IoU=0.5 PR curve, with the precision and recall that produced it.

    mAP alone does not give an operating point, and the project's metric list asks for
    precision, recall and F1. This reads them off the curve pycocotools already computed
    rather than running a second, possibly inconsistent, matcher.

    Precision and recall are reported AT THE BEST-F1 POINT, not at a fixed confidence
    threshold. A fixed threshold means different things to different models - a score of
    0.25 is not comparable between YOLO and SSD - so it would make these columns
    incomparable across exactly the models the table exists to compare.

    Returns (macro F1, per-class F1, macro precision, macro recall).
    """
    precision = coco_eval.eval["precision"]  # [T, R, K, A, M]
    rec_thrs = coco_eval.params.recThrs
    per_class = np.full(n_cat, np.nan)
    per_class_p = np.full(n_cat, np.nan)
    per_class_r = np.full(n_cat, np.nan)
    for k in range(n_cat):
        pr = precision[0, :, k, 0, -1]
        valid = pr > -1
        if not valid.any():
            continue
        p, r = pr[valid], rec_thrs[valid]
        with np.errstate(invalid="ignore", divide="ignore"):
            f1 = np.where((p + r) > 0, 2 * p * r / (p + r), 0.0)
        best = int(np.nanargmax(f1))
        per_class[k] = float(f1[best])
        per_class_p[k] = float(p[best])
        per_class_r[k] = float(r[best])

    def macro(values):
        return float(np.nanmean(values)) if not np.all(np.isnan(values)) else 0.0

    return macro(per_class), per_class, macro(per_class_p), macro(per_class_r)


def mean_tp_iou(scored_gt: dict, detections: list[dict], thresh: float = 0.5) -> float:
    """Mean IoU of true-positive detections - localisation quality, not detection rate.

    CLAUDE.md's metric list asks for IoU, and mAP does not answer it: a model can score
    well on mAP while placing every box loosely, because mAP asks whether a box cleared
    an IoU threshold, never by how much. This reports how tight the boxes that DID match
    actually are, which is what matters when the box is what you aim at.

    Matched greedily in score order against the ignore-policy-filtered ground truth - the
    same detections and boxes the mAP above is computed from. Written out rather than
    read from pycocotools internals: its `ious` matrices are indexed by an internal
    re-sorting of the ground truth, and quietly mis-indexing them would produce a
    plausible wrong number rather than an error.
    """
    gt_by_key: dict = {}
    for ann in scored_gt["annotations"]:
        if ann.get("iscrowd"):
            continue
        gt_by_key.setdefault((ann["image_id"], ann["category_id"]), []).append(ann["bbox"])

    det_by_key: dict = {}
    for det in detections:
        det_by_key.setdefault((det["image_id"], det["category_id"]), []).append(det)

    total, count = 0.0, 0
    for key, dets in det_by_key.items():
        gts = gt_by_key.get(key)
        if not gts:
            continue
        taken = [False] * len(gts)
        for det in sorted(dets, key=lambda d: -d["score"]):
            dx1, dy1, dx2, dy2 = _xywh_to_xyxy(det["bbox"])
            best_iou, best_index = 0.0, -1
            for index, gt in enumerate(gts):
                if taken[index]:
                    continue
                gx1, gy1, gx2, gy2 = _xywh_to_xyxy(gt)
                ix1, iy1 = max(dx1, gx1), max(dy1, gy1)
                ix2, iy2 = min(dx2, gx2), min(dy2, gy2)
                iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
                inter = iw * ih
                if inter <= 0:
                    continue
                union = ((dx2 - dx1) * (dy2 - dy1)
                         + (gx2 - gx1) * (gy2 - gy1) - inter)
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou, best_index = iou, index
            if best_index >= 0 and best_iou >= thresh:
                taken[best_index] = True
                total += best_iou
                count += 1
    return total / count if count else float("nan")


# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions", type=Path, help="COCO-format detections JSON.")
    parser.add_argument(
        "--gt-as-predictions",
        action="store_true",
        help="Self-test: score the ground truth against itself. mAP must be 1.0 - "
        "this is what proves image ids and coordinate spaces line up.",
    )
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--name", default=None, help="Config name for the results row.")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--no-write", action="store_true", help="Do not append a row.")
    parser.add_argument(
        "--ignore-ioa",
        type=float,
        default=0.5,
        help="Drop a prediction whose intersection-over-own-area with any ignore "
        "region reaches this (default: 0.5).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Bootstrap resamples over val images for the CI (0 disables).",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.predictions and not args.gt_as_predictions:
        sys.exit("Pass --predictions PATH, or --gt-as-predictions for the self-test.")
    if not args.gt.is_file():
        sys.exit(f"Missing {args.gt}\nRun: python scripts/make_splits.py")

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        sys.exit(f"{exc}\nInstall dependencies - see README.md (Setup).")

    gt_raw = json.loads(args.gt.read_text(encoding="utf-8"))

    if args.gt_as_predictions:
        name = args.name or "SELFTEST_gt_as_predictions"
        predictions = [
            {
                "image_id": a["image_id"],
                "category_id": a["category_id"],
                "bbox": a["bbox"],
                "score": 1.0,
            }
            for a in gt_raw["annotations"]
        ]
    else:
        name = args.name or args.predictions.stem
        predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
        if isinstance(predictions, dict):  # tolerate {"annotations": [...]}
            predictions = predictions.get("annotations", [])

    scored_gt, kept, drop_stats = apply_ignore_policy(
        gt_raw, predictions, args.ignore_ioa
    )

    print(f"config           : {name}")
    print(f"ground truth     : {args.gt.relative_to(ROOT)}")
    print(f"images           : {len(scored_gt['images'])}")
    print(f"scored instances : {len(scored_gt['annotations'])}")
    print(f"predictions      : {len(predictions)} -> {len(kept)} after ignore policy")
    for k, v in drop_stats.items():
        print(f"  {k:<32} {v}")

    if not kept:
        sys.exit("\nNo predictions survived the ignore policy - nothing to score.")

    coco_gt = COCO()
    coco_gt.dataset = scored_gt
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(copy.deepcopy(kept))

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    print()
    coco_eval.summarize()

    params = coco_eval.params
    cat_ids = list(params.catIds)
    names = {c["id"]: c["name"] for c in scored_gt["categories"]}

    # Reproduce the official number with the local accumulator before trusting it
    # for the bootstrap. A mismatch means the reimplementation has drifted.
    all_positions = list(range(len(params.imgIds)))
    m_all, m50, by_class = accumulate_ap(coco_eval.evalImgs, params, all_positions)
    official = float(coco_eval.stats[0])
    if abs(m_all - official) > 1e-6:
        sys.exit(
            f"\nInternal accumulator disagrees with pycocotools "
            f"({m_all:.6f} vs {official:.6f}). Refusing to report bootstrap CIs."
        )

    f1_macro, f1_class, p_macro, r_macro = best_f1(coco_eval, len(cat_ids))
    tp_iou = mean_tp_iou(scored_gt, kept)

    print(f"precision {p_macro:.4f}   recall {r_macro:.4f}   "
          f"mean TP IoU {tp_iou:.4f}   (best-F1 point, IoU>=0.5)")
    print(f"\n{'class':<12}{'AP@[.5:.95]':>13}{'bestF1@.5':>12}")
    for k, cid in enumerate(cat_ids):
        ap = by_class[k]
        print(
            f"{names.get(cid, cid):<12}"
            f"{'n/a' if np.isnan(ap) else f'{ap:.4f}':>13}"
            f"{'n/a' if np.isnan(f1_class[k]) else f'{f1_class[k]:.4f}':>12}"
        )

    # Fingerprint of the predictions this row was computed from. Rows in this CSV once
    # drifted from their prediction files without anyone noticing: predict_to_coco.py was
    # fixed, every SSD's predictions were regenerated, and the CSV kept the scores from
    # the old ones - ssd_small_960 read 0.1193 when its file actually scored 0.0626. The
    # numbers looked plausible, so nothing flagged it. Recording the hash makes the
    # mismatch detectable instead of invisible.
    digest = hashlib.sha1(args.predictions.read_bytes()).hexdigest()[:12]

    row = {
        "name": name,
        "images": len(scored_gt["images"]),
        "predictions_sha1": digest,
        "mAP_50_95": round(m_all, 6),
        "mAP_50": round(m50, 6),
        "mAP_75": round(float(coco_eval.stats[2]), 6),
        "AR_100": round(float(coco_eval.stats[8]), 6),
        "best_f1_macro": round(f1_macro, 6),
        # Precision and recall at the best-F1 operating point, plus how tight the boxes
        # that matched actually are. All three are on the project's metric list and none
        # is recoverable from mAP.
        "precision_at_best_f1": round(p_macro, 6),
        "recall_at_best_f1": round(r_macro, 6),
        "mean_tp_iou": ("" if np.isnan(tp_iou) else round(float(tp_iou), 6)),
        "ci_low": "",
        "ci_high": "",
        "bootstrap_n": args.bootstrap,
    }
    for k, cid in enumerate(cat_ids):
        row[f"AP_{names.get(cid, cid)}"] = (
            "" if np.isnan(by_class[k]) else round(float(by_class[k]), 6)
        )

    if args.bootstrap > 0:
        rng = random.Random(args.seed)
        n = len(params.imgIds)
        samples = []
        for _ in range(args.bootstrap):
            positions = [rng.randrange(n) for _ in range(n)]
            samples.append(accumulate_ap(coco_eval.evalImgs, params, positions)[0])
        low, high = np.percentile(samples, [2.5, 97.5])
        row["ci_low"], row["ci_high"] = round(float(low), 6), round(float(high), 6)
        print(
            f"\nmAP@[.5:.95] {m_all:.4f}   95% CI [{low:.4f}, {high:.4f}]  "
            f"({args.bootstrap} resamples over {n} images)"
        )
        print(
            "CI is within-match only: val is a single clip, so clip-level n = 1 and\n"
            "this does not estimate across-match variance."
        )
    else:
        print(f"\nmAP@[.5:.95] {m_all:.4f}   (no bootstrap)")

    if not args.no_write:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        exists = args.results.is_file()
        with args.results.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"\nappended to {args.results.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
