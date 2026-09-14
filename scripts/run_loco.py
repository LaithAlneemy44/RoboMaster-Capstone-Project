"""Leave-one-clip-out cross-validation: the project's first across-match variance estimate.

WHY THIS EXISTS
    The 2655 detection images come from only SEVEN match recordings, and the committed
    split holds out one of them. Every confidence interval in results/detection.csv
    therefore bootstraps over IMAGES WITHIN ONE MATCH. CAMERA_HANDOFF.md issue 11 states
    the consequence plainly: clip-level n = 1, so nothing in this project estimates how
    performance varies across venues, lighting or robot liveries.

    This trains the same config seven times, holding out each clip in turn. The spread of
    the seven scores is the missing number.

WHAT THESE ROWS ARE NOT
    Each fold is scored on a DIFFERENT val set, so a fold's mAP is not comparable to any
    row in results/detection.csv and the two must never be averaged together. That is why
    these land in results/loco.csv instead. What IS meaningful is the mean and standard
    deviation ACROSS folds.

    Fold val sizes range 289-420 images because the clips differ in length, so per-fold
    confidence intervals are not equally wide either. Report the across-fold std as the
    headline; it is the quantity that was missing.

ISOLATION
    Each fold writes its split into runs/loco/<clip>/splits/ via make_splits.py --out-dir
    and trains against that fold's own roco_central.yaml. data/splits/ is never touched,
    so other work can keep reading the committed split while this runs for hours.

HYPERPARAMETERS ARE COPIED FROM THE COMMITTED RUN, NOT CHOSEN
    fast_640 was trained with yolo11n, 100 epochs, patience 30, batch 16, seed 0
    (runs/detect/fast_640/args.yaml). The folds use exactly those. Changing any of them
    would make the spread a measure of the change rather than of the clips.

Usage:
    python scripts/run_loco.py                 # all seven folds, resumable
    python scripts/run_loco.py --summary       # just re-print the summary
    touch STOP                                 # finish the current fold, then exit
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
ASSIGNMENT = ROOT / "data" / "splits" / "assignment.csv"
LOCO_DIR = ROOT / "runs" / "loco"
RESULTS = ROOT / "results" / "loco.csv"
PREDICTIONS = ROOT / "results" / "predictions"
STOP = ROOT / "STOP"

# Copied from runs/detect/fast_640/args.yaml. See the module docstring.
MODEL = "fast"
IMGSZ = 640
EPOCHS = 100
BATCH = 16
PATIENCE = 30
SEED = 0


def clips() -> list[str]:
    """The seven match recordings, in a stable order."""
    if not ASSIGNMENT.is_file():
        sys.exit(f"Missing {ASSIGNMENT}. Run: python scripts/make_splits.py --from-assignment")
    with ASSIGNMENT.open(newline="", encoding="utf-8") as fh:
        return sorted({row["clip"] for row in csv.DictReader(fh)})


def fold_name(clip: str) -> str:
    """A path- and CLI-safe name. Several clips begin with '-', which argparse eats."""
    return "loco_" + clip.lstrip("-").replace("-", "_")


def done_folds() -> set[str]:
    if not RESULTS.is_file():
        return set()
    with RESULTS.open(newline="", encoding="utf-8") as fh:
        return {row["name"] for row in csv.DictReader(fh)}


def run(args: list[str], label: str) -> None:
    print(f"    $ {' '.join(str(a) for a in args[1:4])} ...", flush=True)
    done = subprocess.run(args, cwd=ROOT)
    if done.returncode != 0:
        sys.exit(f"{label} failed with exit code {done.returncode}. "
                 f"Fix it and re-run; completed folds are skipped.")


def do_fold(clip: str) -> None:
    name = fold_name(clip)
    fold_dir = LOCO_DIR / name
    splits = fold_dir / "splits"
    weights = ROOT / "runs" / "detect" / name / "weights" / "best.pt"
    preds = PREDICTIONS / f"{name}.json"

    print(f"\n=== {name}   (holding out {clip}) ===", flush=True)

    # 1. Fold split, isolated from data/splits/.
    run([PY, "scripts/make_splits.py", f"--holdout={clip}", "--out-dir", str(splits)],
        "make_splits")

    # 2. Train, unless a PREVIOUS RUN FINISHED TRAINING THIS FOLD.
    #
    # The marker matters. Ultralytics writes best.pt every time val fitness improves, so
    # a fold killed at epoch 50 still has one - and an earlier version of this check
    # skipped training whenever best.pt existed, which would have silently scored a
    # half-trained fold alongside fully-trained ones and quietly widened the across-clip
    # spread this study exists to measure. Existence of weights proves training started,
    # never that it finished.
    marker = fold_dir / "TRAINED"
    if marker.is_file():
        print(f"    training already completed for this fold, skipping")
    else:
        if weights.is_file():
            print(f"    found weights with no completion marker - previous run was "
                  f"interrupted, retraining from scratch")
        run([PY, "scripts/train_yolo.py", "--model", MODEL, "--imgsz", str(IMGSZ),
             "--epochs", str(EPOCHS), "--batch", str(BATCH), "--patience", str(PATIENCE),
             "--seed", str(SEED), "--data", str(splits / "roco_central.yaml"),
             "--name", name], "train_yolo")
        fold_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text("training completed", encoding="utf-8")
    if not weights.is_file():
        sys.exit(f"Training reported success but {weights} is missing.")

    # 3. Predict on THIS fold's val set, not the committed one.
    run([PY, "scripts/predict_to_coco.py", "--family", "yolo", "--weights", str(weights),
         "--imgsz", str(IMGSZ), "--gt", str(splits / "coco_val.json"),
         "--out", str(preds), "--device", "0"], "predict_to_coco")

    # 4. Score into loco.csv - deliberately NOT detection.csv, see the docstring.
    run([PY, "scripts/evaluate_detection.py", "--predictions", str(preds),
         "--gt", str(splits / "coco_val.json"), "--name", name,
         "--results", str(RESULTS)], "evaluate_detection")


def summarise() -> None:
    if not RESULTS.is_file():
        print("No folds scored yet.")
        return
    with RESULTS.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["name"].startswith("loco_")]
    if not rows:
        print("No fold rows in results/loco.csv.")
        return

    print(f"\n{'fold':<34}{'mAP':>9}{'armor AP':>10}{'val imgs':>10}")
    for r in sorted(rows, key=lambda r: r["name"]):
        imgs = r.get("n_images") or r.get("images") or "-"
        print(f"{r['name']:<34}{float(r['mAP_50_95']):>9.4f}"
              f"{float(r['AP_armor']):>10.4f}{imgs:>10}")

    for field, label in (("mAP_50_95", "mAP"), ("AP_armor", "armor AP")):
        vals = [float(r[field]) for r in rows]
        mean = statistics.fmean(vals)
        if len(vals) > 1:
            sd = statistics.stdev(vals)
            print(f"\n{label:<10} across {len(vals)} clips: mean {mean:.4f}  "
                  f"sd {sd:.4f}  min {min(vals):.4f}  max {max(vals):.4f}")
        else:
            print(f"\n{label:<10} only one fold so far: {mean:.4f}")

    if len(rows) == len(clips()):
        print("\nThis across-clip sd is the across-match variance the bootstrapped CIs in\n"
              "results/detection.csv cannot capture. Fold scores use different val sets\n"
              "and are not comparable to detection.csv rows individually.")
    else:
        print(f"\n{len(rows)} of {len(clips())} folds complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", action="store_true",
                        help="Print the summary for completed folds and exit.")
    parser.add_argument("--only", nargs="+", metavar="CLIP",
                        help="Run only these clips (default: all seven).")
    args = parser.parse_args()

    if args.summary:
        summarise()
        return

    STOP.unlink(missing_ok=True)
    all_clips = clips()
    wanted = args.only or all_clips
    unknown = sorted(set(wanted) - set(all_clips))
    if unknown:
        sys.exit(f"Unknown clip(s): {unknown}\nKnown: {all_clips}")

    already = done_folds()
    todo = [c for c in wanted if fold_name(c) not in already]
    print(f"{len(all_clips)} clips, {len(already)} already scored, {len(todo)} to run")
    if not todo:
        summarise()
        return

    LOCO_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for i, clip in enumerate(todo, 1):
        if STOP.is_file():
            print(f"\nSTOP found - stopping cleanly after {i - 1} of {len(todo)} folds.")
            break
        print(f"\n[{i}/{len(todo)}] elapsed {(time.perf_counter() - started) / 60:.0f} min")
        do_fold(clip)

    print(f"\ntotal {(time.perf_counter() - started) / 3600:.2f} h")
    summarise()


if __name__ == "__main__":
    main()
