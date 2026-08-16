"""Train, predict and score every detection config, unattended and resumable.

The full ladder is roughly 40 GPU-hours across 12 configs, so this is a multi-day
run on a card the desktop shares. Three properties matter more than speed:

  RESUMABLE   Re-running skips any config already scored, so an interrupted sweep
              picks up where it stopped instead of starting over.
  FAULT-TOLERANT  One config failing (OOM, a bad checkpoint) records the failure and
              moves on. A 40-hour job must not die at hour 30 over one run.
  ORDERED CHEAPEST FIRST  Real results start landing within a couple of hours rather
              than at the end, so a mistake in the pipeline surfaces early.

Each config runs train -> predict_to_coco -> evaluate_detection, so every model is
scored through the one shared evaluator.

STOPPING AND RESUMING (you do not need to leave the machine on overnight):

  Ctrl-C          Stops now. Both trainers checkpoint every epoch, so you lose at
                  most the epoch in flight. Re-run this script to carry on - it
                  resumes the partial config from last.pt with its optimizer and
                  LR schedule intact, then continues down the list.
  STOP file       `New-Item STOP` makes the sweep finish the CONFIG it is on and
                  then exit cleanly. Delete the file before re-running.
  --stop-after H  Refuses to START a config that cannot finish within H hours, so
                  an evening session winds itself down.

A config counts as trained only when its run directory holds a `.trained` marker.
best.pt alone is not proof - both trainers write it mid-run, so treating it as
"done" would silently score a half-trained model.

Usage:
    python scripts/run_sweep.py --dry-run     # show the plan and exit
    python scripts/run_sweep.py               # run everything not yet scored
    python scripts/run_sweep.py --stop-after 8
    python scripts/run_sweep.py --only fast_320 ssd_small_320
    python scripts/run_sweep.py --epochs 60   # trim the budget
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
RESULTS = ROOT / "results" / "detection.csv"
PRED_DIR = ROOT / "results" / "predictions"
SWEEP_LOG = ROOT / "results" / "sweep_log.csv"
# Create this file to make the sweep stop after the config it is currently running,
# rather than killing it mid-epoch.
STOP_FILE = ROOT / "STOP"

# (family, variant, imgsz, batch, estimated hours at 100 epochs)
# Batches come from results/probe.csv - measured, not guessed. The yolo11s rungs at
# 640 and 960 are held below their probed maximum because those peaked at 92-96% of
# available VRAM, and the desktop drives the same card; Ultralytics accumulates to a
# nominal batch of 64, so a smaller batch barely perturbs the optimisation.
CONFIGS = [
    ("yolo", "fast", 320, 32, 1.66),
    ("yolo", "yolo", 320, 32, 2.28),
    ("ssd", "small", 320, 32, 1.50),
    ("ssd", "large", 320, 32, 1.60),
    ("ssd", "small", 640, 16, 2.50),
    ("ssd", "large", 640, 16, 2.94),
    ("yolo", "fast", 640, 16, 3.21),
    ("ssd", "small", 960, 8, 4.50),
    ("yolo", "fast", 960, 8, 4.93),
    ("yolo", "yolo", 640, 12, 4.50),
    ("ssd", "large", 960, 8, 5.53),
    ("yolo", "yolo", 960, 6, 8.59),
]


def config_name(family: str, variant: str, imgsz: int) -> str:
    return f"{variant}_{imgsz}" if family == "yolo" else f"ssd_{variant}_{imgsz}"


def run_dir(family: str, variant: str, imgsz: int) -> Path:
    if family == "yolo":
        return ROOT / "runs" / "detect" / f"{variant}_{imgsz}"
    return ROOT / "runs" / "ssd" / f"ssd_{variant}_{imgsz}"


def weights_path(family: str, variant: str, imgsz: int) -> Path:
    base = run_dir(family, variant, imgsz)
    return base / "weights" / "best.pt" if family == "yolo" else base / "best.pt"


def last_path(family: str, variant: str, imgsz: int) -> Path:
    base = run_dir(family, variant, imgsz)
    return base / "weights" / "last.pt" if family == "yolo" else base / "last.pt"


def already_scored() -> set[str]:
    if not RESULTS.is_file():
        return set()
    with RESULTS.open(newline="", encoding="utf-8") as fh:
        return {row["name"] for row in csv.DictReader(fh)}


def log_outcome(name: str, stage: str, status: str, seconds: float, detail: str) -> None:
    SWEEP_LOG.parent.mkdir(parents=True, exist_ok=True)
    exists = SWEEP_LOG.is_file()
    with SWEEP_LOG.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if not exists:
            writer.writerow(["name", "stage", "status", "seconds", "detail"])
        writer.writerow([name, stage, status, round(seconds, 1), detail])


def run(label: str, command: list[str]) -> tuple[bool, float]:
    """Run a stage, streaming nothing but reporting cleanly. Returns (ok, seconds)."""
    print(f"    [{label}] {' '.join(str(c) for c in command[1:])}", flush=True)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        tail = (result.stdout + result.stderr).strip().splitlines()[-6:]
        print(f"    [{label}] FAILED after {elapsed / 60:.1f} min", flush=True)
        for line in tail:
            print(f"      | {line}", flush=True)
        return False, elapsed
    print(f"    [{label}] ok ({elapsed / 60:.1f} min)", flush=True)
    return True, elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="Run only these config names (see --dry-run).")
    parser.add_argument("--force", action="store_true",
                        help="Re-run configs that already have a results row.")
    parser.add_argument("--stop-after", type=float, default=None, metavar="HOURS",
                        help="Do not START a config whose estimate would run past this "
                             "many hours. Lets the sweep wind down on its own before "
                             "you shut the machine down for the night.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scored = set() if args.force else already_scored()

    pending = []
    for family, variant, imgsz, batch, hours in CONFIGS:
        name = config_name(family, variant, imgsz)
        if args.only and name not in args.only:
            continue
        if name in scored:
            print(f"[skip] {name:<18} already in results/detection.csv")
            continue
        pending.append((family, variant, imgsz, batch, hours, name))

    if not pending:
        print("\nNothing to do - every requested config is already scored.")
        return

    budget = sum(p[4] for p in pending) * args.epochs / 100
    print(f"\n{len(pending)} config(s) to run, ~{budget:.1f} GPU-hours at "
          f"{args.epochs} epochs (before early stopping)\n")
    header = f"{'config':<18}{'family':>8}{'imgsz':>7}{'batch':>7}{'est h':>8}"
    if args.stop_after is not None:
        header += f"{'cumul':>8}  within budget"
    print(header)

    cumulative = 0.0
    for family, variant, imgsz, batch, hours, name in pending:
        estimate = hours * args.epochs / 100
        line = (f"{name:<18}{family:>8}{imgsz:>7}{batch:>7}{estimate:>8.1f}")
        if args.stop_after is not None:
            fits = cumulative + estimate <= args.stop_after
            line += f"{cumulative + estimate:>8.1f}  {'yes' if fits else 'no'}"
            if fits:
                cumulative += estimate
        print(line)

    if args.stop_after is not None:
        print(f"\n~{cumulative:.1f} h of the {args.stop_after:.1f} h budget would run "
              f"before the sweep stops on its own.")

    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()
    done, failed = [], []

    for index, (family, variant, imgsz, batch, hours, name) in enumerate(pending, start=1):
        if STOP_FILE.is_file():
            print(f"\n{STOP_FILE.name} found - stopping cleanly before {name}.")
            print(f"Delete it and re-run this script to carry on.")
            break

        spent = (time.perf_counter() - started_all) / 3600
        estimate = hours * args.epochs / 100
        if args.stop_after is not None and spent + estimate > args.stop_after:
            print(f"\n[{index}/{len(pending)}] {name} needs ~{estimate:.1f} h and "
                  f"{spent:.1f} h of the {args.stop_after:.1f} h budget is gone.")
            print("Stopping here rather than starting a run that cannot finish.")
            break

        print(f"\n{'=' * 70}")
        print(f"[{index}/{len(pending)}] {name}   (~{estimate:.1f} h, "
              f"{spent:.1f} h elapsed)")
        print("=" * 70, flush=True)

        weights = weights_path(family, variant, imgsz)
        finished = (run_dir(family, variant, imgsz) / ".trained").is_file()
        if finished:
            print("    [train] already finished (.trained marker), skipping")
        else:
            # best.pt existing is NOT proof of completion - both trainers write it
            # mid-run. Only the .trained marker is, and last.pt means resume.
            resuming = last_path(family, variant, imgsz).is_file()
            if resuming:
                print("    [train] partial run found, resuming from last.pt")
            if family == "yolo":
                command = [PYTHON, "scripts/train_yolo.py", "--model", variant,
                           "--imgsz", str(imgsz), "--batch", str(batch),
                           "--epochs", str(args.epochs)]
            else:
                command = [PYTHON, "scripts/train_ssd.py", "--backbone", variant,
                           "--imgsz", str(imgsz), "--batch", str(batch),
                           "--epochs", str(args.epochs)]
            if resuming:
                command.append("--resume")
            ok, seconds = run("train", command)
            log_outcome(name, "train", "ok" if ok else "failed", seconds, str(weights))
            if not ok:
                failed.append(name)
                continue
            if not weights.is_file():
                print(f"    [train] finished but {weights} is missing")
                log_outcome(name, "train", "missing_weights", 0, str(weights))
                failed.append(name)
                continue

        preds = PRED_DIR / f"{name}.json"
        ok, seconds = run("predict", [
            PYTHON, "scripts/predict_to_coco.py", "--family", family,
            "--weights", str(weights), "--imgsz", str(imgsz), "--out", str(preds),
        ])
        log_outcome(name, "predict", "ok" if ok else "failed", seconds, str(preds))
        if not ok:
            failed.append(name)
            continue

        ok, seconds = run("evaluate", [
            PYTHON, "scripts/evaluate_detection.py",
            "--predictions", str(preds), "--name", name,
        ])
        log_outcome(name, "evaluate", "ok" if ok else "failed", seconds, "")
        (done if ok else failed).append(name)

    total = (time.perf_counter() - started_all) / 3600
    print(f"\n{'=' * 70}")
    print(f"sweep finished in {total:.1f} h - {len(done)} scored, {len(failed)} failed")
    if failed:
        print(f"failed: {', '.join(failed)}")
        print("Re-run this script to retry them; scored configs are skipped.")
    print(f"results: {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
