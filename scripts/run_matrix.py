"""Benchmark every detector x tracker pairing on the target CPU.

Proposal 5.2 defines the experiment as "every combination of an object detection and an
object tracking model". The earlier tracking benchmark fixed the detector at yolo_960 and
varied the tracker, which answers "which tracker is cheapest" but not "which perception
model should go on the robot" - and the second question is the research question.

This is scripts/run_benchmark.py's shape applied to that grid: one subprocess per cell,
strictly sequential, resume from the results CSV, STOP file for a clean exit. The two
rules it enforces are the same and for the same reasons - core capping has to happen
before torch initialises, and two benchmarks at once measure each other's contention.

DETECTOR REPRESENTATIVES
    One per family rather than every config, because the grid multiplies out fast and
    the tracker axis is what is new here. Each is the best case for its family, so the
    comparison is between the families at their strongest.

        yolo_960                best accuracy overall
        fast_960                best accuracy per millisecond
        ssd_small_960_anchor    best SSD, on the corrected anchors
        classical_strict        the classical detector

CORE LEVELS
    1 and 6. One core is the constrained competition hardware CLAUDE.md is about; six is
    this machine's full physical width, and the pair brackets the deployment range
    without spending the grid on intermediate points already covered by the detection
    sweep.

GOTURN RUNS FEWER FRAMES
    It costs ~4.2 s/frame at six cores and worse at one, so at the grid's frame count it
    alone would outlast every other cell combined. Per-frame latency is stable, so it
    gets a shorter run; `frames` is recorded per row, so the difference is visible in the
    results rather than hidden.

Usage:
    python scripts/run_matrix.py                 # the whole grid
    python scripts/run_matrix.py --cores 6       # one core level
    python scripts/run_matrix.py --dry-run       # list the cells, run nothing
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
RESULTS = ROOT / "results" / "tracking_performance.csv"
STOP_FILE = ROOT / "STOP"

# (row label, family, weights-or-config, imgsz)
DETECTORS = (
    ("yolo_960", "yolo", ROOT / "runs" / "detect" / "yolo_960" / "weights" / "best.pt", 960),
    ("fast_960", "yolo", ROOT / "runs" / "detect" / "fast_960" / "weights" / "best.pt", 960),
    ("ssd_small_960_anchor", "ssd", ROOT / "runs" / "ssd" / "ssd_small_960_anchor" / "best.pt", 960),
    ("strict", "classical", "strict", 960),
)
TRACKERS = ("classical", "sort", "vit", "goturn")

# SORT is a complete published system, not a tracker to be mixed and matched: CLAUDE.md
# says it is "tested standalone, not combined with other models". Its detector therefore
# pairs ONLY with its own tracker and is deliberately kept out of the cross product -
# frcnn + vit is not a thing anyone proposed and would not mean anything.
STANDALONE = (
    ("frcnn_resnet50_640", "frcnn",
     ROOT / "runs" / "frcnn" / "frcnn_resnet50_640" / "best.pt", 640, "sort"),
)

# arc02 is a moderately busy clip; arc06 carries the most ground-truth objects of the
# seven, so the single-object trackers' per-robot cost shows up as a difference between
# the two rather than having to be argued for.
SEQUENCES = ("arc02", "arc06")
CORE_LEVELS = (1, 6)

FRAMES = 30
GOTURN_FRAMES = 10


def already_done() -> set[tuple[str, str, str, int]]:
    if not RESULTS.is_file():
        return set()
    with RESULTS.open(newline="", encoding="utf-8") as fh:
        return {
            (row["sequence"], row["tracker"], row["detector"], int(row["cores"]))
            for row in csv.DictReader(fh)
        }


def run_cell(seq: str, tracker: str, label: str, family: str, weights, imgsz: int,
             cores: int, frames: int) -> tuple[bool, float]:
    command = [
        PYTHON, str(ROOT / "scripts" / "benchmark_tracking.py"),
        "--seq", str(ROOT / "data" / "tracking" / seq),
        "--tracker", tracker,
        "--detector-family", family,
        "--imgsz", str(imgsz),
        "--cores", str(cores),
        "--frames", str(frames),
    ]
    command += (["--detector-config", str(weights)] if family == "classical"
                else ["--detector", str(weights)])

    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - started

    # Ctrl-C reaches the whole process group, so the child exits 130. Treating that as an
    # ordinary failure would march on to the next cell - the opposite of the intent.
    if result.returncode == 130:
        raise KeyboardInterrupt
    if result.returncode != 0:
        print(f"    FAILED after {elapsed:.0f}s", flush=True)
        for line in (result.stdout + result.stderr).strip().splitlines()[-6:]:
            print(f"      | {line}", flush=True)
        return False, elapsed

    for line in result.stdout.splitlines():
        if line.startswith(("detect ", "total ", "tracks ", "cpu ")) or "WARNING" in line:
            print(f"      {line}", flush=True)
    return True, elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cores", type=int, nargs="+", default=list(CORE_LEVELS))
    parser.add_argument("--seq", nargs="+", default=list(SEQUENCES))
    parser.add_argument("--trackers", nargs="+", default=list(TRACKERS))
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--goturn-frames", type=int, default=GOTURN_FRAMES)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if STOP_FILE.is_file():
        sys.exit(f"{STOP_FILE.name} exists - delete it before starting a run.")

    done = already_done()
    cells = []
    for seq in args.seq:
        for label, family, weights, imgsz in DETECTORS:
            if family != "classical" and not Path(weights).is_file():
                print(f"skip {label}: no weights at {weights}")
                continue
            for tracker in args.trackers:
                for cores in args.cores:
                    if (seq, tracker, label, cores) in done:
                        continue
                    frames = args.goturn_frames if tracker == "goturn" else args.frames
                    cells.append(
                        (seq, tracker, label, family, weights, imgsz, cores, frames)
                    )

    for label, family, weights, imgsz, tracker in STANDALONE:
        if not Path(weights).is_file():
            print(f"skip {label}: no weights at {weights} "
                  f"(train it: python scripts/train_frcnn.py)")
            continue
        if args.trackers != list(TRACKERS) and tracker not in args.trackers:
            continue
        for seq in args.seq:
            for cores in args.cores:
                if (seq, tracker, label, cores) in done:
                    continue
                cells.append((seq, tracker, label, family, weights, imgsz, cores,
                              args.frames))

    total = len(cells)
    if not total:
        print(f"Nothing to do - all cells present in {RESULTS.name}.")
        return

    print(f"grid   : {len(args.seq)} clips x {len(DETECTORS)} detectors x "
          f"{len(args.trackers)} trackers x {len(args.cores)} core levels")
    print(f"cells  : {total} to run, {len(done)} already done")
    print(f"frames : {args.frames} ({args.goturn_frames} for goturn)")
    print("Benchmarks run one at a time; leave the machine otherwise idle.\n")
    if args.dry_run:
        for cell in cells:
            print(f"  {cell[0]:6} {cell[2]:22} {cell[1]:10} {cell[6]} core(s)")
        return

    started_all = time.perf_counter()
    ok = failed = 0
    for i, cell in enumerate(cells, 1):
        if STOP_FILE.is_file():
            print(f"\n{STOP_FILE.name} found - stopping cleanly.")
            break
        seq, tracker, label, family, weights, imgsz, cores, frames = cell
        eta = ""
        if ok:
            per = (time.perf_counter() - started_all) / ok
            eta = f"   eta {per * (total - i + 1) / 60:.0f} min"
        print(f"[{i}/{total}] {seq}  {label} + {tracker} @ {cores} core(s){eta}",
              flush=True)
        success, _ = run_cell(seq, tracker, label, family, weights, imgsz, cores, frames)
        ok, failed = ok + success, failed + (not success)

    mins = (time.perf_counter() - started_all) / 60
    print(f"\n{ok} ok, {failed} failed in {mins:.1f} min -> {RESULTS}")


if __name__ == "__main__":
    main()
