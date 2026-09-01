"""Drive scripts/benchmark_cpu.py across every (config, core-count) combination.

Mirrors scripts/run_sweep.py, which already solved this shape for training: a grid, one
subprocess per cell, resume from the results CSV, and a STOP file for clean exits.

TWO RULES THIS ENFORCES

  1. One subprocess per cell. Core capping has to be applied before torch initialises,
     so it cannot be changed inside a running process. A crash is also contained to a
     single cell rather than losing the grid.

  2. Strictly sequential. Two benchmarks running at once measure each other's
     contention, not the models. This is why there is no parallelism here and why the
     child warns when the machine is otherwise busy.

STOPPING AND RESUMING
  Ctrl-C            stops now; the cell in flight is lost, everything finished is kept.
  STOP file         `New-Item STOP` finishes the current cell, then exits cleanly.
  Re-run            skips every (config, cores) pair already in results/performance.csv.

Usage:
    python scripts/run_benchmark.py                 # the whole grid
    python scripts/run_benchmark.py --frames 100    # quicker, coarser
    python scripts/run_benchmark.py --cores 1 4     # only these core levels
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
RESULTS = ROOT / "results" / "performance.csv"
BENCH_LOG = ROOT / "results" / "benchmark_log.csv"
STOP_FILE = ROOT / "STOP"

sys.path.insert(0, str(ROOT / "scripts"))
from run_sweep import CONFIGS, config_name, weights_path  # noqa: E402

# The classical detector has no trained weights - a config IS the model - so it is
# listed separately rather than squeezed into run_sweep's (family, variant, imgsz)
# tuple. Only the configs actually scored in detection.csv are benchmarked.
CLASSICAL_CONFIGS = ("strict", "balanced", "tight", "loose")

# Levels are (physical cores, use hyperthread siblings). Counted in PHYSICAL cores
# because siblings share execution resources - see core_ids() in benchmark_cpu.py, where
# pinning to logical 0 and 1 came out slower than pinning to logical 0 alone. The small
# levels stand in for constrained competition hardware; the last two are this machine's
# full compute and its true ceiling with SMT on.
PHYSICAL = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)
CORE_LEVELS = [(c, False) for c in (1, 2, 4) if c < PHYSICAL]
CORE_LEVELS += [(PHYSICAL, False), (PHYSICAL, True)]


def already_done() -> set[tuple[str, int, bool]]:
    if not RESULTS.is_file():
        return set()
    with RESULTS.open(newline="", encoding="utf-8") as fh:
        return {
            (row["name"], int(row["cores"]), bool(int(row.get("smt", 0))))
            for row in csv.DictReader(fh)
        }


def log_outcome(name: str, cores: int, smt: bool, status: str, seconds: float,
                detail: str) -> None:
    BENCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    exists = BENCH_LOG.is_file()
    with BENCH_LOG.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if not exists:
            writer.writerow(["name", "cores", "smt", "status", "seconds", "detail"])
        writer.writerow([name, cores, int(smt), status, round(seconds, 1), detail])


def run_cell(name: str, family: str, imgsz: int, weights: Path, cores: int, smt: bool,
             frames: int) -> tuple[bool, float]:
    command = [
        PYTHON, str(ROOT / "scripts" / "benchmark_cpu.py"),
        "--family", family, "--name", name,
        "--imgsz", str(imgsz), "--cores", str(cores),
    ]
    # For classical, `weights` carries the config name instead of a file path.
    command += (["--config", str(weights)] if family == "classical"
                else ["--weights", str(weights)])
    if smt:
        command.append("--smt")
    if frames:
        command += ["--frames", str(frames)]

    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.perf_counter() - started

    # Ctrl-C reaches the whole process group, so the child exits 130. Treating that as
    # an ordinary failure would march on to the next cell - the opposite of the intent.
    if result.returncode == 130:
        raise KeyboardInterrupt
    if result.returncode != 0:
        tail = (result.stdout + result.stderr).strip().splitlines()[-6:]
        print(f"    FAILED after {elapsed:.0f}s", flush=True)
        for line in tail:
            print(f"      | {line}", flush=True)
        return False, elapsed

    for line in result.stdout.splitlines():
        if line.startswith(("latency", "fps", "cpu ", "rss")):
            print(f"      {line}", flush=True)
        if "WARNING" in line:
            print(f"      {line}", flush=True)
    return True, elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--frames", type=int, default=0, help="0 = all val images.")
    parser.add_argument("--cores", type=int, nargs="+", default=None,
                        help="Physical-core levels to run. Default: "
                             f"{[c for c, s in CORE_LEVELS if not s]} plus an SMT "
                             "reference at full width.")
    parser.add_argument("--smt", action="store_true",
                        help="With --cores, also run an SMT reference at the widest.")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Limit to these config names.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if STOP_FILE.is_file():
        sys.exit(f"{STOP_FILE.name} exists - delete it before starting a run.")

    levels = CORE_LEVELS
    if args.cores:
        levels = [(c, False) for c in args.cores]
        if args.smt:
            levels.append((max(args.cores), True))

    done = already_done()
    cells = []
    for family, variant, imgsz in CONFIGS:
        name = config_name(family, variant, imgsz)
        if args.only and name not in args.only:
            continue
        weights = weights_path(family, variant, imgsz)
        if not weights.is_file():
            print(f"skip {name}: no weights at {weights}")
            continue
        for cores, smt in levels:
            if (name, cores, smt) not in done:
                cells.append((name, family, imgsz, weights, cores, smt))

    for cfg in CLASSICAL_CONFIGS:
        name = f"classical_{cfg}"
        if args.only and name not in args.only:
            continue
        for cores, smt in levels:
            if (name, cores, smt) not in done:
                cells.append((name, "classical", 960, Path(cfg), cores, smt))

    total = len(cells)
    if not total:
        print(f"Nothing to do - all cells present in {RESULTS.name}.")
        return

    pretty = ", ".join(f"{c}{'+SMT' if s else ''}" for c, s in levels)
    print(f"machine: {PHYSICAL} physical cores   levels: {pretty}")
    print(f"cells  : {total} to run, {len(done)} already done")
    print(f"frames : {args.frames or 'all val images'}")
    print("Benchmarks run one at a time; leave the machine otherwise idle.\n")

    started_all = time.perf_counter()
    ok = failed = 0
    for i, (name, family, imgsz, weights, cores, smt) in enumerate(cells, 1):
        if STOP_FILE.is_file():
            print(f"\n{STOP_FILE.name} found - stopping cleanly.")
            break

        eta = ""
        if ok:
            per = (time.perf_counter() - started_all) / ok
            eta = f"   eta {per * (total - i + 1) / 60:.0f} min"
        label = f"{cores} core(s){'+SMT' if smt else ''}"
        print(f"[{i}/{total}] {name} @ {label}{eta}", flush=True)

        success, elapsed = run_cell(
            name, family, imgsz, weights, cores, smt, args.frames
        )
        log_outcome(name, cores, smt, "ok" if success else "failed", elapsed,
                    f"frames={args.frames or 'all'}")
        ok, failed = ok + success, failed + (not success)

    mins = (time.perf_counter() - started_all) / 60
    print(f"\n{ok} ok, {failed} failed in {mins:.1f} min -> {RESULTS}")
    if ok:
        print("Build the combined table:  python scripts/report_table.py")


if __name__ == "__main__":
    main()
