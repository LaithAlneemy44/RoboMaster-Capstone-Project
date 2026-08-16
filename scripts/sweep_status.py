"""Show what the sweep is doing right now, from disk rather than from its stdout.

The sweep is a multi-day job that may be running in another terminal, or have been
started and interrupted several times. This reads its state off the filesystem, so it
works no matter who launched it or whether that terminal is still open.

Usage:
    python scripts/sweep_status.py
    python scripts/sweep_status.py --watch     # refresh every 30s
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from run_sweep import (  # noqa: E402
    CONFIGS, RESULTS, config_name, load_probes, probe_key, run_dir,
)


def gpu_line() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            util, mem, temp = (p.strip() for p in out.stdout.strip().split(","))
            busy = "TRAINING" if int(util) > 20 else "idle"
            return f"GPU {util}% util, {int(mem) / 1024:.1f} GiB, {temp}C   -> {busy}"
    except Exception:  # noqa: BLE001 - nvidia-smi absent or slow is not fatal here
        pass
    return "GPU status unavailable"


def scored() -> dict[str, str]:
    if not RESULTS.is_file():
        return {}
    with RESULTS.open(newline="", encoding="utf-8") as fh:
        return {r["name"]: r["mAP_50_95"] for r in csv.DictReader(fh)}


def report() -> None:
    done = scored()
    probes = load_probes()
    now = time.time()

    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}   {gpu_line()}")
    print()
    print(f"{'config':<18}{'state':<14}{'epoch':>7}{'mAP':>9}{'est h':>8}")
    print("-" * 56)

    remaining = 0.0
    for family, variant, imgsz in CONFIGS:
        name = config_name(family, variant, imgsz)
        probe = probes.get((probe_key(family, variant), imgsz))
        hours = probe[1] if probe else 0.0
        base = run_dir(family, variant, imgsz)
        last = (base / "weights" / "last.pt") if family == "yolo" else (base / "last.pt")

        epoch, state = "", "pending"
        if name in done:
            state = "SCORED"
        elif (base / ".trained").is_file():
            state = "trained"
        elif last.is_file():
            # Touched in the last two minutes means this is the live one.
            fresh = now - last.stat().st_mtime < 120
            state = "RUNNING" if fresh else "partial"
            try:
                import torch
                epoch = str(torch.load(last, map_location="cpu",
                                       weights_only=False).get("epoch", ""))
            except Exception:  # noqa: BLE001 - a checkpoint mid-write is expected
                epoch = "?"

        if state not in ("SCORED",):
            remaining += hours
        print(f"{name:<18}{state:<14}{epoch:>7}"
              f"{done.get(name, ''):>9}{hours:>8.1f}")

    print("-" * 56)
    print(f"{len(done)}/{len(CONFIGS)} scored, ~{remaining:.1f} GPU-hours left")
    if not done:
        print("\nNo scored results yet - the first config takes about 45 minutes.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true", help="Refresh every 30s.")
    args = parser.parse_args()

    if not args.watch:
        report()
        return
    try:
        while True:
            print("\033[2J\033[H", end="")  # clear
            report()
            print("\nCtrl-C to stop watching (does NOT stop the sweep).")
            time.sleep(30)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
