"""Fine-tune YOLO on DJI ROCO Central, starting from COCO-pretrained weights.

TRAINING RUNS ON THE GPU. This script deliberately refuses to fall back to CPU:
a silent CPU fallback here would waste hours, and CPU numbers from a training run
are not the CPU numbers this project reports anyway - those come from the
benchmark harness measuring *inference* on the target machine.

Model presets map to the two YOLO entries in the project scope:
    fast : yolo11n - the smaller/faster variant ("Fast YOLO")
    yolo : yolo11s - the baseline ("YOLO")

Resolution is an EXPERIMENTAL AXIS in this project, not a constant. Armor is 66% of
all instances at a median 22 px in a 1920x1080 frame, so it shrinks to ~3.6 px at 320
and ~11 px at 960 - input size is expected to dominate accuracy, and it trades
directly against the CPU latency the project exists to measure. Both YOLO models and
both MobileNet-SSD backbones therefore run the same 320/640/960 ladder.

Usage:
    .venv/Scripts/python.exe scripts/check_gpu.py          # do this first
    .venv/Scripts/python.exe scripts/train_yolo.py --smoke # 2 epochs, 5% of data
    .venv/Scripts/python.exe scripts/train_yolo.py --probe --model yolo --imgsz 960
    .venv/Scripts/python.exe scripts/train_yolo.py --model fast --imgsz 640

Results land in runs/detect/<name>/ (gitignored). weights/best.pt is the one the
benchmark harness should load.

Run --probe before committing to a real run. It trains a single epoch and reports
wall time and peak VRAM, so batch size per rung is measured rather than guessed -
this card has ~5.08 GiB free because the desktop drives the same GPU, which leaves
very little margin at 960. Probe results append to results/probe.csv.

On CUDA out-of-memory: lower --batch first (16 -> 12 -> 8). Do NOT reduce --imgsz to
escape OOM - it is the variable under study, so changing it silently would put two
models on different rungs of the ladder.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "roco_central.yaml"
PROBE_RESULTS = ROOT / "results" / "probe.csv"

PRESETS = {"fast": "yolo11n.pt", "yolo": "yolo11s.pt"}

# The shared resolution ladder. Held identical across both YOLO presets and both
# MobileNet-SSD backbones so each rung is a like-for-like comparison.
LADDER = (320, 640, 960)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(PRESETS), default="fast",
                        help="Which YOLO preset to fine-tune (default: fast).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16,
                        help="Lower this first on CUDA OOM.")
    parser.add_argument("--imgsz", type=int, default=640, choices=LADDER,
                        help="Rung of the shared resolution ladder to train at.")
    parser.add_argument("--patience", type=int, default=30,
                        help="Stop if val fitness has not improved in this many epochs.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Dataloader workers. Windows spawns processes; 4 is safe.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None, help="Run name under runs/detect/.")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision. Pascal has no tensor cores, so "
                             "AMP mostly saves VRAM here rather than adding speed.")
    parser.add_argument("--smoke", action="store_true",
                        help="2 epochs on 5%% of the data - proves the pipeline runs "
                             "end to end before committing to a real run.")
    parser.add_argument("--probe", action="store_true",
                        help="Train ONE epoch on the full data and report wall time "
                             "and peak VRAM, to size --batch for this rung.")
    return parser.parse_args()


def record_probe(row: dict) -> None:
    PROBE_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    exists = PROBE_RESULTS.is_file()
    with PROBE_RESULTS.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"appended to {PROBE_RESULTS.relative_to(ROOT)}")


def main() -> None:
    args = parse_args()

    if not DATA_YAML.is_file():
        sys.exit(f"Missing {DATA_YAML}\nRun: python scripts/make_splits.py")

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        sys.exit(f"{exc}\nInstall dependencies - see README.md (Setup).")

    if not torch.cuda.is_available():
        sys.exit(
            "CUDA is not available - refusing to train on CPU.\n"
            "Diagnose with: .venv/Scripts/python.exe scripts/check_gpu.py"
        )

    if args.smoke and args.probe:
        sys.exit("--smoke and --probe are different checks; run them separately.")

    epochs = 2 if args.smoke else (1 if args.probe else args.epochs)
    fraction = 0.05 if args.smoke else 1.0
    if args.probe:
        name = f"probe_{args.model}_{args.imgsz}_b{args.batch}"
    elif args.smoke:
        name = f"{args.model}_smoke"
    else:
        name = f"{args.model}_{args.imgsz}"
    name = args.name or name

    print(f"model    : {PRESETS[args.model]}")
    print(f"data     : {DATA_YAML.relative_to(ROOT)}")
    print(f"device   : {torch.cuda.get_device_name(0)}")
    print(f"epochs   : {epochs}   batch: {args.batch}   imgsz: {args.imgsz}")
    free_gib = torch.cuda.mem_get_info(0)[0] / 1024**3
    print(f"VRAM free: {free_gib:.2f} GiB")
    if args.smoke:
        print("MODE     : smoke test (5% of train, 2 epochs) - not a real result\n")
    if args.probe:
        print("MODE     : probe (1 epoch, full data) - sizing --batch, not a result\n")

    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()

    model = YOLO(PRESETS[args.model])
    try:
        model.train(
            data=str(DATA_YAML),
            epochs=epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            workers=args.workers,
            fraction=fraction,
            patience=args.patience,
            device=0,                 # never CPU - see module docstring
            amp=not args.no_amp,
            seed=args.seed,
            deterministic=True,       # reproducibility matters more than the last few %
            project=str(ROOT / "runs" / ("probe" if args.probe else "detect")),
            name=name,
            # A probe is meant to be re-run while tuning batch; a real run is not.
            exist_ok=args.probe,
        )
    except Exception as exc:  # noqa: BLE001 - need the raw CUDA message
        oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc)
        if not oom:
            raise
        peak = torch.cuda.max_memory_reserved(0) / 1024**3
        print(f"\nCUDA OUT OF MEMORY at batch={args.batch}, imgsz={args.imgsz} "
              f"(peaked at {peak:.2f} GiB of {free_gib:.2f} GiB free)")
        if args.probe:
            record_probe({
                "model": args.model, "imgsz": args.imgsz, "batch": args.batch,
                "status": "oom", "epoch_seconds": "", "peak_vram_gib": round(peak, 2),
                "est_100ep_hours": "",
            })
        sys.exit("Lower --batch and probe again. Do not lower --imgsz - it is the "
                 "variable under study.")

    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_reserved(0) / 1024**3

    if args.probe:
        print(f"\n1 epoch (incl. validation): {elapsed / 60:.1f} min")
        print(f"peak VRAM reserved        : {peak:.2f} GiB of {free_gib:.2f} GiB free")
        print(f"estimated 100 epochs      : {elapsed * 100 / 3600:.1f} h")
        record_probe({
            "model": args.model, "imgsz": args.imgsz, "batch": args.batch,
            "status": "ok", "epoch_seconds": round(elapsed, 1),
            "peak_vram_gib": round(peak, 2),
            "est_100ep_hours": round(elapsed * 100 / 3600, 2),
        })
        return

    out = ROOT / "runs" / "detect" / name / "weights" / "best.pt"
    print(f"\nbest weights: {out}")
    print(f"peak VRAM   : {peak:.2f} GiB    wall time: {elapsed / 3600:.2f} h")

    if args.smoke:
        # Ultralytics warms up for 3 epochs by default, so a 2-epoch run never
        # leaves warmup, and the freshly-initialised 5-class head learns nothing.
        # All-zero P/R/mAP here is the expected result, not a failure. What the
        # smoke test actually proves is that images and labels resolve - check the
        # val instance count against the totals make_splits.py printed (11466).
        print("\nSmoke test: all-zero P/R/mAP is EXPECTED (2 epochs < 3 warmup epochs).")
        print("What matters is the val instance count matching make_splits.py.")
    else:
        print("\nScore it through the shared evaluator - Ultralytics' own mAP is not")
        print("comparable with torchvision's:")
        print(f"  python scripts/predict_to_coco.py --family yolo --weights {out} "
              f"--imgsz {args.imgsz} --out preds_{name}.json")
        print(f"  python scripts/evaluate_detection.py --predictions preds_{name}.json "
              f"--name {name}")
        print("\nBenchmark latency on the target CPU - not on this GPU.")


if __name__ == "__main__":
    # Required on Windows: Ultralytics dataloader workers spawn subprocesses, which
    # re-import this module. Without the guard they re-run training recursively.
    main()
