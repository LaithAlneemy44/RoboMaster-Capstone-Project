"""Fine-tune YOLO on DJI ROCO Central, starting from COCO-pretrained weights.

TRAINING RUNS ON THE GPU. This script deliberately refuses to fall back to CPU:
a silent CPU fallback here would waste hours, and CPU numbers from a training run
are not the CPU numbers this project reports anyway - those come from the
benchmark harness measuring *inference* on the target machine.

Model presets map to the two YOLO entries in the project scope:
    fast : yolo11n - the smaller/faster variant ("Fast YOLO")
    yolo : yolo11s - the baseline ("YOLO")

Usage:
    .venv/Scripts/python.exe scripts/check_gpu.py          # do this first
    .venv/Scripts/python.exe scripts/train_yolo.py --smoke # 2 epochs, 5% of data
    .venv/Scripts/python.exe scripts/train_yolo.py --model fast
    .venv/Scripts/python.exe scripts/train_yolo.py --model yolo --batch 8

Results land in runs/detect/<name>/ (gitignored). weights/best.pt is the one the
benchmark harness should load.

On CUDA out-of-memory: lower --batch first (16 -> 12 -> 8). Only reduce --imgsz
after that, and if you do, hold it fixed across every model or the comparison
stops being like-for-like.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "roco_central.yaml"

PRESETS = {"fast": "yolo11n.pt", "yolo": "yolo11s.pt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(PRESETS), default="fast",
                        help="Which YOLO preset to fine-tune (default: fast).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16,
                        help="Lower this first on CUDA OOM.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Keep identical across all models being compared.")
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
    return parser.parse_args()


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

    epochs = 2 if args.smoke else args.epochs
    fraction = 0.05 if args.smoke else 1.0
    name = args.name or (f"{args.model}_smoke" if args.smoke else f"{args.model}_{args.imgsz}")

    print(f"model    : {PRESETS[args.model]}")
    print(f"data     : {DATA_YAML.relative_to(ROOT)}")
    print(f"device   : {torch.cuda.get_device_name(0)}")
    print(f"epochs   : {epochs}   batch: {args.batch}   imgsz: {args.imgsz}")
    if args.smoke:
        print("MODE     : smoke test (5% of train, 2 epochs) - not a real result\n")

    model = YOLO(PRESETS[args.model])
    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        fraction=fraction,
        device=0,                 # never CPU - see module docstring
        amp=not args.no_amp,
        seed=args.seed,
        deterministic=True,       # reproducibility matters more than the last few %
        project=str(ROOT / "runs" / "detect"),
        name=name,
        exist_ok=False,           # don't silently overwrite a previous run's results
    )

    out = ROOT / "runs" / "detect" / name / "weights" / "best.pt"
    print(f"\nbest weights: {out}")

    if args.smoke:
        # Ultralytics warms up for 3 epochs by default, so a 2-epoch run never
        # leaves warmup, and the freshly-initialised 5-class head learns nothing.
        # All-zero P/R/mAP here is the expected result, not a failure. What the
        # smoke test actually proves is that images and labels resolve - check the
        # val instance count against the totals make_splits.py printed (11466).
        print("\nSmoke test: all-zero P/R/mAP is EXPECTED (2 epochs < 3 warmup epochs).")
        print("What matters is the val instance count matching make_splits.py.")
    else:
        print("Benchmark this on the target CPU - do not report GPU inference numbers.")


if __name__ == "__main__":
    # Required on Windows: Ultralytics dataloader workers spawn subprocesses, which
    # re-import this module. Without the guard they re-run training recursively.
    main()
