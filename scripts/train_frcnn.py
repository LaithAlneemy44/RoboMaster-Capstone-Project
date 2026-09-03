"""Fine-tune Faster R-CNN on ROCO, for SORT as a complete perception model.

WHY THIS EXISTS SEPARATELY FROM THE TRACKER
    CLAUDE.md lists SORT as a perception model in its own right - "Faster R-CNN
    detection + Kalman tracking ... tested standalone, not combined with other models".
    What scripts/sort.py provides is only the Kalman half. Benchmarking that half behind
    a YOLO detector measures a hybrid this project invented, not the published pipeline,
    so SORT as specified needs its own detector and this trains it.

    That distinction matters for the write-up: `sort` rows elsewhere in the results are
    "our detector + SORT's tracker", while `frcnn_sort` is the reference pipeline.

WHY RESNET50-FPN
    SORT's paper pairs the tracker with a Faster R-CNN detector, and ResNet50-FPN is the
    standard modern instantiation of that. A lighter backbone would be faster on CPU and
    would flatter the baseline - the point of a baseline is to represent the published
    method, not to win.

REUSED WHOLESALE FROM train_ssd.py
    RocoCoco, collate and evaluate_map, so this model sees exactly the data the SSDs saw,
    pre-resized the same way, and its checkpoint is selected on the same val mAP under
    the same ignore policy. A second dataset implementation here would be a place for the
    comparison to silently stop being like-for-like.

CATEGORY 0 IS UNUSED, WHICH IS WHAT MAKES THIS SAFE
    torchvision reserves label 0 for background. The COCO export's category 0
    ("robomaster-car") is a supercategory carrying zero annotations - verified, not
    assumed - so passing category ids through as labels does not collide with it.

VRAM
    6 GB on the project's GTX 1060, and Faster R-CNN with FPN is far heavier than SSDLite.
    Batch 2 at 640px with AMP is the default for that reason. On CUDA OOM, lower --batch
    before anything else, exactly as CLAUDE.md advises.

Usage:
    python scripts/train_frcnn.py --imgsz 640 --epochs 20
    python scripts/train_frcnn.py --probe          # one batch, check VRAM, exit
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

SPLITS = ROOT / "data" / "splits"
TRAIN_JSON = SPLITS / "coco_train.json"
VAL_JSON = SPLITS / "coco_val.json"


def build_frcnn(backbone: str, imgsz: int, num_classes: int):
    """COCO-pretrained Faster R-CNN with the box head resized to ROCO's classes."""
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    if backbone == "resnet50":
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    else:
        weights = (torchvision.models.detection
                   .FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # RocoCoco already resized to a square imgsz on the CPU. Pinning the internal
    # transform to that size makes its rescale a no-op; leaving the defaults (800/1333)
    # would silently upsample every image and put predictions in a space that
    # to_native() does not undo, mis-scaling every box that is finally scored.
    model.transform.min_size = (imgsz,)
    model.transform.max_size = imgsz
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backbone", choices=("resnet50", "mobilenet"),
                        default="resnet50")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=2, help="Lower first on CUDA OOM.")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--probe", action="store_true",
                        help="One forward/backward, report VRAM, exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    from torch.utils.data import DataLoader

    from train_ssd import RocoCoco, collate, evaluate_map

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA. Training on CPU will take many hours.\n")

    name = args.name or f"frcnn_{args.backbone}_{args.imgsz}"
    out_dir = ROOT / "runs" / "frcnn" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = RocoCoco(TRAIN_JSON, train=True, imgsz=args.imgsz)
    val_set = RocoCoco(VAL_JSON, train=False, imgsz=args.imgsz)
    num_classes = train_set.num_classes

    model = build_frcnn(args.backbone, args.imgsz, num_classes).to(device)
    params = sum(p.numel() for p in model.parameters())

    print(f"run      : {name}")
    print(f"backbone : {args.backbone} (COCO-pretrained)   imgsz {args.imgsz}")
    print(f"data     : {len(train_set)} train / {len(val_set)} val   "
          f"classes {num_classes} (background + 5)")
    print(f"params   : {params / 1e6:.2f}M   device {device}")
    print(f"batch    : {args.batch}   amp {not args.no_amp}\n")

    loader = DataLoader(
        train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        collate_fn=collate, pin_memory=True, drop_last=True,
        persistent_workers=args.workers > 0,
    )

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp and device == "cuda")

    steps_per_epoch = max(1, len(loader))
    warmup_steps = min(500, steps_per_epoch)
    total_steps = args.epochs * steps_per_epoch

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    if args.probe:
        model.train()
        images, targets = next(iter(loader))
        images = [i.to(device) for i in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.amp.autocast("cuda", enabled=not args.no_amp and device == "cuda"):
            loss = sum(model(images, targets).values())
        scaler.scale(loss).backward()
        if device == "cuda":
            peak = torch.cuda.max_memory_allocated(0) / 2**20
            total = torch.cuda.get_device_properties(0).total_memory / 2**20
            print(f"probe OK: loss {float(loss.detach()):.4f}   peak VRAM {peak:.0f} MiB "
                  f"of {total:.0f} MiB")
        else:
            print(f"probe OK: loss {float(loss.detach()):.4f}")
        return

    best_map, best_epoch, step, start_epoch = -1.0, -1, 0, 0

    def save(path: Path, epoch: int, score: float, full: bool = False) -> None:
        payload = {
            "model": model.state_dict(), "backbone": args.backbone,
            "imgsz": args.imgsz, "num_classes": num_classes,
            "epoch": epoch, "val_map": score, "seed": args.seed,
            "arch": "fasterrcnn",
        }
        if full:
            payload |= {
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict(), "best_map": best_map,
                "best_epoch": best_epoch, "step": step,
            }
        # Temp file then replace: interrupting mid-write would otherwise leave a
        # truncated checkpoint, which is the one file a resume depends on.
        tmp = path.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    if args.resume and last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"]
        step = state.get("step", start_epoch * steps_per_epoch)
        best_map, best_epoch = state.get("best_map", -1.0), state.get("best_epoch", -1)
        print(f"Resumed from epoch {start_epoch} "
              f"(best val mAP {best_map:.4f} at epoch {best_epoch})\n")

    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running, seen = 0.0, 0
        for images, targets in loader:
            images = [i.to(device, non_blocking=True) for i in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()}
                       for t in targets]
            # An image with no boxes makes the RPN loss undefined; skip rather than NaN.
            if any(t["boxes"].numel() == 0 for t in targets):
                continue
            with torch.amp.autocast("cuda",
                                    enabled=not args.no_amp and device == "cuda"):
                loss = sum(model(images, targets).values())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            running += float(loss.detach())
            seen += 1
            if seen % 100 == 0:
                done = time.perf_counter() - started
                print(f"\r  epoch {epoch + 1}/{args.epochs}  "
                      f"{seen}/{steps_per_epoch}  loss {running / seen:.4f}  "
                      f"lr {scheduler.get_last_lr()[0]:.5f}  {done / 60:.0f} min",
                      end="", flush=True)
        print()

        save(last_path, epoch + 1, best_map, full=True)
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            score = evaluate_map(model, val_set, device, VAL_JSON, args.batch)
            print(f"  epoch {epoch + 1}: val mAP@[.5:.95] {score:.4f}")
            if score > best_map:
                best_map, best_epoch = score, epoch + 1
                save(best_path, epoch + 1, score)
                print(f"  new best -> {best_path}")

    mins = (time.perf_counter() - started) / 60
    print(f"\ndone in {mins:.1f} min. best val mAP {best_map:.4f} at epoch {best_epoch}")
    print(f"weights: {best_path}")


if __name__ == "__main__":
    main()
