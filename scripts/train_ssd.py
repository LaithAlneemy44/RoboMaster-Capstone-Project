"""Fine-tune MobileNet-SSD (SSDLite) on DJI ROCO Central.

TRAINING RUNS ON THE GPU, for the same reason train_yolo.py refuses to fall back:
a silent CPU fallback wastes hours, and CPU numbers from training are not the CPU
numbers this project reports.

Two axes, matching the project scope:

  backbone   large = mobilenet_v3_large, small = mobilenet_v3_small
  imgsz      320 / 640 / 960, the same ladder train_yolo.py uses

Both backbones are ImageNet-pretrained. torchvision has NO pretrained weights below
width_mult 1.0 - `mobilenet_v3_large(width_mult=0.75)` builds but fails to load
weights with a size mismatch - so varying width_mult would train from scratch on 2258
images and lose for reasons unrelated to width. large-vs-small is a genuine capacity
axis where both ends start from pretrained weights.

torchvision's own `ssdlite320_mobilenet_v3_large` cannot be used directly: it hard-codes
320 and rejects `size=`, so the model is assembled here from the same parts.

Usage:
    .venv/Scripts/python.exe scripts/train_ssd.py --probe --backbone large --imgsz 960
    .venv/Scripts/python.exe scripts/train_ssd.py --backbone large --imgsz 640
    .venv/Scripts/python.exe scripts/train_ssd.py --backbone small --imgsz 320 --batch 32

Checkpoints land in runs/ssd/<name>/ (gitignored). Score best.pt through the shared
evaluator, never through a framework's own mAP:
    python scripts/predict_to_coco.py --family ssd --weights runs/ssd/<name>/best.pt ...
    python scripts/evaluate_detection.py --predictions ...
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

COCO_ROOT = ROOT / "Datasets" / "DJI ROCO Central.v1i.coco"
SPLITS = ROOT / "data" / "splits"
PROBE_RESULTS = ROOT / "results" / "probe.csv"

BACKBONES = ("large", "small")
LADDER = (320, 640, 960)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

def build_ssd(
    backbone_name: str, imgsz: int, num_classes: int, pretrained_backbone: bool = True
):
    """Assemble SSDLite at an arbitrary input size, from either MobileNetV3 backbone."""
    import torch
    from torch import nn
    from torchvision.models import (
        MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights,
        mobilenet_v3_large, mobilenet_v3_small,
    )
    from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
    from torchvision.models.detection.ssd import SSD
    from torchvision.models.detection.ssdlite import SSDLiteHead, _mobilenet_extractor

    if backbone_name not in BACKBONES:
        sys.exit(f"Unknown backbone {backbone_name!r}; expected one of {BACKBONES}")

    # eps/momentum match torchvision's ssdlite recipe.
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    if backbone_name == "large":
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        net = mobilenet_v3_large(weights=weights, norm_layer=norm_layer)
    else:
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        net = mobilenet_v3_small(weights=weights, norm_layer=norm_layer)

    backbone = _mobilenet_extractor(net, trainable_layers=6, norm_layer=norm_layer)
    anchor_gen = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)

    # Probe the feature-map channel counts. Must be done in EVAL mode with a batch of
    # more than one: at 320 the deepest pyramid level is 1x1, and a train-mode
    # BatchNorm over a single 1x1 sample raises "Expected more than 1 value per
    # channel". Training itself is safe because the loader uses drop_last.
    backbone.eval()
    with torch.no_grad():
        channels = [f.shape[1] for f in backbone(torch.zeros(2, 3, imgsz, imgsz)).values()]
    backbone.train()

    head = SSDLiteHead(channels, anchor_gen.num_anchors_per_location(), num_classes, norm_layer)
    return SSD(backbone, anchor_gen, (imgsz, imgsz), num_classes, head=head)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

class RocoCoco:
    """Merged-COCO reader returning torchvision-detection targets.

    Boxes stay in NATIVE 1920x1080 coordinates: SSD's GeneralizedRCNNTransform does
    the resize itself and maps predictions back, which is what lets predict_to_coco
    emit native-scale boxes with no manual rescaling.
    """

    def __init__(self, json_path: Path, train: bool):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        self.train = train
        self.images = data["images"]
        self.num_classes = max(c["id"] for c in data["categories"]) + 1

        by_image: dict[int, list[dict]] = {}
        for ann in data["annotations"]:
            by_image.setdefault(ann["image_id"], []).append(ann)
        self.by_image = by_image

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        import torch
        from PIL import Image
        from torchvision.transforms import functional as TF

        record = self.images[index]
        image = Image.open(COCO_ROOT / record["file_name"]).convert("RGB")

        boxes, labels = [], []
        for ann in self.by_image.get(record["id"], ()):
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:  # degenerate boxes make the loss NaN
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        tensor = TF.to_tensor(image)
        if self.train and boxes and torch.rand(1).item() < 0.5:
            tensor = tensor.flip(-1)
            width = tensor.shape[-1]
            boxes = [[width - b[2], b[1], width - b[0], b[3]] for b in boxes]
        if self.train:
            # Mild photometric jitter. Deliberately not SSD's full zoom-out/crop
            # recipe: that changes effective object scale, which is the variable
            # under study here, and would confound the resolution ladder.
            tensor = tensor.mul(0.8 + 0.4 * torch.rand(1).item()).clamp(0, 1)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }
        return tensor, target


def collate(batch):
    return tuple(zip(*batch))


# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backbone", choices=BACKBONES, default="large")
    parser.add_argument("--imgsz", type=int, choices=LADDER, default=640)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16, help="Lower first on CUDA OOM.")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=4e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--probe", action="store_true",
                        help="One epoch on full data; report wall time and peak VRAM.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_json = SPLITS / "coco_train.json"
    val_json = SPLITS / "coco_val.json"
    for path in (train_json, val_json):
        if not path.is_file():
            sys.exit(f"Missing {path}\nRun: python scripts/make_splits.py")

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        sys.exit(f"{exc}\nInstall dependencies - see README.md (Setup).")

    if not torch.cuda.is_available():
        sys.exit(
            "CUDA is not available - refusing to train on CPU.\n"
            "Diagnose with: .venv/Scripts/python.exe scripts/check_gpu.py"
        )

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    epochs = 1 if args.probe else args.epochs
    name = args.name or (
        f"probe_{args.backbone}_{args.imgsz}_b{args.batch}"
        if args.probe else f"ssd_{args.backbone}_{args.imgsz}"
    )
    out_dir = ROOT / "runs" / ("probe" if args.probe else "ssd") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = RocoCoco(train_json, train=True)
    val_set = RocoCoco(val_json, train=False)
    num_classes = train_set.num_classes

    print(f"backbone : mobilenet_v3_{args.backbone}")
    print(f"imgsz    : {args.imgsz}   batch: {args.batch}   epochs: {epochs}")
    print(f"device   : {torch.cuda.get_device_name(0)}")
    print(f"train    : {len(train_set)} images    val: {len(val_set)} images")
    print(f"classes  : {num_classes} (background + 5)")
    free_gib = torch.cuda.mem_get_info(0)[0] / 1024**3
    print(f"VRAM free: {free_gib:.2f} GiB")
    if args.probe:
        print("MODE     : probe - sizing --batch, not a result\n")

    model = build_ssd(args.backbone, args.imgsz, num_classes).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"params   : {params / 1e6:.2f}M\n")

    loader = DataLoader(
        train_set, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        collate_fn=collate, pin_memory=True,
        # drop_last matters: at 320 the deepest feature map is 1x1, and a trailing
        # batch of one would fail BatchNorm.
        drop_last=True, persistent_workers=args.workers > 0,
    )

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=not args.no_amp)
    steps_per_epoch = max(1, len(loader))
    warmup_steps = min(500, steps_per_epoch)
    total_steps = epochs * steps_per_epoch

    def lr_at(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    step = 0

    try:
        for epoch in range(epochs):
            model.train()
            running, seen = 0.0, 0
            for images, targets in loader:
                images = [i.to(device, non_blocking=True) for i in images]
                targets = [
                    {k: v.to(device, non_blocking=True) for k, v in t.items()}
                    for t in targets
                ]
                with torch.amp.autocast("cuda", enabled=not args.no_amp):
                    losses = model(images, targets)
                    loss = sum(losses.values())

                if not torch.isfinite(loss):
                    sys.exit(
                        f"\nLoss became {loss.item()} at step {step}. Lower --lr "
                        f"(currently {args.lr}) and rerun."
                    )

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                step += 1

                running += float(loss) * len(images)
                seen += len(images)
                if step % 20 == 0:
                    print(
                        f"\r  epoch {epoch + 1}/{epochs}  step {step}/{total_steps}  "
                        f"loss {running / max(1, seen):.4f}  "
                        f"lr {scheduler.get_last_lr()[0]:.5f}",
                        end="",
                    )
            print()
    except torch.cuda.OutOfMemoryError:
        peak = torch.cuda.max_memory_reserved(0) / 1024**3
        print(f"\nCUDA OUT OF MEMORY at batch={args.batch}, imgsz={args.imgsz} "
              f"(peaked at {peak:.2f} GiB of {free_gib:.2f} GiB free)")
        if args.probe:
            record_probe({
                "model": f"ssd_{args.backbone}", "imgsz": args.imgsz,
                "batch": args.batch, "status": "oom", "epoch_seconds": "",
                "peak_vram_gib": round(peak, 2), "est_100ep_hours": "",
            })
        sys.exit("Lower --batch and probe again. Do not lower --imgsz - it is the "
                 "variable under study.")

    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_reserved(0) / 1024**3

    if args.probe:
        print(f"\n1 epoch: {elapsed / 60:.1f} min")
        print(f"peak VRAM reserved: {peak:.2f} GiB of {free_gib:.2f} GiB free")
        print(f"estimated 100 epochs: {elapsed * 100 / 3600:.1f} h")
        record_probe({
            "model": f"ssd_{args.backbone}", "imgsz": args.imgsz, "batch": args.batch,
            "status": "ok", "epoch_seconds": round(elapsed, 1),
            "peak_vram_gib": round(peak, 2),
            "est_100ep_hours": round(elapsed * 100 / 3600, 2),
        })
        return

    checkpoint = out_dir / "best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "backbone": args.backbone,
            "imgsz": args.imgsz,
            "num_classes": num_classes,
            "epochs": epochs,
            "seed": args.seed,
        },
        checkpoint,
    )
    print(f"\ncheckpoint: {checkpoint}")
    print(f"peak VRAM : {peak:.2f} GiB    wall time: {elapsed / 3600:.2f} h")
    print("\nScore it through the shared evaluator:")
    print(f"  python scripts/predict_to_coco.py --family ssd --weights {checkpoint} "
          f"--imgsz {args.imgsz} --out preds_{name}.json")
    print(f"  python scripts/evaluate_detection.py --predictions preds_{name}.json "
          f"--name {name}")


def record_probe(row: dict) -> None:
    PROBE_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    exists = PROBE_RESULTS.is_file()
    with PROBE_RESULTS.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"appended to {PROBE_RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    # Required on Windows: DataLoader workers spawn subprocesses that re-import this
    # module. Without the guard they re-run training recursively.
    main()
