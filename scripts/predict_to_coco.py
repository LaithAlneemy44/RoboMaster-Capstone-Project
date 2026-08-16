"""Run a trained detector over the val split and write COCO-format detections.

This is the adapter layer that makes scripts/evaluate_detection.py able to score
every model family through one code path. Each family predicts in its own idiom and
its own class indexing; all of it is normalised here, so the evaluator only ever
sees `[{image_id, category_id, bbox, score}, ...]`.

Class indices are mapped by NAME, never by a hardcoded offset - YOLO indexes its
classes from 0 while the COCO export numbers categories from 1, and silently sliding
every prediction one class sideways would still produce plausible-looking mAP.

Note on device: predictions may be generated on the GPU, because only the DETECTIONS
matter here and they are identical either way. Latency, FPS, CPU and RAM must still
be measured on the target CPU by the benchmark harness - never taken from this script.

Usage:
    python scripts/predict_to_coco.py --family yolo \\
        --weights runs/detect/fast_640/weights/best.pt --imgsz 640 --out preds.json

    python scripts/evaluate_detection.py --predictions preds.json --name fast_640
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GT = ROOT / "data" / "splits" / "coco_val.json"
COCO_ROOT = ROOT / "Datasets" / "DJI ROCO Central.v1i.coco"

# Low on purpose. mAP integrates the whole precision/recall curve, so throwing away
# low-confidence detections truncates the curve and understates the model. An
# operating-point threshold is chosen later, at scoring time.
DEFAULT_CONF = 0.001


def load_targets(gt_path: Path) -> tuple[list[tuple[int, Path]], dict[str, int]]:
    """Return [(image_id, absolute path)] and a class-name -> category_id map."""
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    targets = []
    for img in gt["images"]:
        path = COCO_ROOT / img["file_name"]
        if not path.is_file():
            sys.exit(f"Missing image {path}\nRun: python scripts/download_data.py")
        targets.append((img["id"], path))
    name_to_cat = {c["name"]: c["id"] for c in gt["categories"]}
    return targets, name_to_cat


def predict_yolo(
    weights: Path, targets, name_to_cat: dict[str, int], imgsz: int, conf: float,
    device: str, batch: int,
) -> list[dict]:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    idx_to_cat = {}
    for idx, name in model.names.items():
        if name not in name_to_cat:
            sys.exit(
                f"Model class {name!r} (index {idx}) is not a category in the ground "
                f"truth ({sorted(name_to_cat)}). Refusing to guess a mapping."
            )
        idx_to_cat[int(idx)] = name_to_cat[name]
    print(f"class map (model idx -> coco id): {idx_to_cat}")

    detections: list[dict] = []
    for start in range(0, len(targets), batch):
        chunk = targets[start : start + batch]
        results = model.predict(
            [str(p) for _, p in chunk],
            imgsz=imgsz, conf=conf, device=device, verbose=False,
        )
        for (image_id, _), res in zip(chunk, results):
            # Ultralytics returns xyxy already rescaled to the ORIGINAL image size.
            for box in res.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                detections.append({
                    "image_id": image_id,
                    "category_id": idx_to_cat[int(box.cls[0])],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(box.conf[0]),
                })
        print(f"\r  {min(start + batch, len(targets))}/{len(targets)} images", end="")
    print()
    return detections


def predict_ssd(
    weights: Path, targets, name_to_cat: dict[str, int], imgsz: int, conf: float,
    device: str, batch: int,
) -> list[dict]:
    import torch
    from PIL import Image
    from torchvision.transforms import functional as TF

    sys.path.insert(0, str(ROOT / "scripts"))
    from train_ssd import build_ssd  # noqa: PLC0415 - built in Phase 3

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    model = build_ssd(ckpt["backbone"], ckpt["imgsz"], ckpt["num_classes"])
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    detections: list[dict] = []
    with torch.inference_mode():
        for start in range(0, len(targets), batch):
            chunk = targets[start : start + batch]
            images = [
                TF.to_tensor(Image.open(p).convert("RGB")).to(device) for _, p in chunk
            ]
            for (image_id, _), out in zip(chunk, model(images)):
                # GeneralizedRCNNTransform.postprocess already rescales boxes back to
                # the original image size; assert_native_scale() checks that holds.
                for box, label, score in zip(
                    out["boxes"].cpu(), out["labels"].cpu(), out["scores"].cpu()
                ):
                    if float(score) < conf:
                        continue
                    x1, y1, x2, y2 = (float(v) for v in box)
                    detections.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    })
            print(f"\r  {min(start + batch, len(targets))}/{len(targets)} images", end="")
    print()
    return detections


def assert_native_scale(detections: list[dict], gt_path: Path) -> None:
    """Catch predictions left in resized-input coordinates instead of native ones.

    A model whose boxes were never rescaled produces mAP near zero for a reason that
    looks exactly like undertraining, so it is worth failing loudly here instead.
    """
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    sizes = {i["id"]: (i["width"], i["height"]) for i in gt["images"]}
    outside = 0
    for det in detections:
        w, h = sizes[det["image_id"]]
        x, y, bw, bh = det["bbox"]
        if x < -1 or y < -1 or x + bw > w + 1 or y + bh > h + 1:
            outside += 1
    if outside:
        print(f"  WARNING: {outside} boxes fall outside their image bounds.")
    if detections:
        widest = max(d["bbox"][0] + d["bbox"][2] for d in detections)
        print(f"  rightmost box edge: {widest:.1f}px (images are 1920px wide)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--family", choices=("yolo", "ssd"), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--device", default="0", help="'0' for GPU, 'cpu' for CPU.")
    parser.add_argument("--batch", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        sys.exit(f"Missing weights: {args.weights}")
    if not args.gt.is_file():
        sys.exit(f"Missing {args.gt}\nRun: python scripts/make_splits.py")

    targets, name_to_cat = load_targets(args.gt)
    print(f"family : {args.family}")
    print(f"weights: {args.weights}")
    print(f"images : {len(targets)}   imgsz: {args.imgsz}   conf: {args.conf}")

    predict = predict_yolo if args.family == "yolo" else predict_ssd
    start = time.perf_counter()
    detections = predict(
        args.weights, targets, name_to_cat, args.imgsz, args.conf,
        args.device, args.batch,
    )
    elapsed = time.perf_counter() - start

    print(f"\n{len(detections)} detections in {elapsed:.1f}s")
    assert_native_scale(detections, args.gt)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(detections), encoding="utf-8")
    print(f"wrote {args.out}")
    print(f"\nScore it:\n  python scripts/evaluate_detection.py "
          f"--predictions {args.out} --name <config-name>")


if __name__ == "__main__":
    main()
