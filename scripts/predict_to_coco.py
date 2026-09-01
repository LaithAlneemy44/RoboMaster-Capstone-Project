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


def normalise_device(device: str) -> str:
    """Turn Ultralytics' device spelling into one torch also accepts.

    Ultralytics takes "0" to mean the first GPU, but torch.device("0") raises
    "Invalid device string". Both understand "cuda:0", so everything is normalised to
    that and each backend gets a string it can parse.
    """
    device = device.strip()
    return f"cuda:{device}" if device.isdigit() else device


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


def load_yolo(weights: Path, name_to_cat: dict[str, int], quiet: bool = False):
    """Load a YOLO checkpoint and build its model-index -> COCO-category map.

    Shared with scripts/benchmark_cpu.py so the timed model is built exactly the way
    the scored one was - a benchmark of a differently-constructed model would report
    latency for something other than the row it sits beside.
    """
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
    if not quiet:
        print(f"class map (model idx -> coco id): {idx_to_cat}")
    return model, idx_to_cat


def yolo_result_to_dets(res, image_id: int, idx_to_cat: dict[int, int]) -> list[dict]:
    """Convert one Ultralytics Result into COCO detection dicts."""
    # Ultralytics returns xyxy already rescaled to the ORIGINAL image size.
    dets = []
    for box in res.boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        dets.append({
            "image_id": image_id,
            "category_id": idx_to_cat[int(box.cls[0])],
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(box.conf[0]),
        })
    return dets


def predict_yolo(
    weights: Path, targets, name_to_cat: dict[str, int], imgsz: int, conf: float,
    device: str, batch: int,
) -> list[dict]:
    model, idx_to_cat = load_yolo(weights, name_to_cat)

    detections: list[dict] = []
    for start in range(0, len(targets), batch):
        chunk = targets[start : start + batch]
        results = model.predict(
            [str(p) for _, p in chunk],
            imgsz=imgsz, conf=conf, device=device, verbose=False,
        )
        for (image_id, _), res in zip(chunk, results):
            detections.extend(yolo_result_to_dets(res, image_id, idx_to_cat))
        print(f"\r  {min(start + batch, len(targets))}/{len(targets)} images", end="")
    print()
    return detections


def load_ssd(weights: Path, imgsz: int, device: str, quiet: bool = False):
    """Rebuild an SSD from its checkpoint. Returns (model, checkpoint imgsz)."""
    import torch

    sys.path.insert(0, str(ROOT / "scripts"))
    from train_ssd import build_ssd  # noqa: PLC0415 - built in Phase 3

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    # The checkpoint's own imgsz wins: the model's anchors and head were built for it,
    # and preprocessing at a different size would silently mis-scale every box.
    ckpt_imgsz = int(ckpt["imgsz"])
    if ckpt_imgsz != imgsz and not quiet:
        print(f"  note: --imgsz {imgsz} ignored; checkpoint was trained at {ckpt_imgsz}")
    # The checkpoint records its normalisation; rebuilding with the wrong one
    # fails on missing keys, since GroupNorm and BatchNorm have different state.
    model = build_ssd(ckpt["backbone"], ckpt_imgsz, ckpt["num_classes"],
                      norm=ckpt.get("norm", "batch"),
                      min_ratio=ckpt.get("min_ratio", 0.2),
                      max_ratio=ckpt.get("max_ratio", 0.95))
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, ckpt_imgsz


def ssd_preprocess(image, ckpt_imgsz: int, device: str):
    """Decoded PIL image -> (input tensor, (sx, sy) native-rescale factors).

    Takes an already-decoded image rather than a path so the benchmark harness can
    time JPEG decode separately from preprocessing.
    """
    from PIL import Image
    from torchvision.transforms import functional as TF

    native_w, native_h = image.size
    # Must mirror RocoCoco.__getitem__: the dataset pre-resizes on the CPU, so the
    # model was trained on squashed (imgsz, imgsz) input and emits boxes in that
    # space rather than in native pixels.
    image = image.resize((ckpt_imgsz, ckpt_imgsz), Image.BILINEAR)
    return TF.to_tensor(image).to(device), (native_w / ckpt_imgsz, native_h / ckpt_imgsz)


def ssd_out_to_dets(out, image_id: int, sx: float, sy: float, conf: float) -> list[dict]:
    """Convert one SSD output dict into COCO detections, rescaled to native pixels."""
    dets = []
    for box, label, score in zip(
        out["boxes"].cpu(), out["labels"].cpu(), out["scores"].cpu()
    ):
        if float(score) < conf:
            continue
        x1, y1, x2, y2 = (float(v) for v in box)
        x1, x2 = x1 * sx, x2 * sx
        y1, y2 = y1 * sy, y2 * sy
        dets.append({
            "image_id": image_id,
            "category_id": int(label),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(score),
        })
    return dets


def predict_ssd(
    weights: Path, targets, name_to_cat: dict[str, int], imgsz: int, conf: float,
    device: str, batch: int,
) -> list[dict]:
    import torch
    from PIL import Image

    model, ckpt_imgsz = load_ssd(weights, imgsz, device)

    detections: list[dict] = []
    with torch.inference_mode():
        for start in range(0, len(targets), batch):
            chunk = targets[start : start + batch]
            images, scales = [], []
            for _, path in chunk:
                tensor, scale = ssd_preprocess(
                    Image.open(path).convert("RGB"), ckpt_imgsz, device
                )
                images.append(tensor)
                scales.append(scale)

            for (image_id, _), (sx, sy), out in zip(chunk, scales, model(images)):
                detections.extend(ssd_out_to_dets(out, image_id, sx, sy, conf))
            print(f"\r  {min(start + batch, len(targets))}/{len(targets)} images", end="")
    print()
    return detections


def predict_classical(
    config_name: str, targets, name_to_cat: dict[str, int], conf: float,
) -> list[dict]:
    """The classical detector. No weights file - a config IS the model here.

    It has no training stage, so nothing is loaded from disk except the template bank,
    and it detects a single class. Everything else goes through the same COCO shape as
    the DL families so the evaluator cannot tell them apart.
    """
    import cv2

    sys.path.insert(0, str(ROOT / "scripts"))
    from classical_detector import CONFIGS, ClassicalDetector  # noqa: PLC0415

    if config_name not in CONFIGS:
        sys.exit(f"Unknown classical config {config_name!r}. "
                 f"Choose from: {', '.join(sorted(CONFIGS))}")
    detector = ClassicalDetector(CONFIGS[config_name])

    detections: list[dict] = []
    candidates = 0
    for index, (image_id, path) in enumerate(targets):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        for (x, y, w, h), score, cls in detector.detect(frame):
            if score < conf:
                continue
            if cls not in name_to_cat:
                sys.exit(f"Detector emitted class {cls!r}, absent from the ground "
                         f"truth ({sorted(name_to_cat)}).")
            detections.append({
                "image_id": image_id,
                "category_id": name_to_cat[cls],
                "bbox": [x, y, w, h],
                "score": score,
            })
        candidates += detector.last_candidates
        print(f"\r  {index + 1}/{len(targets)} images", end="")
    print(f"\n  mean candidate regions/frame: {candidates / max(1, len(targets)):.0f}")
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
    parser.add_argument("--family", choices=("yolo", "ssd", "classical"), required=True)
    # Not required for classical: it has no trained weights, only a parameter set.
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--config", help="Classical config name, e.g. strict.")
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--device", default="0", help="'0' for GPU, 'cpu' for CPU.")
    parser.add_argument("--batch", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.family == "classical":
        if not args.config:
            sys.exit("--config is required for --family classical")
    elif not args.weights or not args.weights.is_file():
        sys.exit(f"Missing weights: {args.weights}")
    if not args.gt.is_file():
        sys.exit(f"Missing {args.gt}\nRun: python scripts/make_splits.py")

    targets, name_to_cat = load_targets(args.gt)
    device = normalise_device(args.device)
    print(f"family : {args.family}")
    print(f"model  : {args.config if args.family == 'classical' else args.weights}")
    print(f"images : {len(targets)}   imgsz: {args.imgsz}   conf: {args.conf}")
    print(f"device : {device}")

    start = time.perf_counter()
    if args.family == "classical":
        detections = predict_classical(args.config, targets, name_to_cat, args.conf)
    else:
        predict = predict_yolo if args.family == "yolo" else predict_ssd
        detections = predict(
            args.weights, targets, name_to_cat, args.imgsz, args.conf,
            device, args.batch,
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
