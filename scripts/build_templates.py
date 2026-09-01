"""Build the edge-map template bank the classical detector matches against.

Proposal 5.2 specifies template matching plus normalised cross correlation. That needs
templates, and they have to come from somewhere principled: these are averaged edge maps
of real armor plates cropped from the TRAINING split.

WHY A BANK RATHER THAN ONE TEMPLATE
    Measured over the 31,896 train-split armor instances:

        size       median 21.5px, p10 17.0 -> p90 28.5     1.7x spread
        aspect w/h median 1.29,   p10 0.60 -> p90 2.15      3.6x spread

    Scale barely varies, which is unusually kind to template matching. Aspect ratio does
    vary, because a plate seen edge-on is narrow and the same plate seen square-on is
    wide. One template would match well at one viewing angle and poorly everywhere else,
    so the crops are binned by aspect and averaged within each bin.

WHY EDGE MAPS RATHER THAN RAW PIXELS
    An armor plate is two bright LED bars flanking a digit. Raw intensity is dominated by
    how blown-out the LEDs are in that particular frame; the edge structure is what is
    actually consistent between plates, and NCC on edges is what 5.2 describes.

TRAIN SPLIT ONLY
    Building templates from val would fold the evaluation set into the model and inflate
    every number downstream - the same hazard make_splits.py exists to prevent for the DL
    side. This refuses any COCO file that is not the train split.

Usage:
    python scripts/build_templates.py
    python scripts/build_templates.py --limit 8000 --canonical-height 24
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COCO_ROOT = ROOT / "Datasets" / "DJI ROCO Central.v1i.coco"
TRAIN_JSON = ROOT / "data" / "splits" / "coco_train.json"
OUT = ROOT / "models" / "classical" / "templates.npz"

TARGET_CLASS = "armor"

# Bins chosen to span the measured p10-p90 aspect range (0.60 - 2.15). A crop is
# assigned to the nearest bin centre.
ASPECT_BINS = (0.7, 1.0, 1.4, 2.0)

# Crops below this are mostly JPEG noise once edge-detected.
MIN_CROP_PX = 8


def edge_map(gray):
    """Sobel gradient magnitude, normalised to 0-1.

    Sobel rather than Canny: Canny thresholds to a binary map, and at ~20px a plate has
    so few edge pixels that a threshold either keeps almost everything or almost nothing.
    A continuous magnitude keeps the structure NCC needs.
    """
    import cv2
    import numpy as np

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    peak = float(mag.max())
    return mag / peak if peak > 0 else mag


def collect(train_json: Path, limit: int, seed: int):
    """Crop armor instances from the train split, grouped by nearest aspect bin."""
    import cv2

    data = json.loads(train_json.read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in data["categories"]}
    images = {i["id"]: i for i in data["images"]}

    wanted = [a for a in data["annotations"] if names[a["category_id"]] == TARGET_CLASS]
    print(f"[scan]  {len(wanted)} {TARGET_CLASS} instances in {train_json.name}")
    random.Random(seed).shuffle(wanted)

    # Group by image so each file is decoded once, not once per instance.
    by_image = {}
    for ann in wanted[:limit] if limit else wanted:
        by_image.setdefault(ann["image_id"], []).append(ann)

    bins = {a: [] for a in ASPECT_BINS}
    used = 0
    for index, (image_id, anns) in enumerate(by_image.items()):
        path = COCO_ROOT / images[image_id]["file_name"]
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            continue
        height, width = frame.shape
        for ann in anns:
            x, y, w, h = (int(round(v)) for v in ann["bbox"])
            if w < MIN_CROP_PX or h < MIN_CROP_PX:
                continue
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(width, x + w), min(height, y + h)
            if x1 - x0 < MIN_CROP_PX or y1 - y0 < MIN_CROP_PX:
                continue
            crop = frame[y0:y1, x0:x1]
            aspect = (x1 - x0) / (y1 - y0)
            nearest = min(ASPECT_BINS, key=lambda b: abs(b - aspect))
            bins[nearest].append(edge_map(crop))
            used += 1
        print(f"\r[crop]  {index + 1}/{len(by_image)} images, {used} crops",
              end="", flush=True)
    print()
    return bins, used


def average(bins, canonical_height: int):
    """Resize each bin's crops to its canonical size and average them."""
    import cv2
    import numpy as np

    templates, meta = {}, {}
    for aspect, crops in bins.items():
        if not crops:
            print(f"  aspect {aspect}: no crops, skipped")
            continue
        width = max(4, int(round(canonical_height * aspect)))
        stack = np.zeros((canonical_height, width), dtype=np.float32)
        for crop in crops:
            stack += cv2.resize(crop, (width, canonical_height),
                                interpolation=cv2.INTER_AREA)
        mean = stack / len(crops)
        # Zero-mean so NCC responds to structure rather than overall brightness.
        mean -= mean.mean()
        peak = float(np.abs(mean).max())
        if peak > 0:
            mean /= peak
        templates[f"ar{aspect}"] = mean
        meta[f"ar{aspect}"] = (len(crops), width, canonical_height)
        print(f"  aspect {aspect}: {len(crops):6} crops -> {width}x{canonical_height}")
    return templates, meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--coco", type=Path, default=TRAIN_JSON)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--limit", type=int, default=6000,
                        help="Max instances to average. 0 = all.")
    parser.add_argument("--canonical-height", type=int, default=20,
                        help="Template height; width follows the aspect bin.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-any-split", action="store_true",
                        help="Override the train-split guard. Almost certainly wrong.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.coco.is_file():
        sys.exit(f"Missing {args.coco}\nRun: python scripts/make_splits.py")
    if "train" not in args.coco.stem and not args.allow_any_split:
        sys.exit(
            f"{args.coco.name} is not the train split.\n"
            "Templates built from val or test leak the evaluation set into the model, "
            "and every mAP measured afterwards would be inflated for a reason that is "
            "invisible in the results.\nPass --allow-any-split only if you are certain."
        )

    import numpy as np

    bins, used = collect(args.coco, args.limit, args.seed)
    if not used:
        sys.exit("No usable crops - check the dataset is downloaded.")
    print(f"\n[build] averaging {used} crops into {len(ASPECT_BINS)} aspect bins")
    templates, meta = average(bins, args.canonical_height)
    if not templates:
        sys.exit("No templates produced.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        source=str(args.coco.name),
        target_class=TARGET_CLASS,
        crops_used=used,
        canonical_height=args.canonical_height,
        **templates,
    )
    print(f"\n[done]  {args.out}  ({len(templates)} templates from {used} crops)")
    print("Next:\n    python scripts/classical_detector.py --selftest")


if __name__ == "__main__":
    main()
