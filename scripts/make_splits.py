"""Build train/val splits for detection without touching the pinned export.

All 2655 images are frames sampled from just SEVEN match recordings. A random
split therefore puts near-duplicate consecutive frames on both sides and inflates
val mAP badly - the model recognises the frame, not the object. The default here
is a GROUP-AWARE split: whole clips are held out, so train and val share no match.

Holding out `-VsBorn-of-Fire_BO2_1` alone gives 2258/397 = 85.05/14.95, which is
the 85/15 the project proposal specifies, with zero clip leakage. That is why it
is the default. Its one weakness: val is a single match, so val mAP has no
across-clip variance in it - confidence intervals must come from bootstrapping
over val images, and should be reported as such.

Rather than moving files (which would break scripts/verify_data.py, since it fails
on both missing and extra files under Datasets/), this script writes the splits as
*lists* into data/splits/. Datasets/ stays byte-identical to the pinned export.

The YOLO and COCO exports use identical filenames, so one assignment drives both
formats and the two model families train on exactly the same images.

Usage:
    python scripts/make_splits.py                      # group-aware 85/15 (default)
    python scripts/make_splits.py --list-clips         # show the 7 clips and exit
    python scripts/make_splits.py --holdout -VsCUBOT_BO2_1 -VsRPS_BO2_2
    python scripts/make_splits.py --from-assignment    # reproduce the committed split
    python scripts/make_splits.py --val-frac 0.15      # random re-split (LEAKY)
    python scripts/make_splits.py --keep-export-split  # export's valid split (LEAKY)

Outputs (all outside Datasets/, all gitignored except assignment.csv):
    data/roco_central.yaml        Ultralytics data config -> point train_yolo.py at this
    data/splits/train.txt         absolute image paths, one per line
    data/splits/val.txt
    data/splits/coco_train.json   merged + re-indexed COCO annotations
    data/splits/coco_val.json
    data/splits/assignment.csv    filename,clip,orig_split,new_split - COMMITTED

assignment.csv is the only committed artifact because it is small, diffable, and
fully determines the split; everything else regenerates from it in seconds via
--from-assignment, and the absolute paths in the other files are machine-local.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
YOLO_DIR = ROOT / "Datasets" / "DJI ROCO Central.v1i.yolov11"
COCO_DIR = ROOT / "Datasets" / "DJI ROCO Central.v1i.coco"
OUT_DIR = ROOT / "data" / "splits"
YAML_PATH = ROOT / "data" / "roco_central.yaml"
ASSIGNMENT_PATH = OUT_DIR / "assignment.csv"

ORIG_SPLITS = ("train", "valid", "test")
# Index order must match the export's data.yaml `names` list.
CLASS_NAMES = ["armor", "base", "car", "ignore", "watcher"]

# "ignore" marks robots that are real but too ambiguous to score against - 387
# instances, 0.67% of the dataset. Models still TRAIN on it (learning to tag an
# ambiguous robot as `ignore` is better than misfiring it as `car`), but its
# predictions are discarded at evaluation. Marking it iscrowd=1 records that in
# the annotations themselves; scripts/evaluate_detection.py does the actual
# suppression, because COCO's iscrowd only excludes matches within the SAME
# category and an ignore region has to suppress detections of any class.
IGNORE_CLASS = "ignore"

# Roboflow names every frame "<clip>_<frame>_jpg.rf.<hash>.jpg". The frame index is
# always the last numeric field, so a greedy prefix takes the clip name intact -
# clip names themselves contain underscores and digits ("-VsBorn-of-Fire_BO2_1").
CLIP_RE = re.compile(r"^(?P<clip>.+)_(?P<frame>\d+)_jpg\.rf\.[0-9a-f]+\.jpg$", re.IGNORECASE)

# 397 of 2655 images = 14.95%, i.e. the proposal's 85/15 without any clip leakage.
# Every class is present in it (armor 6793, base 272, car 2342, ignore 19,
# watcher 383), which a held-out clip must be for per-class mAP to be defined.
DEFAULT_HOLDOUT = ("-VsBorn-of-Fire_BO2_1",)


def parse_clip(name: str) -> str:
    match = CLIP_RE.match(name)
    if not match:
        sys.exit(
            f"Cannot parse a clip name out of {name!r}.\n"
            "Expected Roboflow's '<clip>_<frame>_jpg.rf.<hash>.jpg' form. If the "
            "export version changed, update CLIP_RE in this script - do not fall "
            "back to a random split, it leaks frames across train/val."
        )
    return match.group("clip")


def collect_images() -> dict[str, str]:
    """Map each image filename to the original split folder it lives in."""
    origin: dict[str, str] = {}
    for split in ORIG_SPLITS:
        img_dir = YOLO_DIR / split / "images"
        if not img_dir.is_dir():
            sys.exit(f"Missing {img_dir}. Run: python scripts/download_data.py")
        for img in img_dir.iterdir():
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                if img.name in origin:
                    sys.exit(f"Duplicate filename across splits: {img.name}")
                origin[img.name] = split
    if not origin:
        sys.exit(f"No images found under {YOLO_DIR}")
    return origin


def assign_by_clip(clips: dict[str, str], holdout: tuple[str, ...]) -> dict[str, str]:
    """Hold out whole match recordings, so no clip appears on both sides."""
    known = set(clips.values())
    unknown = [c for c in holdout if c not in known]
    if unknown:
        sys.exit(
            "Unknown clip(s): " + ", ".join(unknown) + "\n"
            "Available: " + ", ".join(sorted(known)) + "\n"
            "List them with: python scripts/make_splits.py --list-clips"
        )
    if set(holdout) >= known:
        sys.exit("--holdout covers every clip, which leaves no training data.")
    return {
        name: ("val" if clip in holdout else "train") for name, clip in clips.items()
    }


def assign_random(names_all: list[str], val_frac: float, seed: int) -> dict[str, str]:
    """Pool all images and draw a fresh validation set. Leaks frames across clips."""
    if not 0.0 < val_frac < 1.0:
        sys.exit(f"--val-frac must be strictly between 0 and 1, got {val_frac}")
    names = sorted(names_all)  # sorted first so the seed alone determines the draw
    random.Random(seed).shuffle(names)
    val = set(names[: round(len(names) * val_frac)])
    return {name: ("val" if name in val else "train") for name in names_all}


def assign_from_csv(path: Path, expected: set[str]) -> dict[str, str]:
    """Replay a previously written assignment.csv exactly."""
    if not path.is_file():
        sys.exit(f"No assignment to replay at {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = {r["filename"]: r["new_split"] for r in csv.DictReader(fh)}

    if set(rows) != expected:
        missing = sorted(expected - set(rows))[:5]
        extra = sorted(set(rows) - expected)[:5]
        sys.exit(
            f"{path.name} does not describe the dataset on disk "
            f"({len(rows)} rows vs {len(expected)} images).\n"
            + (f"  not in csv : {', '.join(missing)}\n" if missing else "")
            + (f"  not on disk: {', '.join(extra)}\n" if extra else "")
            + "The export version probably changed - rebuild instead of replaying."
        )
    bad = {v for v in rows.values()} - {"train", "val"}
    if bad:
        sys.exit(f"{path.name} has unexpected new_split values: {sorted(bad)}")
    return rows


def class_counts(names: list[str], origin: dict[str, str]) -> Counter[str]:
    """Count labelled instances per class, read from the YOLO label files."""
    counts: Counter[str] = Counter()
    for name in names:
        label = YOLO_DIR / origin[name] / "labels" / f"{Path(name).stem}.txt"
        if not label.is_file():
            continue
        for line in label.read_text(encoding="utf-8").splitlines():
            if line.strip():
                counts[CLASS_NAMES[int(line.split()[0])]] += 1
    return counts


def write_image_lists(split_names: dict[str, list[str]], origin: dict[str, str]) -> None:
    """Write one absolute image path per line.

    Ultralytics derives label paths by swapping /images/ for /labels/, so pointing
    into the original split folders resolves labels correctly with no copying.
    """
    for split, names in split_names.items():
        lines = [
            (YOLO_DIR / origin[name] / "images" / name).as_posix() for name in names
        ]
        (OUT_DIR / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_yaml(description: str) -> None:
    YAML_PATH.write_text(
        "# Generated by scripts/make_splits.py - do not hand-edit, do not commit.\n"
        "# The paths below are absolute and therefore machine-local; regenerate\n"
        "# after moving the project with: python scripts/make_splits.py --from-assignment\n"
        f"# Split: {description}\n"
        f"path: {YOLO_DIR.as_posix()}\n"
        f"train: {(OUT_DIR / 'train.txt').as_posix()}\n"
        f"val: {(OUT_DIR / 'val.txt').as_posix()}\n"
        "\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)),
        encoding="utf-8",
    )


def write_coco(split_names: dict[str, list[str]], origin: dict[str, str]) -> None:
    """Merge the per-split COCO files, re-indexing ids so they cannot collide.

    Every split's export numbers its images and annotations from 0, so a naive
    concatenation would silently alias annotations onto the wrong images. file_name
    keeps a split-relative prefix, so load with root=Datasets/DJI ROCO Central.v1i.coco.
    """
    source = {}
    for split in ORIG_SPLITS:
        path = COCO_DIR / split / "_annotations.coco.json"
        if not path.is_file():
            sys.exit(f"Missing {path}")
        source[split] = json.loads(path.read_text(encoding="utf-8"))

    categories = source["train"]["categories"]
    for split in ORIG_SPLITS:
        if source[split]["categories"] != categories:
            sys.exit(f"Category list in {split} differs from train - cannot merge.")

    ignore_id = next((c["id"] for c in categories if c["name"] == IGNORE_CLASS), None)
    if ignore_id is None:
        sys.exit(
            f"No {IGNORE_CLASS!r} category in the COCO export "
            f"(found: {[c['name'] for c in categories]}).\n"
            "The evaluator's ignore-region policy depends on it - see IGNORE_CLASS."
        )

    # image_id -> annotations, per original split, so lookup stays O(1).
    by_image = {split: _group(data["annotations"]) for split, data in source.items()}
    image_by_name = {
        split: {img["file_name"]: img for img in data["images"]}
        for split, data in source.items()
    }

    for split, names in split_names.items():
        images, annotations = [], []
        marked = 0
        # ids are 1-based ON PURPOSE. COCOeval stores the matched ground-truth id in
        # dtMatches and then tests it with logical_and(dtm, ...) - so an annotation
        # with id 0 is falsy, its match is silently scored as a false positive, and
        # the AP of whichever class owns it is quietly deflated.
        for new_id, name in enumerate(names, start=1):
            orig = origin[name]
            src = image_by_name[orig].get(name)
            if src is None:
                sys.exit(f"{name} is missing from the {orig} COCO annotations.")
            img = dict(src)
            old_id = img["id"]
            img["id"] = new_id
            img["file_name"] = f"{orig}/{name}"
            images.append(img)
            for ann in by_image[orig].get(old_id, []):
                ann = dict(ann)
                ann["id"] = len(annotations) + 1
                ann["image_id"] = new_id
                if ann["category_id"] == ignore_id:
                    ann["iscrowd"] = 1
                    marked += 1
                annotations.append(ann)

        merged = {
            "info": source["train"]["info"],
            "licenses": source["train"]["licenses"],
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }
        (OUT_DIR / f"coco_{split}.json").write_text(json.dumps(merged), encoding="utf-8")
        print(f"  coco_{split}.json: {marked} {IGNORE_CLASS} annotations marked iscrowd=1")


def _group(annotations: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for ann in annotations:
        grouped.setdefault(ann["image_id"], []).append(ann)
    return grouped


def describe(
    assignment: dict[str, str], clips: dict[str, str], fallback: str
) -> str:
    """Describe what the split IS, not how it was invoked.

    A clip-disjoint split gets the same description whether it was built with
    --holdout or replayed with --from-assignment, so replaying reproduces every
    output file byte for byte.
    """
    sides: dict[str, set[str]] = {"train": set(), "val": set()}
    for name, split in assignment.items():
        sides[split].add(clips[name])
    if sides["train"] & sides["val"]:
        return fallback
    return "group-aware, held-out clip(s): " + ", ".join(sorted(sides["val"]))


def print_clip_table(clips: dict[str, str], assignment: dict[str, str] | None) -> None:
    """Per-clip image counts, and which side of the split each clip landed on."""
    per: dict[str, Counter[str]] = defaultdict(Counter)
    for name, clip in clips.items():
        per[clip]["n"] += 1
        if assignment:
            per[clip][assignment[name]] += 1
    total = len(clips)

    print(f"\n{'clip':<28}{'images':>8}{'share':>8}{'train':>8}{'val':>7}")
    for clip, counts in sorted(per.items(), key=lambda kv: -kv[1]["n"]):
        share = f"{100 * counts['n'] / total:.1f}%"
        train = str(counts["train"]) if assignment else "-"
        val = str(counts["val"]) if assignment else "-"
        print(f"{clip:<28}{counts['n']:>8}{share:>8}{train:>8}{val:>7}")

    if assignment:
        straddling = [c for c, k in per.items() if k["train"] and k["val"]]
        if straddling:
            print(
                f"\n  LEAKAGE: {len(straddling)}/{len(per)} clips appear on BOTH sides. "
                "Consecutive frames\n  straddle the split, so val mAP is inflated and is "
                "not a generalization estimate."
            )
        else:
            print(f"\n  No leakage: train and val share none of the {len(per)} clips.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--holdout",
        nargs="+",
        metavar="CLIP",
        default=None,
        help="Clip name(s) to hold out as val. Default: " + " ".join(DEFAULT_HOLDOUT),
    )
    mode.add_argument(
        "--val-frac",
        type=float,
        default=None,
        help="LEAKY. Randomly re-split all images to this validation fraction, "
        "ignoring clip boundaries.",
    )
    mode.add_argument(
        "--keep-export-split",
        action="store_true",
        help="LEAKY. Keep the export's own valid split and fold test into train (80/20).",
    )
    mode.add_argument(
        "--from-assignment",
        nargs="?",
        const=str(ASSIGNMENT_PATH),
        default=None,
        metavar="CSV",
        help="Replay a previous assignment.csv exactly (default: the committed one). "
        "Use this to regenerate the machine-local files after moving the project.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for --val-frac.")
    parser.add_argument(
        "--list-clips",
        action="store_true",
        help="Print the clips found in the dataset and exit without writing anything.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    origin = collect_images()
    clips = {name: parse_clip(name) for name in origin}

    if args.list_clips:
        print(f"{len(origin)} images from {len(set(clips.values()))} clips")
        print_clip_table(clips, None)
        return

    if args.from_assignment:
        assignment = assign_from_csv(Path(args.from_assignment), set(origin))
        fallback = f"replayed from {Path(args.from_assignment).name} (LEAKY)"
    elif args.keep_export_split:
        assignment = {
            name: ("val" if split == "valid" else "train")
            for name, split in origin.items()
        }
        fallback = "export's valid split, test folded into train (LEAKY)"
    elif args.val_frac is not None:
        assignment = assign_random(list(origin), args.val_frac, args.seed)
        fallback = f"random val_frac={args.val_frac} seed={args.seed} (LEAKY)"
    else:
        holdout = tuple(args.holdout) if args.holdout else DEFAULT_HOLDOUT
        assignment = assign_by_clip(clips, holdout)
        fallback = "unreachable - assign_by_clip is always clip-disjoint"

    description = describe(assignment, clips, fallback)

    split_names: dict[str, list[str]] = {"train": [], "val": []}
    for name in sorted(assignment):
        split_names[assignment[name]].append(name)
    for split, names in split_names.items():
        if not names:
            sys.exit(f"The {split} split came out empty - refusing to write it.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_image_lists(split_names, origin)
    write_yaml(description)
    write_coco(split_names, origin)

    with ASSIGNMENT_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filename", "clip", "orig_split", "new_split"])
        for name in sorted(assignment):
            writer.writerow([name, clips[name], origin[name], assignment[name]])

    total = len(assignment)
    print(f"{total} images  ({description})")
    for split in ("train", "val"):
        names = split_names[split]
        counts = class_counts(names, origin)
        share = 100 * len(names) / total
        print(
            f"  {split:<5} {len(names):>5} images ({share:5.2f}%)  "
            f"{sum(counts.values()):>6} instances  "
            + "  ".join(f"{c}={counts[c]}" for c in CLASS_NAMES)
        )
        empty = [c for c in CLASS_NAMES if not counts[c]]
        if empty:
            print(f"        WARNING: no {', '.join(empty)} instances in {split}.")

    print_clip_table(clips, assignment)
    print(f"\nwrote {YAML_PATH.relative_to(ROOT)} and {OUT_DIR.relative_to(ROOT)}/")
    print("Datasets/ untouched - `python scripts/verify_data.py` should still pass.")


if __name__ == "__main__":
    main()
