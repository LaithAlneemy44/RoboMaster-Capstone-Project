"""Split the labelled tracking sequences into train / val / test, whole clips only.

The same lesson as detection, for the same reason. scripts/make_splits.py holds out
entire match recordings because consecutive frames of one moment landing on both sides
of a split inflates the score - a model is then tested on frames it effectively trained
on. Tracking data is nothing but consecutive frames, so the hazard is worse here, and a
sequence is never divided.

WHY THIS IS NOT PART OF make_splits.py
    That script is bound to the detection dataset: Roboflow filename parsing, YOLO label
    trees, COCO json emission, a fixed five-class list. None of it applies to MOT
    sequences. The pattern is reused - group-aware assignment, a printed leakage
    verdict, one committed assignment file - but the code is not.

THE RATIO IS PROVISIONAL
    Proposal 5.3 specifies 70/15/15, on the assumption that GOTURN would be trained on
    the training portion. No cv2 tracker can be fine-tuned through the OpenCV API, and
    CLAUDE.md warns against training GOTURN at all, so the trackers may well stay frozen
    and pretrained. If they do, NOTHING consumes the training split, and those frames
    are hand-labelled for no reader while val and test go short - something closer to
    20/40/40 would then buy far more evaluation data for the same labelling hours.
    Hence --ratio. Record whichever is used, and why, in the write-up.

Usage:
    python scripts/make_tracking_splits.py
    python scripts/make_tracking_splits.py --ratio 20 40 40
    python scripts/make_tracking_splits.py --from-assignment
"""

from __future__ import annotations

import argparse
import configparser
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACKING = ROOT / "data" / "tracking"
ASSIGNMENT = TRACKING / "assignment.csv"

SPLITS = ("train", "val", "test")
DEFAULT_RATIO = (70.0, 15.0, 15.0)  # proposal 5.3; see module docstring

# Below this, a split's score is dominated by which clips happened to land in it.
MIN_SEQS_PER_SPLIT = 2


def discover() -> dict[str, int]:
    """Sequence name -> frame count, for every labelled sequence."""
    if not TRACKING.is_dir():
        sys.exit(f"No {TRACKING}\nRun: python scripts/fetch_clips.py")

    sequences = {}
    for path in sorted(TRACKING.iterdir()):
        info = path / "seqinfo.ini"
        if path.name.startswith("_") or not info.is_file():
            continue
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(info, encoding="utf-8")
        length = int(config["Sequence"]["seqLength"])

        if not (path / "gt" / "gt.txt").is_file():
            print(f"  skip {path.name}: no gt/gt.txt yet ({length} frames extracted)")
            continue
        sequences[path.name] = length
    return sequences


def assign(sequences: dict[str, int], ratio: tuple[float, float, float]) -> dict[str, str]:
    """Greedily place whole sequences to approach the target share of FRAMES.

    Balancing frames rather than sequence count: clips differ in length, so an even
    split of clips can be a lopsided split of data. Longest first, each sequence going
    to whichever split is furthest below its target - the standard greedy approach, and
    with a handful of clips it lands close enough.
    """
    total = sum(sequences.values())
    targets = {s: total * r / sum(ratio) for s, r in zip(SPLITS, ratio)}
    current = {s: 0 for s in SPLITS}
    out = {}

    for name, length in sorted(sequences.items(), key=lambda kv: -kv[1]):
        pick = max(SPLITS, key=lambda s: targets[s] - current[s])
        out[name] = pick
        current[pick] += length
    return out


def verdict(sequences: dict[str, int], assignment: dict[str, str],
            ratio: tuple[float, float, float]) -> bool:
    """Print the split and check it for leakage. Returns False if it is unusable."""
    total = sum(sequences.values())
    print(f"\n{'sequence':28}{'frames':>9}  split")
    print("-" * 50)
    for name in sorted(sequences, key=lambda n: (assignment[n], n)):
        print(f"{name:28}{sequences[name]:9}  {assignment[name]}")
    print("-" * 50)

    ok = True
    print(f"\n{'split':8}{'seqs':>6}{'frames':>9}{'actual':>9}{'target':>9}")
    for split, want in zip(SPLITS, ratio):
        members = [n for n, s in assignment.items() if s == split]
        frames = sum(sequences[n] for n in members)
        share = 100.0 * frames / total if total else 0.0
        print(f"{split:8}{len(members):6}{frames:9}{share:8.1f}%{want:8.1f}%")
        if not members:
            print(f"         ERROR: {split} is empty.")
            ok = False
        elif len(members) < MIN_SEQS_PER_SPLIT:
            # Exactly the caveat already carried for detection, where val is one clip.
            print(f"         WARNING: only {len(members)} sequence(s) - results for "
                  f"{split} cannot estimate across-clip variance.")

    # Whole sequences are assigned, so leakage is structurally impossible. Assert it
    # anyway: this is the property the whole script exists to guarantee.
    overlap = [n for n in sequences if sum(assignment[n] == s for s in SPLITS) != 1]
    print(f"\nleakage: {'FAIL' if overlap else 'none'} - "
          f"every sequence appears in exactly one split")
    return ok and not overlap


def write_assignment(sequences: dict[str, int], assignment: dict[str, str]) -> None:
    ASSIGNMENT.parent.mkdir(parents=True, exist_ok=True)
    with ASSIGNMENT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sequence", "frames", "split"])
        for name in sorted(sequences):
            writer.writerow([name, sequences[name], assignment[name]])
    print(f"\nwrote {ASSIGNMENT}")


def read_assignment(sequences: dict[str, int]) -> dict[str, str]:
    if not ASSIGNMENT.is_file():
        sys.exit(f"Missing {ASSIGNMENT} - run without --from-assignment first.")
    with ASSIGNMENT.open(newline="", encoding="utf-8") as fh:
        assignment = {r["sequence"]: r["split"] for r in csv.DictReader(fh)}

    missing = sorted(set(sequences) - set(assignment))
    extra = sorted(set(assignment) - set(sequences))
    if missing:
        sys.exit(f"Sequences with no assignment: {', '.join(missing)}\n"
                 "Re-run without --from-assignment to reassign.")
    if extra:
        print(f"  note: {len(extra)} assigned sequence(s) not present locally: "
              f"{', '.join(extra)}")
    return assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ratio", type=float, nargs=3, default=list(DEFAULT_RATIO),
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--from-assignment", action="store_true",
                        help="Reuse the committed assignment.csv instead of reassigning.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequences = discover()
    if not sequences:
        sys.exit("No labelled sequences found. Label one with "
                 "scripts/label_tracks.py first.")

    ratio = tuple(args.ratio)
    if args.from_assignment:
        assignment = read_assignment(sequences)
    else:
        assignment = assign(sequences, ratio)

    if not verdict(sequences, assignment, ratio):
        sys.exit("\nSplit is unusable - see errors above.")
    if args.dry_run:
        print("\n(dry run - nothing written)")
        return
    if not args.from_assignment:
        write_assignment(sequences, assignment)


if __name__ == "__main__":
    main()
