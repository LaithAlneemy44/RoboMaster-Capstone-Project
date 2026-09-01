"""Seed a sequence with detector-proposed tracks so labelling is correction, not drawing.

Hand-drawing all seven clips is roughly 15 hours. This runs the trained detector over a
sequence, links its boxes into tracks, and writes the .labelstate.json that
scripts/label_tracks.py opens - turning the job into reviewing and fixing, which is
several times faster.

WHAT THIS COSTS, STATED PLAINLY
    The proposing detector is yolo_960, which is also one of the models under
    comparison. A box a human accepts unchanged makes that detector's localisation into
    ground truth, which flatters it. Three things keep it honest:

      1. The primary tracking result is GT-FED - every tracker receives the same
         ground-truth boxes, so no detector sits in that loop and the proposer stops
         mattering for the headline comparison.
      2. Detection accuracy is measured separately on ROCO, whose ground truth was never
         near a model of ours.
      3. Every frame still requires human confirmation; nothing is written as verified.

    The end-to-end detector+tracker number is the one carrying a caveat, and the write-up
    should say so. Plan verification step 2 measures the residual bias rather than
    asserting it is small.

WHY A LOW CONFIDENCE THRESHOLD
    Deliberately over-proposing. Deleting a spurious box is a visible action a reviewer
    takes; failing to notice an object that was never proposed is invisible, and missing
    objects are the failure mode that silently corrupts ground truth - every detector is
    then penalised for finding something that is really there.

WHY SPARSE KEYFRAMES
    Only every Nth detection is kept. Writing every frame would make the ground truth
    literally the detector output, jitter included, and leave 300 dense boxes per clip to
    check. Keyframes let interpolate() smooth between them and cut the review to roughly
    30 points per track.

Usage:
    python scripts/pre_annotate.py --seq data/tracking/arc04
    python scripts/pre_annotate.py --seq data/tracking/arc02 --conf 0.10 --force
"""

from __future__ import annotations

import argparse
import configparser
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "yolo_960" / "weights" / "best.pt"


def read_seqinfo(seq: Path) -> dict:
    path = seq / "seqinfo.ini"
    if not path.is_file():
        sys.exit(f"Missing {path}\nRun: python scripts/fetch_clips.py")
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(path, encoding="utf-8")
    return dict(config["Sequence"])


def detect(frames, weights: Path, imgsz: int, conf: float, device: str, target: str):
    """Per-frame list of xywh boxes for the target class."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = dict(model.names.items())
    if target not in names.values():
        sys.exit(f"Model has no {target!r} class: {sorted(names.values())}")

    per_frame = []
    for index, path in enumerate(frames):
        result = model.predict(str(path), imgsz=imgsz, conf=conf, device=device,
                               verbose=False)[0]
        boxes = []
        for box in result.boxes:
            if names[int(box.cls[0])] != target:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            boxes.append([x1, y1, x2 - x1, y2 - y1])
        per_frame.append(boxes)
        print(f"\r[detect] {index + 1}/{len(frames)}", end="", flush=True)
    print()
    return per_frame


def associate(per_frame, iou_thresh: float, max_age: int):
    """Greedy IoU linking of per-frame boxes into tracks. Frame numbers are 1-based.

    This is SORT-like, which would flatter SORT if it were the last word - it is not.
    The frames where linking fails are exactly the identity switches the metric counts,
    and those are the frames a human corrects. What survives review is human-decided.
    """
    from verify_tracking_data import iou  # noqa: PLC0415

    tracks: dict[int, dict[int, list]] = {}
    last_box: dict[int, list] = {}
    last_seen: dict[int, int] = {}
    velocity: dict[int, tuple[float, float]] = {}
    next_id = 1

    def predict(tid: int, frame: int) -> list[float]:
        """Where the track should be now, given where it was going.

        Matching a stale box against a detection several frames later fails once the
        robot has moved, which splits one robot into a string of short tracks - the
        first attempt produced 33 tracks for about 5 robots. Carrying the last observed
        velocity forward keeps the match alive across a dropout.
        """
        box = last_box[tid]
        gap = frame - last_seen[tid]
        vx, vy = velocity.get(tid, (0.0, 0.0))
        return [box[0] + vx * gap, box[1] + vy * gap, box[2], box[3]]

    for offset, boxes in enumerate(per_frame):
        frame = offset + 1
        live = [t for t, seen in last_seen.items() if frame - seen <= max_age]

        pairs = sorted(
            ((iou(predict(t, frame), b), t, i) for t in live for i, b in enumerate(boxes)),
            reverse=True,
        )
        used_tracks, used_boxes = set(), set()
        for score, tid, index in pairs:
            if score < iou_thresh:
                break
            if tid in used_tracks or index in used_boxes:
                continue
            used_tracks.add(tid)
            used_boxes.add(index)
            new = boxes[index]
            gap = frame - last_seen[tid]
            # Exponential smoothing, so one noisy detection does not throw the
            # prediction off for every frame after it.
            vx = (new[0] - last_box[tid][0]) / gap
            vy = (new[1] - last_box[tid][1]) / gap
            old_vx, old_vy = velocity.get(tid, (vx, vy))
            velocity[tid] = (0.5 * vx + 0.5 * old_vx, 0.5 * vy + 0.5 * old_vy)

            tracks[tid][frame] = new
            last_box[tid], last_seen[tid] = new, frame

        for index, box in enumerate(boxes):
            if index in used_boxes:
                continue
            tracks[next_id] = {frame: box}
            last_box[next_id], last_seen[next_id] = box, frame
            next_id += 1
    return tracks


def to_keyframes(tracks, stride: int, min_len: int):
    """Keep every Nth box per track, always including its first and last frame."""
    out = {}
    for tid, boxes in tracks.items():
        frames = sorted(boxes)
        if len(frames) < min_len:
            continue  # a two-frame flicker is noise, not a track worth reviewing
        keep = {f for i, f in enumerate(frames) if i % stride == 0}
        keep.add(frames[0])
        keep.add(frames[-1])
        out[tid] = {f: boxes[f] for f in sorted(keep)}
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seq", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.15,
                        help="Low on purpose - over-propose. See module docstring.")
    parser.add_argument("--class", dest="target", default="car")
    parser.add_argument("--keyframe-stride", type=int, default=10)
    # Tuned on arc04: 0.30/5 gave 35 tracks for roughly 8-12 real robots, 0.15/30
    # gives 24. Residual fragments are cheaper to fix now that label_tracks.py can
    # merge two tracks with one keypress.
    parser.add_argument("--iou-thresh", type=float, default=0.15)
    parser.add_argument("--max-age", type=int, default=30,
                        help="Frames a track may go unmatched before it is closed.")
    parser.add_argument("--min-len", type=int, default=5)
    parser.add_argument("--device", default="0", help="Analysis, not a benchmark: GPU.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_path = args.seq / ".labelstate.json"
    if state_path.is_file() and not args.force:
        sys.exit(f"{state_path} already exists - refusing to discard existing labels.\n"
                 "Pass --force only if you are sure that work can be lost.")
    if (args.seq / "gt" / "gt.txt").is_file() and not args.force:
        sys.exit(f"{args.seq} already has human labels in gt/gt.txt. Refusing.")

    info = read_seqinfo(args.seq)
    image_dir = args.seq / info["imDir"]
    frames = sorted(image_dir.glob("*" + info["imExt"]))
    if not frames:
        sys.exit(f"No frames in {image_dir}")

    print(f"sequence : {args.seq.name}  ({len(frames)} frames)")
    print(f"proposer : {args.weights.name} at {args.imgsz}px, conf {args.conf}")
    per_frame = detect(frames, args.weights, args.imgsz, args.conf, args.device,
                       args.target)

    raw = associate(per_frame, args.iou_thresh, args.max_age)
    tracks = to_keyframes(raw, args.keyframe_stride, args.min_len)
    dropped = len(raw) - len(tracks)

    # Written directly rather than through label_tracks.save_state, which would also
    # emit gt/gt.txt. That file should mean "a human produced this", and nothing here
    # has been reviewed yet.
    state = {
        "class": args.target,
        "tracks": {str(t): {str(f): b for f, b in kf.items()}
                   for t, kf in tracks.items()},
        "verified": [],  # every frame still needs human confirmation
    }
    state_path.write_text(json.dumps(state, indent=1), encoding="utf-8")

    total_boxes = sum(len(b) for b in per_frame)
    keyframes = sum(len(kf) for kf in tracks.values())
    print(f"\ndetections : {total_boxes} across {len(frames)} frames")
    print(f"tracks     : {len(tracks)} kept, {dropped} dropped under "
          f"--min-len {args.min_len}")
    print(f"keyframes  : {keyframes} to review (every {args.keyframe_stride}th box)")
    print(f"wrote      : {state_path}")
    print("\nCorrect it - nothing is marked verified yet:")
    print(f"    python scripts/label_tracks.py --seq {args.seq}")


if __name__ == "__main__":
    main()
