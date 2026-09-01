"""Run a tracker over a MOT sequence and write MOT-format results.

One driver for every tracker in scope, so they are fed identical input and their output
is scored by one evaluator:

    classical   scripts/classical_tracker.py   Kalman + greedy association
    sort        scripts/sort.py                Kalman + Hungarian association
    goturn      cv2.TrackerGOTURN              pretrained CNN, single-object
    vit         cv2.TrackerVit                 pretrained transformer, single-object

TWO INPUT MODES
    --gt-fed      every tracker receives the SAME ground-truth boxes. This is the
                  primary result: no detector sits in the loop, so differences between
                  trackers are differences in tracking, and the identity of whoever
                  pre-annotated the labels stops mattering.
    --detector    end-to-end, detector feeding tracker. Reported as secondary, since it
                  confounds detection quality with tracking quality.

SINGLE-OBJECT TRACKERS DOING MULTI-OBJECT WORK
    GOTURN and VitTrack each follow ONE target. Multi-object tracking with them means
    running an instance per target and managing births and deaths around them, which is
    what CvMultiTracker below does. That is a real disadvantage for them - cost grows
    linearly with the number of robots, where the Kalman trackers pay almost nothing per
    extra target - and the per-frame timings should be read with that in mind rather
    than treated as an implementation detail.

Usage:
    python scripts/run_trackers.py --seq data/tracking/arc04 --tracker sort --gt-fed
    python scripts/run_trackers.py --seq data/tracking/arc04 --tracker classical \\
        --detector runs/detect/yolo_960/weights/best.pt
"""

from __future__ import annotations

import argparse
import collections
import configparser
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

TRACKERS = ("classical", "sort", "goturn", "vit")
DEFAULT_OUT = ROOT / "results" / "tracking"


def read_seqinfo(seq: Path) -> dict:
    path = seq / "seqinfo.ini"
    if not path.is_file():
        sys.exit(f"Missing {path}\nRun: python scripts/fetch_clips.py")
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(path, encoding="utf-8")
    return dict(config["Sequence"])


def read_gt(seq: Path, override: Path | None = None):
    """frame -> [(x, y, w, h)] from gt/gt.txt, or from `override`."""
    path = override or (seq / "gt" / "gt.txt")
    if not path.is_file():
        sys.exit(f"Missing {path}\n--gt-fed needs labels; run scripts/label_tracks.py")
    frames = collections.defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        frames[int(parts[0])].append(tuple(float(v) for v in parts[2:6]))
    return frames


class CvMultiTracker:
    """Multi-object wrapper around OpenCV's single-object trackers.

    Keeps one cv2 tracker per target. Each frame every instance is stepped forward, then
    detections are associated to the results by IoU; unmatched detections start new
    instances and instances that fail or go unmatched too long are dropped.
    """

    def __init__(self, kind: str, iou_gate=0.2, max_age=10, min_hits=2):
        from fetch_tracker_weights import goturn_params, vit_params  # noqa: PLC0415

        self.kind = kind
        self.params = goturn_params() if kind == "goturn" else vit_params()
        self.iou_gate, self.max_age, self.min_hits = iou_gate, max_age, min_hits
        self.entries = []          # [tracker, id, box, misses, hits]
        self._next_id = 1

    def _make(self):
        import cv2

        return (cv2.TrackerGOTURN.create(self.params) if self.kind == "goturn"
                else cv2.TrackerVit.create(self.params))

    def update(self, frame, detections):
        from classical_tracker import iou  # noqa: PLC0415

        alive = []
        for entry in self.entries:
            tracker, tid, box, misses, hits = entry
            ok, new = tracker.update(frame)
            if ok:
                alive.append([tracker, tid, tuple(float(v) for v in new), misses, hits])
        self.entries = alive

        used = set()
        for entry in self.entries:
            best, best_i = 0.0, None
            for di, det in enumerate(detections):
                if di in used:
                    continue
                score = iou(entry[2], det)
                if score > best:
                    best, best_i = score, di
            if best >= self.iou_gate:
                used.add(best_i)
                # Re-init on the detection: these trackers drift, and a detection is a
                # better estimate than an extrapolation.
                entry[0] = self._make()
                entry[0].init(frame, tuple(int(v) for v in detections[best_i]))
                entry[2] = detections[best_i]
                entry[3] = 0
                entry[4] += 1
            else:
                entry[3] += 1

        for di, det in enumerate(detections):
            if di in used:
                continue
            tracker = self._make()
            tracker.init(frame, tuple(int(v) for v in det))
            self.entries.append([tracker, self._next_id, det, 0, 1])
            self._next_id += 1

        self.entries = [e for e in self.entries if e[3] <= self.max_age]
        return [(e[1], e[2]) for e in self.entries
                if e[4] >= self.min_hits and e[3] == 0]


def build(kind: str):
    if kind == "classical":
        from classical_tracker import ClassicalTracker  # noqa: PLC0415

        return ClassicalTracker(), False
    if kind == "sort":
        from sort import Sort  # noqa: PLC0415

        return Sort(), False
    return CvMultiTracker(kind), True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seq", type=Path, required=True)
    parser.add_argument("--tracker", choices=TRACKERS, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--gt-fed", action="store_true",
                        help="Feed ground-truth boxes. The primary protocol.")
    source.add_argument("--detector", type=Path, help="YOLO weights for end-to-end.")
    parser.add_argument("--gt", type=Path, default=None,
                        help="Ground-truth file, if not seq/gt/gt.txt.")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--raw-boxes", action="store_true",
                        help="Skip the per-frame cleanup the labeller applied. "
                             "Measures detection noise, not tracking.")
    parser.add_argument("--device", default="0", help="Detector device; not a benchmark.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    import cv2

    args = parse_args()
    info = read_seqinfo(args.seq)
    images = sorted((args.seq / info["imDir"]).glob("*" + info["imExt"]))
    if args.limit:
        images = images[:args.limit]
    if not images:
        sys.exit(f"No frames in {args.seq / info['imDir']}")

    gt = read_gt(args.seq, args.gt) if args.gt_fed else None
    model = None
    if args.detector:
        from ultralytics import YOLO  # noqa: PLC0415

        model = YOLO(str(args.detector))

    box_median = 120.0
    clean_frame = None
    if args.detector and not args.raw_boxes:
        from auto_label import clean_frame  # noqa: PLC0415
        # Same size envelope the labeller used, derived from the frame width so
        # it tracks the sequence rather than being hardcoded per clip.
        box_median = float(info.get("imWidth", 1280)) / 11.0

    tracker, needs_frame = build(args.tracker)
    mode = "gt-fed" if args.gt_fed else f"detector={args.detector.name}"
    print(f"sequence : {args.seq.name}  ({len(images)} frames)")
    print(f"tracker  : {args.tracker}   input: {mode}")

    rows = []
    for index, path in enumerate(images):
        frame_no = index + 1
        frame = cv2.imread(str(path))
        if frame is None:
            continue

        if gt is not None:
            detections = gt.get(frame_no, [])
        else:
            result = model.predict(str(path), imgsz=args.imgsz, conf=args.conf,
                                   device=args.device, verbose=False)[0]
            detections = []
            for box in result.boxes:
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                detections.append((x1, y1, x2 - x1, y2 - y1))
            if not args.raw_boxes:
                # The reference was built from CLEANED detections, so a tracker fed
                # raw ones is scored against a different world: on arc04 that gave
                # MOTA -0.89 and 99 tracks against the reference 9, dominated by
                # duplicate and nested boxes the labeller had already resolved.
                # Applying the identical cleanup leaves online-vs-offline association
                # as the only difference, which is the thing being compared.
                detections = clean_frame(detections, median=box_median)

        outputs = (tracker.update(frame, detections) if needs_frame
                   else tracker.update(detections))
        for tid, (x, y, w, h) in outputs:
            rows.append(f"{frame_no},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
        print(f"\r  {frame_no}/{len(images)} frames, {len(rows)} rows", end="")
    print()

    out_dir = args.out or (DEFAULT_OUT / args.seq.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "gt" if args.gt_fed else "det"
    out_path = out_dir / f"{args.tracker}_{suffix}.txt"
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    ids = len({r.split(",")[1] for r in rows})
    print(f"\n[done]  {out_path}  ({len(rows)} rows, {ids} track ids)")
    print(f"\nScore it:\n    python scripts/eval_tracking.py --seq {args.seq} "
          f"--results {out_path} --name {args.tracker}_{suffix}")


if __name__ == "__main__":
    main()
