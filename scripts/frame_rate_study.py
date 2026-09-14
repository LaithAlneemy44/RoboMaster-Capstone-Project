"""How much accuracy do the trackers lose below 30 fps, and does rescaling recover it?

THE PROBLEM, FROM CAMERA_HANDOFF ISSUE 5
    Every tracker here counts FRAMES, not seconds. `max_age` and `min_hits` are frame
    counts, and both Kalman filters hardcode a transition matrix with dt = 1 frame, so
    velocity is in pixels PER FRAME. All of it was set for 30 fps footage.

    The deployed pipeline does not run at 30 fps. It runs at 6.31 FPS for fast_960 + sort
    and 24.66 for fast_640 + sort. At 10 fps, max_age=10 stops meaning "a third of a
    second" and starts meaning "a full second" - long enough to hold a track through a
    real disappearance. Nothing had ever measured the consequence.

THE TWO ARMS, AND WHY BOTH ARE NEEDED
    naive     stride the frames, keep the 30 fps constants. This is what deploying the
              current code at a lower frame rate actually does.
    rescaled  stride the frames AND divide max_age/min_hits by the stride, so they cover
              the same DURATION as before.

    Reporting only the naive arm would overstate the problem: some of the loss is a
    misconfiguration that costs nothing to fix. Reporting only the rescaled arm would
    understate it: at one frame in N the detector supplies N times less evidence, and no
    parameter change returns information that was never sampled. The gap between the arms
    separates the two.

WHY THIS IS CHEAP
    Detections are cached once per clip by tune_classical_tracker.cached_detections and
    replayed for every (step, arm, tracker) combination. The detector runs zero times if
    the cache is warm - the whole study is tracker arithmetic.

VAL CLIP ONLY
    arc03 is the validation clip named in data/tracking/assignment.csv, the same one the
    tracker sweep used. arc04 stays untouched for the final report.

Usage:
    python scripts/frame_rate_study.py
    python scripts/frame_rate_study.py --steps 1 2 3 5 --trackers classical sort
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

PY = sys.executable
OUT_DIR = ROOT / "results" / "tracking" / "framerate"
RESULTS = ROOT / "results" / "frame_rate.csv"
DEFAULT_SEQ = ROOT / "data" / "tracking" / "arc03"
DEFAULT_DET = ROOT / "runs" / "detect" / "yolo_960" / "weights" / "best.pt"


def run_arm(seq: Path, detections: list[list], tracker_kind: str, step: int,
            rescale: bool, source_fps: float) -> Path:
    """Replay cached detections at a stride and write a MOT file. Returns its path."""
    from run_trackers import build

    tracker, needs_frame = build(tracker_kind, "default", step, rescale)
    if needs_frame:
        sys.exit(f"{tracker_kind} needs image frames; this study replays cached boxes "
                 f"only. Use classical or sort.")

    arm = "rescaled" if rescale else "naive"
    name = f"{tracker_kind}_{arm}_step{step}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.txt"

    rows = []
    # Original frame numbering, so ground truth lines up and eval_tracking --step can
    # filter both sides to the same subset.
    for index in range(0, len(detections), step):
        boxes = [tuple(b) for b in detections[index]]
        for tid, (x, y, w, h) in tracker.update(boxes):
            rows.append(f"{index + 1},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,-1,-1,-1")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def score(seq: Path, results_path: Path, name: str, step: int) -> None:
    done = subprocess.run(
        [PY, "scripts/eval_tracking.py", "--seq", str(seq), "--results", str(results_path),
         "--name", name, "--step", str(step), "--csv", str(RESULTS), "--bootstrap", "200"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if done.returncode != 0:
        print(done.stdout[-2000:])
        print(done.stderr[-2000:])
        sys.exit(f"scoring {name} failed")


def summarise(source_fps: float) -> None:
    if not RESULTS.is_file():
        print("nothing scored")
        return
    with RESULTS.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if "_step" in r["name"]]
    if not rows:
        return

    by = {}
    for r in rows:
        kind, arm, step = r["name"].rsplit("_", 2)
        by[(kind, arm, int(step.replace("step", "")))] = r

    steps = sorted({k[2] for k in by})
    print(f"\n{'tracker':<11}{'arm':<10}{'step':>5}{'fps':>7}{'MOTA':>9}"
          f"{'IDF1':>9}{'switches':>10}")
    for kind in sorted({k[0] for k in by}):
        for arm in ("naive", "rescaled"):
            for step in steps:
                r = by.get((kind, arm, step))
                if not r:
                    continue
                print(f"{kind:<11}{arm:<10}{step:>5}{source_fps / step:>7.1f}"
                      f"{float(r['mota']):>9.4f}{float(r['idf1']):>9.4f}"
                      f"{r['id_switches']:>10}")

    print("\nThe naive arm is what deploying today's constants at a lower frame rate\n"
          "actually does. The gap to the rescaled arm is the part that is a\n"
          "misconfiguration rather than lost information.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq", type=Path, default=DEFAULT_SEQ)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DET)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3, 5],
                        help="Strides to test. 1 is the full-rate baseline.")
    parser.add_argument("--trackers", nargs="+", default=["classical", "sort"],
                        choices=("classical", "sort"))
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    from run_trackers import read_seqinfo
    from tune_classical_tracker import cached_detections

    info = read_seqinfo(args.seq)
    source_fps = float(info.get("frameRate", 30.0))

    if args.summary:
        summarise(source_fps)
        return

    detections = cached_detections(args.seq, args.detector, "yolo", None,
                                   args.imgsz, args.conf, args.device)
    print(f"clip     : {args.seq.name}  {len(detections)} frames @ {source_fps:.2f} fps")

    for kind in args.trackers:
        for step in args.steps:
            # At step 1 the two arms are identical by construction, so run one.
            arms = [False] if step == 1 else [False, True]
            for rescale in arms:
                arm = "rescaled" if rescale else "naive"
                name = f"{kind}_{arm}_step{step}"
                print(f"\n--- {name}  ({source_fps / step:.1f} fps) ---")
                path = run_arm(args.seq, detections, kind, step, rescale, source_fps)
                score(args.seq, path, name, step)

    summarise(source_fps)


if __name__ == "__main__":
    main()
