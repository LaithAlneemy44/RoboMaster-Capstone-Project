"""Hand-label tracks on a MOT-layout sequence and write gt/gt.txt.

No labelled RoboMaster tracking dataset exists, so this is the tool that creates one.
CLAUDE.md calls that dataset the biggest hidden cost in the project, which makes the
labelling rate the thing worth optimising here.

KEYFRAMES, NOT TRACKER PROPAGATION
    Boxes are set on keyframes and linearly interpolated between them, and every frame
    is stepped through and confirmed by a human. It is tempting to propagate boxes with
    one of the OpenCV trackers instead - far less clicking - but the trackers under
    evaluation must not label the data that evaluates them, or each one is scored partly
    on its own output.

    Interpolation is not perfectly neutral either: constant velocity between keyframes
    is the same assumption the classical Kalman tracker makes, which would flatter it.
    The human confirmation pass is what keeps that honest - interpolation only seeds a
    box, a person accepts it. The write-up should describe the procedure plainly.

    Frames confirmed by a human are recorded in .labelstate.json, so the coverage claim
    is auditable rather than asserted.

OUTPUT
    gt/gt.txt in MOT Challenge format:
        frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
    1-based frame numbers, matching img1/%06d.jpg. motmetrics reads this directly, SORT
    consumes it natively, and a single-object GOTURN sequence is one id filtered out.

CONTROLS
    d / a or arrows  next / previous frame        n     new track (drag to draw)
    space            confirm frame                e     edit selected box (drag)
    tab              cycle selected track         x     delete selected box here
    s                save                         X     delete whole track
    q                save and quit                g     jump to frame

Usage:
    python scripts/label_tracks.py --seq data/tracking/arc01
    python scripts/label_tracks.py --selftest        # logic only, no GUI
"""

from __future__ import annotations

import argparse
import configparser
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# MOT reserves conf=1 for a real detection and visibility=1 for fully visible. These are
# hand-drawn ground truth, so both are constant.
GT_CONF = 1
GT_VISIBILITY = 1
DEFAULT_CLASS = "car"


# --------------------------------------------------------------------------- logic
# Kept free of cv2 so it can be tested without a display - see --selftest.

def interpolate(keyframes: dict[int, list[float]], frame: int) -> list[float] | None:
    """Box for `frame` from a track's keyframes, or None outside its span.

    Linear in all four coordinates. Outside the first/last keyframe the track simply
    does not exist - deliberately not extrapolated, since an invented box beyond where
    a human looked would be ground truth nobody verified.
    """
    if not keyframes:
        return None
    if frame in keyframes:
        return list(keyframes[frame])

    marks = sorted(keyframes)
    if frame < marks[0] or frame > marks[-1]:
        return None

    before = max(m for m in marks if m < frame)
    after = min(m for m in marks if m > frame)
    span = after - before
    t = (frame - before) / span
    lo, hi = keyframes[before], keyframes[after]
    return [lo[i] + (hi[i] - lo[i]) * t for i in range(4)]


def to_mot_rows(tracks: dict[int, dict[int, list[float]]], cls: str) -> list[str]:
    """Expand keyframes into one MOT row per frame per track, sorted as MOT expects."""
    rows = []
    for tid, keyframes in tracks.items():
        if not keyframes:
            continue
        marks = sorted(keyframes)
        for frame in range(marks[0], marks[-1] + 1):
            box = interpolate(keyframes, frame)
            if box is None:
                continue
            x, y, w, h = (round(v, 2) for v in box)
            rows.append((frame, tid, f"{frame},{tid},{x},{y},{w},{h},"
                                     f"{GT_CONF},{cls},{GT_VISIBILITY}"))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [r[2] for r in rows]


def save_state(seq: Path, tracks: dict, verified: set[int], cls: str) -> Path:
    """Write gt/gt.txt plus the resumable keyframe state beside it."""
    gt_dir = seq / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_path = gt_dir / "gt.txt"
    gt_path.write_text("\n".join(to_mot_rows(tracks, cls)) + "\n", encoding="utf-8")

    state = {
        "class": cls,
        # json object keys must be strings; ints are restored on load.
        "tracks": {str(t): {str(f): b for f, b in kf.items()}
                   for t, kf in tracks.items()},
        "verified": sorted(verified),
    }
    (seq / ".labelstate.json").write_text(json.dumps(state, indent=1), encoding="utf-8")
    return gt_path


def load_state(seq: Path) -> tuple[dict, set[int], str]:
    path = seq / ".labelstate.json"
    if not path.is_file():
        return {}, set(), DEFAULT_CLASS
    state = json.loads(path.read_text(encoding="utf-8"))
    tracks = {int(t): {int(f): list(b) for f, b in kf.items()}
              for t, kf in state["tracks"].items()}
    return tracks, set(state.get("verified", [])), state.get("class", DEFAULT_CLASS)


def read_seqinfo(seq: Path) -> dict:
    path = seq / "seqinfo.ini"
    if not path.is_file():
        sys.exit(f"Missing {path}\nRun: python scripts/fetch_clips.py")
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(path, encoding="utf-8")
    return dict(config["Sequence"])


# ------------------------------------------------------------------------- selftest

def selftest() -> None:
    """Exercise interpolation and the save/load round-trip without a display.

    The motmetrics check is the one that matters: scoring the written file against
    itself must give a perfect MOTA, which proves the format actually parses as MOT
    rather than merely looking like it.
    """
    print("[test] interpolation")
    kf = {1: [0.0, 0.0, 10.0, 10.0], 5: [40.0, 20.0, 10.0, 10.0]}
    mid = interpolate(kf, 3)
    assert mid == [20.0, 10.0, 10.0, 10.0], mid
    assert interpolate(kf, 1) == [0.0, 0.0, 10.0, 10.0]
    assert interpolate(kf, 6) is None, "must not extrapolate past the last keyframe"
    assert interpolate({}, 1) is None
    print("       ok - midpoint, endpoint, no extrapolation")

    print("[test] round-trip through gt.txt")
    with tempfile.TemporaryDirectory() as tmp:
        seq = Path(tmp) / "seq01"
        seq.mkdir()
        tracks = {1: dict(kf), 2: {2: [5.0, 5.0, 8.0, 8.0], 4: [9.0, 5.0, 8.0, 8.0]}}
        gt_path = save_state(seq, tracks, {1, 2, 3}, "car")
        rows = gt_path.read_text(encoding="utf-8").strip().splitlines()
        # track 1 spans frames 1-5, track 2 spans 2-4
        assert len(rows) == 5 + 3, len(rows)
        assert rows[0].startswith("1,1,"), rows[0]

        back, verified, cls = load_state(seq)
        assert back == tracks, back
        assert verified == {1, 2, 3}
        assert cls == "car"
        print(f"       ok - {len(rows)} rows, keyframes and verified set survive")

        print("[test] motmetrics parses it as MOT")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from mot_compat import mm  # noqa: PLC0415 - NumPy 2 shim, see that module

        gt = mm.io.loadtxt(str(gt_path), fmt="mot15-2D")
        acc = mm.utils.compare_to_groundtruth(gt, gt, "iou", distth=0.5)
        summary = mm.metrics.create().compute(acc, metrics=["mota", "num_switches"])
        mota = float(summary["mota"].iloc[0])
        switches = int(summary["num_switches"].iloc[0])
        assert mota == 1.0, f"MOTA {mota} scoring a file against itself"
        assert switches == 0, switches
        print(f"       ok - MOTA {mota:.1f}, {switches} id switches against itself")

    print("\nAll logic checks passed.")


# ------------------------------------------------------------------------------ gui

def run_gui(seq: Path, cls: str) -> None:
    import cv2

    info = read_seqinfo(seq)
    images = sorted((seq / info["imDir"]).glob(f"*{info['imExt']}"))
    if not images:
        sys.exit(f"No frames in {seq / info['imDir']}")

    tracks, verified, cls = load_state(seq) or ({}, set(), cls)
    state = {"frame": 0, "selected": None, "drag": None, "mode": None}

    def current_boxes(index: int):
        out = {}
        for tid, kf in tracks.items():
            box = interpolate(kf, index + 1)  # gt.txt is 1-based
            if box is not None:
                out[tid] = (box, (index + 1) in kf)
        return out

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and state["mode"] in ("new", "edit"):
            state["drag"] = [x, y, x, y]
        elif event == cv2.EVENT_MOUSEMOVE and state["drag"]:
            state["drag"][2:] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP and state["drag"]:
            x0, y0, x1, y1 = state["drag"]
            box = [float(min(x0, x1)), float(min(y0, y1)),
                   float(abs(x1 - x0)), float(abs(y1 - y0))]
            state["drag"] = None
            if box[2] < 3 or box[3] < 3:
                return
            if state["mode"] == "new":
                tid = max(tracks, default=0) + 1
                tracks[tid] = {}
                state["selected"] = tid
            tid = state["selected"]
            if tid is not None:
                tracks[tid][state["frame"] + 1] = box
            state["mode"] = None

    window = f"label_tracks - {seq.name}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    print(__doc__.split("CONTROLS")[1].split("Usage:")[0])

    while True:
        index = state["frame"]
        canvas = cv2.imread(str(images[index]))
        for tid, (box, is_key) in current_boxes(index).items():
            x, y, w, h = (int(v) for v in box)
            chosen = tid == state["selected"]
            colour = (0, 255, 0) if is_key else (0, 200, 255)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, 3 if chosen else 1)
            cv2.putText(canvas, f"#{tid}{'*' if is_key else ''}", (x, max(12, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        if state["drag"]:
            x0, y0, x1, y1 = state["drag"]
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (255, 255, 0), 1)

        done = "OK" if (index + 1) in verified else "--"
        cv2.putText(canvas, f"{index + 1}/{len(images)} [{done}] tracks:{len(tracks)} "
                            f"sel:{state['selected']} mode:{state['mode'] or '-'}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(window, canvas)

        key = cv2.waitKey(20) & 0xFF
        if key in (ord("d"), 83):
            state["frame"] = min(index + 1, len(images) - 1)
        elif key in (ord("a"), 81):
            state["frame"] = max(index - 1, 0)
        elif key == ord(" "):
            verified.add(index + 1)
            state["frame"] = min(index + 1, len(images) - 1)
        elif key == ord("n"):
            state["mode"] = "new"
        elif key == ord("e"):
            state["mode"] = "edit"
        elif key == 9:  # tab
            ids = sorted(tracks)
            if ids:
                nxt = 0 if state["selected"] not in ids \
                    else (ids.index(state["selected"]) + 1) % len(ids)
                state["selected"] = ids[nxt]
        elif key == ord("x") and state["selected"] in tracks:
            tracks[state["selected"]].pop(index + 1, None)
        elif key == ord("X") and state["selected"] in tracks:
            tracks.pop(state["selected"])
            state["selected"] = None
        elif key == ord("g"):
            try:
                state["frame"] = max(0, min(len(images) - 1,
                                            int(input("jump to frame: ")) - 1))
            except ValueError:
                pass
        elif key in (ord("s"), ord("q")):
            path = save_state(seq, tracks, verified, cls)
            print(f"[save]  {path}  ({len(tracks)} tracks, "
                  f"{len(verified)}/{len(images)} frames confirmed)")
            if key == ord("q"):
                break
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seq", type=Path, help="Sequence dir, e.g. data/tracking/arc01")
    parser.add_argument("--class", dest="cls", default=DEFAULT_CLASS)
    parser.add_argument("--selftest", action="store_true", help="Logic only, no GUI.")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.seq:
        parser.error("--seq is required unless --selftest is given")
    run_gui(args.seq, args.cls)


if __name__ == "__main__":
    main()
