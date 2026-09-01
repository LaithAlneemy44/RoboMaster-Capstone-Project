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
    h                mark absent here (occluded / left frame)
    m                merge selected track INTO another (prompts for its id)
    s                save                         X     delete whole track
    q                save and quit                g     jump to frame

Usage:
    python scripts/label_tracks.py --seq data/tracking/arc01 --tracks   # plan the merges
    python scripts/label_tracks.py --seq data/tracking/arc01
    python scripts/label_tracks.py --selftest        # logic only, no GUI
"""

from __future__ import annotations

import argparse
import configparser
import json
import statistics
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

def interpolate(keyframes: dict[int, list[float] | None], frame: int) -> list[float] | None:
    """Box for `frame` from a track's keyframes, or None where the track is absent.

    Linear in all four coordinates. Two ways to get None:

      Outside the first/last keyframe - deliberately not extrapolated, since an
      invented box beyond where a human looked would be ground truth nobody verified.

      Across an ABSENCE marker (a keyframe whose value is None). Robots leave frame and
      come back, and plain interpolation would fill the gap with boxes for an object
      that is not there - fabricated ground truth that every detector is then penalised
      for missing. Marking the exit and the return breaks the span instead.
    """
    if not keyframes:
        return None
    if frame in keyframes:
        marked = keyframes[frame]
        return list(marked) if marked is not None else None

    marks = sorted(keyframes)
    if frame < marks[0] or frame > marks[-1]:
        return None

    before = max(m for m in marks if m < frame)
    after = min(m for m in marks if m > frame)
    lo, hi = keyframes[before], keyframes[after]
    if lo is None or hi is None:
        return None  # the track is absent across this span

    t = (frame - before) / (after - before)
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
    # `b is None` is an absence marker, not a box - list(None) would raise, so a
    # sequence could not be reopened after marking any robot as having left frame.
    tracks = {int(t): {int(f): (list(b) if b is not None else None)
                       for f, b in kf.items()}
              for t, kf in state["tracks"].items()}
    return tracks, set(state.get("verified", [])), state.get("class", DEFAULT_CLASS)


def report_tracks(seq: Path, max_gap: int, max_drift: float) -> None:
    """List tracks and flag likely fragments, so the merge pass is not done blind.

    Pre-annotation splits one robot into several tracks whenever the detector loses it.
    Those splits are what make the identity metrics meaningless if left in, and tabbing
    through 40 tracks in the GUI hunting for them is slow. This finds the candidates;
    a human still decides, because identity IS the measurement here and auto-merging
    would put an algorithm back in charge of the thing being measured.
    """
    tracks, verified, _ = load_state(seq)
    if not tracks:
        sys.exit(f"No .labelstate.json in {seq} - run scripts/pre_annotate.py first.")

    spans = {t: (min(kf), max(kf)) for t, kf in tracks.items()}
    print(f"{seq.name}: {len(tracks)} tracks, {len(verified)} frames confirmed\n")
    print(f"{'track':>6}{'first':>7}{'last':>7}{'span':>7}{'keyframes':>11}")
    for tid in sorted(tracks, key=lambda t: spans[t][0]):
        first, last = spans[tid]
        print(f"{tid:6}{first:7}{last:7}{last - first + 1:7}{len(tracks[tid]):11}")

    def box_at(tid, frame):
        return interpolate(tracks[tid], frame)

    candidates = []
    for a in tracks:
        for b in tracks:
            if a == b:
                continue
            gap = spans[b][0] - spans[a][1]
            if not 0 <= gap <= max_gap:
                continue
            end, start = box_at(a, spans[a][1]), box_at(b, spans[b][0])
            if end is None or start is None:
                continue
            size = max(1.0, (end[2] * end[3]) ** 0.5)
            drift = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5 / size
            if drift <= max_drift:
                candidates.append((drift, a, b, gap))

    print(f"\nSEQUENTIAL fragments - one robot lost and re-found "
          f"(gap <= {max_gap} frames, drift <= {max_drift:.1f}x size):")
    if candidates:
        for drift, a, b, gap in sorted(candidates):
            verdict = ("merge it" if drift < 0.3 else
                       "look first" if drift < 0.7 else "skip - probably different")
            print(f"  #{a:<3} ends frame {spans[a][1]:>4}  ->  "
                  f"#{b:<3} starts frame {spans[b][0]:>4}   "
                  f"{gap:>3} frames apart, drift {drift:.2f}   {verdict}")
    else:
        print("  none found")

    # A different failure, and the one the sequential search cannot see: two tracks
    # alive AT THE SAME TIME sitting on the same robot. The detector proposes two boxes
    # for one object and association happily keeps both. Left in, the ground truth
    # claims two robots where there is one, and every tracker is charged a false
    # positive for the robot it correctly did not see twice.
    overlaps, nested = [], []
    for a in tracks:
        for b in tracks:
            if a >= b:
                continue
            lo = max(spans[a][0], spans[b][0])
            hi = min(spans[a][1], spans[b][1])
            if hi - lo < 10:
                continue
            scores, covered = [], []
            for frame in range(lo, hi + 1, 5):
                box_a, box_b = box_at(a, frame), box_at(b, frame)
                if box_a and box_b:
                    scores.append(_iou(box_a, box_b))
                    covered.append(_containment(box_a, box_b))
            if not scores:
                continue
            score, cover = statistics.median(scores), statistics.median(covered)
            # Which one to drop: the physically smaller box is the partial detection.
            areas = {}
            for tid in (a, b):
                boxes = [box_at(tid, f) for f in range(lo, hi + 1, 5)]
                sizes = [bx[2] * bx[3] for bx in boxes if bx]
                areas[tid] = statistics.median(sizes) if sizes else 0.0
            smaller = a if areas[a] <= areas[b] else b
            larger = b if smaller == a else a

            if score > 0.5:
                overlaps.append((score, smaller, larger, hi - lo + 1))
            elif cover >= 0.7:
                # IoU is blind to nesting: a small box sitting entirely inside a large
                # one scores low because the union is dominated by the big box. On arc04
                # frame 1, #2 lies 96% inside #6 - obviously the same robot - at IoU
                # 0.24. Containment catches the partial detections IoU misses.
                nested.append((cover, smaller, larger, hi - lo + 1))

    print("\nNESTED boxes - one box sits INSIDE another, so it is a part of that robot")
    print("(turret, wheel) rather than a second robot. DELETE the smaller one with X:")
    if nested:
        for cover, small, big, shared in sorted(nested, reverse=True):
            print(f"  delete #{small:<3} - it is {cover:.0%} inside #{big}, "
                  f"over {shared} frames")
    else:
        print("  none found")

    print("\nSIMULTANEOUS duplicates - two boxes on one robot at the same time.")
    print("DELETE one with X, do not merge: their keyframes are not aligned, so merging")
    print("interleaves them and the box alternates between two positions every frame.")
    if overlaps:
        for score, small, big, shared in sorted(overlaps, reverse=True):
            lo = max(spans[small][0], spans[big][0])
            hi = min(spans[small][1], spans[big][1])
            print(f"  delete #{small:<3} - overlaps #{big} by {score:.0%} across "
                  f"frames {lo}-{hi} ({shared} frames)")
    else:
        print("  none found")

    # ---- merge groups -------------------------------------------------------------
    # Pairs alone are awkward to act on, because merging changes the ids and the rest of
    # the list goes stale: arc04 lists #4 and #6 as duplicates AND #6 -> #16 as a
    # fragment, so doing them in the wrong order leaves you re-running this three times.
    # Treating every confident pair as a graph edge and taking connected components
    # gives one target per group that is guaranteed to still exist when you reach it.
    # Only STRONG evidence is chained. Chaining every confident pair joins separate
    # robots through shared neighbours: on arc07 it produced a single group of ten,
    # containing contradictions (#25 continued by both #41 and #43) because robots in a
    # scrum overlap each other 53-66%. Requiring 65% overlap or 0.20 drift breaks that
    # into four clean pairs. Weaker pairs are still reported, just one at a time.
    #
    # SEQUENTIAL PAIRS ONLY. Merging two tracks that are alive at the same time is
    # wrong: their keyframes are not aligned (#4 has 5, #6 has 8, sharing only 2), so a
    # merge interleaves them and the box alternates between two positions - on arc04 the
    # merged #4/#6 jumps 30px back at frame 68 and 56px forward at frame 73. Co-alive
    # duplicates must have one side DELETED, not merged. Sequential fragments have
    # disjoint spans, so merging them is a clean concatenation.
    STRONG_OVERLAP, STRONG_DRIFT = 0.65, 0.20
    edges = [(a, b) for drift, a, b, _ in candidates if drift <= STRONG_DRIFT]

    parent = {t: t for t in tracks}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)   # lowest id becomes the target

    groups: dict[int, list[int]] = {}
    for t in tracks:
        groups.setdefault(find(t), []).append(t)
    groups = {k: sorted(v) for k, v in groups.items() if len(v) > 1}

    print("\nMERGE GROUPS - each group should be ONE robot:")
    if groups:
        by_pair = {}
        for drift, a, b, gap in candidates:
            if drift < 0.3:
                by_pair[(a, b)] = f"ends {spans[a][1]} -> starts {spans[b][0]}, drift {drift:.2f}"
        for score, a, b, shared in overlaps:
            by_pair[(a, b)] = f"co-alive {shared} frames, overlap {score:.0%}"

        for target in sorted(groups):
            members = [t for t in groups[target] if t != target]
            size = len(members) + 1
            print(f"\n  group of {size}: merge {', '.join(f'#{m}' for m in members)} "
                  f"into #{target}")
            if size > 2:
                # Chained through shared neighbours, so it can silently swallow a second
                # robot: A links to B, B links to C, and A and C are different robots.
                # The edges that built the group are printed so a wrong link is visible
                # and can be broken by merging only the pairs you accept.
                print(f"    chained from {size - 1} links - verify before merging:")
                for (a, b), why in sorted(by_pair.items()):
                    if find(a) == target:
                        print(f"      #{a} - #{b}: {why}")
    else:
        print("  none - nothing left to merge")

    # Confident enough to report, not strong enough to chain. Kept separate so a weak
    # link cannot quietly drag a second robot into a group.
    weak = []
    for drift, a, b, _ in candidates:
        if STRONG_DRIFT < drift < 0.3 and find(a) != find(b):
            weak.append((f"#{a} -> #{b}", f"drift {drift:.2f}"))
    for score, a, b, shared in overlaps:
        if score < STRONG_OVERLAP and find(a) != find(b):
            weak.append((f"#{a} + #{b}", f"overlap {score:.0%} over {shared} frames"))

    if weak:
        print(f"\nWEAKER PAIRS - judgement calls, do these one at a time (or skip):")
        for pair, why in sorted(weak):
            print(f"  {pair:<12} {why}")

    print(f"\n{len(groups)} group(s) to merge, {len(weak)} judgement call(s).")
    print("The # numbers are TRACK IDS - the labels drawn above each box - not frames.")
    print("To merge: press tab until the top bar reads sel:<source id>, press m, then "
          "type the target id.")
    print("A group of 3 or more was chained through a shared track, which can join two "
          "robots\nthrough a common neighbour - check those before merging.")
    print("When unsure, DO NOT merge. A missed merge penalises every tracker equally;")
    print("a wrong one invents a robot and rewards trackers that make the same mistake.")


def _containment(a, b) -> float:
    """Intersection over the SMALLER box: how much of one lies inside the other.

    IoU cannot see nesting. A turret-sized box inside a whole-robot box has a large
    intersection but a union dominated by the big box, so IoU stays low while the two
    plainly describe the same robot. On arc04 frame 1, #2 lies 96% inside #6 at an IoU
    of only 0.24.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    smaller = min(aw * ah, bw * bh)
    return inter / smaller if smaller > 0 else 0.0


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (aw * ah + bw * bh - inter)


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

    # An absence marker must break the span, not be interpolated across.
    gapped = {1: [0.0, 0.0, 10.0, 10.0], 3: None, 5: [40.0, 20.0, 10.0, 10.0]}
    assert interpolate(gapped, 2) is None, "must not bridge into an absence"
    assert interpolate(gapped, 3) is None, "an absence frame has no box"
    assert interpolate(gapped, 4) is None, "must not bridge out of an absence"
    assert interpolate(gapped, 1) == [0.0, 0.0, 10.0, 10.0]
    assert interpolate(gapped, 5) == [40.0, 20.0, 10.0, 10.0]
    print("       ok - midpoint, endpoint, no extrapolation, absence breaks the span")

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
                out[tid] = (box, kf.get(index + 1) is not None and (index + 1) in kf)
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

        # tab cycles every track in the clip, not just the ones visible here, so a
        # selected track with no box on this frame looks identical to a deleted one.
        # Saying where it does live stops that being mistaken for junk worth deleting.
        chosen = state["selected"]
        if chosen in tracks and chosen not in current_boxes(index):
            marks = sorted(tracks[chosen])
            cv2.putText(canvas,
                        f"#{chosen} is not in this frame - it runs {marks[0]}-{marks[-1]}"
                        f"  (press g, then {marks[0]})",
                        (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
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
            # Lowercase x removes only THIS frame's keyframe. On an interpolated
            # (orange) box there is no keyframe here, so it silently does nothing -
            # which reads as the tool ignoring the key. Say so instead.
            removed = tracks[state["selected"]].pop(index + 1, None)
            if removed is None:
                print(f"  #{state['selected']} has no keyframe on frame {index + 1} "
                      f"(orange = interpolated). Press X to delete the whole track.")
            else:
                print(f"  removed #{state['selected']} keyframe at frame {index + 1}")
        elif key == ord("h") and state["selected"] in tracks:
            # Absence marker. Use this the frame a robot leaves view and the frame it
            # returns; interpolation will not bridge the gap.
            tracks[state["selected"]][index + 1] = None
        elif key == ord("X") and state["selected"] in tracks:
            gone = state["selected"]
            marks = sorted(tracks.pop(gone))
            state["selected"] = None
            print(f"  deleted track #{gone} entirely "
                  f"({len(marks)} keyframes, frames {marks[0]}-{marks[-1]}). "
                  f"{len(tracks)} tracks left.")
        elif key == ord("m") and state["selected"] in tracks:
            # Pre-annotation splits one robot into several tracks whenever the detector
            # drops it for a few frames - one clip proposed 24 tracks for roughly 8-12
            # real robots. Without merging, joining two fragments means deleting one and
            # redrawing it, which would undo most of what pre-annotation saves.
            source = state["selected"]
            try:
                target = int(input(f"merge track {source} INTO track id: "))
            except ValueError:
                continue
            if target not in tracks or target == source:
                print(f"  no track {target} to merge into")
                continue
            moved = tracks.pop(source)
            for f, box in moved.items():
                # Keyframes already on the target win: a human put them there.
                tracks[target].setdefault(f, box)
            state["selected"] = target
            print(f"  merged {source} -> {target} "
                  f"({len(tracks[target])} keyframes)")
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
    parser.add_argument("--tracks", action="store_true",
                        help="List tracks and flag likely fragments, then exit.")
    parser.add_argument("--max-gap", type=int, default=45)
    parser.add_argument("--max-drift", type=float, default=1.5)
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.seq:
        parser.error("--seq is required unless --selftest is given")
    if args.tracks:
        report_tracks(args.seq, args.max_gap, args.max_drift)
        return
    run_gui(args.seq, args.cls)


if __name__ == "__main__":
    main()
