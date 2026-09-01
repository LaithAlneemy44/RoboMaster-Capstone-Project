"""Measure CPU cost of a detector+tracker pipeline on the target CPU.

This is the tracking headline. Proposal 5.4 asks for mean FPS, latency, CPU and RAM, and
CLAUDE.md is explicit that CPU-constrained numbers are the contribution - most published
tracking benchmarks report GPU throughput, which says nothing about a robot.

NEEDS NO LABELS
    Timing does not care what is in the frame, so this runs over all seven clips whether
    they are annotated or not. That matters here: the tracking ground truth is machine
    generated and its accuracy carries a caveat, but none of that touches these numbers.

DETECTION AND TRACKING ARE TIMED SEPARATELY
    A tracker that is cheap beside its detector is a completely different deployment
    proposition from one that dominates the frame budget, and the combined figure hides
    which it is. Both are reported, plus their sum.

SINGLE-OBJECT TRACKERS COST MORE AS THE SCENE FILLS
    GOTURN and VitTrack each follow one target, so multi-object tracking runs one
    instance per robot (see CvMultiTracker in run_trackers.py). Their per-frame cost
    therefore scales with the number of robots on screen, where the Kalman trackers pay
    almost nothing extra. Tracks-in-flight is recorded alongside the timings so that
    shows up as a measurement rather than an anomaly.

Core capping, resource sampling and confidence intervals are reused from
benchmark_cpu.py unchanged, so tracking rows are directly comparable with detection rows.

Usage:
    python scripts/benchmark_tracking.py --seq data/tracking/arc04 --tracker sort --cores 4
"""

from __future__ import annotations

# Same ordering rule as benchmark_cpu.py: nothing that pulls in torch may be imported
# before the core cap is applied.
import argparse
import csv
import platform
import statistics
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_cpu import (  # noqa: E402
    ResourceSampler, apply_core_cap, bootstrap_ci, check_machine_idle,
)

RESULTS = ROOT / "results" / "tracking_performance.csv"
DEFAULT_DETECTOR = ROOT / "runs" / "detect" / "yolo_960" / "weights" / "best.pt"
WARMUP = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seq", type=Path, required=True)
    parser.add_argument("--tracker", choices=("classical", "sort", "goturn", "vit"),
                        required=True)
    parser.add_argument("--detector", type=Path, default=DEFAULT_DETECTOR)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--cores", type=int, required=True,
                        help="Distinct PHYSICAL cores to pin to.")
    parser.add_argument("--smt", action="store_true")
    parser.add_argument("--frames", type=int, default=100,
                        help="Frames to time. Latency is stable, so a subset suffices.")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    affinity = apply_core_cap(args.cores, args.smt)  # BEFORE torch is imported

    import cv2
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(len(affinity))

    from auto_label import clean_frame  # noqa: PLC0415
    from run_trackers import build, read_seqinfo  # noqa: PLC0415

    info = read_seqinfo(args.seq)
    images = sorted((args.seq / info["imDir"]).glob("*" + info["imExt"]))
    if args.frames:
        images = images[:args.frames]
    if not images:
        sys.exit(f"No frames in {args.seq / info['imDir']}")

    model = YOLO(str(args.detector))
    names = dict(model.names.items())
    tracker, needs_frame = build(args.tracker)
    box_median = float(info.get("imWidth", 1280)) / 11.0

    print(f"sequence : {args.seq.name}   tracker: {args.tracker}")
    print(f"detector : {args.detector.name} at {args.imgsz}px")
    print(f"cores    : {args.cores} physical{' +SMT' if args.smt else ''} "
          f"(affinity {affinity})")
    print(f"frames   : {len(images)} (+{WARMUP} warmup)   device: cpu")
    baseline = check_machine_idle()

    def detect(frame):
        result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, device="cpu",
                               verbose=False)[0]
        boxes = []
        for box in result.boxes:
            if names[int(box.cls[0])] != "car":
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            boxes.append((x1, y1, x2 - x1, y2 - y1))
        # Matches what the labeller and run_trackers.py feed the trackers, so the timing
        # corresponds to the pipeline that produced the accuracy numbers.
        return clean_frame(boxes, median=box_median)

    for path in images[:WARMUP]:
        frame = cv2.imread(str(path))
        dets = detect(frame)
        tracker.update(frame, dets) if needs_frame else tracker.update(dets)

    det_ms, trk_ms, decode_ms, in_flight = [], [], [], []
    proc = psutil.Process()
    with ResourceSampler(proc, len(affinity)) as sampler:
        for index, path in enumerate(images):
            t0 = time.perf_counter()
            frame = cv2.imread(str(path))
            t1 = time.perf_counter()
            dets = detect(frame)
            t2 = time.perf_counter()
            outputs = (tracker.update(frame, dets) if needs_frame
                       else tracker.update(dets))
            t3 = time.perf_counter()

            decode_ms.append((t1 - t0) * 1e3)
            det_ms.append((t2 - t1) * 1e3)
            trk_ms.append((t3 - t2) * 1e3)
            in_flight.append(len(outputs))
            print(f"\r  {index + 1}/{len(images)} frames", end="", flush=True)
    print()

    total_ms = [d + t for d, t in zip(det_ms, trk_ms)]
    lo, hi = bootstrap_ci(total_ms, args.bootstrap, args.seed)
    mean_cpu, norm_cpu, peak_rss = sampler.summary()

    row = {
        "sequence": args.seq.name,
        "tracker": args.tracker,
        "detector": args.detector.parent.parent.name,
        "cores": args.cores,
        "smt": int(args.smt),
        "logical_cpus": len(affinity),
        "frames": len(images),
        "detect_ms": round(statistics.fmean(det_ms), 3),
        "track_ms": round(statistics.fmean(trk_ms), 3),
        "total_ms": round(statistics.fmean(total_ms), 3),
        "total_ci_low": round(lo, 3),
        "total_ci_high": round(hi, 3),
        "total_p95_ms": round(sorted(total_ms)[int(0.95 * (len(total_ms) - 1))], 3),
        "fps": round(1000.0 / statistics.fmean(total_ms), 3),
        "decode_ms": round(statistics.fmean(decode_ms), 3),
        # The number the single-object trackers pay per frame.
        "tracks_in_flight": round(statistics.fmean(in_flight), 1),
        "track_share": round(
            statistics.fmean(trk_ms) / max(statistics.fmean(total_ms), 1e-9), 3),
        "cpu_pct_mean": round(mean_cpu, 1),
        "cpu_pct_of_cap": round(norm_cpu, 1),
        "rss_peak_mib": round(peak_rss, 1),
        "baseline_cpu_pct": round(baseline, 1),
        "cpu_model": platform.processor(),
    }

    print(f"detect   {row['detect_ms']:.1f} ms      track {row['track_ms']:.1f} ms "
          f"({row['track_share']:.0%} of the pipeline)")
    print(f"total    {row['total_ms']:.1f} ms  95% CI "
          f"[{row['total_ci_low']:.1f}, {row['total_ci_high']:.1f}]   "
          f"{row['fps']:.2f} FPS")
    print(f"tracks   {row['tracks_in_flight']:.1f} in flight per frame")
    print(f"cpu      {row['cpu_pct_of_cap']:.0f}% of the {len(affinity)}-thread cap   "
          f"rss {row['rss_peak_mib']:.0f} MiB")

    if args.no_write:
        return
    args.results.parent.mkdir(parents=True, exist_ok=True)
    is_new = not args.results.exists()
    with args.results.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    print(f"\nappended to {args.results}")


if __name__ == "__main__":
    main()
