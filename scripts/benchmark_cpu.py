"""Measure single-frame CPU latency, throughput, CPU% and RAM for one detector.

This is the other half of every results row. scripts/evaluate_detection.py answers
"how accurate is this model"; this answers "can it run on the robot", which is the
part of the project that is actually novel - most published benchmarks report GPU
numbers, and CLAUDE.md is explicit that the CPU figures are the contribution.

ONE (config, core-count) combination per process, by design. Core capping has to
happen before torch initialises its thread pools, so it cannot be done in a loop
inside a single process. scripts/run_benchmark.py drives the grid.

WHY AFFINITY AND NOT THREAD COUNT
    Both documented ways of limiting PyTorch's CPU use silently do nothing on this
    build. Measured on yolo_960, 1 core vs all:

        torch.set_num_threads()   181 ms vs 167 ms   (no effect)
        OMP_NUM_THREADS env       152 ms vs 151 ms   (no effect - and
                                  torch.get_num_threads() reported 8 either way)
        psutil cpu_affinity()     429 ms vs 205 ms   (works)

    Only OS-enforced affinity constrains anything. Had this been built the obvious
    way it would have produced four near-identical sets of numbers labelled as four
    different hardware budgets - a fabricated result, not a measurement. So affinity
    is applied first and then ASSERTED, and the script aborts if it did not take.

WHAT IS TIMED
    Headline latency is preprocess + inference + postprocess. JPEG decode is timed
    but excluded, because on a robot frames arrive from a camera sensor rather than
    as JPEGs on disk; charging every model for a decode it would never perform would
    also penalise the high-resolution configs unevenly. Both numbers are recorded.

    Batch size is always 1. Batching inflates throughput and does not reflect
    frame-at-a-time deployment.

Usage:
    python scripts/benchmark_cpu.py --family yolo --name yolo_960 \\
        --weights runs/detect/yolo_960/weights/best.pt --imgsz 960 --cores 4
"""

from __future__ import annotations

# Ordering here is load-bearing: argparse and psutil are stdlib/pure-python, but torch
# must NOT be imported until after cpu_affinity has been applied, so every torch import
# in this file is deliberately function-local.
import argparse
import csv
import json
import platform
import random
import statistics
import sys
import threading
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GT = ROOT / "data" / "splits" / "coco_val.json"
DEFAULT_RESULTS = ROOT / "results" / "performance.csv"

WARMUP_FRAMES = 5      # first calls pay lazy allocation and cache warming
SAMPLE_HZ = 20.0       # resource sampler polls at 50 ms
BUSY_THRESHOLD = 25.0  # system CPU% above which the machine is too loaded to trust


class ResourceSampler:
    """Polls process CPU% and RSS on a background thread while inference runs.

    Sampling rather than measuring once at the end: RSS peaks mid-run and a single
    reading after the loop would miss it entirely.
    """

    def __init__(self, proc: psutil.Process, n_cores: int) -> None:
        self.proc = proc
        self.n_cores = n_cores
        self.cpu: list[float] = []
        self.rss: list[int] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        self.proc.cpu_percent()  # first call primes the delta; its return is garbage
        while not self._stop.wait(1.0 / SAMPLE_HZ):
            try:
                self.cpu.append(self.proc.cpu_percent())
                self.rss.append(self.proc.memory_info().rss)
            except psutil.Error:
                break

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def summary(self) -> tuple[float, float, float]:
        """(mean CPU% of one core, mean CPU% normalised to the core cap, peak RSS MiB)."""
        # psutil reports CPU% relative to a single core, so a fully-loaded 4-core cap
        # reads as 400%. The normalised column divides by the cap, giving "fraction of
        # the budget actually used" - comparable across core levels, unlike the raw one.
        cpu = [c for c in self.cpu if c > 0]
        mean_cpu = statistics.fmean(cpu) if cpu else 0.0
        peak_rss = max(self.rss) / 1024**2 if self.rss else 0.0
        return mean_cpu, mean_cpu / self.n_cores, peak_rss


def core_ids(n_physical: int, smt: bool) -> list[int]:
    """Logical CPU ids for `n_physical` DISTINCT physical cores.

    Not range(n). Adjacent logical CPUs are hyperthread siblings sharing one physical
    core, so range(2) pins two threads to a single core - which is slower than one
    thread, not faster, because the second thread adds parallel overhead without
    adding any compute. Measured on this machine with a 1400x1400 matmul:

        affinity [0, 1]   46.7 ms   <- siblings, one physical core
        affinity [0, 2]   29.4 ms   <- two physical cores
        affinity [0, 6]   29.6 ms

    An early version of this harness used range(n) and duly reported "2 cores" as
    20% SLOWER than "1 core" across every config. Striding by the SMT ratio fixes it.
    """
    logical = psutil.cpu_count(logical=True)
    physical = psutil.cpu_count(logical=False) or logical
    stride = max(1, logical // physical)

    ids = []
    for i in range(n_physical):
        base = i * stride
        ids.append(base)
        if smt:  # also take the siblings, for the full-machine reference run
            ids.extend(range(base + 1, base + stride))
    return sorted(ids)


def apply_core_cap(n_physical: int, smt: bool) -> list[int]:
    """Pin this process to `n_physical` cores and prove it took effect.

    Must run before torch is imported. Returns the affinity list actually in force.
    """
    physical = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True)
    if n_physical < 1 or n_physical > physical:
        sys.exit(f"--cores {n_physical} outside 1..{physical} physical cores here.")

    wanted = core_ids(n_physical, smt)
    proc = psutil.Process()
    try:
        proc.cpu_affinity(wanted)
    except (AttributeError, psutil.Error) as exc:
        sys.exit(
            f"cpu_affinity is unavailable here ({exc}).\n"
            "Refusing to continue: without it the core cap is a no-op and every "
            "core level would report the same numbers under different labels."
        )

    actual = proc.cpu_affinity()
    if sorted(actual) != wanted:
        sys.exit(
            f"Core cap did not take: asked for {wanted}, process is on {actual}. "
            "Refusing to report unconstrained numbers as constrained."
        )
    return actual


def check_machine_idle() -> float:
    """Warn if something else is competing for the CPU. Returns the baseline load."""
    baseline = psutil.cpu_percent(interval=0.5)
    if baseline > BUSY_THRESHOLD:
        print(
            f"  WARNING: system CPU is {baseline:.0f}% busy before this run started.\n"
            "           Latency will be inflated. Stop other work and re-run."
        )
    return baseline


def load_frames(
    gt_path: Path, limit: int | None, seed: int
) -> tuple[list[Path], dict[str, int]]:
    """The val images, optionally subsampled - the same frames the accuracy eval used.

    Returns (paths, class-name -> category-id), the latter only because YOLO's loader
    needs it to build its index map.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from predict_to_coco import load_targets  # noqa: PLC0415

    targets, name_to_cat = load_targets(gt_path)
    paths = [p for _, p in targets]
    if limit and limit < len(paths):
        paths = random.Random(seed).sample(paths, limit)
    return paths, name_to_cat


def time_yolo(weights: Path, name_to_cat, imgsz: int, conf: float, frames: list[Path]):
    """Per-frame (decode_s, headline_s) for a YOLO model, plus its stage breakdown.

    Decode is done here rather than letting Ultralytics load from the path, so that
    the decode cost can be separated out. model.predict() then covers preprocess +
    inference + NMS, which is exactly the headline span.
    """
    import cv2

    from predict_to_coco import load_yolo, yolo_result_to_dets

    model, idx_to_cat = load_yolo(weights, name_to_cat, quiet=True)
    predict = lambda arr: model.predict(  # noqa: E731
        arr, imgsz=imgsz, conf=conf, device="cpu", verbose=False
    )

    for path in frames[:WARMUP_FRAMES]:
        predict(cv2.imread(str(path)))

    decodes, headlines, stages = [], [], []
    for i, path in enumerate(frames):
        t0 = time.perf_counter()
        arr = cv2.imread(str(path))  # BGR, which is what Ultralytics expects
        t1 = time.perf_counter()
        results = predict(arr)
        t2 = time.perf_counter()

        yolo_result_to_dets(results[0], 0, idx_to_cat)  # exercise the real postprocess
        decodes.append(t1 - t0)
        headlines.append(t2 - t1)
        # Ultralytics reports its own preprocess/inference/postprocess split in ms.
        s = results[0].speed
        stages.append((s["preprocess"], s["inference"], s["postprocess"]))
        _progress(i + 1, len(frames))
    return decodes, headlines, stages


def time_ssd(weights: Path, imgsz: int, conf: float, frames: list[Path]):
    """Per-frame (decode_s, headline_s) for an SSD, with an explicit stage breakdown."""
    import torch
    from PIL import Image

    from predict_to_coco import load_ssd, ssd_out_to_dets, ssd_preprocess

    model, ckpt_imgsz = load_ssd(weights, imgsz, "cpu", quiet=True)

    def one(path):
        t0 = time.perf_counter()
        image = Image.open(path).convert("RGB")
        t1 = time.perf_counter()
        tensor, (sx, sy) = ssd_preprocess(image, ckpt_imgsz, "cpu")
        t2 = time.perf_counter()
        out = model([tensor])[0]
        t3 = time.perf_counter()
        ssd_out_to_dets(out, 0, sx, sy, conf)
        t4 = time.perf_counter()
        return t1 - t0, t4 - t1, ((t2 - t1) * 1e3, (t3 - t2) * 1e3, (t4 - t3) * 1e3)

    with torch.inference_mode():
        for path in frames[:WARMUP_FRAMES]:
            one(path)

        decodes, headlines, stages = [], [], []
        for i, path in enumerate(frames):
            d, h, s = one(path)
            decodes.append(d)
            headlines.append(h)
            stages.append(s)
            _progress(i + 1, len(frames))
    return decodes, headlines, stages


def time_classical(config_name: str, conf: float, frames):
    """Per-frame timing for the classical detector. No torch, no device - pure OpenCV.

    Decode is separated the same way as for the DL families so the headline number means
    the same thing in every row. Candidate count is tracked because the classical
    detector's cost is dominated by how many regions its HSV gate lets through, and a
    slow config should be attributable to its gate rather than mysterious.
    """
    import cv2

    from classical_detector import CONFIGS, ClassicalDetector  # noqa: PLC0415

    if config_name not in CONFIGS:
        sys.exit(f"Unknown classical config {config_name!r}. "
                 f"Choose from: {', '.join(sorted(CONFIGS))}")
    detector = ClassicalDetector(CONFIGS[config_name])

    for path in frames[:WARMUP_FRAMES]:
        detector.detect(cv2.imread(str(path)))

    decodes, headlines, stages, candidates = [], [], [], []
    for i, path in enumerate(frames):
        t0 = time.perf_counter()
        image = cv2.imread(str(path))
        t1 = time.perf_counter()
        detector.detect(image)
        t2 = time.perf_counter()
        decodes.append(t1 - t0)
        headlines.append(t2 - t1)
        # The classical pipeline is one fused stage, so the DL split does not apply.
        stages.append((0.0, (t2 - t1) * 1e3, 0.0))
        candidates.append(detector.last_candidates)
        _progress(i + 1, len(frames))
    print()
    time_classical.last_candidates = (
        statistics.fmean(candidates) if candidates else 0.0)
    return decodes, headlines, stages


def _progress(done: int, total: int) -> None:
    print(f"\r  {done}/{total} frames", end="", flush=True)


def bootstrap_ci(values: list[float], n: int, seed: int) -> tuple[float, float]:
    """Percentile bootstrap over the mean, matching evaluate_detection.py's method.

    Same resampling scheme as the accuracy CIs so the two intervals in a results row
    mean the same thing.
    """
    if n <= 0 or not values:
        return float("nan"), float("nan")
    import numpy as np

    rng = random.Random(seed)
    k = len(values)
    means = [
        statistics.fmean([values[rng.randrange(k)] for _ in range(k)]) for _ in range(n)
    ]
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--family", choices=("yolo", "ssd", "classical"),
                        required=True)
    # Not required for classical: it has no trained weights, only a parameter set.
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--config", help="Classical config name, e.g. strict.")
    parser.add_argument("--name", required=True, help="Config name, e.g. yolo_960.")
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--cores", type=int, required=True,
                        help="Number of distinct PHYSICAL cores to pin to.")
    parser.add_argument("--smt", action="store_true",
                        help="Also use each core's hyperthread sibling.")
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--frames", type=int, default=0, help="0 = all val images.")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-write", action="store_true")
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

    affinity = apply_core_cap(args.cores, args.smt)  # BEFORE torch is imported

    import torch

    # Match thread count to the mask. On its own this does nothing (see module
    # docstring), but leaving 8 threads fighting over 1 core measures contention
    # rather than the model.
    torch.set_num_threads(len(affinity))

    frames, name_to_cat = load_frames(args.gt, args.frames or None, args.seed)
    print(f"config : {args.name}   family: {args.family}   imgsz: {args.imgsz}")
    print(f"cores  : {args.cores} physical{' +SMT' if args.smt else ''} "
          f"(affinity {affinity})   torch threads: {torch.get_num_threads()}")
    print(f"frames : {len(frames)} (+{WARMUP_FRAMES} warmup)   device: cpu")
    baseline = check_machine_idle()

    proc = psutil.Process()
    with ResourceSampler(proc, len(affinity)) as sampler:
        if args.family == "yolo":
            decodes, headlines, stages = time_yolo(
                args.weights, name_to_cat, args.imgsz, args.conf, frames
            )
        elif args.family == "classical":
            decodes, headlines, stages = time_classical(
                args.config, args.conf, frames
            )
        else:
            decodes, headlines, stages = time_ssd(
                args.weights, args.imgsz, args.conf, frames
            )
    print()

    lat_ms = [h * 1e3 for h in headlines]
    lo, hi = bootstrap_ci(lat_ms, args.bootstrap, args.seed)
    mean_cpu, norm_cpu, peak_rss = sampler.summary()
    pre, inf, post = (statistics.fmean(s[i] for s in stages) for i in range(3))

    row = {
        "name": args.name,
        "family": args.family,
        "imgsz": args.imgsz,
        "cores": args.cores,
        "smt": int(args.smt),
        "logical_cpus": len(affinity),
        "frames": len(frames),
        "lat_mean_ms": round(statistics.fmean(lat_ms), 3),
        "lat_std_ms": round(statistics.stdev(lat_ms), 3) if len(lat_ms) > 1 else 0.0,
        "lat_median_ms": round(statistics.median(lat_ms), 3),
        "lat_p95_ms": round(sorted(lat_ms)[int(0.95 * (len(lat_ms) - 1))], 3),
        "lat_ci_low": round(lo, 3),
        "lat_ci_high": round(hi, 3),
        "fps_mean": round(1000.0 / statistics.fmean(lat_ms), 3),
        "decode_mean_ms": round(statistics.fmean(decodes) * 1e3, 3),
        "preprocess_mean_ms": round(pre, 3),
        "infer_mean_ms": round(inf, 3),
        "postprocess_mean_ms": round(post, 3),
        "cpu_pct_mean": round(mean_cpu, 1),
        "cpu_pct_of_cap": round(norm_cpu, 1),
        "rss_peak_mib": round(peak_rss, 1),
        "torch_threads": torch.get_num_threads(),
        "baseline_cpu_pct": round(baseline, 1),
        "cpu_model": platform.processor(),
        "candidates_per_frame": round(
            getattr(time_classical, "last_candidates", 0.0), 1),
    }

    print(f"latency  {row['lat_mean_ms']:.1f} ms  "
          f"95% CI [{row['lat_ci_low']:.1f}, {row['lat_ci_high']:.1f}]  "
          f"p95 {row['lat_p95_ms']:.1f}")
    print(f"fps      {row['fps_mean']:.2f}   decode {row['decode_mean_ms']:.1f} ms "
          f"(excluded from latency)")
    print(f"cpu      {row['cpu_pct_mean']:.0f}% of one core "
          f"({row['cpu_pct_of_cap']:.0f}% of the {len(affinity)}-thread cap)")
    print(f"rss peak {row['rss_peak_mib']:.0f} MiB")

    if args.no_write:
        return
    args.results.parent.mkdir(parents=True, exist_ok=True)
    is_new = not args.results.exists()
    if not is_new:
        # The row's shape must match the header already on disk. classical adds a
        # candidates_per_frame column the DL families do not, and appending a wider row
        # under a narrower header silently corrupts the file for every later reader.
        with args.results.open(newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh), [])
        missing = [k for k in row if k not in header]
        if missing and header:
            sys.exit(
                f"{args.results.name} has no column(s) for {missing}. "
                "Appending would produce rows wider than the header. Migrate the file "
                "first, or write to a different --results path."
            )
        row = {k: row.get(k, "") for k in header}
    with args.results.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    print(f"\nappended to {args.results}")


if __name__ == "__main__":
    main()
