"""Join tracking accuracy and tracking CPU cost into the table for the write-up.

The tracking counterpart to scripts/report_table.py, kept separate because the two
halves are keyed differently: accuracy is per (tracker, sequence), while CPU cost is per
(tracker, sequence, core count).

READING ORDER MATTERS HERE
    The CPU table is the headline. Tracking ground truth for this project is machine
    generated, so the accuracy figures carry a bias that cannot be fully removed - labels
    derived from motion association favour SORT and the classical tracker over the
    appearance-based GOTURN and VitTrack. The CPU numbers need no labels at all and are
    unaffected, which is why they lead and accuracy follows with the caveat printed
    beside it rather than buried in a methods section.

Emits results/tracking_report.md and results/tracking_combined.csv.

Usage:
    python scripts/report_tracking.py
    python scripts/report_tracking.py --cores 4
"""

from __future__ import annotations

import argparse
import collections
import csv
import statistics
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
ACCURACY = ROOT / "results" / "tracking.csv"
PERFORMANCE = ROOT / "results" / "tracking_performance.csv"
QUALITY = ROOT / "results" / "label_quality.csv"
OUT_CSV = ROOT / "results" / "tracking_combined.csv"
OUT_MD = ROOT / "results" / "tracking_report.md"


def read(path: Path, required: bool = True):
    if not path.is_file():
        if required:
            sys.exit(f"Missing {path}")
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(row, key, default=float("nan")):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def mean_by_pairing(rows, keys):
    """Average across sequences, so each perception model gets one row per core level.

    Keyed by DETECTOR AND TRACKER, not tracker alone. Proposal 5.2 compares perception
    models - a detector paired with a tracker - and averaging the detector axis away
    would blend yolo_960 with the classical detector into a number describing neither.
    With a single detector present this reads exactly as a tracker-only table.
    """
    grouped = collections.defaultdict(list)
    for row in rows:
        key = (row.get("detector", "?"), row["tracker"],
               int(row["cores"]), int(row.get("smt", 0)))
        grouped[key].append(row)
    out = {}
    for key, group in grouped.items():
        out[key] = {k: statistics.fmean([f(r, k) for r in group]) for k in keys}
        out[key]["sequences"] = len(group)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument("--fps-target", type=float, default=30.0)
    args = parser.parse_args()

    perf = read(PERFORMANCE, required=False)
    acc = read(ACCURACY, required=False)
    quality = read(QUALITY, required=False)
    if not perf and not acc:
        sys.exit("Nothing to report - run benchmark_tracking.py / eval_tracking.py first.")

    lines = ["# Tracking — CPU cost and accuracy", ""]

    if perf:
        keys = ["detect_ms", "track_ms", "total_ms", "fps", "track_share",
                "tracks_in_flight", "cpu_pct_of_cap", "rss_peak_mib"]
        summary = mean_by_pairing(perf, keys)
        lines += [
            "## CPU cost of each perception model (the headline — needs no labels)",
            "",
            "One row per detector+tracker pairing, averaged over sequences. "
            f"`RT` marks >= {args.fps_target:g} FPS. `track %` is the tracker's share of "
            "the pipeline: a tracker that is cheap beside its detector is a completely "
            "different deployment proposition from one that dominates the frame budget.",
            "",
            "| cores | detector | tracker | detect ms | track ms | total ms | FPS | RT "
            "| track % | tracks/frame | CPU% of cap | RAM MiB |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        shown = []
        for (detector, tracker, cores, smt), row in sorted(
            summary.items(), key=lambda kv: (kv[0][2], kv[0][3], -kv[1]["fps"])
        ):
            if args.cores is not None and cores != args.cores:
                continue
            label = f"{cores}{'+SMT' if smt else ''}"
            rt = "OK" if row["fps"] >= args.fps_target else ""
            shown.append((cores, smt, detector, tracker, row))
            lines.append(
                f"| {label} | `{detector}` | `{tracker}` | {row['detect_ms']:.1f} | "
                f"{row['track_ms']:.1f} | {row['total_ms']:.1f} | {row['fps']:.1f} | "
                f"{rt} | {row['track_share']:.0%} | {row['tracks_in_flight']:.1f} | "
                f"{row['cpu_pct_of_cap']:.0f} | {row['rss_peak_mib']:.0f} |"
            )
        lines.append("")

        # The fastest pairing per core level, stated rather than left to be read off the
        # table - this is the deployment answer the research question asks for.
        best, pool = {}, collections.Counter()
        for cores, smt, detector, tracker, row in shown:
            key = (cores, smt)
            pool[key] += 1
            if key not in best or row["fps"] > best[key][2]["fps"]:
                best[key] = (detector, tracker, row)
        if best:
            lines.append("Fastest pairing at each core level:")
            lines.append("")
            for (cores, smt), (detector, tracker, row) in sorted(best.items()):
                label = f"{cores}{'+SMT' if smt else ''}"
                # The count matters. The full detector x tracker matrix was run at 1 and
                # 6 cores only; the intermediate levels carry the earlier sweep's single
                # detector. Without this, "fastest" would read as a claim over the whole
                # grid at every level rather than over whatever was measured there.
                lines.append(
                    f"- **{label} core(s)** — `{detector}` + `{tracker}`, "
                    f"{row['total_ms']:.1f} ms ({row['fps']:.1f} FPS) "
                    f"— best of {pool[(cores, smt)]} pairings measured at this level"
                )
            lines.append("")

    if acc:
        lines += [
            "## Accuracy (secondary — see the caveat below)",
            "",
            "**Read IDF1 and ID switches, not MOTA.** Every tracker in a given row "
            "group is fed by the same detector, and MOTA is dominated by that "
            "detector's misses and false positives, which are therefore identical "
            "across trackers; only the ID-switch term varies, and it is small next to "
            "the others. On arc01 `classical`, `sort` and `vit` return MOTA 0.570 and "
            "IDF1 0.651 alike — the trackers genuinely agreed on that clip's identity "
            "assignment. Where they diverge, IDF1 separates them: GOTURN scores 0.493 "
            "against 0.651 on the same frames, with twice the ID switches.",
            "",
            "| tracker | sequence | MOTA | 95% CI | IDF1 | ID switches | MT / ML |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in sorted(acc, key=lambda r: (r["name"], r["sequence"])):
            lines.append(
                f"| `{row['name']}` | {row['sequence']} | {f(row, 'mota'):.3f} | "
                f"[{f(row, 'mota_ci_low'):.3f}, {f(row, 'mota_ci_high'):.3f}] | "
                f"{f(row, 'idf1'):.3f} | {int(f(row, 'id_switches', 0))} | "
                f"{int(f(row, 'mostly_tracked', 0))} / "
                f"{int(f(row, 'mostly_lost', 0))} |"
            )
        lines += [
            "",
            "**Caveat.** Ground truth here was generated automatically "
            "(`scripts/auto_label.py`), because hand-labelling was not affordable within "
            "the project. Labels derived from motion association share their assumptions "
            "with SORT and the classical tracker and not with the appearance-based "
            "GOTURN and VitTrack, so this table is biased toward the former. The CPU "
            "table above needs no labels and is unaffected.",
            "",
            "Evaluation is detector-fed, not ground-truth-fed. Feeding trackers the "
            "reference boxes proved degenerate, and this was measured rather than "
            "assumed: on arc01 the classical tracker and SORT both scored MOTA 0.991651 "
            "— identical to six decimal places, which is not two trackers agreeing but "
            "one answer arrived at twice. Handed boxes that were themselves produced by "
            "association, a Kalman tracker returns the labeller's own tracks. "
            "Detector-fed asks a real question instead — how much an online tracker "
            "loses against an offline reference that saw the whole clip.",
            "",
            "Rows ending `_det150` are the four-way head-to-head: all four trackers on "
            "the same 3 clips x 150 frames, fed by `yolo_960`. Rows ending `_det` are "
            "the broader 7-clip sample, which only `classical` and `sort` are cheap "
            "enough to run in full — GOTURN alone would need ~17 hours for it. Compare "
            "within a suffix, not across.",
            "",
        ]

    if quality:
        lines += [
            "## Label quality (no ground truth exists, so this measures implausibility)",
            "",
            "| sequence | tracks | median track len | tracks/robot | p99 step/size | flags |",
            "|---|---|---|---|---|---|",
        ]
        for row in quality:
            lines.append(
                f"| {row['sequence']} | {row['tracks']} | "
                f"{row['median_track_len']} | {row.get('tracks_per_robot', '-')} | "
                f"{row['p99_step_over_size']} | {row['flags']} |"
            )
        lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    if perf:
        with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(perf[0]))
            writer.writeheader()
            writer.writerows(perf)

    print("\n".join(lines))
    print(f"\nwrote {OUT_MD}")
    if perf:
        print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
