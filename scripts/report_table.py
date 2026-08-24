"""Join accuracy and CPU performance into the table that goes in the write-up.

results/detection.csv holds one row per config (mAP, per-class AP, bootstrap CIs).
results/performance.csv holds one row per (config, core-count) (latency, FPS, CPU, RAM).
Neither answers the research question alone: "yolo_960 reaches 0.668 mAP" is only a
finding once paired with what it costs to run. This joins them on `name`.

Emits results/combined.csv (everything, for analysis) and results/combined.md (a
readable table per core level, for the report).

Usage:
    python scripts/report_table.py
    python scripts/report_table.py --cores 4       # one core level
    python scripts/report_table.py --fps-target 60
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# The table uses a few non-ASCII glyphs and Windows consoles default to cp1252, which
# raises UnicodeEncodeError on them. The files are written as UTF-8 regardless; this
# only keeps the on-screen copy from taking the whole script down with it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DETECTION = ROOT / "results" / "detection.csv"
PERFORMANCE = ROOT / "results" / "performance.csv"
OUT_CSV = ROOT / "results" / "combined.csv"
OUT_MD = ROOT / "results" / "combined.md"


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise SystemExit(f"Missing {path}")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def f(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def join(accuracy: list[dict], performance: list[dict]) -> list[dict]:
    by_name = {r["name"]: r for r in accuracy}
    missing = sorted({r["name"] for r in performance} - set(by_name))
    if missing:
        print(f"note: no accuracy row for {', '.join(missing)} - left blank")

    rows = []
    for perf in performance:
        acc = by_name.get(perf["name"], {})
        rows.append({
            "name": perf["name"],
            "family": perf["family"],
            "imgsz": perf["imgsz"],
            "cores": int(perf["cores"]),
            "smt": int(perf.get("smt", 0)),
            "logical_cpus": perf.get("logical_cpus", ""),
            "mAP_50_95": acc.get("mAP_50_95", ""),
            "mAP_50": acc.get("mAP_50", ""),
            "map_ci_low": acc.get("ci_low", ""),
            "map_ci_high": acc.get("ci_high", ""),
            "AP_armor": acc.get("AP_armor", ""),
            "AP_base": acc.get("AP_base", ""),
            "AP_car": acc.get("AP_car", ""),
            "AP_watcher": acc.get("AP_watcher", ""),
            "lat_mean_ms": perf["lat_mean_ms"],
            "lat_ci_low": perf["lat_ci_low"],
            "lat_ci_high": perf["lat_ci_high"],
            "lat_p95_ms": perf["lat_p95_ms"],
            "fps_mean": perf["fps_mean"],
            "decode_mean_ms": perf["decode_mean_ms"],
            "cpu_pct_mean": perf["cpu_pct_mean"],
            "cpu_pct_of_cap": perf["cpu_pct_of_cap"],
            "rss_peak_mib": perf["rss_peak_mib"],
            "frames": perf["frames"],
            "cpu_model": perf["cpu_model"],
        })
    return sorted(rows, key=lambda r: (r["cores"], r["smt"], -f(r, "mAP_50_95", -1)))


def markdown(rows: list[dict], fps_target: float) -> str:
    out = [
        "# Detection models — accuracy and CPU cost",
        "",
        f"Latency excludes JPEG decode (reported separately); batch size 1; "
        f"CPU `{rows[0]['cpu_model']}`.",
        f"Core counts are OS-enforced affinity caps on DISTINCT PHYSICAL cores, "
        f"standing in for constrained hardware. `RT` marks ≥ {fps_target:g} FPS.",
        "",
    ]
    for cores, smt in sorted({(r["cores"], r["smt"]) for r in rows}):
        group = [r for r in rows if r["cores"] == cores and r["smt"] == smt]
        suffix = " + SMT (full machine)" if smt else ""
        out += [
            f"## {cores} physical core{'s' if cores != 1 else ''}{suffix}",
            "",
            "| config | mAP@.5:.95 | 95% CI | armor AP | latency ms | 95% CI | "
            "p95 ms | FPS | RT | CPU% of cap | RAM MiB |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in group:
            rt = "✓" if f(r, "fps_mean", 0) >= fps_target else ""
            out.append(
                f"| `{r['name']}` | {f(r, 'mAP_50_95'):.4f} | "
                f"[{f(r, 'map_ci_low'):.3f}, {f(r, 'map_ci_high'):.3f}] | "
                f"{f(r, 'AP_armor'):.4f} | {f(r, 'lat_mean_ms'):.1f} | "
                f"[{f(r, 'lat_ci_low'):.1f}, {f(r, 'lat_ci_high'):.1f}] | "
                f"{f(r, 'lat_p95_ms'):.1f} | {f(r, 'fps_mean'):.1f} | {rt} | "
                f"{f(r, 'cpu_pct_of_cap'):.0f} | {f(r, 'rss_peak_mib'):.0f} |"
            )
        out.append("")
    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cores", type=int, default=None, help="Only this core level.")
    parser.add_argument("--fps-target", type=float, default=30.0,
                        help="FPS at or above which a config is marked real-time.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = join(read_csv(DETECTION), read_csv(PERFORMANCE))
    if args.cores is not None:
        rows = [r for r in rows if r["cores"] == args.cores]
    if not rows:
        raise SystemExit("No rows to report.")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    md = markdown(rows, args.fps_target)
    OUT_MD.write_text(md, encoding="utf-8")

    print(md)
    print(f"\nwrote {OUT_CSV}\nwrote {OUT_MD}")

    blank = [r["name"] for r in rows if not r["mAP_50_95"]]
    if blank:
        print(f"\nWARNING: {len(blank)} row(s) have no accuracy: {sorted(set(blank))}")


if __name__ == "__main__":
    main()
