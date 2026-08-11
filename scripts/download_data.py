"""Download the DJI ROCO Central detection dataset from Roboflow.

The dataset is deliberately not committed to this repository (see README.md).
This script re-fetches the exact pinned export in both formats the project needs:

    yolov11 -> Ultralytics training for the YOLO / Fast YOLO models
    coco    -> MobileNet-SSD training

Both exports contain byte-identical images; only the annotation format differs.

Usage:
    python scripts/download_data.py            # both formats
    python scripts/download_data.py --format coco

Requires a Roboflow API key, read from the ROBOFLOW_API_KEY environment
variable or from a .env file at the repository root (.env is gitignored).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "Datasets"

# Pinned export identity. These values come from the exported data.yaml and must
# not be changed casually: bumping VERSION produces a different dataset and
# invalidates data/manifest.sha256 along with every benchmark number derived
# from it.
WORKSPACE = "enterprise-9gout"
PROJECT = "dji-roco-central"
VERSION = 1

# Roboflow export format -> local directory name, matching the paths recorded in
# data/manifest.sha256.
FORMATS = {
    "yolov11": "DJI ROCO Central.v1i.yolov11",
    "coco": "DJI ROCO Central.v1i.coco",
}


def load_api_key() -> str:
    """Read ROBOFLOW_API_KEY from the environment, falling back to .env."""
    import os

    key = os.environ.get("ROBOFLOW_API_KEY")
    if key:
        return key.strip()

    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() == "ROBOFLOW_API_KEY":
                return value.strip().strip("'\"")

    sys.exit(
        "No Roboflow API key found.\n"
        "Set it in your environment:\n"
        '    $env:ROBOFLOW_API_KEY = "your_key"      (PowerShell)\n'
        "or create a .env file at the repository root containing:\n"
        "    ROBOFLOW_API_KEY=your_key\n"
        "Your key is at https://app.roboflow.com/settings/api\n"
        "(.env is gitignored - never commit the key.)"
    )


def download(fmt: str, api_key: str, overwrite: bool) -> None:
    target = DEST / FORMATS[fmt]
    if target.exists() and not overwrite:
        print(f"[skip] {target.name} already exists (use --overwrite to refetch)")
        return

    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("The roboflow package is not installed. Run: pip install roboflow")

    print(f"[fetch] {PROJECT} v{VERSION} as '{fmt}' -> {target}")
    rf = Roboflow(api_key=api_key)
    version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
    version.download(fmt, location=str(target), overwrite=overwrite)
    print(f"[done] {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=sorted(FORMATS),
        action="append",
        dest="formats",
        help="Export format to download (repeatable). Defaults to all.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Refetch even if the target directory already exists.",
    )
    args = parser.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    api_key = load_api_key()

    for fmt in args.formats or sorted(FORMATS):
        download(fmt, api_key, args.overwrite)

    print("\nNext step - verify integrity against the committed manifest:")
    print("    python scripts/verify_data.py")


if __name__ == "__main__":
    main()
