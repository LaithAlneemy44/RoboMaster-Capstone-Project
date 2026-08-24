"""Download the weights OpenCV's trackers need but does not ship.

OpenCV compiles GOTURN and VitTrack into the library while distributing their trained
networks separately, from locations that are neither stable nor uniform. CLAUDE.md flags
GOTURN dependency rot as a known trap; this script is the mitigation, and it verifies by
constructing each tracker rather than by trusting a file to be present.

GOTURN    ~343 MiB Caffe model. Lived in opencv_extra/testdata/tracking, was removed
          from master, and now survives only at the historical commit that
          testdata/dnn/gsoc2016-goturn/README.md points back to. GitHub caps blobs at
          100 MiB, so it is four split-zip parts that must be concatenated to unzip.
          The commit SHA below is pinned deliberately - master 404s.

VitTrack  ~700 KiB ONNX from opencv_zoo, stored via Git LFS. raw.githubusercontent
          serves the LFS *pointer* (131 bytes) rather than the model, so it is fetched
          through media.githubusercontent.com instead.

NanoTrack is deliberately absent. Its weights are in neither opencv_extra nor opencv_zoo
(the zoo has NanoDet, a detector, which is a different thing), so cv2.TrackerNano would
need third-party ONNX files. Left out rather than pulled from an unverified source.

Every cv2 tracker defaults to looking for its weights in the CURRENT WORKING DIRECTORY,
which would force each caller to chdir. Use the *_params() accessors below instead.

Usage:
    python scripts/fetch_tracker_weights.py
    python scripts/fetch_tracker_weights.py --overwrite
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "models" / "trackers"

# Pinned: master no longer has the caffemodel. Do not float this to a branch.
GOTURN_SHA = "c4219d5eb3105ed8e634278fad312a1a8d2c182d"
GOTURN_RAW = (
    f"https://raw.githubusercontent.com/opencv/opencv_extra/{GOTURN_SHA}"
    "/testdata/tracking"
)
GOTURN_PROTOTXT = ("goturn.prototxt", 7_949)
GOTURN_CHUNKS = [
    ("goturn.caffemodel.zip.001", 99_614_720),
    ("goturn.caffemodel.zip.002", 99_614_720),
    ("goturn.caffemodel.zip.003", 99_614_720),
    ("goturn.caffemodel.zip.004", 61_040_147),
]

# media.githubusercontent.com, not raw - raw returns the LFS pointer.
VIT_NAME = "object_tracking_vittrack_2023sep.onnx"
VIT_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main"
    f"/models/object_tracking_vittrack/{VIT_NAME}"
)
VIT_SIZE = 714_726


def goturn_params():
    """cv2.TrackerGOTURN_Params pointing at this project's downloaded weights."""
    import cv2

    params = cv2.TrackerGOTURN_Params()
    params.modelTxt = str(DEST / "goturn.prototxt")
    params.modelBin = str(DEST / "goturn.caffemodel")
    return params


def vit_params():
    """cv2.TrackerVit_Params pointing at this project's downloaded ONNX."""
    import cv2

    params = cv2.TrackerVit_Params()
    params.net = str(DEST / VIT_NAME)
    return params


def fetch(name: str, url: str, expected: int, overwrite: bool) -> Path:
    import requests

    target = DEST / name
    if target.is_file() and not overwrite:
        actual = target.stat().st_size
        if actual == expected:
            print(f"[skip]  {name} ({actual:,} bytes)")
            return target
        # A truncated part still unzips into a corrupt model, so the size is checked
        # rather than mere existence.
        print(f"[refetch] {name}: {actual:,} bytes, expected {expected:,}")

    print(f"[fetch] {name} ...", end="", flush=True)
    response = requests.get(url, stream=True, timeout=120)
    if not response.ok:
        sys.exit(
            f"\nHTTP {response.status_code} for {url}\n"
            "The pinned source may have moved or been garbage-collected."
        )

    got = 0
    with target.open("wb") as fh:
        for block in response.iter_content(chunk_size=1 << 20):
            fh.write(block)
            got += len(block)
            print(f"\r[fetch] {name} {got / 1e6:.0f} MB ", end="", flush=True)
    print()

    if got != expected:
        target.unlink(missing_ok=True)
        sys.exit(f"{name}: got {got:,} bytes, expected {expected:,}. Removed.")
    return target


def assemble_goturn(overwrite: bool) -> Path:
    """Concatenate the split parts and unzip the caffemodel out of them."""
    model = DEST / "goturn.caffemodel"
    if model.is_file() and not overwrite:
        print(f"[skip]  goturn.caffemodel ({model.stat().st_size:,} bytes)")
        return model

    joined = DEST / "goturn.caffemodel.zip"
    print("[join]  concatenating 4 parts ...")
    with joined.open("wb") as out:
        for name, _ in GOTURN_CHUNKS:
            out.write((DEST / name).read_bytes())

    print("[unzip] extracting caffemodel ...")
    with zipfile.ZipFile(joined) as zf:
        inner = next(n for n in zf.namelist() if n.endswith(".caffemodel"))
        with zf.open(inner) as src, model.open("wb") as dst:
            while block := src.read(1 << 20):
                dst.write(block)
    joined.unlink()  # the 343 MiB intermediate is not worth keeping
    return model


def verify() -> None:
    """Construct each tracker for real. Nothing else proves the weights load."""
    import cv2
    import numpy as np

    # A textured patch, not a blank frame. VitTrack scores its match and reports lost
    # on uniform black - correct behaviour that reads like a failure in the log.
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[60:100, 80:120] = 255
    for label, make in (
        ("GOTURN", lambda: cv2.TrackerGOTURN.create(goturn_params())),
        ("VitTrack", lambda: cv2.TrackerVit.create(vit_params())),
    ):
        tracker = make()
        tracker.init(frame, (80, 60, 40, 40))
        ok, _ = tracker.update(frame)
        print(f"[check] {label} initialised and ran one update (ok={ok})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)
    name, size = GOTURN_PROTOTXT
    fetch(name, f"{GOTURN_RAW}/{name}", size, args.overwrite)
    for name, size in GOTURN_CHUNKS:
        fetch(name, f"{GOTURN_RAW}/{name}", size, args.overwrite)
    assemble_goturn(args.overwrite)
    fetch(VIT_NAME, VIT_URL, VIT_SIZE, args.overwrite)
    verify()

    print(f"\n[done]  {DEST}")
    print("Build trackers via the params accessors:")
    print("    from fetch_tracker_weights import goturn_params, vit_params")
    print("    cv2.TrackerGOTURN.create(goturn_params())")
    print("    cv2.TrackerVit.create(vit_params())")


if __name__ == "__main__":
    main()
