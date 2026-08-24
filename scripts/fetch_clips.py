"""Download an ARC Robotics clip and extract frames as a MOT-layout sequence.

Produces the directory structure MOT Challenge tools already understand, so motmetrics
and SORT read it without an adapter:

    data/tracking/<seq>/
        img1/000001.jpg ...      extracted frames, 1-based like MOT
        seqinfo.ini              MOT's own metadata file
        gt/gt.txt                written later by scripts/label_tracks.py

EXTRACTION RATE IS THE WHOLE POINT
    ROCO failed as tracking data because its frames were sampled several video frames
    apart, leaving armor plates displaced more than their own width between
    "consecutive" frames. That is unrecoverable after the fact. So this defaults to
    extracting EVERY frame at the source rate, records the rate in seqinfo.ini, and
    warns loudly when asked to downsample. Run scripts/verify_tracking_data.py on the
    result before labelling anything.

COPYRIGHT
    ARC footage is not ours to redistribute. Raw downloads land in data/tracking/_raw/,
    which is gitignored, and neither the video nor the extracted frames should be
    committed or published. Only annotations are project artefacts.

No ffmpeg required: video-only formats need no muxing, and OpenCV does the decoding.

Usage:
    python scripts/fetch_clips.py --url https://youtu.be/XXXX --name arc01 \\
        --start 120 --end 150
"""

from __future__ import annotations

import argparse
import configparser
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACKING = ROOT / "data" / "tracking"
RAW = TRACKING / "_raw"


def download(url: str, max_height: int) -> Path:
    """Fetch a video-only stream. Video-only avoids muxing, so ffmpeg is not needed."""
    import yt_dlp

    RAW.mkdir(parents=True, exist_ok=True)
    options = {
        "format": f"bv[height<={max_height}]/b[height<={max_height}]",
        "outtmpl": str(RAW / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)
        target = Path(ydl.prepare_filename(info))
        if target.is_file():
            print(f"[skip]  {target.name} already downloaded")
            return target
        print(f"[fetch] {info.get('title', url)}")
        print(f"        {info.get('width')}x{info.get('height')} "
              f"@ {info.get('fps')} fps, {info.get('duration')}s")
        ydl.download([url])
    if not target.is_file():
        sys.exit(f"yt-dlp reported success but {target} is missing.")
    return target


def extract(video: Path, out_dir: Path, start: float, end: float, fps: float) -> dict:
    """Decode frames into out_dir/img1. Returns metadata for seqinfo.ini."""
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        sys.exit(f"OpenCV could not open {video}")

    native_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_fps = fps or native_fps
    stride = max(1, round(native_fps / target_fps))
    if stride > 1:
        print(f"  WARNING: taking every {stride}th frame ({native_fps:.1f} -> "
              f"{native_fps / stride:.1f} fps).\n"
              "           This is how ROCO became untrackable. Prefer --fps 0.")

    images = out_dir / "img1"
    images.mkdir(parents=True, exist_ok=True)
    for stale in images.glob("*.jpg"):
        stale.unlink()

    capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    written = 0
    source_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        position = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if end and position > end:
            break
        if source_index % stride == 0:
            written += 1
            # 1-based %06d, exactly as MOT names its frames.
            cv2.imwrite(str(images / f"{written:06d}.jpg"), frame)
            print(f"\r[frames] {written} written ({position:.1f}s)", end="", flush=True)
        source_index += 1
    capture.release()
    print()

    if not written:
        sys.exit("No frames extracted - check --start/--end against the video length.")
    return {
        "native_fps": native_fps,
        "frame_rate": native_fps / stride,
        "seq_length": written,
        "width": width,
        "height": height,
    }


def write_seqinfo(out_dir: Path, name: str, meta: dict, url: str, start: float,
                  end: float) -> None:
    """MOT's seqinfo.ini, plus a provenance section of our own."""
    config = configparser.ConfigParser()
    config.optionxform = str  # MOT keys are CamelCase; do not lowercase them
    config["Sequence"] = {
        "name": name,
        "imDir": "img1",
        "frameRate": f"{meta['frame_rate']:.6g}",
        "seqLength": str(meta["seq_length"]),
        "imWidth": str(meta["width"]),
        "imHeight": str(meta["height"]),
        "imExt": ".jpg",
    }
    # Not part of the MOT spec, but the extraction rate is the thing that silently ruins
    # tracking data, so it is recorded where it cannot be lost.
    config["Provenance"] = {
        "sourceUrl": url,
        "startSeconds": str(start),
        "endSeconds": str(end or "eof"),
        "nativeFps": f"{meta['native_fps']:.6g}",
        "downsampled": str(abs(meta["frame_rate"] - meta["native_fps"]) > 1e-6),
    }
    with (out_dir / "seqinfo.ini").open("w", encoding="utf-8") as fh:
        config.write(fh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", required=True)
    parser.add_argument("--name", required=True, help="Sequence name, e.g. arc01.")
    parser.add_argument("--start", type=float, default=0.0, help="Seconds.")
    parser.add_argument("--end", type=float, default=0.0, help="Seconds, 0 = to end.")
    parser.add_argument("--fps", type=float, default=0.0,
                        help="Extraction fps. 0 = every frame (recommended).")
    parser.add_argument("--max-height", type=int, default=1080)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = TRACKING / args.name
    if (out_dir / "gt" / "gt.txt").is_file():
        sys.exit(f"{out_dir} already has labels - refusing to re-extract over them.")

    video = download(args.url, args.max_height)
    meta = extract(video, out_dir, args.start, args.end, args.fps)
    write_seqinfo(out_dir, args.name, meta, args.url, args.start, args.end)

    print(f"\n[done]  {out_dir}  ({meta['seq_length']} frames @ "
          f"{meta['frame_rate']:.1f} fps, {meta['width']}x{meta['height']})")
    print("\nGate it BEFORE labelling:")
    print(f"    python scripts/verify_tracking_data.py --frames {out_dir / 'img1'} "
          "--class car")


if __name__ == "__main__":
    main()
