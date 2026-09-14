"""Export a trained YOLO checkpoint to ONNX, optionally INT8-quantized.

WHY
    CAMERA_HANDOFF.md calls export and quantization "the largest available speedup and
    entirely unquantified" - every CPU number in this project is native fp32 PyTorch. It
    matters more since deployment moved to an Apple Silicon laptop, where ONNX Runtime is
    strong and CUDA is irrelevant.

THE DESIGN IS "IT IS JUST ANOTHER WEIGHTS FILE"
    Ultralytics' YOLO() loads a .onnx through onnxruntime and keeps the same preprocess
    and postprocess it uses for a .pt. So an exported model flows through
    predict_to_coco.py, evaluate_detection.py and benchmark_cpu.py unchanged, and the
    comparison is genuinely like-for-like: the only thing that differs between a `.pt` row
    and its `_onnx` row is the inference engine.

SCOPE: YOLO FAMILY ONLY
    load_ssd and load_frcnn rebuild CUSTOM modules from raw state_dicts, so exporting them
    means hand-writing torch.onnx.export and re-validating anchor decoding. Faster R-CNN
    is disqualified on latency regardless (0.45 FPS on one core). Say so in the write-up
    rather than implying the whole field was exported.

TWO THINGS THIS SCRIPT REFUSES TO GET WRONG
    1. CALIBRATION SPLIT. INT8 static quantization needs real images to measure activation
       ranges. Ultralytics defaults that to `split: val` - which is the split every
       reported number comes from, so calibrating there would leak val statistics into the
       model being scored on val. This script hard-codes split="train" and asserts what it
       resolved to. It is deliberately NOT a CLI flag.

    2. NMS STAYS IN PYTHON. Exporting with nms=True moves non-max suppression inside the
       ONNX graph, which silently shifts its cost out of the "postprocess" bucket and into
       "inference". Total latency would still be right, but the stage breakdown would stop
       meaning what it means for the .pt rows. Left at the default nms=False.

Usage:
    python scripts/export_onnx.py --weights runs/detect/fast_640/weights/best.pt \\
        --imgsz 640 --variant fp32
    python scripts/export_onnx.py --weights runs/detect/fast_640/weights/best.pt \\
        --imgsz 640 --variant int8
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import os

# Ultralytics auto-pip-installs missing optional deps by default, and for ONNX it reaches
# for onnxruntime-GPU. That would add a CUDA execution provider to a project whose entire
# contribution is CPU measurement - a silent methodology breach, not a convenience. Set
# before importing ultralytics anywhere.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "roco_central.yaml"

# Not a flag. See the module docstring - making this configurable is how val leakage
# gets reintroduced by someone in a hurry six months from now.
CALIBRATION_SPLIT = "train"


def check_calibration_source(data_yaml: Path) -> int:
    """Assert the calibration split resolves to the train list, and count its images."""
    import yaml  # noqa: PLC0415

    if not data_yaml.is_file():
        sys.exit(f"Missing {data_yaml}\nRun: python scripts/make_splits.py --from-assignment")
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    listed = config.get(CALIBRATION_SPLIT)
    if not listed:
        sys.exit(f"{data_yaml} has no '{CALIBRATION_SPLIT}:' key; cannot calibrate safely.")

    path = Path(listed)
    if not path.is_absolute():
        path = (data_yaml.parent / path).resolve()
    if not path.is_file():
        sys.exit(f"Calibration list {path} does not exist.")

    count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    val_listed = config.get("val")
    if val_listed and Path(val_listed).name == path.name:
        sys.exit(f"Calibration list and val list are the same file ({path}). Refusing: "
                 f"calibrating on val leaks into every number reported on val.")

    print(f"calibration : {path}")
    print(f"              {count} images from the '{CALIBRATION_SPLIT}' split (never val)")
    return count


def verify_export(onnx_path: Path, source_names: dict) -> None:
    """Reload the artifact and confirm class names survived.

    onnxruntime's quantizer rebuilds the ModelProto, and if it drops the metadata that
    carries class names, predict_to_coco's name-based category mapping fails. It fails
    loudly rather than mis-mapping, but catching it here is cheaper than catching it
    three scripts downstream.
    """
    from ultralytics import YOLO  # noqa: PLC0415

    reloaded = YOLO(str(onnx_path))
    names = reloaded.names
    if names != source_names:
        print(f"WARNING: class names differ after export.")
        print(f"  checkpoint: {source_names}")
        print(f"  exported  : {names}")
        sys.exit("Refusing to report a model whose class mapping changed in export.")
    print(f"verified    : {onnx_path.name} reloads with {len(names)} classes intact")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True,
                        help="Trained .pt checkpoint (YOLO family only).")
    parser.add_argument("--imgsz", type=int, required=True,
                        help="Export at this size. Static shape, so it must match the "
                             "size the config is benchmarked and scored at.")
    parser.add_argument("--variant", choices=("fp32", "int8"), default="fp32")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Dataset yaml supplying INT8 calibration images.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Copy the artifact here as well (optional).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.weights.is_file():
        sys.exit(f"Missing {args.weights}")
    if args.weights.suffix != ".pt":
        sys.exit(f"Expected a .pt checkpoint, got {args.weights.suffix}")

    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(str(args.weights))
    source_names = dict(model.names)
    print(f"checkpoint  : {args.weights}")
    print(f"imgsz       : {args.imgsz}   variant: {args.variant}")

    # device="cpu" is load-bearing, not tidiness. ultralytics/engine/exporter.py picks
    # its dependency as `"onnxruntime-gpu" if "cuda" in self.device.type else
    # "onnxruntime"`, so exporting on CUDA makes it pip-install the GPU runtime - which
    # shares the onnxruntime namespace, wins the import, and puts CUDA and TensorRT ahead
    # of CPU in the provider order. Every later latency row would then be a GPU
    # measurement wearing a CPU label. Export device does not change the traced graph or
    # the weights, so this costs nothing but calibration wall time.
    kwargs = dict(format="onnx", imgsz=args.imgsz, batch=1, dynamic=False,
                  simplify=True, nms=False, device="cpu")
    if args.variant == "int8":
        check_calibration_source(args.data)
        # quantize=8, not int8=True: the latter is a deprecated alias in ultralytics
        # 8.4.120 and only survives via a deprecation shim.
        kwargs.update(quantize=8, data=str(args.data), split=CALIBRATION_SPLIT)

    produced = Path(model.export(**kwargs))

    # An fp32 export lands as best.onnx, and a later int8 export of the same checkpoint
    # DELETES that intermediate on its way to best_int8.onnx. Renaming the fp32 artifact
    # to a variant-specific name keeps both on disk regardless of the order they are run
    # in - otherwise exporting fp32 then int8 silently leaves you with only int8.
    if args.variant == "fp32" and produced.stem == args.weights.stem:
        stable = produced.with_name(f"{produced.stem}_fp32.onnx")
        produced.replace(stable)
        produced = stable
    print(f"exported    : {produced}  ({produced.stat().st_size / 1024**2:.1f} MiB)")

    baseline = args.weights.stat().st_size / 1024**2
    ratio = produced.stat().st_size / 1024**2 / baseline
    print(f"size        : {baseline:.1f} MiB (.pt) -> "
          f"{produced.stat().st_size / 1024**2:.1f} MiB  ({ratio:.2f}x)")

    verify_export(produced, source_names)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, args.out)
        print(f"copied      : {args.out}")

    print("\nScore it, then benchmark it:")
    print(f"  python scripts/predict_to_coco.py --family yolo --weights {produced} "
          f"--imgsz {args.imgsz} --out results/predictions/<name>.json")
    print(f"  python scripts/benchmark_cpu.py --family yolo --weights {produced} "
          f"--imgsz {args.imgsz} --name <name> --cores 1")


if __name__ == "__main__":
    main()
