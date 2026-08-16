"""Verify the GPU is genuinely usable for training - not just detected.

torch.cuda.is_available() is NOT a sufficient check on this machine. The GTX 1060
is Pascal (compute capability 6.1 / sm_61), and CUDA 12.8 dropped Pascal support.
A torch wheel built without sm_61 kernels still reports is_available() == True and
still lets you move tensors to the device; it only fails at the first kernel launch
with "no kernel image is available for execution on the device" - which, in a
training run, means it dies several minutes in rather than immediately.

So this checks three things in order of increasing strength:
  1. CUDA is available and a device is present
  2. the device's compute capability appears in torch's compiled arch list
  3. a real kernel actually launches and produces the right answer

Usage:
    .venv/Scripts/python.exe scripts/check_gpu.py

Exits non-zero if the GPU cannot actually run a kernel.
"""

from __future__ import annotations

import sys


def main() -> None:
    try:
        import torch
    except ImportError:
        sys.exit("torch is not installed. See README.md (Setup).")

    print(f"torch            : {torch.__version__}")
    print(f"built for CUDA   : {torch.version.cuda}")

    if not torch.cuda.is_available():
        sys.exit(
            "\nFAILED - torch.cuda.is_available() is False.\n"
            "You probably have the CPU-only wheel. Reinstall with:\n"
            "  pip install --index-url https://download.pytorch.org/whl/cu126 "
            "torch torchvision"
        )

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch_list = torch.cuda.get_arch_list()
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free_b, _ = torch.cuda.mem_get_info(0)

    print(f"device           : {name}")
    print(f"capability       : sm_{major}{minor}")
    print(f"compiled archs   : {' '.join(arch_list)}")
    print(f"VRAM total       : {total_gb:.2f} GiB")
    print(f"VRAM free now    : {free_b / 1024**3:.2f} GiB")

    # 2. Is this GPU's architecture actually in the wheel?
    target = f"sm_{major}{minor}"
    if target not in arch_list:
        print(f"\nWARNING - {target} is not in torch's compiled arch list.")
        print("This wheel was likely built with CUDA >= 12.8, which dropped Pascal.")

    # 3. The only check that really counts: launch a kernel.
    try:
        a = torch.randn(512, 512, device="cuda")
        result = (a @ a.T).sum().item()
        torch.cuda.synchronize()
        if result != result:  # NaN
            raise RuntimeError("matmul produced NaN")
    except Exception as exc:  # noqa: BLE001 - want the raw CUDA message shown
        sys.exit(
            f"\nFAILED - the GPU was detected but could not run a kernel:\n  {exc}\n\n"
            "For a Pascal card (sm_61), install the cu126 build - cu128 and newer\n"
            "do not ship Pascal kernels:\n"
            "  pip install --force-reinstall --index-url "
            "https://download.pytorch.org/whl/cu126 torch torchvision"
        )

    print("\nOK - CUDA kernel launched and returned a real result. Safe to train.")

    # The display drives this same card, so usable VRAM is well under the 6 GiB
    # on the box. Budget batch size against free memory, not total.
    if free_b / 1024**3 < 5.0:
        print(
            f"Note: only {free_b / 1024**3:.2f} GiB is free - the desktop is using the "
            "rest.\n      If you hit CUDA OOM, lower --batch before touching --imgsz."
        )


if __name__ == "__main__":
    main()
