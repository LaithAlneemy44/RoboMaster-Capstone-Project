# Running this on the MacBook

The deployment target is an Apple Silicon MacBook. Everything in this repo was built and
measured on an x86 Windows desktop, so this is a first run on new hardware, not a copy of
a working setup. Expect a surprise or two and read the "known differences" section before
concluding something is broken.

## 1. Clone and create the environment

```bash
git clone <your-repo-url> Capstone
cd Capstone
python3 -m venv .venv
source .venv/bin/activate
```

## 2. Install PyTorch — NOT the version in requirements.txt's comment

`requirements.txt` tells you to install torch from the `cu126` index. **That is for the
Windows machine with the GTX 1060 and is wrong here.** Apple Silicon has no CUDA. Install
the plain build:

```bash
pip install torch torchvision
pip install -r requirements.txt
```

If pip tries to pull `onnxruntime-gpu` at any point, stop and remove it — it shares the
`onnxruntime` namespace, wins the import, and would put CUDA ahead of CPU in the
execution-provider order. `scripts/benchmark_cpu.py` has a guard that refuses to record a
row if the session is not on CPU, but it is better not to install it at all.

## 3. Grant camera permission

macOS blocks camera access **silently** — `cv2.VideoCapture` returns a handle that reads
black frames or nothing, with no error. Before debugging anything camera-related:

**System Settings → Privacy & Security → Camera** → enable whichever app runs Python
(Terminal, iTerm, or VS Code).

## 4. Check what you actually have

```bash
python scripts/robot/camera_checkup.py --list
python scripts/robot/camera_checkup.py --device 0
```

This reports the negotiated pixel format, the **delivered** frame rate against the
claimed one, capture latency and jitter, and whether exposure can be locked. All of it is
information the project does not yet have for any real camera.

## 5. The camera test, with no robot

Play `results/test_footage.mp4` fullscreen (10 seconds of real match footage, loops
fine), point the camera at the screen, and run:

```bash
python scripts/robot_gui.py --driver mock --source 0 \
    --weights runs/detect/fast_640/weights/best.pt --imgsz 640
```

Press **T** for auto-tracking. Green boxes on the robots means camera, model, tracker and
display all work end to end.

**This proves the plumbing, not the optics.** A robot on a screen is a different angular
size than a robot across an arena, so do not read anything into detection quality here.
That question needs a real plate at a real distance — the plate measurement in
`camera_checkup.py` answers it.

## 6. Exposure lock — untested code, first run

```bash
brew install uvc-util        # or build from source
python scripts/robot/camera.py -I 0        # list the controls THIS unit advertises
```

`scripts/robot/camera.py` was written against documentation and **has never run against
real hardware** — it is macOS-only and the repo's machine is Windows. Its name resolution,
parsing and failure reporting are covered by selftests; the actual UVC transaction is not.

Two things in it are still open:

- The control names vary by unit. The script matches candidates against what the device
  advertises and tells you if none match. `-I 0` prints the real list.
- `--exposure N` needs a value tuned once against arena lighting. Omitting it locks manual
  mode at the current exposure, which stops the hunting but does not set the level.

```bash
python scripts/robot_gui.py --source 0 --lock-exposure --exposure 200
```

## Known differences from the Windows machine

**No weights except two.** `runs/` is gitignored. Only `fast_640` and `fast_960` are
committed — the deployment front-runner and the armor leader. Everything else
(`yolo_960`, the SSD ladder, Faster R-CNN, the ONNX exports) stays on the Windows machine
and must be copied if needed.

**No datasets.** `Datasets/` and the tracking clip frames are not in git. Training,
accuracy scoring and the benchmark grid cannot run here without them. The camera test
does not need them.

**Generated splits are absent.** If a script complains about `data/splits/coco_val.json`
or `data/roco_central.yaml`, run `python scripts/make_splits.py --from-assignment` — but
it needs `Datasets/` present.

**`psutil.cpu_affinity()` does not exist on macOS.** `benchmark_cpu.py` will refuse to run
its core-capped grid and tell you so. Use `--no-core-cap`, which records
`core_constraint=none` so those rows can never be averaged with the affinity-capped
Windows ones. For an efficiency-core figure, wrap the command in `taskpolicy -b`. That
gives a two-point comparison, not a 1/2/4/6 sweep — Apple Silicon cores are heterogeneous
and a bare core count is meaningless there.

**Nothing measured here is comparable to the committed numbers.** Different instruction
set, different BLAS, different threading. Re-measuring on this machine is the single
largest open item in the project (CAMERA_HANDOFF.md issue 1), and the results belong in
their own rows, not merged with the Windows ones.

**ONNX INT8 may behave differently.** It loses on this project's Zen 3 desktop because
that CPU has AVX2 but no VNNI, so INT8 gets no hardware acceleration. Apple Silicon's
integer path is different, and that conclusion should be re-tested rather than carried
over.
