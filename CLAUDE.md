# CLAUDE.md

Context for Claude Code working on this project. Read this before making changes.

## What this project is

A semester-long computer vision research project comparing **classical vs deep learning
perception models** for real-time autonomous targeting in RoboMaster-style robot competitions.

The research question: *How do classical and deep learning perception models compare in
real-time autonomous targeting in robot competitions?*

The core contribution / gap in the literature: most existing benchmarks run on GPUs. This
project measures **CPU-only performance** — the case where no discrete GPU is available.
Keep this framing central; it dictates a lot of the methodology below.

**The motivation wording changed and the write-up must follow.** This clause originally
read "what actually matters for on-robot deployment". Deployment is now a MacBook that runs
the pipeline off the robot and sends commands over a WiFi relay, which is neither on-robot
nor constrained. The research question is
untouched and still answerable — classical versus deep learning, CPU rather than GPU — but
say "commodity CPU without a discrete GPU", not "low-power on-robot board". Wording
problem, not a results problem.

A "perception model" = one **detection** model paired with one **tracking** model.

## Where the project is now — read before planning anything

**The perception work is complete and measured. Do not rebuild it.** 34 detection configs
scored and CPU-benchmarked (170 rows in `results/performance.csv`), the full
detector x tracker matrix (108 rows), all four trackers, both classical models tuned,
seven-fold cross-validation, and a working driver-station GUI with headless tests. Check
`results/` before starting anything that sounds like it has been done.

Two documents carry the detail:
- **`CAMERA_HANDOFF.md`** — every measured number, 12 numbered open issues, and the
  methodology traps. The authoritative record. Read it before proposing work.
- **`SETUP_MACOS.md`** — running on the deployment laptop, and what breaks there.

Both predate the current deployment design (see *Deployment architecture*). Where they call
re-measuring on the MacBook the project's largest open item, read it narrowly: research
benchmarks stay on the Ryzen, and what the MacBook needs is its own deployed frame rate.

Findings that constrain further work:

- **Armor detection failed outside YOLO because of an ANCHOR FLOOR**, not small-object
  difficulty. Faster R-CNN gained armor AP 0.0108 -> 0.3737 *and* overall mAP; SSD gained
  armor only by trading 15% mAP; anchor-free YOLO was immune. One cause, three severities
  set by architecture.
- **Across-match variance dwarfs every other uncertainty.** Leave-one-clip-out over all
  seven clips gives mAP 0.6682 +/- 0.0361, against a within-split bootstrap CI of 0.0116
  and run-to-run of 0.0075. Any claim about an unseen venue must quote the across-clip sd.
- **The committed split is the HARDEST of the seven clips**, so every headline number is
  conservative rather than flattering. Worth stating in the write-up.
- **ONNX fp32 helps small models and hurts large ones**; INT8 loses on both axes here
  (11-17% armor AP for negative speed) because this CPU has AVX2 but no VNNI. That is a
  Zen 3 result — re-test on Apple Silicon before carrying it over.
- **Trackers are frame-rate coupled.** ~9% MOTA lost at 6 fps on the 30 fps constants;
  rescaling `max_age`/`min_hits` recovers about half but raises ID switches. MOTA versus
  identity is a real trade for a targeting system.
- **No model distinguishes red team from blue.** ROCO's classes carry no team dimension,
  so every DL model here is team-blind by construction. Only the classical detector's hue
  bands could tell friend from foe, and only with a colour camera.

## Models in scope

### Detection models
- **YOLO** — fine-tuned from COCO-pretrained weights via Ultralytics
- **Fast YOLO** — smaller/faster YOLO variant
- **MobileNet-SSD** — multiple configs by varying width & resolution multipliers
- **Classical detector** (built from scratch, OpenCV): RGB→HSV → template matching over
  matching-hue regions → grayscale edge map → normalised cross correlation (NCC).
  Multiple parameter configs. No training — parameter tuning only.

### Tracking models
- **GOTURN** — use PRETRAINED weights, light fine-tuning ONLY. Do NOT train from scratch.
- **Classical tracker** (built from scratch): kinematics-based prediction + Kalman filter.
  No training — parameter tuning only.
- **SORT** — a complete perception model already (Faster R-CNN detection + Kalman tracking).
  Tested standalone, not combined with other models.

Roughly 7 combined perception models to compare (excluding parameter-varied configs).

## Hardware — keep the roles separate

- **Windows desktop — training and research benchmarking.** Ryzen 5 5600G (6 physical /
  12 logical) plus a GTX 1060 6GB. **Training happens on the GPU, benchmarking on the CPU;
  never mix them.** The CPU produced every latency, FPS, CPU% and RAM number in
  `results/`, and benchmarks for the research question still run here, CPU only, with the
  1/2/4/6-core sweep. 6GB VRAM is the training limiter — on CUDA OOM, lower batch or image
  size. **Datasets and most weights live only here.**
- **Apple Silicon MacBook — the live demo platform, not the benchmark platform.** It runs
  the perception pipeline on its CPU for the demo. ARM64 shares neither the instruction set
  nor the threading behaviour of the Ryzen, so no committed performance row describes it,
  and nothing measured on it belongs in the research comparison. What it does need
  measured is its own deployed frame rate, because the tracker constants depend on it (see
  *Current status*).
- **The robot — an Arduino Mega, with no compute board.** Nothing runs perception on the
  robot. The two Picos only relay commands (see *Deployment architecture*).

**Never benchmark inference on a GPU.** The CPU numbers are the entire contribution.
`benchmark_cpu.py` now asserts this rather than trusting it: it checks every torch
parameter is on CPU, and for ONNX it reads back which execution provider the session
actually bound and refuses to write a row otherwise.

## Deployment architecture (current)

```
C270 ─────USB────▶ MacBook ──USB──▶ Pico A ···WiFi···▶ Pico B ──UART?──▶ Arduino Mega
                   pipeline,        with the            on the           robot
                   CPU only         MacBook             robot            firmware
```

- **The MacBook runs detection and tracking on CPU** and sends turret commands to Pico A
  over USB.
- **Pico A and Pico B are Raspberry Pi Pico 2 WH boards that only relay.** Pico A stays
  with the MacBook; Pico B sits on the robot, receives commands from Pico A over WiFi, and
  passes them to the Arduino Mega.
- **No wire between the MacBook and the robot.** Reviewers rejected a direct wired
  MacBook-to-Arduino link as a risk to the laptop. Do not design anything that connects the
  MacBook to the robot by wire. Camera-to-MacBook by wire is allowed.
- **Camera capture on macOS goes through AVFoundation, not V4L2**, and OpenCV capture
  properties behave differently from Linux. `scripts/robot/camera.py` already drives
  exposure through `uvc-util` for that reason: `CAP_PROP_EXPOSURE` is unreliable on
  AVFoundation.

### The camera — Logitech C270 HD

The camera is a **Logitech C270 HD webcam**. It replaces an earlier design note that named an
Arducam global-shutter board; ignore any Arducam, OV9281, M12 or 1280×800 references left in
older commits. The C270 is a colour sensor, which matters twice over:
- the classical detector separates red from blue by hue, and
- every DL detector, `fast_960` included, was trained on colour ROCO footage.

Spec-sheet figures (confirm with `camera_checkup.py`, not from this note): native
**1280×720**, not the 1920×1080 every committed number was measured at, but the same 16:9
aspect, so frames letterbox exactly like the training data; **30 fps** maximum, and
commonly only in **MJPG** (uncompressed YUY2 at 720p is usually well under that), so expect
a decode per frame; rolling shutter; fixed focus; 55° diagonal, about 49° horizontal.
It is **USB-A, USB 2.0**, so a USB-C MacBook needs an adapter or hub.

That FOV gives about 26 px/degree — the best of the options the handoff compared
(CAMERA_HANDOFF §4), and plate size is what armor detection lives on. The lens is fixed, so
if plates still measure small at engagement distance there is no cheap lens swap; the
levers are the network input size and cropping.

`camera_checkup.py` compares a clicked plate against the 25×22 px training median, which was
measured at 1920×1080. Both frames are resized to the same network input, so at 1280×720
the equivalent median is about 17×15 px. Scale before reading its verdict.

### Robot control link — open decisions

None of these is decided. **Do not pick defaults for them silently; flag them wherever code
depends on them.**

1. Wire protocol between the MacBook and Pico A — likely USB serial.
2. Protocol between the two Picos over WiFi (e.g. UDP vs TCP), and which Pico hosts the
   network.
3. Interface between Pico B and the Arduino Mega — likely UART. The Pico is 3.3 V logic and
   the Mega 5 V, so the voltage on the Mega TX → Pico RX line must be checked.
4. Command message format (e.g. pitch/yaw angles plus a fire flag), and how stale or
   dropped commands are handled.
5. Where the camera is mounted — on the turret (cabled to the MacBook) or fixed off the
   robot. The design does not say. `aiming.py` steers on pixel error from the image centre,
   which only works if the camera moves with the turret.

**Code already depends on items 1 and 4, through a draft written before the relay existed.**
`docs/ROBOT_PROTOCOL.md` and `scripts/robot/protocol.py` specify a single serial hop from
host to robot microcontroller: 115200 8N1 ASCII lines (`DRV`, `GIM`, `FIRE`, `INTAKE`,
`STOP`, `PING`), turret commands as *rates* rather than angles because the host has no
encoder feedback, and a firmware rule to stop all motion after 300 ms of silence.
`driver.py`, `aiming.py` and the GUI all speak it, and `SerialDriver` would open Pico A's
USB port unchanged. Treat it as a proposal, not a contract: whether it survives the relay,
rates versus angles, and which hop enforces the 300 ms stop are exactly items 1–4.

### On the MacBook specifically

A fresh clone there has the code, `fast_640` and `fast_960`, and a test clip. It does
**not** have `Datasets/`, the tracking clip frames, or any other weights — so training and
accuracy scoring cannot run there, and the research benchmark grid does not belong there.
The camera test can run. See `SETUP_MACOS.md`.

Three things will go wrong in order:
1. **`requirements.txt` says to install torch from the `cu126` index.** That is correct
   for the Windows machine and wrong on Apple Silicon. Install plain `torch torchvision`.
2. **macOS blocks camera access silently.** `cv2.VideoCapture` returns a handle that reads
   nothing, with no error, until Terminal has Camera permission in System Settings.
3. **`psutil.cpu_affinity()` does not exist on Darwin**, so `benchmark_cpu.py`'s
   core-capped grid refuses to run. If you time the pipeline there with it, use
   `--no-core-cap`, which records `core_constraint=none`. Those rows describe the demo
   machine: they are never research rows and never averaged with the Ryzen's. Apple
   Silicon cores are heterogeneous, so a bare core count means nothing there anyway.

## Dataset

### Detection (still images)
- **DJI ROCO Central** from Roboflow — 2655 images.
  https://universe.roboflow.com/enterprise-9gout/dji-roco-central
- Classes: car, armor, base, watcher (+ possibly "ignore").
- Export TWICE: **YOLO format** (YOLOv8/v11) for YOLO models via Ultralytics, and
  **COCO JSON** for MobileNet-SSD. Same data, two formats.
- Prefer the Roboflow export CODE SNIPPET over the manual zip — cleaner Ultralytics integration.
- Do NOT mix in the North/South ROCO variants — it changes class balance and breaks the
  2655-image framing in the proposal.
- Split: 85/15 train/val for detection (no test set needed — combined models are what's tested).
- **SETTLED — the split is group-aware by match clip.** The 2655 images are frames from only
  SEVEN match recordings, so a random split puts consecutive frames of the same moment in
  both train and val and inflates val mAP. Default holds out the whole
  `-VsBorn-of-Fire_BO2_1` clip: 2258/397 = 85.05/14.95, which hits the proposal's 85/15
  with zero leakage. Applies to EVERY detection model. `make_splits.py` prints a per-clip
  leakage verdict every run; `--val-frac`/`--keep-export-split` are leaky and labelled so.
  Limitation to state in the write-up: val is one match, so clip-level n=1 — mAP confidence
  intervals come from bootstrapping over val images and do NOT capture across-match variance.
- Only `data/splits/assignment.csv` is committed. `data/roco_central.yaml`,
  `data/splits/*.txt` and `data/splits/coco_*.json` hold absolute paths or are bulky
  generated JSON, so they are gitignored — rebuild with
  `python scripts/make_splits.py --from-assignment` after a clone or a move.

### Tracking (video) — DONE, but read how it was actually built
- Seven clips from the ARC Robotics YouTube channel, MOT-format, in `data/tracking/`.
  Frames are gitignored; only `seqinfo.ini`, `gt.txt` and `assignment.csv` are committed.
- **The ground truth is MACHINE-generated, not hand-labelled.** `scripts/auto_label.py`
  produced it by motion association. The original plan was manual labelling; that is not
  what happened, and the difference matters: those labels share assumptions with SORT and
  the classical tracker and not with appearance-based GOTURN/VitTrack, so **tracking
  accuracy comparisons are biased toward the Kalman trackers.** CPU timings are
  unaffected — they need no labels. This is CAMERA_HANDOFF issue 7 and belongs in the
  write-up's limitations.
- Split declared in `data/tracking/assignment.csv`: arc03 is val, arc04 is test. It was
  declared *after* results existed, which is defensible only because nothing is ever
  fitted to it — no tracker is trained, GOTURN and VitTrack are frozen. It constrains
  future tuning, not past results (handoff issue 8).

## Frameworks
- **Ultralytics** — YOLO training/inference
- **PyTorch** — underlying DL. **Which build depends on the machine.** On the Windows
  desktop install the `cu126` build (the GTX 1060 is Pascal/sm_61 and CUDA 12.8 dropped
  it) and verify with `torch.cuda.is_available()`. On the MacBook install plain
  `torch torchvision` — there is no CUDA, and `cuda.is_available()` returning False is
  correct there, not a fault. Do not "fix" it by reaching for MPS: that is a GPU, and
  benchmarking on it would invalidate the contribution.
- **OpenCV + NumPy** — classical detector, classical tracker, GOTURN
- **Python** end to end

## Current status

- **Camera capture code exists, but has never run against a real camera.**
  `scripts/robot/vision.py` (`FrameSource`) and `scripts/robot/camera_checkup.py` open
  devices with a plain `cv2.VideoCapture(index)`, which the standard opencv-python build
  resolves to AVFoundation on macOS. Nothing requests `cv2.CAP_AVFOUNDATION` explicitly
  or sets this camera's MJPG format and 1280×720 mode. The next software step is bringing
  that code up against the C270 on the MacBook and hardening it, not writing a second
  capture path beside it.
- **Tracker parameters are hard-coded for 30 fps** and need re-tuning once the deployed
  frame rate on the MacBook is measured. `run_trackers.rescale_temporal()` already rescales
  `max_age`/`min_hits` by frame stride; the GUI's `vision.AutoTracker` does not call it.
  Every tracker score so far was on whole robots, while the GUI tracks armor plates.
- **The Picos will not arrive before the October 2026 prototype showing.** That demo does
  not use the vision system: it runs a scripted move, rotate, shoot and collect routine
  coded into the Arduino Mega firmware and started on power-up. That sketch is
  `firmware/demo_routine/demo_routine.ino`: it builds for the Mega and for an Uno (the
  drive test board), picking the pin set at compile time. Every pin is still a placeholder,
  and the collector and shooter are switched off until the subsystem owners' own sketches
  are merged in.

## Priorities / order of work

The original plan — verify data, confirm the GPU, build the measurement harness before
scaling, one end-to-end pipeline first, then breadth — **is complete**. It is kept here
because it explains why the code is shaped as it is: the harness came first, and that is
why every model goes through one evaluator rather than its own.

What is actually open, roughly in order of value:

1. **Camera bring-up on the MacBook.** Plug in the C270, then run
   `scripts/robot/camera_checkup.py`: negotiated format, delivered frame rate, capture
   latency, and plate pixel size at real engagement distance — the measurement that
   decides whether detection can work at all. Capture cost is still "not measured"
   everywhere in the handoff.
2. **The deployed frame rate**, measured on the MacBook with the real camera, then the
   tracker constants re-tuned for it (see *Current status*).
3. **The exposure lock is untested code.** `scripts/robot/camera.py` was written against
   documentation and has never run against hardware. Its control names need confirming on
   the real unit and its exposure value needs tuning once against arena lighting.
4. **Confidence threshold is hardcoded 0.25 in six files and was never swept** — that is
   the live system's operating point, and it is unexplored (handoff issue 9).
5. **The robot control link** — the open decisions under *Deployment architecture*. Blocked
   until the Picos arrive, and not to be settled by default in code.
6. Remaining smaller gaps are enumerated as the 12 numbered issues in `CAMERA_HANDOFF.md`.

**Finish measurement before adding models.** The comparison is already broad enough to
answer the research question, and its numbers are complete on the benchmark platform.
What the demo lacks is a working camera and a frame rate on the MacBook.

## Known traps (do not repeat)
- **Never train GOTURN from scratch** — semester-eating. Pretrained + light fine-tune only.
- **Don't use Roboflow's hosted ROCO model** — black box, can't control architecture, can't
  measure CPU/RAM locally. Download weights and run locally instead.
- **Don't benchmark inference on the GPU** — the CPU numbers are the whole contribution.
- **GOTURN and SORT have dependency rot** — older repos, may fight current OpenCV/PyTorch.
  Confirm they build before committing to them.
- **Define "accuracy" for the combined detection+tracking pipeline up front** — it's not one
  number. Decide the scoring method before collecting data so it isn't redefined afterward.
- The classical models need MORE coding time than the DL models, not less — you're building
  the algorithms, not importing them.

### Measurement traps learned the hard way — all of these produced wrong numbers first

- **The documented knob is often not the one that binds.** `torch.set_num_threads()` and
  `OMP_NUM_THREADS` both did nothing; only `cpu_affinity()` constrained PyTorch. ONNX
  Runtime ignores affinity entirely and needs `intra_op_num_threads` *and* spin-wait
  disabled — with defaults it reported ONNX as 6.5x slower than PyTorch. Verify a control
  actually took effect; never assume.
- **Diagnose from the SHAPE of an anomaly, not its size.** A penalty that tracks the core
  cap, or a model that gets slower as cores are added, cannot be an engine difference.
  That reasoning caught two separate false results.
- **Benchmarks need an idle machine, and contention is invisible by default.**
  `baseline_cpu_pct` samples only *before* a cell; `external_load_cores` samples
  throughout. A model once measured slower on six cores than two because other work was
  running. The idle floor here is ~0.56 cores, so thresholds must be calibrated, not
  guessed.
- **CSV writers must validate the header before appending.** Building a `DictWriter` from
  the row rather than the file silently mis-maps every later read when the schema grows.
  This happened once (`cpu_model` read back as `0.86`); all three writers now guard.
- **Never let `onnxruntime-gpu` install.** It shares the `onnxruntime` namespace, wins the
  import, and puts CUDA ahead of CPU. Ultralytics pulls it in when exporting on CUDA, so
  `export_onnx.py` passes `device="cpu"`.
- **Quantization calibration defaults to the val split.** That is leakage straight into
  the numbers being reported. `export_onnx.py` hard-codes `split="train"` and asserts it.
- **Behaviour tests do not check documentation.** The GUI's on-screen key legend went
  stale while eleven passing tests asserted the new bindings. Rendering it and looking
  found it in seconds.

## Metrics to compute
Mean accuracy, std of accuracy, precision, recall, F1, mAP, IoU, ID switches, mean FPS,
mean latency, CPU usage, RAM usage, confidence intervals — all comparable across models.

## Ethics / handling notes
- ARC YouTube footage may be licensed — respect copyright, do not redistribute raw video.
- Project is low-risk desktop research; note potential military/surveillance applicability
  is explicitly out of scope and not the intent.
