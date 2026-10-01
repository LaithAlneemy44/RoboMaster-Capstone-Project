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
read "what actually matters for on-robot deployment". Deployment is now a MacBook tethered
to the robot over USB, which is neither on-robot nor constrained. The research question is
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

## Hardware — three roles, do not conflate them

- **Windows desktop — training and all existing measurements.** Ryzen 5 5600G (6 physical
  / 12 logical) plus a GTX 1060 6GB. The GPU trains; the CPU produced every latency, FPS,
  CPU% and RAM number in `results/`. 6GB VRAM is the limiter — on CUDA OOM, lower batch or
  image size. **Datasets and most weights live only here.**
- **Apple Silicon MacBook — the deployment target, and nothing has been measured on it.**
  Camera on the turret, USB tether, inference on the laptop. ARM64 shares neither the
  instruction set nor the threading behaviour of the Windows numbers, so **none of the
  committed performance rows transfer.** Re-measuring here is CAMERA_HANDOFF issue 1, the
  largest open item in the project.
- **The robot — does not exist yet.** The serial contract is specified in
  `docs/ROBOT_PROTOCOL.md` and the driver station speaks it, but nothing has been driven.

**Never benchmark inference on a GPU.** The CPU numbers are the entire contribution.
`benchmark_cpu.py` now asserts this rather than trusting it: it checks every torch
parameter is on CPU, and for ONNX it reads back which execution provider the session
actually bound and refuses to write a row otherwise.

### On the MacBook specifically

A fresh clone there has the code, `fast_640` and `fast_960`, and a test clip. It does
**not** have `Datasets/`, the tracking clip frames, or any other weights — so training,
accuracy scoring and the benchmark grid cannot run there. The camera test can. See
`SETUP_MACOS.md`.

Three things will go wrong in order:
1. **`requirements.txt` says to install torch from the `cu126` index.** That is correct
   for the Windows machine and wrong on Apple Silicon. Install plain `torch torchvision`.
2. **macOS blocks camera access silently.** `cv2.VideoCapture` returns a handle that reads
   nothing, with no error, until Terminal has Camera permission in System Settings.
3. **`psutil.cpu_affinity()` does not exist on Darwin**, so the core-capped grid refuses
   to run. Use `--no-core-cap`, which records `core_constraint=none` so those rows can
   never be averaged with affinity-capped ones. Apple Silicon cores are also
   heterogeneous, so a bare core count is meaningless there — `taskpolicy -b` gives a
   two-point comparison, not a 1/2/4/6 sweep.

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

### Tracking (video) — biggest hidden cost
- No labelled RoboMaster tracking dataset exists. Must be BUILT by hand from clips on the
  ARC Robotics YouTube channel (~267 videos, mostly RoboMaster footage).
- Manual per-frame bounding box labelling. Decide the annotation format BEFORE labelling
  (MOT-style / consistent per-frame boxes) so GOTURN and SORT can both consume it.
- Split: 70/15/15 train/val/test (test set IS used here — evaluates combined models).
- Start this EARLY and run it in PARALLEL with everything else. It is the real bottleneck,
  not model training.

## Frameworks
- **Ultralytics** — YOLO training/inference
- **PyTorch** — underlying DL (install the CUDA build, not CPU-only; verify with
  `torch.cuda.is_available()`)
- **OpenCV + NumPy** — classical detector, classical tracker, GOTURN
- **Python** end to end

## Priorities / order of work

The original plan — verify data, confirm the GPU, build the measurement harness before
scaling, one end-to-end pipeline first, then breadth — **is complete**. It is kept here
because it explains why the code is shaped as it is: the harness came first, and that is
why every model goes through one evaluator rather than its own.

What is actually open, roughly in order of value:

1. **Re-measure on the MacBook.** Nothing has been. It is the deployment target and no
   committed number applies to it. Start with the camera test in `SETUP_MACOS.md`, then
   the benchmark grid with `--no-core-cap`.
2. **The camera.** Plate pixel size at real engagement distance is the measurement that
   decides whether detection can work at all — `scripts/robot/camera_checkup.py` takes it.
   Capture cost is still "not measured" everywhere in the handoff.
3. **The exposure lock is untested code.** `scripts/robot/camera.py` was written against
   documentation and has never run against hardware. Its control names need confirming on
   the real unit and its exposure value needs tuning once against arena lighting.
4. **Confidence threshold is hardcoded 0.25 in six files and was never swept** — that is
   the live system's operating point, and it is unexplored (handoff issue 9).
5. Remaining smaller gaps are enumerated as the 12 numbered issues in `CAMERA_HANDOFF.md`.

**Finish measurement before adding models.** The comparison is already broad enough to
answer the research question; what it lacks is a number on the hardware it will run on.

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
