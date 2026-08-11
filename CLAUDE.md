# CLAUDE.md

Context for Claude Code working on this project. Read this before making changes.

## What this project is

A semester-long computer vision research project comparing **classical vs deep learning
perception models** for real-time autonomous targeting in RoboMaster-style robot competitions.

The research question: *How do classical and deep learning perception models compare in
real-time autonomous targeting in robot competitions?*

The core contribution / gap in the literature: most existing benchmarks run on GPUs. This
project measures **CPU-constrained performance**, which is what actually matters for on-robot
deployment. Keep this framing central — it dictates a lot of the methodology below.

A "perception model" = one **detection** model paired with one **tracking** model.

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

## Hardware — IMPORTANT distinction

Two machine roles that must not be conflated:

- **Training / fine-tuning → GPU.** Local machine has an NVIDIA GTX 1060 6GB (CUDA-capable).
  Fine-tuning YOLO/MobileNet takes minutes to a couple of hours. 6GB VRAM is the limiter —
  if CUDA out-of-memory, lower batch size or image size. Cloud (Colab) is an acceptable
  alternative; trained weights are identical wherever training happens.
- **Benchmarking / inference → target CPU.** ALL latency/FPS/CPU%/RAM numbers must be
  measured on the representative competition CPU, NOT the GPU or a beefy desktop chip.
  The reported CPU must be decided and documented (e.g. Intel N100-class / on-robot board).
  This is the whole point of the project — do not benchmark inference on the GPU.

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
1. Verify dataset integrity and set up the environment (pin versions).
2. Confirm PyTorch sees the GPU (`torch.cuda.is_available()` → True).
3. Build the **measurement harness FIRST**, before scaling to all models. It must log
   mAP, IoU, precision, recall, F1, ID switches, mean FPS, latency, CPU%, RAM, confidence
   intervals — uniformly across all models. This is the most important code in the project.
4. Get ONE end-to-end pipeline working on a trivial case before adding breadth.
5. Then scale to all detection models, then tracking, then combine and benchmark.

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

## Metrics to compute
Mean accuracy, std of accuracy, precision, recall, F1, mAP, IoU, ID switches, mean FPS,
mean latency, CPU usage, RAM usage, confidence intervals — all comparable across models.

## Ethics / handling notes
- ARC YouTube footage may be licensed — respect copyright, do not redistribute raw video.
- Project is low-risk desktop research; note potential military/surveillance applicability
  is explicitly out of scope and not the intent.
