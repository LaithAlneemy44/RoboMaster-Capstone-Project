# Camera + inference handoff

Everything measured in this repo that bears on putting a live camera in front of a
trained model and running it on a CPU. Written for a planning session with no prior
context beyond the project framing.

All numbers below come from committed CSVs in `results/`. Anything never measured is
marked **not measured** rather than estimated.

---

## 1. Benchmark board

**There is no target board yet. Every number in this project was measured on the
development desktop.**

| | |
|---|---|
| CPU | **AMD Ryzen 5 5600G** (Zen 3 APU, 6 physical / 12 logical cores) |
| RAM | 15.9 GiB |
| Reported as | `AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD` in every `cpu_model` column |
| OS | Windows 11 Pro 26200 |

Constrained hardware was simulated, not used. `scripts/benchmark_cpu.py` applies an
OS-enforced affinity cap via `psutil.cpu_affinity()` to 1, 2, 4, or 6 **distinct
physical** cores. Two notes that matter if you re-measure:

- `torch.set_num_threads()` and `OMP_NUM_THREADS` had **no effect** on this build.
  Affinity capping is the only mechanism that actually constrained the process.
- Adjacent logical CPUs are hyperthread siblings. The cap picks strided ids so that
  "2 cores" means two physical cores, not one core's two threads. 6 cores + SMT
  measured *slower* than 6 physical cores (811 vs 710 ms on Faster R-CNN).

**Treat the 1-core column as the constrained-hardware proxy.** A Zen 3 core at desktop
clocks is considerably faster than an N100 or an ARM SBC core, so 1-core Ryzen numbers
are an optimistic bound for a low-power board, not a prediction for one. CLAUDE.md asks
for a representative competition CPU to be chosen and documented — **that decision has
not been made, and nothing has been measured on candidate hardware.** This is the single
largest open item for the hardware phase.

### Was anything reported as CPU actually measured on GPU?

**No.** Verified against how each number was produced:

- Every latency / FPS / CPU% / RAM figure comes from `benchmark_cpu.py` or
  `benchmark_tracking.py`, both of which apply the affinity cap *before* importing
  torch and run inference with `device="cpu"`.
- The **GPU (GTX 1060 6 GB) was used for training only**, plus for generating detection
  predictions that feed accuracy scoring. Accuracy is device-independent, so mAP values
  are unaffected; no timing was taken from those runs.
- Each benchmark row records `baseline_cpu_pct`, the machine's idle load sampled before
  the cell ran. Across all 225 measured rows, **0 exceeded the 25% contamination
  threshold** (max 22.9%). Contaminated cells were re-run, not reported.

---

## 2. Per-model performance

Detection measured over 397 ROCO val images at **1920×1080**. Latency **excludes** JPEG
decode, which is reported separately. `mAP` is COCO mAP@[.5:.95]; `armor AP` is broken
out because armor plates are the actual aim point and behave very differently from
whole-robot detection.

### Detection only

| config | imgsz | cores | latency ms | FPS | CPU% of cap | RAM MiB | mAP | armor AP |
|---|---|---|---|---|---|---|---|---|
| `classical_strict` | native | 1 | 34.2 | 29.27 | 95 | 546 | 0.0028 | 0.0113 |
| `classical_strict` | native | 6 | 25.3 | 39.57 | 27 | 553 | 0.0028 | 0.0113 |
| `fast_320` | 320 | 1 | 33.8 | 29.60 | 94 | 703 | 0.4965 | 0.0068 |
| `fast_320` | 320 | 6 | 19.6 | 50.97 | 87 | 708 | 0.4965 | 0.0068 |
| `yolo_320` | 320 | 1 | 62.7 | 15.96 | 97 | 750 | 0.5169 | 0.0448 |
| `yolo_320` | 320 | 6 | 30.9 | 32.33 | 90 | 749 | 0.5169 | 0.0448 |
| `fast_640` | 640 | 1 | 84.2 | 11.88 | 96 | 725 | 0.6073 | 0.2943 |
| `fast_640` | 640 | 6 | 37.1 | 26.92 | 91 | 726 | 0.6073 | 0.2943 |
| `ssd_small_960_anchor` | 960 | 1 | 133.0 | 7.52 | 97 | 758 | 0.5137 | 0.0930 |
| `ssd_small_960_anchor` | 960 | 6 | 79.8 | 12.53 | 96 | 771 | 0.5137 | 0.0930 |
| `fast_960` | 960 | 1 | 158.2 | 6.32 | 97 | 770 | 0.6399 | **0.4537** |
| `fast_960` | 960 | 6 | 62.6 | 15.97 | 94 | 767 | 0.6399 | **0.4537** |
| `yolo_640` | 640 | 1 | 200.3 | 4.99 | 98 | 786 | 0.6263 | 0.3882 |
| `yolo_640` | 640 | 6 | 72.0 | 13.89 | 95 | 787 | 0.6263 | 0.3882 |
| `ssd_large_960_anchor` | 960 | 1 | 346.0 | 2.89 | 97 | 1016 | 0.5068 | 0.1167 |
| `yolo_960` | 960 | 1 | 394.4 | 2.54 | 98 | 843 | **0.6679** | 0.4409 |
| `yolo_960` | 960 | 6 | 146.4 | 6.83 | 97 | 848 | **0.6679** | 0.4409 |
| `frcnn_resnet50_640` | 640 | 1 | 2210.1 | 0.45 | 98 | 1088 | 0.5198 | 0.0108 |
| `frcnn_resnet50_640` | 640 | 6 | 709.8 | 1.41 | 98 | 1091 | 0.5198 | 0.0108 |

Full grid (25 configs × 5 core levels = 125 rows) is in `results/performance.csv`.

### Detection + tracking combined

Measured on ARC match clips, also **1920×1080, 30 fps**. `total` = detect + track;
decode again separate. Averaged over 2 clips.

| detector | tracker | cores | detect ms | track ms | total ms | FPS | tracks/frame | RAM MiB |
|---|---|---|---|---|---|---|---|---|
| `classical_strict` | classical | 1 | 30.3 | 0.60 | **30.9** | **33.85** | 12.5 | 534 |
| `classical_strict` | sort | 1 | 30.7 | 1.14 | 31.8 | 32.98 | 12.5 | 570 |
| `ssd_small_960_anchor` | classical | 1 | 143.0 | 0.14 | 143.1 | 6.99 | 4.7 | 745 |
| `fast_960` | sort | 1 | 158.2 | 0.37 | 158.6 | 6.31 | 5.4 | 788 |
| `yolo_960` | sort | 1 | 402.5 | 0.35 | 402.8 | 2.48 | 5.1 | 859 |
| `frcnn_resnet50_640` | sort | 1 | 2255.2 | 0.46 | 2255.6 | 0.44 | 6.3 | 1046 |
| `classical_strict` | classical | 6 | 22.3 | 0.59 | **22.9** | **47.88** | 12.5 | 541 |
| `fast_960` | sort | 6 | 66.9 | 0.41 | 67.3 | 14.85 | 5.4 | 786 |
| `ssd_small_960_anchor` | classical | 6 | 91.0 | 0.15 | 91.2 | 10.97 | 4.7 | 758 |
| `yolo_960` | sort | 6 | 143.5 | 0.13 | 143.7 | 6.96 | 5.1 | 838 |
| `frcnn_resnet50_640` | sort | 6 | 736.4 | 0.50 | 736.9 | 1.36 | 6.3 | 1055 |
| `fast_960` | vit | 6 | 67.0 | 108.39 | 175.4 | 5.75 | 5.4 | 796 |
| `fast_960` | goturn | 1 | 163.5 | 4308.65 | 4472.2 | 0.22 | 5.0 | 7211 |
| `classical_strict` | goturn | 1 | 140.1 | 93322.08 | 93462.1 | 0.03 | 13.3 | 17476 |

Full matrix (100 rows) in `results/tracking_performance.csv`.

**The Kalman trackers are free.** `classical` costs 0.13–0.60 ms and `sort` 0.13–1.15 ms
per frame regardless of detector. Detection is ≥99% of every viable pipeline. Any
optimisation effort belongs on the detector.

**GOTURN and VitTrack scale with target count, not image size**, because each tracks one
object and multi-object means one instance per robot. VitTrack: 83 ms at 4.7 tracks,
324 ms at 11.6. GOTURN: 4.3 s at 5 tracks, 93 s and 17 GiB RSS at 13.3. **Neither is
deployable.** Do not plan hardware around them.

### Front-runner

**`fast_960` + `sort` (or + `classical` tracker).** Reasoning from the data:

- **Armor AP 0.4537, the highest of any model measured**, including `yolo_960` (0.4409).
  For a targeting system armor is the aim point, and everything outside the YOLO family
  is effectively blind to it — SSD 0.093, Faster R-CNN 0.011, classical 0.011.
- 62.6 ms at 6 cores (15.97 FPS), 158.2 ms at 1 core (6.31 FPS).
- Overall mAP 0.6399, second only to `yolo_960` (0.6679) which costs 2.3× more time.

If the board cannot sustain that, **`fast_640`** is the fallback: 37.1 ms at 6 cores
(26.92 FPS), but armor AP drops to 0.2943. Below 640 armor detection collapses
(`fast_320` = 0.0068) — **do not go below 640 input if armor matters.**

`classical_strict` + `classical` is the fastest pipeline measured (33.85 FPS at one
core) and is **not a deployment candidate**: mAP 0.0028, and see §7 for what it actually
detects.

**`fast_640` and `fast_320` were never run through the combined tracking benchmark** —
only `fast_960` was. Given the tracker costs <1.2 ms in every measured pairing, combined
cost is detection + ~1 ms, but that specific combination is **not measured**.

---

## 3. Model input requirements

**Nothing is exported or quantized. Every model runs in its native training framework.**
No ONNX, OpenVINO, TensorRT, TFLite, or INT8 artifact exists anywhere in the repo —
verified by search. All CPU numbers above are **native PyTorch / OpenCV fp32**.

That is the largest single optimisation left on the table and it is entirely unexplored:
**export and quantization are not measured, not attempted, and not validated.**

| model | framework | network input | aspect handling |
|---|---|---|---|
| `yolo_*`, `fast_*` | Ultralytics 8.4.120 / PyTorch | **544 × 960** for a 1920×1080 frame at `imgsz=960` | letterboxed: long side → `imgsz`, short side padded to a multiple of 32 |
| `ssd_*` | torchvision SSDLite | **960 × 960** (or 640/320 square) | **squashed** — aspect ratio destroyed |
| `frcnn_resnet50_640` | torchvision Faster R-CNN | **640 × 640** | **squashed** |
| `classical_*` | OpenCV only | **native 1920 × 1080, no resize** | n/a |

Verified by instrumenting the forward pass, not from documentation. A 1920×1080 frame at
`imgsz=960` really does reach the YOLO network as `(1, 3, 544, 960)`.

Runtime versions the numbers were taken under: torch 2.13.0+cu126, torchvision 0.28.0+cu126,
ultralytics 8.4.120, opencv 4.14.0, numpy 2.5.2, Python 3.14.6.

Detector output classes: `{0: armor, 1: base, 2: car, 3: ignore, 4: watcher}` for YOLO.
The torchvision models use COCO category ids where **3 = car** and **1 = armor**
(category 0 is an unused supercategory).

---

## 4. Preprocessing pipeline

In order, per family. This determines what the camera should hand over and what is
wasted work.

### YOLO / Fast YOLO — 2.5–3.3 ms at 960, 0.5–1.1 ms at 320

Frame is handed to Ultralytics as a **raw BGR `uint8` HWC numpy array** — exactly what
`cv2.imread` and `cv2.VideoCapture.read()` produce. Ultralytics then does internally:

1. Letterbox resize to long-side `imgsz`, short side padded to /32 (1920×1080 → 544×960)
2. BGR → RGB
3. HWC → CHW, contiguous
4. `uint8` → `float32`, divide by 255
5. Batch dimension

**No mean/std normalization.** Scaling back to native pixels is handled by Ultralytics.

### MobileNet-SSD and Faster R-CNN — 15.3–17.9 ms (SSD @960), 9.8–10.8 ms (FRCNN @640)

Implemented in `ssd_preprocess()` in `scripts/predict_to_coco.py`:

1. Decode to **PIL, RGB** (`Image.open(...).convert("RGB")`)
2. `image.resize((imgsz, imgsz), BILINEAR)` — **square squash, aspect ratio not preserved**
3. `torchvision.transforms.functional.to_tensor` → CHW float32, /255

then inside the model's `GeneralizedRCNNTransform`:

4. Normalize with **ImageNet mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`**
5. Resize is a no-op — `min_size` and `max_size` are both pinned to `imgsz`, and the
   input is already square. **This pinning is load-bearing.** torchvision defaults to
   800/1333; without the pin the model would silently upsample and every box would come
   back mis-scaled.

Boxes come out in `imgsz` space and are rescaled to native by `(native_w/imgsz,
native_h/imgsz)`.

**This path is a known inefficiency.** 15–18 ms of PIL resize + tensor conversion against
YOLO's 2.5 ms for the same frame, ~20% of the SSD frame budget. A cv2-based resize
feeding a preallocated tensor would very likely cut most of it. **Not measured.**

### Classical detector — 0.0 ms preprocess (fused into detection)

Runs at **native resolution, no resize**. Inside `detect()`:

1. BGR → HSV (`cv2.COLOR_BGR2HSV`) for the hue gate
2. BGR → GRAY (`cv2.COLOR_BGR2GRAY`)
3. Hue-range mask via `cv2.inRange` over configured red/blue bands → candidate regions
4. Sobel gradient magnitude (ksize 3), normalized 0–1, on the grayscale image
5. Per candidate: template resize + `cv2.matchTemplate` NCC against the template bank

### What this means for the camera

- **Hand over BGR `uint8` frames.** Both the YOLO path and the classical path want BGR
  natively. Only the torchvision path wants RGB, and it converts anyway.
- **Do not colour-convert in the capture layer.** Every consumer either wants BGR or
  does its own conversion.
- **Do not pre-resize in the capture layer** unless you commit to one model family.
  YOLO letterboxes, the torchvision models squash, and the classical detector wants
  native. A capture-side resize would have to be undone or would silently change the
  geometry the models were trained on.
- JPEG decode of a 1920×1080 frame cost **6.0–16.7 ms** across detection cells and
  **3.6–15.7 ms** across tracking cells, and is *excluded* from every latency figure
  above. (One 56.7 ms outlier exists, on the GOTURN + classical cell where the machine
  was thrashing 17 GiB — disregard it.) A live camera replaces decode with capture: if
  the camera delivers MJPEG you will pay a comparable cost, if it delivers raw YUV or
  BGR you may avoid it. **Camera capture cost is not measured.**

---

## 5. Detection → tracking handoff

Detector returns a list of `(x, y, w, h)` boxes in **native image pixels**, float. The
YOLO and torchvision paths filter to a single class before handing over — `car` in the
tracking pipeline, since the tracking ground truth contains only robots.

Between detector and tracker sits `clean_frame()` from `scripts/auto_label.py`, applied
in both the accuracy and the timing paths so they describe the same pipeline:

- drops a box **contained ≥50%** inside a larger one (turret detected inside chassis)
- **fuses** two boxes overlapping at IoU ≤ 0.15 into their union when the union stays
  within 1.6× the clip's median box (one robot split across two boxes)

Then per tracker:

- `classical` and `sort` receive **boxes only** — `tracker.update(dets)`
- `goturn` and `vit` receive **frame and boxes** — `tracker.update(frame, dets)`, because
  they are appearance-based and need pixels

Both Kalman trackers return `(x, y, w, h, track_id)`.

### Frame-rate and timing sensitivity — read this before choosing a camera FPS

**The trackers count frames, not seconds. Every temporal parameter is frame-based and
was set for 30 fps footage.**

- `max_age = 10` **frames** — how long a track survives unmatched. At 30 fps that is
  0.33 s. At 10 fps the same constant becomes 1.0 s, and tracks will survive occlusions
  they should not.
- `min_hits = 2` **frames** before a track is reported at all — a 2-frame startup
  latency that becomes 200 ms at 10 fps versus 67 ms at 30 fps.
- The Kalman constant-velocity model uses **dt = 1 frame implicitly**. Velocity is in
  **pixels per frame**, not pixels per second. Run at half the frame rate and every
  robot appears to move twice as fast per step, which degrades the motion prediction the
  association depends on.

**Consequence:** if the deployed pipeline runs at a different frame rate than 30 fps —
and at 6.31 FPS for `fast_960` + `sort` it certainly will — `max_age`, `min_hits`, and
`process_noise` should be rescaled, or the tracker should be rewritten to take a real
`dt`. **This has never been tested at any frame rate other than the clips' native 30 fps.**

A related trap: a *variable* frame interval (whatever the CPU manages that frame) breaks
the constant-velocity assumption differently on every frame. A fixed-rate loop that drops
frames is more predictable for the tracker than a free-running one.

---

## 6. Bottlenecks and constraints

### Where the time goes

Per frame at 6 cores, `fast_960`, 1920×1080. The stage split comes from the detection
benchmark (`results/performance.csv`), which is the only place stages were instrumented;
the tracking row is listed beside it because the two runs used different frames — ROCO
val images versus ARC clip frames — and so differ slightly on the same model.

| stage | ms | share of detect |
|---|---|---|
| JPEG decode (excluded from headline latency) | 6.3 | — |
| preprocess | 2.5 | 4% |
| **inference** | **58.3** | **93%** |
| postprocess (NMS + box decode) | 0.6 | 1% |
| **detection total** | **62.6** | |
| tracking (`sort`, measured in the combined run) | 0.41 | <1% |

The combined run measured the same model at `detect 66.9 + track 0.41 = 67.3 ms` on clip
frames. Use 62.6 / 67.3 as the range for `fast_960` at 6 cores rather than either alone.

**Inference dominates completely.** Preprocessing is worth attacking only on the
torchvision path (§4), and tracking is not worth attacking at all.

### Plainly: what frame rate does this run at?

On the **Ryzen 5 5600G**, with no export or quantization:

- `fast_960` + `sort`: **15.97 FPS at 6 cores, 6.31 FPS at 1 core**
- `fast_640` (detection only): **26.92 FPS at 6 cores, 11.88 FPS at 1 core**
- `yolo_960` + `sort`: **6.96 FPS at 6 cores, 2.48 FPS at 1 core**

**Nothing with usable armor accuracy reaches 30 FPS on any core count tested.** 21 of the
125 detection cells clear 30 FPS. Every one of them is either a 320 px variant
(`fast_320`, `yolo_320`, `ssd_*_320`) or the classical detector. The best armor AP in
that entire set is **`yolo_320` at 0.0448** — against `fast_960`'s 0.4537, a 10× gap.
Put plainly: on this hardware you can have 30 FPS or you can detect armor plates, not
both.

On a lower-power board these numbers will be **worse**, likely substantially.

### What caps camera resolution and FPS

- **Camera resolution above 1920×1080 buys nothing** for the DL models. Every one
  downsamples to ≤960 on the long side immediately; extra pixels cost capture bandwidth
  and resize time, then are thrown away. Capture at 1920×1080 or below.
- **The classical detector is the exception** — it runs at native resolution, so its cost
  scales directly with sensor resolution. Its 34.2 ms was measured at 1920×1080.
- **Camera FPS above the achievable inference rate buys nothing** unless you deliberately
  drop frames. At ~16 FPS achievable, a 60 fps camera means discarding 3 of every 4 frames.
- **RAM is not a constraint** for viable configs: 534–1091 MiB peak RSS. Only GOTURN
  (7.2–17.5 GiB) is memory-hostile, and it is not deployable for other reasons.
- **CPU saturates**: 87–100% of the cap for every DL config. There is no headroom to run
  the pipeline alongside other robot workloads on the same cores. The classical detector
  at 6 cores (27–28%) is the only one that leaves room.

---

## 7. Open issues and incomplete pieces

**Ordered by how much they could change hardware-phase decisions.**

1. **No target board chosen, nothing measured on candidate hardware.** All numbers are
   Ryzen 5 5600G with affinity caps. The project's stated contribution is CPU-constrained
   measurement on a representative competition CPU; that CPU has not been selected.

2. **No export or quantization, at all.** No ONNX/OpenVINO/TFLite/INT8 path exists or has
   been attempted. Every number is native fp32 PyTorch. This is likely the largest
   available speedup and is entirely unquantified.

3. **No live-camera code exists.** There is no capture loop, no frame source other than
   files on disk, and no end-to-end real-time script. `benchmark_tracking.py` reads JPEGs
   from a directory. Camera capture, buffering, frame-drop policy, and
   capture→inference handoff are all unwritten and unmeasured.

4. **The classical detector detects armor plates, not robot chassis.** Its `detect()`
   returns boxes labelled `"armor"` only. When it feeds the tracker, the tracker is
   tracking plates — which is why it reports 12.5 tracks/frame where YOLO reports 5.1.
   Its headline 33.85 FPS is therefore **not comparable** to the other rows as a
   robot-tracking pipeline, and its tracking accuracy was never evaluated against
   plate-level ground truth. Do not read its speed as a usable result.

5. **Tracker parameters are frame-rate-coupled and untested off 30 fps.** See §5. They
   were also **never tuned** — hand-set from problem geometry, no sweep was ever run.
   `data/tracking/assignment.csv` reserves arc03 for validation if you tune them.

6. **Armor detection fails outside the YOLO family.** SSD (0.093), Faster R-CNN (0.011)
   and the classical detector (0.011) are all effectively blind to armor plates at ~21 px.
   Three independent architectures failing the same way suggests a target-size problem,
   not a model problem. If the hardware phase needs armor-level targeting, the model
   choice is constrained to YOLO/Fast YOLO at ≥640 input.

7. **Tracking ground truth is machine-generated** (`scripts/auto_label.py`), not
   hand-labelled. Its labels come from motion association, which shares assumptions with
   SORT and the classical tracker and not with appearance-based GOTURN/VitTrack, so
   tracking accuracy comparisons are biased toward the Kalman trackers. **CPU numbers are
   unaffected** — timing needs no labels.

8. **The tracking train/val/test split was declared after results were produced.** It is
   defensible only because nothing was ever fitted (no tracker is trained; GOTURN and
   VitTrack are frozen). It constrains future tuning, not past results.

9. **Confidence threshold is fixed at 0.25** in the tracking pipeline and was never swept.
   The operating point that matters for a live system (precision vs recall trade-off under
   motion) is unexplored. Detection precision/recall in `results/detection.csv` are quoted
   at the *best-F1* point, which is not necessarily the right deployment threshold.

10. **Faster R-CNN uses torchvision defaults** `score_thresh=0.05, detections_per_img=100`,
    while SSD was deliberately set to `0.001 / 300` to match YOLO. Its mAP is therefore
    slightly pessimistic relative to the others. Does not affect its latency, which
    disqualifies it regardless.

11. **Val is a single match clip** (clip-level n = 1). All confidence intervals bootstrap
    over images or frames and do **not** capture across-match variance. Real-world
    performance on a different venue, lighting, or robot livery is not estimated.

---

## Reference: where the numbers live

| file | contents |
|---|---|
| `results/performance.csv` | detection CPU, 125 rows (25 configs × 5 core levels) |
| `results/detection.csv` | detection accuracy, 25 configs, with `predictions_sha1` staleness guard |
| `results/tracking_performance.csv` | combined detector×tracker CPU, 100 rows |
| `results/tracking.csv` | tracking accuracy, 26 rows |
| `results/combined.md` | detection accuracy joined to cost |
| `results/tracking_report.md` | tracking report, incl. per-pairing CPU table |
| `scripts/benchmark_cpu.py` | detection harness — core capping, resource sampling, CIs |
| `scripts/benchmark_tracking.py` | combined-pipeline harness |
| `scripts/predict_to_coco.py` | `ssd_preprocess`, `load_ssd`, `load_frcnn` — the preprocessing of record |
| `scripts/run_trackers.py` | `make_detector()` — the detector factory all paths share |
