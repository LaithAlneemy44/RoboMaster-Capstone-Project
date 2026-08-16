# RoboMaster Capstone Project

Comparing **classical vs deep learning perception models** for real-time autonomous
targeting in RoboMaster-style robot competitions.

> **Research question:** How do classical and deep learning perception models compare in
> real-time autonomous targeting in robot competitions?

Most published perception benchmarks report GPU numbers. This project measures
**CPU-constrained performance**, which is what actually governs whether a model can be
deployed on a competition robot. That framing drives the methodology throughout: models
are *trained* on a GPU, but every latency, FPS, CPU and RAM figure is measured on the
representative competition CPU.

A "perception model" here means one **detection** model paired with one **tracking** model.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/download_data.py` | Re-fetches the pinned Roboflow dataset export |
| `scripts/verify_data.py` | Checks the local dataset against the committed manifest |
| `scripts/check_gpu.py` | Proves the GPU can actually launch a kernel (not just `is_available()`) |
| `scripts/make_splits.py` | Builds train/val lists for both formats without touching `Datasets/` |
| `scripts/train_yolo.py` | Fine-tunes YOLO from COCO-pretrained weights, GPU-only |
| `scripts/predict_to_coco.py` | Runs a trained model over val, writes COCO detections |
| `scripts/evaluate_detection.py` | **The single evaluator** — scores every model family |
| `data/manifest.sha256` | SHA-256 of all 7973 dataset files (provenance + integrity) |
| `data/splits/assignment.csv` | **The split of record** — every image's clip and train/val side |
| `data/roco_central.yaml` | Generated Ultralytics data config — the training entry point |
| `data/splits/` | Generated image lists + merged COCO annotations (not committed) |
| `Datasets/` | The dataset itself — **not committed**, see below |
| `CLAUDE.md` | Working context and methodology notes |

## Setup

From a fresh clone:

```bash
py -3.14 -m venv .venv
.venv\Scripts\activate          # PowerShell:  .venv\Scripts\Activate.ps1

# PyTorch must come from the cu126 index - see "GPU note" below. Install it first so
# the later installs see torch as already satisfied and don't pull the CPU wheel.
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision
pip install -r requirements.txt

git config core.hooksPath scripts/hooks   # enable the large-file commit guard

python scripts/download_data.py           # fetch the dataset (~1.4 GB)
python scripts/verify_data.py             # confirm it matches the manifest
python scripts/check_gpu.py               # confirm the GPU can actually run a kernel
python scripts/make_splits.py --from-assignment   # rebuild the recorded split
```

That last step is required on every fresh clone and after any move of the project
directory — the files it writes hold absolute paths and are deliberately not committed.
See "Train/val split" below.

The clone itself is under a megabyte — the dataset arrives in the download step.

### GPU note — this machine needs the `cu126` build specifically

The GTX 1060 is **Pascal (sm_61)**, and CUDA 12.8 dropped Pascal support. A `cu128` or
newer wheel installs fine, reports `torch.cuda.is_available() == True`, and even lets you
move tensors to the device — then dies at the first kernel launch with *"no kernel image is
available for execution on the device"*, typically minutes into a training run. The driver
(560.94) caps out at CUDA 12.6 independently, so `cu126` is the right target twice over.

Because of that failure mode, `torch.cuda.is_available()` is **not** a sufficient check.
`scripts/check_gpu.py` also verifies `sm_61` is in torch's compiled arch list and launches a
real kernel:

```bash
python scripts/check_gpu.py     # must end with "Safe to train"
```

Usable VRAM is roughly 4.9 GB, not 6 GB — the desktop display drives the same card.

## Dataset

**DJI ROCO Central** — 2655 images, from Roboflow Universe.
<https://universe.roboflow.com/enterprise-9gout/dji-roco-central>

| | |
|---|---|
| Workspace / project | `enterprise-9gout` / `dji-roco-central` |
| Version | **1** (2023-03-25) |
| Images | 2655 |
| Classes | `armor`, `base`, `car`, `ignore`, `watcher` (`nc: 5`) |
| Split as exported | 1858 train / 531 valid / 266 test (70/20/10) |
| Preprocessing | Auto-orientation, EXIF stripping. No augmentation. |
| License | CC BY 4.0 |

The dataset is exported in **two formats** — `yolov11` for the Ultralytics YOLO models and
`coco` for MobileNet-SSD. Both contain byte-identical images; only the annotations differ.

### Getting the data

```bash
python scripts/download_data.py     # fetches both formats into Datasets/
python scripts/verify_data.py       # confirms it matches data/manifest.sha256
```

You need a Roboflow API key in `ROBOFLOW_API_KEY` (environment variable or a `.env` file at
the repo root). `.env` is gitignored — never commit the key.

### Why the dataset isn't committed

It is ~1.4 GB on disk and would add ~688 MB to the repository permanently. Git history is
immutable, so committed data can never be reclaimed without rewriting history and
force-pushing. Because the export is **version-pinned**, re-downloading it is deterministic,
so nothing is lost by keeping it out — `data/manifest.sha256` still records exactly which
bytes were used, and `verify_data.py` proves a fresh download matches.

The same reasoning covers `runs/`, `*.pt` and other training artifacts: they are
byte-distinct on every run, so Git stores a full copy each time rather than a diff. If a
specific trained weight file needs to be preserved for the write-up, use Git LFS or attach
it to a release rather than committing it directly.

### Keeping the repository small

`git add .` is safe to use here. Two layers stand behind it:

1. **`.gitignore`** excludes everything that would bloat the repo — the dataset, `runs/`,
   model weights in any framework (`*.pt`, `*.pth`, `*.ckpt`, `*.caffemodel`, ...), raw
   video and extracted frames, and the Roboflow SDK's default download directories.
2. **A pre-commit hook** (`scripts/hooks/pre-commit`) rejects any staged file over 50 MB,
   whatever it is called. `.gitignore` only catches patterns anticipated in advance; the
   hook catches everything else. Enable it once per clone:
   `git config core.hooksPath scripts/hooks`.

What *should* be committed: source, configs, benchmark result tables, and the hand-built
tracking annotations.

### Attribution

DJI ROCO Central, provided by a Roboflow Universe user, licensed **CC BY 4.0**.

## Train/val split — group-aware by match clip

**Decision: hold out the whole `-VsBorn-of-Fire_BO2_1` clip. 2258 train / 397 val = 85.05/14.95.**

The 2655 images are not independent photographs. They are frames sampled from just
**seven match recordings**, and Roboflow's own 70/20/10 export splits them at random —
so all seven clips appear on both sides, and consecutive frames of the same moment sit
in train and val simultaneously. Val mAP under that split measures frame memorisation,
not generalization, and with no test set the same inflated split was also doing model
selection.

Splitting by clip removes the leakage. Every clip contains all five classes, so any of
them is a legal holdout; `-VsBorn-of-Fire_BO2_1` is used because at 397 images it is
14.95% of the dataset, which reproduces the proposal's 85/15 essentially exactly.

| Clip | Images | Share | Split |
|---|---:|---:|---|
| `-AresVs-_BO2_2` | 420 | 15.8% | train |
| `-VsHLL_BO2_2` | 420 | 15.8% | train |
| `WMJVs-_BO2_2` | 420 | 15.8% | train |
| `AllianceVsArtisans_BO2_2` | 419 | 15.8% | train |
| **`-VsBorn-of-Fire_BO2_1`** | **397** | **14.95%** | **val** |
| `-VsCUBOT_BO2_1` | 290 | 10.9% | train |
| `-VsRPS_BO2_2` | 289 | 10.9% | train |

Val holds 9809 instances (armor 6793, base 272, car 2342, ignore 19, watcher 383).

**Stated limitation.** Val is a single match, so clip-level *n* = 1. Confidence intervals
on detection mAP must therefore come from bootstrapping over the 397 val images, and they
describe variance *within* one match — they do not estimate across-match variance. The
honest way to get that would be leave-one-clip-out cross-validation over all seven clips,
at 7× the training cost per model; it was not taken, and that trade-off should be stated
in the write-up rather than left implicit.

This split applies to **every** detection model — YOLO, Fast YOLO, MobileNet-SSD and the
classical detector — or the comparison is not like-for-like.

```bash
python scripts/make_splits.py                  # the default: the split above
python scripts/make_splits.py --list-clips     # inspect clips without writing
python scripts/make_splits.py --holdout -VsCUBOT_BO2_1 -VsRPS_BO2_2   # a different holdout
```

`--val-frac` and `--keep-export-split` still exist for comparison against the old
behaviour, but they split at random across clip boundaries. They are labelled `LEAKY` in
the output, and `make_splits.py` prints a per-clip table with an explicit leakage verdict
on every run, so a leaky split cannot be produced by accident.

### What is committed, and regenerating after a move

Only **`data/splits/assignment.csv`** is committed. It is 200 KB of diffable text
(`filename,clip,orig_split,new_split`) and fully determines the split.

`data/roco_central.yaml`, `data/splits/*.txt` and `data/splits/coco_*.json` are
**gitignored**. The first two contain absolute paths, so they are machine-local — they
broke once already when the project moved from `C:` to `E:`, and they cannot work on
Colab. Regenerate them from the committed CSV, byte for byte:

```bash
python scripts/make_splits.py --from-assignment
```

Run that after cloning, after moving the project, or on Colab. It replays the recorded
assignment exactly rather than re-deriving it, so it does not depend on RNG
reproducibility or on the flags originally used.

## Evaluation — one evaluator for every model

Ultralytics and torchvision each compute mAP their own way, and the two do not agree.
If YOLO were scored by one and MobileNet-SSD by the other, the project's headline
comparison would be between two different metrics. So every detection model is scored
by `scripts/evaluate_detection.py` instead, from a COCO-format predictions file:

```bash
python scripts/predict_to_coco.py --family yolo \
    --weights runs/detect/fast_640/weights/best.pt --imgsz 640 --out preds.json
python scripts/evaluate_detection.py --predictions preds.json --name fast_640
```

Results append to `results/detection.csv`, one row per config.

**`ignore` policy.** Models train on all five classes — learning to tag an ambiguous
robot as `ignore` beats misfiring it as `car` — and `ignore` is then excluded at
scoring time: its ground truth is removed so it cannot be a miss, its predictions are
dropped, and any prediction overlapping an ignore region at IoA ≥ 0.5 is suppressed so
it cannot be a false positive. That last step is why COCO's own `iscrowd` is not
sufficient — `iscrowd` only excludes matches within the *same* category.

**Confidence intervals** come from bootstrapping over the 397 val images. They measure
variance *within* one match, not across matches — see the split limitation above.

### Self-test

```bash
python scripts/evaluate_detection.py --gt-as-predictions   # must report mAP 1.0000
```

Scoring the ground truth against itself is what actually proves image ids and
coordinate spaces line up. It is worth keeping: it is what caught the id bug below.

> **COCO annotation ids must be 1-based.** `COCOeval` stores the matched ground-truth
> id in `dtMatches` and then tests it with `logical_and(dtm, ...)`, so an annotation
> with id `0` is falsy and its match is silently rescored as a false positive. With
> 0-based ids the self-test reported 0.9975 instead of 1.0, and the loss landed
> entirely on `armor` — the class holding 66% of all instances. `make_splits.py` now
> numbers images and annotations from 1.

## Open decisions

These need to be settled *before* benchmark data is collected, since each changes every
number that gets reported.

- [x] **Train/val split.** Settled: group-aware by match clip, holding out
      `-VsBorn-of-Fire_BO2_1` for 85.05/14.95. See "Train/val split" above for the
      leakage finding that drove it and the stated limitation it carries.
- [ ] **The `ignore` class.** The export defines it as a real class (`nc: 5`). Decide
      whether `ignore` regions are dropped, treated as background, or excluded from the mAP
      computation. Whatever the choice, it must be identical for YOLO, MobileNet-SSD and the
      classical detector, or the comparison is not like-for-like.
- [ ] **Benchmark CPU.** The representative competition CPU must be chosen and documented,
      since all reported performance figures depend on it.
- [ ] **Combined-pipeline accuracy definition.** "Accuracy" for detection + tracking is not
      a single number. Define the scoring method before collecting data.

## Tracking dataset

No labelled RoboMaster tracking dataset exists; one is being built by hand from ARC
Robotics footage, with a 70/15/15 train/val/test split. Annotation files (small, diffable
text) **are** committed once they exist. Raw video frames are not — the source footage is
licensed and must not be redistributed.
