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
| `data/manifest.sha256` | SHA-256 of all 7973 dataset files (provenance + integrity) |
| `Datasets/` | The dataset itself — **not committed**, see below |
| `CLAUDE.md` | Working context and methodology notes |

## Setup

From a fresh clone:

```bash
python -m venv .venv
.venv\Scripts\activate          # PowerShell:  .venv\Scripts\Activate.ps1
pip install roboflow ultralytics opencv-python numpy

git config core.hooksPath scripts/hooks   # enable the large-file commit guard

python scripts/download_data.py           # fetch the dataset (~1.4 GB)
python scripts/verify_data.py             # confirm it matches the manifest
```

The clone itself is under a megabyte — the dataset arrives in the download step.

Install the **CUDA build** of PyTorch, not the CPU-only wheel, and confirm the GPU is
visible before training:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

> Dependency versions are not yet pinned. A `requirements.txt` with exact versions should
> be added before the first benchmark run, so results stay reproducible.

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

## Open decisions

These need to be settled *before* benchmark data is collected, since both change every
number that gets reported.

- [ ] **Train/val split.** The project proposal specifies 85/15 train/val with no test set
      for detection. The Roboflow export ships **70/20/10**. Either re-split locally or
      amend the proposal — but pick one and apply it uniformly across all detection models.
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
