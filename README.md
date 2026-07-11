# FREUID Challenge 2026 — Reproducible Submission

Identity-document fraud detection. This repo contains everything needed to
reproduce our final submission: training code, the frozen model weights, an
inference entrypoint, and a Docker container that runs **inference only, with
no network access**.

## Final model (one sentence)

An **ensemble of 3 checkpoints (epochs 10, 11, 12)** of `maxvit_base_tf_512`
trained on the full competition training set augmented with self-generated
pseudo-labels, each checkpoint scored with **4-view edge-crop test-time
augmentation** (the 4 edge-midpoint crops of a 1.15× up-scaled image, no flip)
= 12 forward passes/image, averaged in probability space.

## Repository layout

```
infer.py            # entrypoint: /data (flat images) -> /submissions/submission.csv
Dockerfile          # reproducible, network-free inference image
requirements.txt    # pinned inference deps (torch/numpy from base image)
weights/            # 3 frozen checkpoints (Git LFS): ckpt_e10/e11/e12 (fp16)
train/              # training code used to produce the weights (documentation)
REPORT.md           # technical report
LICENSE             # MIT
```

## Input / output contract

* **Input:** a flat directory of images mounted read-only at `/data`
  (`.jpeg .jpg .png .webp .bmp .tif .tiff`). Row id = filename without extension.
  No CSV / manifest / subfolders are read.
* **Output:** `/submissions/submission.csv` with header `id,label`; one row per
  input image; `label` = float fraud score (higher ⇒ more likely fraudulent).

## Build & run (Docker, no network)

```bash
# 1. get the weights (Git LFS)
git lfs install && git lfs pull

# 2. build
docker build -t freuid-infer .

# 3. run — inference only, network disabled.
#    Defaults (--workers 0) need NO extra flags: a plain run just works.
docker run --rm --gpus all --network none \
    -v /abs/path/to/images:/data:ro \
    -v /abs/path/to/output:/submissions \
    freuid-infer
# -> /abs/path/to/output/submission.csv

# Optional ~15% faster decode (multi-worker DataLoader needs a bigger /dev/shm):
#   docker run ... --shm-size=8g freuid-infer --workers 8
```

Run directly (without Docker):

```bash
python infer.py --data /path/to/images --out out/submission.csv --weights weights
```

## Hardware / runtime

* Developed and tested on a single **NVIDIA H100 (80 GB)**, CUDA 12.6.
* Inference uses bf16 autocast + channels-last. TTA is 4 edge-crops ×
  3 checkpoints = **12 forward passes per image**. `--bs` / `--workers` can be
  tuned; CPU-only inference also works (much slower).
* **Measured (Docker, `--network none`):** the plain-default run (`--workers 0`,
  no `--shm-size` needed) scores the **7,821 public images in 428 s (7 min)** on
  one H100 → **≈2.0 h for the 134,997 hidden-test images on one H100**. The
  faster path (`--shm-size=8g --workers 8`) does it in 372 s → ≈1.8 h. On a
  single **A100** (2.0× realistic → 3.0× worst-case slowdown) the hidden set is
  **≈4.1 h (default) / ≈3.6 h (fast path)** up to ≈5–6 h worst-case — within the
  challenge's **6-hour single-A100** limit. Shard `/data` across GPUs to cut it
  further.
* No network access is used at inference (`timm.create_model(pretrained=False)`;
  weights loaded from local files).

## Reproducing the training

See `REPORT.md` and `train/`. Everything is documented; no external datasets are
used (only the competition data + self-generated pseudo-labels).
