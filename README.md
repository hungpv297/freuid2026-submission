# FREUID Challenge 2026 — Reproducible Submission

Identity-document fraud detection. This repo contains everything needed to
reproduce our final submission: training code, the frozen model weights, an
inference entrypoint, and a Docker container that runs **inference only, with
no network access**.

## Final model (one sentence)

An **ensemble of 3 checkpoints (epochs 10, 11, 12)** of `maxvit_base_tf_512`
trained on the full competition training set augmented with self-generated
pseudo-labels, each checkpoint scored with **dense-crop test-time augmentation**
(native image + a 3×3 grid of 1.15× crops, each in the 4 D2 orientations = 40
views), averaged in probability space.

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

# 3. run — inference only, network disabled
docker run --rm --gpus all --network none \
    -v /abs/path/to/images:/data:ro \
    -v /abs/path/to/output:/submissions \
    freuid-infer
# -> /abs/path/to/output/submission.csv

# Faster decode with more dataloader workers needs a larger /dev/shm:
#   docker run ... --shm-size=8g freuid-infer --workers 8
```

Run directly (without Docker):

```bash
python infer.py --data /path/to/images --out out/submission.csv --weights weights
```

## Hardware / runtime

* Developed and tested on a single **NVIDIA H100 (80 GB)**, CUDA 12.6.
* Inference uses bf16 autocast + channels-last. Dense-crop TTA is 40 views ×
  3 checkpoints = **120 forward passes per image**; budget accordingly for large
  private sets. `--bs` / `--workers` can be tuned; CPU-only inference also works
  (much slower).
* No network access is used at inference (`timm.create_model(pretrained=False)`;
  weights loaded from local files).

## Reproducing the training

See `REPORT.md` and `train/`. Everything is documented; no external datasets are
used (only the competition data + self-generated pseudo-labels).
