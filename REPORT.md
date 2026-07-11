# FREUID Challenge 2026 — Technical Report

## 1. Introduction

We address identity-document fraud detection as a binary image-classification
problem: each document image is assigned a continuous fraud score (higher =
more likely fraudulent), evaluated with the challenge's tail-focused DET metric
(APCER at low BPCER). The task exhibits a strong domain shift between the
training data and the (public/private) test data — including print-and-capture
recaptures and document types under-represented in training — which drives our
design choices toward strong local-artifact modelling at inference time.

## 2. Method

**Backbone.** `maxvit_base_tf_512.in21k_ft_in1k` (timm), a hybrid
conv + window/grid-attention model at 512×512. In an earlier fold-0 sweep of
30+ timm backbones, MaxViT @ 512 clearly dominated (mid-size hybrid + high
resolution beat larger pure-conv / pure-ViT models that overfit the digital
domain), which is why it is our final backbone.

**Training data.** All competition training images with the 5 stratified folds
merged (no held-out validation — in-domain OOF is saturated and does not
discriminate models), plus **self-generated pseudo-labels**: confident
predictions on the public test images (score < 0.05 → bona-fide, > 0.95 →
fraud) added to the training set (semi-supervised). **No external datasets are
used.**

**Objective / optimisation.** `BCEWithLogitsLoss` on a single logit; AdamW
(lr 1e-4, weight-decay 1e-5); cosine schedule with 1-epoch warmup; bf16
autocast; channels-last. Image size 512, batch size 10, 12 epochs.

**Augmentation.** "V1": horizontal-flip, vertical-flip, rotate(±20°), then
resize the whole document to 512. (Crop-based *training* was tried and hurt;
crops help only at *test* time — see §5.)

**Checkpoint ensemble.** We save epochs 10, 11, and 12 and ensemble all three;
averaging the three neighbouring epochs denoises the pseudo-label-trained model.

## 3. Inference

For every test image each of the 3 checkpoints is scored with **dense-crop
test-time augmentation**:

* views = the native 512 resize **+** a 3×3 grid of crops taken from a 1.15×
  up-scaled image (9 crops), each shown in the 4 orientations of the D2 group
  (identity, h-flip, v-flip, 180° rotate) → **10 × 4 = 40 views**;
* the 40 view probabilities are averaged, then the 3 checkpoints are averaged
  → one fraud score per image.

Rationale: fraud cues (font/MRZ/splice/GenAI edits) are *local*; a full-image
resize blurs them, whereas averaging many local crops preserves them. The
inference is deterministic and needs no network (`pretrained=False`; weights
from local checkpoints).

## 4. Results

Public leaderboard (AuDET, lower = better):

| Configuration | Public |
|---|---|
| Single checkpoint (e12), dense-crop TTA | 0.00178 |
| **3-checkpoint ensemble (e10+e11+e12), dense-crop TTA — final** | **0.00145** |

## 5. Reproducibility

* **Code / weights frozen** at the tagged commit (see the pinned Kaggle reply
  for the exact SHA); no model, architecture, or training-code changes after the
  freeze date.
* **Deterministic, network-free inference** via the provided Docker image, run
  with `--network none`; weights are baked into the image (Git LFS in the repo).
* **Data sources:** competition training set + self-generated pseudo-labels on
  the public test set. No external data, no runtime downloads.
* **Training entrypoints:** `train/` (`make_folds.py`, `pseudo_gen.py`,
  `train_full.py`) reproduce the checkpoints from the competition data; commands
  are documented in `train/README_TRAIN.md`.
* **Hardware:** single NVIDIA H100 (80 GB), CUDA 12.6, PyTorch 2.7.1, timm
  1.0.27.

## 6. Notes / limitations

The gains at the extreme low-FPR tail are small in absolute terms (a handful of
tail images on the 5% public split), so ranking there is sensitive; the
ensemble + dense-crop TTA is chosen for stability rather than chasing the noise
floor.
