# Training — how the frozen weights were produced

These scripts are provided for **reproducibility / documentation**. They use
absolute paths under `/mnt/ecai/` (our training box) by default; set
`FREUID_ROOT` (used by `pseudo_gen.py`) and/or adjust the `/mnt/ecai/...` paths
in `train_full.py` / `train_sweep.py` to point at your data. Environment:
Python 3.11, PyTorch 2.7.1 (CUDA 12.6), timm 1.0.27, albumentations 2.0.8,
single NVIDIA H100 80 GB.

Data layout assumed:
```
/mnt/ecai/data/train_labels.csv               # columns: id,label,type,is_digital
/mnt/ecai/data/train/train/<id>.jpeg          # training images
/mnt/ecai/data/public_test/public_test/<id>.jpeg
```

## Overview

The final weights are **3 checkpoints (e10/e11/e12) of a single backbone,
`maxvit_base_tf_512`, trained on ALL competition data + semi-supervised
pseudo-labels**. The one subtlety is that the pseudo-labels are the **mean of
SEVEN model predictions**, so a full from-scratch reproduction trains those
seven models first. To make the final step reproducible without that, the
resulting **`pseudo.csv` is shipped in this folder** — you can skip straight to
step 4.

```
step 1  make_folds.py            -> folds.csv
step 2  train_full.py (base)     -> full_maxvit_b512_b12/{ckpt,pred}_e{10,11,12}
step 3a train_sweep.py x4        -> the 4 extra pseudo-source predictions
step 3b pseudo_gen.py            -> pseudo.csv   (or use the shipped one)
step 4  train_full.py --pseudo   -> ps_full_maxvit_b512_b12/ckpt_e{10,11,12}  (FINAL)
```

## 1. Folds
5-fold stratified split (label × document-type), seed 42.
```bash
python make_folds.py            # -> /mnt/ecai/folds.csv
```

## 2. Base all-data model (also supplies 3 of the 7 pseudo sources)
`maxvit_base_tf_512` on ALL folds merged (no val), BCE, V1 aug, 12 epochs;
writes a checkpoint AND public-test predictions per saved epoch.
```bash
python train_full.py --exp full_maxvit_b512_b12 \
    --model maxvit_base_tf_512.in21k_ft_in1k \
    --imgsz 512 --bs 10 --lr 1e-4 --epochs 12 --save_epochs 10,11,12 --aug v1
# -> exp/full_maxvit_b512_b12/{ckpt_e10,e11,e12}.pt + pred_e{10,11,12}.csv
```

## 3a. The 4 extra pseudo-source models (fold-0 sweep / epoch study)
Each is a single fold-0 model whose `pred_public.csv` (hflip-TTA) feeds the
pseudo ensemble. Configs (imgsz/bs/lr) are from `models_cfg.py`.
```bash
python train_sweep.py --exp ep08_maxvit_l512  --model maxvit_large_tf_512.in21k_ft_in1k --imgsz 512 --bs 8  --lr 1e-4 --epochs 8  --aug v1 --fold 0
python train_sweep.py --exp 07_maxvit_l512_e15 --model maxvit_large_tf_512.in21k_ft_in1k --imgsz 512 --bs 8  --lr 1e-4 --epochs 15 --aug v1 --fold 0
python train_sweep.py --exp 03_maxvit_b512_e15 --model maxvit_base_tf_512.in21k_ft_in1k  --imgsz 512 --bs 10 --lr 1e-4 --epochs 15 --aug v1 --fold 0
python train_sweep.py --exp 23_maxvit_b384_e15 --model maxvit_base_tf_384.in21k_ft_in1k  --imgsz 384 --bs 16 --lr 1e-4 --epochs 15 --aug v1 --fold 0
# -> exp/<exp>/pred_public.csv  (one per model)
```

## 3b. Pseudo-labels — ensemble the 7 predictions
Mean of the 7 public-test predictions; keep confident rows (score < 0.05 →
bona-fide, > 0.95 → fraud). **No test labels are used.** The script now
**requires all 7 sources** (it errors instead of silently using fewer).
```bash
python pseudo_gen.py            # -> /mnt/ecai/pseudo.csv  (7,301 confident rows)
```
**Shortcut:** the produced `pseudo.csv` is shipped here as `train/pseudo.csv`
(id, filepath relative to the public-test dir, label, score) — use it directly
in step 4 to skip steps 2's predictions + 3a + 3b.

## 4. Final model (the shipped weights)
Retrain `maxvit_base_tf_512` on ALL folds + pseudo rows, same recipe as step 2,
saving epochs 10/11/12. **These three checkpoints are `../weights/`.**
```bash
python train_full.py --exp ps_full_maxvit_b512_b12 \
    --model maxvit_base_tf_512.in21k_ft_in1k \
    --imgsz 512 --bs 10 --lr 1e-4 --epochs 12 --save_epochs 10,11,12 \
    --aug v1 --pseudo train/pseudo.csv
# -> exp/ps_full_maxvit_b512_b12/{ckpt_e10,e11,e12}.pt   (final)
```

Inference (4-edge-crop TTA + 3-checkpoint ensemble, 12 passes/image) is in
`../infer.py`.

## Files
* `make_folds.py`   — stratified 5-fold split.
* `train_full.py`   — all-data (no val) trainer; `--pseudo` concatenates the
  pseudo CSV; imports `DS` + `build_aug` from `train_sweep.py`.
* `train_sweep.py`  — fold-0 trainer + dataset/aug (`build_aug`: V1..V5); also
  the source of the 4 extra pseudo predictions.
* `pseudo_gen.py`   — builds the confident pseudo-label CSV from the 7 sources
  (strict: requires all 7).
* `pseudo.csv`      — the shipped pseudo-label artifact (7,301 rows).
* `models_cfg.py`   — backbone/config table (imgsz/bs/lr per model).

## Notes
* The shipped weights are stored in **fp16** (halved size); inference casts them
  into an fp32 model — numerically equivalent to the fp32 training checkpoints
  (mean |Δscore| ≈ 5e-4 on a 30-image check).
* No external datasets are used. Only the competition train set + self-generated
  pseudo-labels on the public test images.
* Pseudo-labels are a **domain-adaptation bet for the private set**; they do NOT
  improve the public leaderboard (non-pseudo base + dense-crop TTA actually
  scores better on public). See the report for the LB breakdown.
