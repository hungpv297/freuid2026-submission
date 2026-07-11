# Training — how the frozen weights were produced

These scripts are provided for **reproducibility / documentation**. They use
absolute paths under `/mnt/ecai/` (our training box); a reproducer should adjust
`ROOT`/paths to point at the competition data. Environment: Python 3.11, PyTorch
2.7.1 (CUDA 12.6), timm 1.0.27, albumentations 2.0.8, single NVIDIA H100 80 GB.

Data layout assumed:
```
/mnt/ecai/data/train_labels.csv               # columns: id,label,type,is_digital
/mnt/ecai/data/train/train/<id>.jpeg          # training images
/mnt/ecai/data/public_test/public_test/<id>.jpeg
```

## Pipeline (4 steps)

**1. Folds** — 5-fold stratified split (label × document-type), seed 42.
```bash
python make_folds.py            # -> /mnt/ecai/folds.csv
```

**2. Base model** — train `maxvit_base_tf_512` on ALL folds merged (no val),
BCE, V1 aug, 12 epochs; this also writes public-test predictions per checkpoint.
```bash
python train_full.py --exp full_maxvit_b512_b12 \
    --model maxvit_base_tf_512.in21k_ft_in1k \
    --imgsz 512 --bs 10 --lr 1e-4 --epochs 12 --save_epochs 10,11,12 --aug v1
# -> exp/full_maxvit_b512_b12/{ckpt_e10,e11,e12}.pt + pred_e{10,11,12}.csv
```

**3. Pseudo-labels** — ensemble several strong maxvit checkpoints' public-test
predictions; keep only confident ones (score < 0.05 → bona-fide, > 0.95 →
fraud) as extra training rows. **No test labels are used** — the pseudo-labels
are the models' own confident predictions (semi-supervised).
```bash
python pseudo_gen.py            # -> /mnt/ecai/pseudo.csv
```

**4. Final model** — retrain `maxvit_base_tf_512` on ALL folds + pseudo rows,
same recipe as step 2, saving epochs 10/11/12. **These three checkpoints are the
final weights** (see `../weights/`).
```bash
python train_full.py --exp ps_full_maxvit_b512_b12 \
    --model maxvit_base_tf_512.in21k_ft_in1k \
    --imgsz 512 --bs 10 --lr 1e-4 --epochs 12 --save_epochs 10,11,12 \
    --aug v1 --pseudo /mnt/ecai/pseudo.csv
# -> exp/ps_full_maxvit_b512_b12/{ckpt_e10,e11,e12}.pt   (final)
```

Inference (dense-crop TTA + 3-checkpoint ensemble) is in `../infer.py`.

## Files
* `make_folds.py`   — stratified 5-fold split.
* `train_full.py`   — all-data (no val) trainer; `--pseudo` concatenates the
  pseudo CSV; imports `DS` + `build_aug` from `train_sweep.py`.
* `train_sweep.py`  — dataset + augmentation definitions (`build_aug`: V1..V5).
* `pseudo_gen.py`   — builds the confident pseudo-label CSV.
* `models_cfg.py`   — backbone/config table used by the wider sweep.

## Notes
* The shipped weights are stored in **fp16** (halved size); inference casts them
  into an fp32 model — numerically equivalent to the fp32 training checkpoints
  (mean |Δscore| ≈ 5e-4 on a 30-image check).
* No external datasets are used. Only the competition train set + self-generated
  pseudo-labels on the public test images.
