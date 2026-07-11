"""Generate the semi-supervised pseudo-labels used by the FINAL model.

The pseudo-labels are the MEAN of the public-test predictions of SEVEN strong,
diverse maxvit checkpoints (no ground-truth test labels are used). Only very
confident rows are kept (score < 0.05 -> bona-fide, > 0.95 -> fraud).

The seven source prediction CSVs and how each is produced:
  from train_full.py  (all-data base model, step 2 of README_TRAIN.md):
    exp/full_maxvit_b512_b12/pred_e12.csv
    exp/full_maxvit_b512_b12/pred_e11.csv
    exp/full_maxvit_b512_b12/pred_e10.csv
  from train_sweep.py (fold-0 models, exact commands in README_TRAIN.md):
    exp/ep08_maxvit_l512/pred_public.csv     # maxvit_large_tf_512, 8 epochs
    exp/07_maxvit_l512_e15/pred_public.csv   # maxvit_large_tf_512, 15 epochs
    exp/03_maxvit_b512_e15/pred_public.csv   # maxvit_base_tf_512, 15 epochs
    exp/23_maxvit_b384_e15/pred_public.csv   # maxvit_base_tf_384, 15 epochs

NOTE: the resulting pseudo.csv is ALSO shipped alongside this script
(../train/pseudo.csv) so the final model can be reproduced WITHOUT retraining
all seven source models. Re-run this script only to regenerate from scratch.
"""
import os, argparse, numpy as np, pandas as pd

ROOT = os.environ.get("FREUID_ROOT", "/mnt/ecai")
SRC = [
    f"{ROOT}/exp/full_maxvit_b512_b12/pred_e12.csv",
    f"{ROOT}/exp/full_maxvit_b512_b12/pred_e11.csv",
    f"{ROOT}/exp/full_maxvit_b512_b12/pred_e10.csv",
    f"{ROOT}/exp/ep08_maxvit_l512/pred_public.csv",
    f"{ROOT}/exp/07_maxvit_l512_e15/pred_public.csv",
    f"{ROOT}/exp/03_maxvit_b512_e15/pred_public.csv",
    f"{ROOT}/exp/23_maxvit_b384_e15/pred_public.csv",
]
LO, HI = 0.05, 0.95

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{ROOT}/pseudo.csv")
    ap.add_argument("--public_dir", default=f"{ROOT}/data/public_test/public_test")
    a = ap.parse_args()

    # STRICT: all seven sources are required — do NOT silently drop missing files,
    # otherwise the pseudo set (and hence the final weights) changes without notice.
    missing = [s for s in SRC if not os.path.exists(s)]
    if missing:
        raise SystemExit(
            "[pseudo_gen] missing %d/%d source predictions:\n  %s\n"
            "Produce them first (see README_TRAIN.md), or use the shipped "
            "train/pseudo.csv to reproduce the final model directly."
            % (len(missing), len(SRC), "\n  ".join(missing)))

    print(f"[pseudo_gen] ensembling {len(SRC)} prediction files")
    dfs = [pd.read_csv(s).set_index("id")["label"].rename(f"p{i}") for i, s in enumerate(SRC)]
    M = pd.concat(dfs, axis=1)
    assert not M.isna().any().any(), "id mismatch across source prediction files"
    score = M.mean(axis=1)

    df = pd.DataFrame({"id": score.index, "score": score.values})
    df["filepath"] = df["id"].map(lambda x: os.path.join(a.public_dir, f"{x}.jpeg"))
    conf = df[(df.score < LO) | (df.score > HI)].copy()
    conf["label"] = (conf.score > 0.5).astype(int)
    conf[["id", "filepath", "label", "score"]].to_csv(a.out, index=False)
    print(f"[pseudo_gen] public_test={len(df)} confident(<{LO} or >{HI})={len(conf)} "
          f"dropped={len(df)-len(conf)}")
    print(f"[pseudo_gen] label balance: {conf.label.value_counts().to_dict()}")
    print(f"[pseudo_gen] saved {a.out}")

if __name__ == "__main__":
    main()
