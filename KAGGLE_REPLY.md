# FREUID 2026 — Reproducibility reply

Paste the block below into the pinned Kaggle discussion thread (fills the
official REPLY_TEMPLATE). Replace every `<...>` placeholder.

```
---
FREUID Challenge 2026 — Reproducibility Package
---

Team name: <YOUR TEAM NAME>
Kaggle usernames: <comma-separated kaggle usernames>
Final Kaggle submission: <selected submission> — AuDET 0.00115 (2026-07-11)

Repository (this should be public git repository): https://github.com/hungpv297/freuid2026-submission
Commit SHA: <FROZEN 40-char SHA — run `git rev-parse HEAD` at freeze>
Technical report (PDF): https://github.com/hungpv297/freuid2026-submission/blob/main/REPORT.pdf

We confirm this repository at the stated commit reproduces our selected final
submission and complies with the competition rules.

Signed (team captain): <YOUR KAGGLE USERNAME>
Date (UTC): 2026-07-__
```

Notes for reviewers:
- Docker: `docker run --rm --gpus all --network none -v imgs:/data:ro -v out:/submissions freuid-infer`
- Runtime: **12 forward passes/image** (3 checkpoints × 4 edge-crops). Plain
  default run (`--workers 0`, no `--shm-size` needed): **428 s for the 7,821
  public images on one H100 → ≈2.0 h for the full 134,997 hidden-test images on
  one H100**; extrapolated to a single **A100** ≈**4.1 h** (realistic 2.0×),
  ≤ ~6 h even at the compute-peak worst case — within the 6-hour single-A100
  limit. A `--shm-size=8g --workers 8` run is ~15% faster.
- Weights are fp16 (Git LFS); the leaderboard number was produced with the fp32
  training checkpoints — numerically equivalent (mean |Δscore| ~5e-4).
