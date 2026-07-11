# FREUID 2026 — Reproducibility reply

Paste the block below into the pinned Kaggle discussion thread (fills the
official REPLY_TEMPLATE). Replace every `<...>` placeholder.

```
---
FREUID Challenge 2026 — Reproducibility Package
---

Team name: <YOUR TEAM NAME>
Kaggle usernames: <comma-separated kaggle usernames>
Final Kaggle submission: psb512_ens.csv — AuDET 0.00145 (2026-07-11)

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
- Weights are fp16 (Git LFS); the leaderboard number was produced with the fp32
  training checkpoints — numerically equivalent (mean |Δscore| ~5e-4).
