# FREUID 2026 — Reproducibility reply (paste into the pinned Kaggle thread)

```
Team name: <YOUR TEAM NAME>
Kaggle usernames: <comma-separated kaggle usernames>
Final Kaggle submission: psb512_ens.csv — AuDET 0.00145 (2026-07-11)

Repository (this should be public git repository): https://github.com/hungpv297/freuid2026-submission
Commit SHA: <FILL WITH FROZEN 40-char SHA — see below>
Technical report (PDF): https://github.com/hungpv297/freuid2026-submission/blob/main/REPORT.pdf

We confirm this repository at the stated commit reproduces our selected final
submission and complies with the competition rules.

Signed (team captain): <YOUR KAGGLE USERNAME>
Date (UTC): 2026-07-__
```

Notes:
- Fill <...> fields. Commit SHA = the frozen commit (run `git rev-parse HEAD`
  after your last change, before the July-13 freeze).
- Shipped weights are fp16 (Git LFS); leaderboard number produced with the fp32
  training checkpoints — numerically equivalent (mean |Δscore| ~5e-4).
- Docker: `docker run --rm --gpus all --network none -v imgs:/data:ro -v out:/submissions freuid-infer`.
