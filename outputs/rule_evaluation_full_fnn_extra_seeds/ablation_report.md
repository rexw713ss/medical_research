# Formal 6-hour FNN Ablation Study

Values are mean +/- SD across random seeds. Brier and ECE use validation-only Platt calibration.
Rule Stability is mean pairwise Top-10 Jaccard similarity.

| Model | AUROC | AUPRC | Brier | ECE | Rule Concordance | Rule Stability | Rule Drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full Knowledge-Guided Temporal FNN | 0.642 +/- 0.002 | 0.122 +/- 0.000 | 0.052 +/- 0.000 | 0.001 +/- 0.000 | 0.503 +/- 0.106 | 0.667 | 0.224 +/- 0.005 |
