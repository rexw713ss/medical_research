# Formal 6-hour FNN Ablation Study

Values are mean +/- SD across random seeds. Brier and ECE use validation-only Platt calibration.
Rule Stability is mean pairwise Top-10 Jaccard similarity.

| Model | AUROC | AUPRC | Brier | ECE | Rule Concordance | Rule Stability | Rule Drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| Randomly initialized FNN | 0.640 +/- 0.001 | 0.118 +/- 0.002 | 0.052 +/- 0.000 | 0.001 +/- 0.000 | 0.496 +/- 0.127 | 0.207 | 0.165 +/- 0.010 |
| Guideline-guided FNN without temporal features | 0.595 +/- 0.000 | 0.084 +/- 0.001 | 0.053 +/- 0.000 | 0.001 +/- 0.000 | 0.647 +/- 0.021 | 0.818 | 0.245 +/- 0.003 |
| Temporal FNN without clinical consistency regularization | 0.646 +/- 0.002 | 0.123 +/- 0.000 | 0.052 +/- 0.000 | 0.001 +/- 0.000 | 0.536 +/- 0.119 | 0.587 | 0.222 +/- 0.007 |
| Full Knowledge-Guided Temporal FNN | 0.646 +/- 0.001 | 0.123 +/- 0.001 | 0.052 +/- 0.000 | 0.001 +/- 0.000 | 0.528 +/- 0.123 | 0.674 | 0.224 +/- 0.011 |
