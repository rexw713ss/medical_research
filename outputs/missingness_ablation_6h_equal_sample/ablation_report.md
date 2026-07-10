# Formal 6-hour FNN Ablation Study

Values are mean +/- SD across random seeds. Brier and ECE use validation-only Platt calibration.
Rule Stability is mean pairwise Top-10 Jaccard similarity.

| Model | AUROC | AUPRC | Brier | ECE | Rule Concordance | Rule Stability | Rule Drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| Explicit KG-TFNN without missingness channels | 0.603 +/- 0.000 | 0.088 +/- 0.000 | 0.053 +/- 0.000 | 0.001 +/- 0.000 | 0.735 +/- 0.022 | 0.879 | 0.231 +/- 0.001 |
| Missingness-only Temporal FNN | 0.595 +/- 0.000 | 0.090 +/- 0.000 | 0.053 +/- 0.000 | 0.002 +/- 0.000 | NA | 0.879 | 0.241 +/- 0.001 |
