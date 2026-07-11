# Feature-Matched Baselines for the Primary 6-Hour Task

## Design

- Outcome: SOFA increase >=2 in `(t, t+6]`.
- Observation history: 24 hourly steps.
- Split: fixed patient-level `subject_id` manifest.
- Samples: 200,000 train windows, 50,000 validation windows, and all 830,839 independent test windows.
- GRU input: 39 hourly channels comprising 13 clinical values, 13 raw missingness indicators, and 13 time-since-last-measurement channels.
- XGBoost/LightGBM input: 143 deterministic current-state and temporal summaries derived from the same 39 channels.
- Calibration and model selection: MIMIC validation data only.
- Paired inference: 1,000 patient-clustered bootstrap replicates on identical test windows.

## Test Performance

| Model | AUROC | AUPRC |
|---|---:|---:|
| Explicit KG-TFNN | 0.6448 | 0.1236 |
| Feature-matched GRU | 0.6587 | 0.1272 |
| Feature-matched XGBoost | 0.6870 | 0.1665 |
| Feature-matched LightGBM | 0.6904 | 0.1710 |

## Paired KG-TFNN Differences

| Comparator | Delta AUROC (95% CI) | P | Delta AUPRC (95% CI) | P |
|---|---:|---:|---:|---:|
| GRU | -0.0139 (-0.0192 to -0.0088) | <0.002 | -0.0036 (-0.0085 to 0.0008) | 0.10 |
| XGBoost | -0.0422 (-0.0461 to -0.0381) | <0.002 | -0.0429 (-0.0470 to -0.0389) | <0.002 |
| LightGBM | -0.0456 (-0.0499 to -0.0413) | <0.002 | -0.0474 (-0.0518 to -0.0432) | <0.002 |

Empirical zero bootstrap values are reported as `P<2/1000`, not `P=0`.

## Interpretation

The feature-matched analysis does not support architecture superiority for KG-TFNN. The manuscript therefore frames KG-TFNN as an intrinsic-interpretability/performance tradeoff: its explicit fuzzy memberships and IF-THEN rule structure are available by construction, while feature-matched boosted trees achieved higher discrimination. This experiment evaluates matched information availability; the tree representation remains a deterministic summary of the same hourly channels rather than an identical neural architecture.
