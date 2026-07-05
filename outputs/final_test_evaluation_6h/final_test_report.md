# Final Frozen-Model Test Evaluation

- Primary outcome: `label_sofa_increase_ge2_6h`.
- Frozen checkpoint SHA-256: `158427a5c358016f35b435b1ab5f75c7194a3ff3f9b6c9d68c5190a8a9125688`.
- Checkpoint epoch: 15.
- Test windows: 830,839; patients: 7,287.
- Bootstrap: 1000 replicates clustered by `subject_id`.
- Calibration and operating thresholds were fitted on validation only.

## Performance

| Metric | Estimate | Patient-clustered 95% CI |
|---|---:|---:|
| AUROC | 0.6559 | 0.6492-0.6628 |
| AUPRC | 0.1309 | 0.1250-0.1375 |
| Brier score | 0.0521 | 0.0507-0.0534 |
| ECE | 0.0012 | 0.0006-0.0026 |
| Calibration intercept | -0.020 | - |
| Calibration slope | 0.997 | - |

## Fixed-Specificity Operating Points

| Target specificity | Threshold | Observed specificity | Sensitivity | PPV | NPV | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 90% | 0.0924 | 0.9006 (0.8946-0.9062) | 0.2667 (0.2534-0.2811) | 0.1394 (0.1343-0.1450) | 0.9532 (0.9518-0.9545) | 0.1831 (0.1767-0.1901) |
| 95% | 0.1164 | 0.9507 (0.9467-0.9542) | 0.1755 (0.1645-0.1880) | 0.1767 (0.1689-0.1851) | 0.9503 (0.9489-0.9517) | 0.1761 (0.1678-0.1848) |

F1 at probability 0.5 is retained in `advanced_metrics.csv` for supplement use.
