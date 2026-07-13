# eICU External Validation

## Design

- Development database: MIMIC-IV.
- External test database: eICU-CRD.
- Outcome: future 6-hour SOFA increase >= 2.
- Observation window: 24 hours.
- Model, membership functions, rules, attention and weights were frozen.
- Platt calibration, fixed-specificity thresholds and risk strata were fitted on MIMIC validation only.
- No eICU outcome was used for fitting, selection or recalibration.

## Cohort

- Windows: 6,215,890
- Patients: 80,239
- ICU stays: 99,262
- Hospitals: 205
- Event prevalence: 0.0475

## Performance

| Probability | AUROC | AUPRC | Brier | ECE | Calibration intercept | Calibration slope |
|---|---:|---:|---:|---:|---:|---:|
| Raw checkpoint | 0.6221 | 0.0922 | 0.2444 | 0.4340 | -3.041 | 0.780 |
| MIMIC-calibrated | 0.6221 | 0.0922 | 0.0459 | 0.0267 | -1.127 | 0.732 |

Patient-clustered 95% CI: AUROC 0.6192-0.6249; AUPRC 0.0902-0.0942; Brier 0.0455-0.0463.

## MIMIC-Defined Operating Points

| Target specificity | Threshold | External specificity | External sensitivity | PPV | NPV |
|---:|---:|---:|---:|---:|---:|
| 90% | 0.0924 | 0.790 (0.787-0.793) | 0.377 (0.372-0.383) | 0.082 (0.081-0.083) | 0.962 (0.962-0.963) |
| 95% | 0.1164 | 0.882 (0.880-0.884) | 0.262 (0.257-0.267) | 0.100 (0.098-0.101) | 0.960 (0.960-0.960) |

## Provenance

- Checkpoint: `outputs\explicit_temporal_fnn_formal_6h\seed_42\best_model.pt`
- Checkpoint SHA-256: `158427a5c358016f35b435b1ab5f75c7194a3ff3f9b6c9d68c5190a8a9125688`
- MIMIC validation windows: 819,573
- Cluster bootstrap unit: subject_id (500 replicates).
