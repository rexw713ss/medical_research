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
| Raw checkpoint | 0.6034 | 0.0762 | 0.2236 | 0.4109 | -2.939 | 0.734 |
| MIMIC-calibrated | 0.6034 | 0.0762 | 0.0458 | 0.0223 | -1.187 | 0.690 |

Patient-clustered 95% CI: AUROC 0.6006-0.6059; AUPRC 0.0748-0.0778; Brier 0.0454-0.0461.

## Provenance

- Checkpoint: `outputs\explicit_temporal_observation_sensitivity_6h\seed_42\observation_24h_explicit\best_model.pt`
- Checkpoint SHA-256: `49a1a914fe2a55609bbfb26a92425bbaa1fd07c05909c04b5a7a8a18e60f6b76`
- MIMIC validation windows: 50,000
- Cluster bootstrap unit: subject_id (200 replicates).
