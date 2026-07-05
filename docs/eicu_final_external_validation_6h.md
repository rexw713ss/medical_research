# eICU Final External Validation

更新日期：2026-07-05

## Design

- Development：MIMIC-IV train/validation。
- Internal test：MIMIC-IV frozen final-model evaluation。
- External test：eICU-CRD。
- Outcome：未來 6 小時 SOFA increase >= 2。
- Observation window：24 小時。
- Frozen checkpoint SHA-256：`158427a5c358016f35b435b1ab5f75c7194a3ff3f9b6c9d68c5190a8a9125688`。
- 模型、membership functions、rules、attention 與 weights 均未在 eICU 重新訓練。
- Platt calibration 與 operating thresholds 僅來自完整 MIMIC validation set；沒有使用 eICU outcome fitting 或 recalibration。

## Cohort

| Patients | ICU stays | Hospitals | Windows | Event prevalence |
|---:|---:|---:|---:|---:|
| 80,239 | 99,262 | 205 | 6,215,890 | 4.75% |

## External Performance

| Metric | Estimate | Patient-clustered 95% CI |
|---|---:|---:|
| AUROC | 0.6221 | 0.6192–0.6249 |
| AUPRC | 0.0922 | 0.0902–0.0942 |
| Brier, MIMIC-calibrated | 0.0459 | 0.0455–0.0463 |
| ECE, MIMIC-calibrated | 0.0267 | - |
| Calibration intercept | -1.127 | - |
| Calibration slope | 0.732 | - |

## Transported Operating Points

| MIMIC target specificity | Fixed threshold | eICU specificity | eICU sensitivity | PPV | NPV |
|---:|---:|---:|---:|---:|---:|
| 90% | 0.0924 | 0.790（0.787–0.793） | 0.377（0.372–0.383） | 0.082 | 0.962 |
| 95% | 0.1164 | 0.882（0.880–0.884） | 0.262（0.257–0.267） | 0.100 | 0.960 |

MIMIC 的固定 thresholds 在 eICU 未維持原目標 specificity，顯示 probability calibration 與 operating point 存在跨資料庫 transportability gap。這不應以 eICU recalibration 修飾 primary external result；可在 supplement 另做 recalibration sensitivity analysis，但必須與原始 frozen transport 結果分開。

## Provenance

- Bootstrap：500 次，以 `subject_id` 為 cluster。
- 完整報告：`outputs/eicu_external_validation/final_frozen_model_evaluation/eicu_external_validation_report.md`
- Metrics：`outputs/eicu_external_validation/final_frozen_model_evaluation/external_metrics.json`
- Operating points：`outputs/eicu_external_validation/final_frozen_model_evaluation/external_fixed_specificity.csv`
- Figures：`outputs/eicu_external_validation/final_frozen_model_evaluation/figures/`
