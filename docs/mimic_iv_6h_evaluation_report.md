# MIMIC-IV Primary 6-Hour Evaluation

## Evaluation Design

- Outcome: SOFA increase >=2 within the next 6 hours.
- Inputs: the same 13 clinical predictors and 24-hour history; the proposed model additionally represents prespecified missingness and time-since channels from those predictors.
- Equal-sample cohort: 200,000 training windows, 50,000 validation windows, and all 830,839 independent test windows from 7,287 test patients.
- All compared models use the same patient split, validation/test windows, outcomes, and validation-only probability calibration.
- Confidence intervals and model differences use 1,000 `subject_id`-clustered bootstrap replicates for the prespecified paired comparison.
- Test outcomes were not used for tuning, checkpoint selection, calibration, or threshold selection.

The key/outcome audit passed for all paired models: explicit KG-TFNN, Logistic Regression, EBM, XGBoost, and GRU.

## Primary Equal-Sample Results

| Model | AUROC (95% CI) | AUPRC (95% CI) | Brier | ECE |
|---|---:|---:|---:|---:|
| Explicit Knowledge-Guided Temporal FNN | 0.6448 (0.6379-0.6515) | 0.1236 (0.1177-0.1297) | 0.0523 | 0.0013 |
| GRU | 0.6238 (0.6170-0.6306) | 0.1037 (0.0992-0.1082) | 0.0529 | 0.0014 |
| XGBoost | 0.6073 (0.6007-0.6141) | 0.0896 (0.0859-0.0934) | 0.0532 | 0.0013 |
| Explainable Boosting Machine | 0.6072 (0.6008-0.6141) | 0.0891 (0.0853-0.0929) | 0.0532 | 0.0012 |
| Logistic Regression | 0.5795 (0.5718-0.5872) | 0.0794 (0.0760-0.0827) | 0.0534 | 0.0013 |

Other prespecified baselines remain available in `table_3_model_performance.csv`; the old sequence-only FNN has been removed from the primary proposed-model row.

## Paired Comparisons

Compared with GRU on identical test windows, explicit KG-TFNN improved:

- AUROC by 0.0210 (patient-clustered 95% CI 0.0152-0.0267; bootstrap P<0.002).
- AUPRC by 0.0199 (95% CI 0.0151-0.0248; P<0.002).
- Brier score by -0.00056 (95% CI -0.00068 to -0.00044; P<0.002).
- Sensitivity at the 90% specificity operating point by 0.0333.
- Sensitivity at the 95% specificity operating point by 0.0294.

The explicit KG-TFNN also outperformed Logistic Regression, EBM, and XGBoost in paired AUROC and AUPRC. An empirical bootstrap P value stored as zero is reported as `P<2/1000`, not `P=0`.

## Operating Characteristics

| Validation target specificity | Threshold | Test sensitivity | Test specificity | PPV | NPV |
|---:|---:|---:|---:|---:|---:|
| 90% | 0.0925 | 0.2523 | 0.9024 | 0.1350 | 0.9524 |
| 95% | 0.1152 | 0.1650 | 0.9518 | 0.1713 | 0.9497 |

Validation-defined risk strata showed monotonic test event rates of 3.89% (low), 6.35% (medium), and 11.99% (high). The window-based lead-time analysis detected 1,291 of 3,710 first deterioration events, with median lead time 4 hours (IQR 2-6).

After validation-only calibration, test calibration intercept was -0.074 and slope was 0.977. Event-level alert burden with a 6-hour refractory period is reported separately in `outputs/clinical_sensitivity_analyses_6h/`.

## Primary Artifacts

- `outputs/explicit_kg_tfnn_paired_comparison_6h/input_audit.json`
- `outputs/explicit_kg_tfnn_paired_comparison_6h/evaluation/advanced_metrics.csv`
- `outputs/explicit_kg_tfnn_paired_comparison_6h/evaluation/paired_model_comparisons.csv`
- `outputs/explicit_kg_tfnn_paired_comparison_6h/evaluation/fixed_specificity_metrics.csv`
- `outputs/explicit_kg_tfnn_paired_comparison_6h/evaluation/decision_curve.csv`
- `outputs/explicit_kg_tfnn_paired_comparison_6h/evaluation/lead_time_summary.csv`
- `outputs/explicit_kg_tfnn_paired_comparison_6h/evaluation/risk_stratification.csv`
- `outputs/explicit_kg_tfnn_paired_comparison_6h/evaluation/figures/`
