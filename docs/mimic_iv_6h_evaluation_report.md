# MIMIC-IV 6-hour formal evaluation report

## Evaluation design

- Outcome: future 6-hour SOFA increase >= 2.
- Predictors: the same 13 protocol predictors for every model.
- Observation history: 24 hourly measurements.
- Equal-sample training cohort: 200,000 windows.
- Equal-sample validation cohort: 50,000 windows.
- Independent test cohort: 830,839 windows from 7,287 patients.
- Test prevalence: 5.69%.
- Train, validation, and test patients are disjoint according to `patient_split.csv`.
- All models use exactly the same test windows.
- Confidence intervals use 500 patient-clustered bootstrap replicates. All hourly
  windows belonging to the same `subject_id` are resampled together.
- Operating thresholds and Platt calibration parameters are estimated from validation
  data only and applied unchanged to the independent test set.

The independent test audit passed for all 12 models with no patient overlap or window
mismatch.

## Main equal-sample results

| Model | AUROC | AUPRC | Brier after validation-only calibration |
|---|---:|---:|---:|
| GRU | 0.6238 | 0.1037 | 0.05285 |
| LSTM | 0.6156 | 0.1002 | 0.05292 |
| XGBoost | 0.6073 | 0.0896 | 0.05315 |
| EBM | 0.6072 | 0.0891 | 0.05316 |
| Random Forest | 0.6038 | 0.0866 | 0.05322 |
| GAM | 0.6003 | 0.0878 | 0.05320 |
| LightGBM | 0.6002 | 0.0872 | 0.05322 |
| Logistic Regression | 0.5795 | 0.0794 | 0.05341 |
| KG-Temporal FNN | 0.5747 | 0.0738 | 0.05349 |
| Decision Tree | 0.5741 | 0.0754 | 0.05345 |
| NEWS2 | 0.5699 | 0.0736 | 0.05349 |
| SOFA | 0.4978 | 0.0558 | 0.05368 |

KG-Temporal FNN AUROC was 0.5747 (patient-clustered 95% CI 0.5678-0.5820),
and AUPRC was 0.0738 (95% CI 0.0706-0.0768). GRU had the highest discrimination:
AUROC 0.6238 (95% CI 0.6171-0.6304) and AUPRC 0.1037 (95% CI 0.0994-0.1082).

The full-cohort FNN analysis previously reached AUROC 0.5880 and AUPRC 0.0775.
It should be reported as a supplementary sample-size sensitivity analysis rather than
mixed into the equal-sample primary comparison.

## Paired comparisons

Paired differences use identical subject-level bootstrap replicates for both models.

- FNN versus NEWS2: AUROC difference +0.0049, 95% CI -0.0020 to 0.0118,
  p = 0.184; not statistically significant.
- FNN versus Decision Tree: AUROC difference +0.0006, 95% CI -0.0055 to 0.0064,
  p = 0.768; not statistically significant.
- FNN versus Logistic Regression: AUROC difference -0.0047, 95% CI -0.0129 to
  0.0030, p = 0.260; not statistically significant.
- FNN versus EBM: AUROC difference -0.0325, 95% CI -0.0385 to -0.0273,
  bootstrap p < 0.002.
- FNN versus GRU: AUROC difference -0.0490, 95% CI -0.0551 to -0.0429,
  bootstrap p < 0.002.

With 500 bootstrap replicates, an empirical p value reported as zero means
`p < 2 / 500 = 0.004` for a two-sided comparison; it should not be written as p = 0.

## Operating characteristics

Validation-defined KG-Temporal FNN thresholds produced:

| Target specificity | Test sensitivity | Observed test specificity | PPV | NPV |
|---:|---:|---:|---:|---:|
| 90% | 0.1548 | 0.9004 | 0.0857 | 0.9464 |
| 95% | 0.0861 | 0.9496 | 0.0935 | 0.9451 |

At the 90% specificity threshold, FNN detected 390 of 3,710 deterioration events
(10.5%), with median lead time 6 hours. GRU detected 880 events (23.7%), with median
lead time 5 hours.

Validation-defined FNN risk strata had test event rates of 4.75% (low), 6.41%
(medium), and 8.37% (high). This shows monotonic risk separation, although the absolute
separation is modest.

After validation-only Platt calibration, FNN calibration intercept was -0.272 and
slope was 0.909. Calibration curves use equal-frequency quantile bins because the
outcome prevalence is low.

## Completed checklist

- Independent patient-level test set: complete.
- Patient-clustered bootstrap 95% CI: complete, 500 replicates.
- Paired model comparison: complete for all 12 models.
- Unified 6-hour ROC, PR, and calibration report: complete.
- Sensitivity at 90% and 95% specificity: complete.
- Decision Curve Analysis: complete for thresholds 0.01-0.20.
- Lead Time Analysis: complete.
- Low, medium, and high risk stratification: complete.
- Calibration intercept and slope: complete.

Rule-quality evaluation for the new explicit-temporal FNN remains pending and is not
part of this baseline benchmark report.

## Primary artifacts

- `outputs/advanced_evaluation_6h_equal_sample/advanced_metrics.csv`
- `outputs/advanced_evaluation_6h_equal_sample/paired_model_comparisons.csv`
- `outputs/advanced_evaluation_6h_equal_sample/fixed_specificity_metrics.csv`
- `outputs/advanced_evaluation_6h_equal_sample/decision_curve.csv`
- `outputs/advanced_evaluation_6h_equal_sample/lead_time_summary.csv`
- `outputs/advanced_evaluation_6h_equal_sample/risk_stratification.csv`
- `outputs/advanced_evaluation_6h_equal_sample/figures/`
