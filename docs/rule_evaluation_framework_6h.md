# Rule Evaluation Framework: Experimental Results

## Summary

| Analysis | Experimental result |
|---|---:|
| Rule Complexity | Top-10 mean antecedents = 1.44 |
| Rule Stability | 5-seed mean pairwise Top-10 Jaccard = 0.720 |
| Guideline-Direction Alignment | Mean prespecified direction alignment = 1.000 |
| Rule Drift | Median center shift = 0.264 initial sigmas (mean 1.421) |

Five-seed complexity/stability uses the fixed equal-sample training protocol (200,000 train and 50,000 validation windows per seed). Direction alignment, membership drift, activated rules and case studies use the frozen full-cohort final model.

## Rule Complexity And Stability

Five seeds produced 10 pairwise comparisons. Values below use the same Top-10 rule definition.

| Seed pair | Jaccard similarity |
|---|---:|
| 42 vs 52 | 0.818 |
| 42 vs 62 | 0.538 |
| 42 vs 72 | 0.818 |
| 42 vs 82 | 0.818 |
| 52 vs 62 | 0.667 |
| 52 vs 72 | 0.667 |
| 52 vs 82 | 1.000 |
| 62 vs 72 | 0.538 |
| 62 vs 82 | 0.667 |
| 72 vs 82 | 0.667 |

## Guideline-Direction Alignment Rubric

Each rule receives one point for guideline-consistent static direction and one for a clinically worsening/persistent temporal direction. Cross-feature rules additionally require a predefined clinical rule combination. NA criteria are excluded from the denominator.

| Rank | Rule | Static | Temporal | Cross-rule | Overall |
|---:|---|---:|---:|---:|---:|
| 1 | IF FiO2 requirement IS supplemental o2 AND FiO2 requirement measurements were frequently abnormal THEN deterioration risk IS high | 1 | 1 | NA | 1.00 |
| 2 | IF platelet count IS low AND platelet count abnormality persisted across the observation window THEN deterioration risk IS high | 1 | 1 | NA | 1.00 |
| 3 | IF creatinine IS very high AND creatinine-related fuzzy risk increased over the 24-hour window THEN deterioration risk IS high | 1 | 1 | NA | 1.00 |
| 4 | IF GCS IS severely altered AND GCS-related fuzzy risk increased over the 24-hour window THEN deterioration risk IS high | 1 | 1 | NA | 1.00 |
| 5 | IF systolic blood pressure IS very low AND systolic blood pressure-related fuzzy risk increased over the 24-hour window THEN deterioration risk IS high | 1 | 1 | NA | 1.00 |
| 6 | IF bilirubin IS critical high AND bilirubin-related fuzzy risk increased over the 24-hour window THEN deterioration risk IS high | 1 | 1 | NA | 1.00 |
| 7 | IF lactate IS severe AND lactate-related fuzzy risk increased over the 24-hour window THEN deterioration risk IS high | 1 | 1 | NA | 1.00 |
| 8 | IF GCS IS altered AND SpO2 IS low AND GCS-related fuzzy risk increased over the 24-hour window THEN deterioration risk IS high | 1 | 1 | 1 | 1.00 |
| 9 | IF creatinine IS high AND platelet count IS low AND bilirubin IS high AND creatinine-related fuzzy risk increased over the 24-hour window THEN deterioration risk IS high | 1 | 1 | 1 | 1.00 |
| 10 | IF PaO2/FiO2 ratio IS very low AND FiO2 requirement IS high support AND PaO2/FiO2 ratio-related fuzzy risk increased during the last hour THEN deterioration risk IS high | 1 | 1 | 1 | 1.00 |

## Membership-Function Drift

| Parameter | Mean shift | Median shift |
|---|---:|---:|
| Center, normalized by initial sigma | 1.421 | 0.264 |
| Sigma, relative change | 0.658 | 0.299 |
| Rule weight, absolute change | 1.469 | 1.350 |

## Activated Rules

Rule activation is measured at the attention-selected hour using normalized cross-rule activation > 0.1.

| Outcome group | Windows | Mean activated rules | Median (IQR) |
|---|---:|---:|---:|
| Negative windows | 783,547 | 1.819 | 2.0 (1.0-2.0) |
| Positive windows | 47,292 | 1.833 | 2.0 (1.0-2.0) |

## TP/FP/FN Case Studies

Cases use the MIMIC-validation threshold for 90% specificity (0.0924). One probability-median representative was selected per error group.

| Case | Subject | Stay | Index hour | True outcome | Calibrated risk | Activated rules |
|---|---:|---:|---:|---:|---:|---:|
| TP | 11436396 | 31660563 | 58 | 1 | 0.136 | 3 |
| FP | 15117765 | 37635377 | 209 | 0 | 0.116 | 3 |
| FN | 16846280 | 30755921 | 563 | 1 | 0.052 | 2 |

Figures: `outputs/rule_evaluation_6h/figures/`. Detailed numeric tables are stored beside this report.
