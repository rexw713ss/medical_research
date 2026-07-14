# Frozen Baseline External Validation on eICU

## Design

- Outcome: future 6-hour SOFA increase >= 2.
- Observation window: 24 hours.
- Complete external cohort: 80,239 patients, 99,262 ICU stays, and 6,215,890 windows.
- Predictive parameters were frozen after MIMIC-IV development.
- Calibration and fixed-specificity thresholds used MIMIC validation only.
- No eICU outcome was used for fitting, model selection, calibration, or threshold selection.
- Uncertainty used 200 patient-clustered bootstrap replicates.
- EBM is a current-state comparator and is not architecture matched.

## External Performance

| display_name | input_design | auroc | auroc_ci_low | auroc_ci_high | auprc | auprc_ci_low | auprc_ci_high | brier | ece | calibration_intercept | calibration_slope | sensitivity_at_spec_90 | sensitivity_at_spec_95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KG-TFNN (equal-sample) | 24h x 39 feature-matched | 0.6100 | 0.6074 | 0.6125 | 0.0862 | 0.0844 | 0.0880 | 0.0456 | 0.0222 | -1.1318 | 0.7107 | 0.3281 | 0.2253 |
| LightGBM (feature-matched) | 24h x 39 feature-matched | 0.6247 | 0.6220 | 0.6273 | 0.0949 | 0.0931 | 0.0970 | 0.0460 | 0.0203 | -1.3998 | 0.5868 | 0.3190 | 0.2355 |
| XGBoost (feature-matched) | 24h x 39 feature-matched | 0.6323 | 0.6297 | 0.6350 | 0.0999 | 0.0979 | 0.1019 | 0.0455 | 0.0180 | -1.1829 | 0.6653 | 0.3147 | 0.2222 |
| GRU (feature-matched) | 24h x 39 feature-matched | 0.6036 | 0.6008 | 0.6065 | 0.0721 | 0.0707 | 0.0735 | 0.0470 | 0.0309 | -1.6334 | 0.5363 | 0.3943 | 0.2854 |
| EBM (current state) | current-state 13 features | 0.5869 | 0.5842 | 0.5895 | 0.0695 | 0.0683 | 0.0706 | 0.0452 | 0.0120 | -0.9282 | 0.7449 | 0.2168 | 0.1408 |

## Paired Equal-Sample Comparisons

Differences are candidate minus equal-sample KG-TFNN on identical eICU windows.

| comparison | candidate | reference | metric | difference | ci_low | ci_high | paired_bootstrap_p | bootstrap_unit | replicates | direction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm_matched minus kg_tfnn_equal_sample | lightgbm_matched | kg_tfnn_equal_sample | auroc | 0.0146 | 0.0129 | 0.0169 | 0.0100 | subject_id | 200.0000 | higher is better |
| lightgbm_matched minus kg_tfnn_equal_sample | lightgbm_matched | kg_tfnn_equal_sample | auprc | 0.0087 | 0.0073 | 0.0100 | 0.0100 | subject_id | 200.0000 | higher is better |
| lightgbm_matched minus kg_tfnn_equal_sample | lightgbm_matched | kg_tfnn_equal_sample | brier | 0.0003 | 0.0003 | 0.0004 | 0.0100 | subject_id | 200.0000 | lower is better |
| xgboost_matched minus kg_tfnn_equal_sample | xgboost_matched | kg_tfnn_equal_sample | auroc | 0.0223 | 0.0205 | 0.0244 | 0.0100 | subject_id | 200.0000 | higher is better |
| xgboost_matched minus kg_tfnn_equal_sample | xgboost_matched | kg_tfnn_equal_sample | auprc | 0.0137 | 0.0124 | 0.0149 | 0.0100 | subject_id | 200.0000 | higher is better |
| xgboost_matched minus kg_tfnn_equal_sample | xgboost_matched | kg_tfnn_equal_sample | brier | -0.0002 | -0.0002 | -0.0001 | 0.0100 | subject_id | 200.0000 | lower is better |
| gru_matched minus kg_tfnn_equal_sample | gru_matched | kg_tfnn_equal_sample | auroc | -0.0064 | -0.0085 | -0.0040 | 0.0100 | subject_id | 200.0000 | higher is better |
| gru_matched minus kg_tfnn_equal_sample | gru_matched | kg_tfnn_equal_sample | auprc | -0.0141 | -0.0156 | -0.0128 | 0.0100 | subject_id | 200.0000 | higher is better |
| gru_matched minus kg_tfnn_equal_sample | gru_matched | kg_tfnn_equal_sample | brier | 0.0014 | 0.0013 | 0.0015 | 0.0100 | subject_id | 200.0000 | lower is better |
| ebm_current_state minus kg_tfnn_equal_sample | ebm_current_state | kg_tfnn_equal_sample | auroc | -0.0232 | -0.0255 | -0.0211 | 0.0100 | subject_id | 200.0000 | higher is better |
| ebm_current_state minus kg_tfnn_equal_sample | ebm_current_state | kg_tfnn_equal_sample | auprc | -0.0167 | -0.0180 | -0.0154 | 0.0100 | subject_id | 200.0000 | higher is better |
| ebm_current_state minus kg_tfnn_equal_sample | ebm_current_state | kg_tfnn_equal_sample | brier | -0.0004 | -0.0005 | -0.0003 | 0.0100 | subject_id | 200.0000 | lower is better |

## Formal Full-Cohort KG-TFNN Context

The prespecified formal KG-TFNN used a different, full-cohort training protocol and is therefore shown as context rather than included in the equal-sample architecture comparison.

- External AUROC: 0.6221
- External AUPRC: 0.0922
- External Brier score: 0.0459
- External ECE: 0.0267

## Interpretation Boundary

This analysis evaluates frozen transportability. It does not establish architecture superiority because the formal full-cohort KG-TFNN and the equal-sample comparison answer different questions.
