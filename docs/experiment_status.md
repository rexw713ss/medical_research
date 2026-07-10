# 實驗完成度與下一步

更新日期：2026-07-10

## Outcome Priority

- Primary：未來 6 小時 SOFA increase >=2。
- Secondary：未來 12/24 小時 SOFA increase >=2。
- Primary observation history：24 小時；4/6/12/24 小時已作 observation-window sensitivity。

## Primary 6-Hour Experiments

| 實驗 | 狀態 | 正式輸出 |
|---|---|---|
| Leakage-free SOFA labels、explicit temporal features | 完成 | `sofa_scores_hourly.csv`, `model_hourly_features_v3.csv` |
| Patient-level split、equal-sample protocol、cohort fingerprints | 完成 | `patient_split.csv`, `comparison_protocol.json` |
| Explicit KG-TFNN Optuna 與 full-cohort final model | 完成 | `outputs/explicit_temporal_fnn_tuning_6h/`, `outputs/explicit_temporal_fnn_formal_6h/` |
| Frozen one-time internal test evaluation | 完成 | `outputs/final_test_evaluation_6h/` |
| Equal-sample explicit KG-TFNN paired comparison | 完成；1,000 次 subject-clustered bootstrap | `outputs/explicit_kg_tfnn_paired_comparison_6h/` |
| Four-component model ablation | 完成；4 variants x 3 seeds | `outputs/fnn_ablation_6h_equal_sample/` |
| Missingness-only / no-missingness ablation | 執行中；3 seeds | `outputs/missingness_ablation_6h_equal_sample/` |
| SOFA outcome-definition sensitivity | 完成；500 次 subject-clustered bootstrap | `outputs/clinical_sensitivity_analyses_6h/` |
| Event-level alarm burden、lead time、false-alert burden | 完成 | `outputs/clinical_sensitivity_analyses_6h/` |
| Age、sex、ethnicity、ICU type、current SOFA subgroups | 完成 | `outputs/clinical_sensitivity_analyses_6h/` |
| Rule extraction and Rule Evaluation Framework | 完成 | `outputs/temporal_rule_extraction_6h/`, `outputs/rule_evaluation_6h/` |
| Frozen eICU external validation | 完成；無 retraining/recalibration | `outputs/eicu_external_validation/final_frozen_model_evaluation/` |
| eICU hospital-clustered sensitivity and site heterogeneity | 完成；205 hospitals | `outputs/eicu_hospital_sensitivity_6h/` |
| Cohort flow、Table 1-5、Figure 1-5 | 完成 | `outputs/manuscript_tables_figures_6h/` |
| TRIPOD+AI reporting matrix | 完成；頁碼待投稿排版 | `docs/TRIPOD_AI_checklist.md` |
| PROBAST+AI self-assessment | 完成；待獨立 reviewer 確認 | `docs/PROBAST_AI_checklist.md` |

## Newly Locked Results

- Equal-sample explicit KG-TFNN：AUROC 0.6448（95% CI 0.6379-0.6515），AUPRC 0.1236（0.1177-0.1297）。
- Paired versus GRU：AUROC +0.0210（0.0152-0.0267），AUPRC +0.0199（0.0151-0.0248）。
- SOFA complete-case sensitivity：176,130 windows；AUROC 0.6237、AUPRC 0.0923，低於 primary outcome definition。
- Event-level 90% specificity analysis：event sensitivity 0.3434，48.23 false alerts/100 patient-days，median lead time 3 hours。
- Event-level 95% specificity analysis：event sensitivity 0.2380，24.30 false alerts/100 patient-days，median lead time 2 hours。
- Neuro ICU subgroup discrimination最低：AUROC 0.5643；subgroup estimates are exploratory。
- eICU hospital-clustered AUROC 95% CI：0.6127-0.6326；142 hospitals met per-site reporting thresholds。

## Remaining Before Submission

1. Finish and summarize the running 3-seed missingness ablation.
2. Add a labeled Discussion section covering retrospective selection, surrogate outcome, alarm burden, subgroup uncertainty, and external calibration shift.
3. Add final manuscript page numbers to TRIPOD+AI and obtain independent clinical/methodological review of PROBAST+AI and the rule-concordance rubric.
4. Archive package versions, configs, split/protocol hashes, and checkpoint hashes in a public code release.

Scale sensitivity and external comparator transport remain optional supplementary analyses; they are not required to interpret the seven prespecified robustness experiments completed here.

## Secondary Analysis

- Refit and evaluate 12-hour outcome models after the 6-hour manuscript is locked.
- Refit and evaluate 24-hour outcome models after the 6-hour manuscript is locked.
- Do not mix secondary-outcome validation results into the primary 6-hour claims.

## Non-Negotiable Reporting Rules

- Full-cohort and equal-sample estimates must be labeled separately.
- Test outcomes cannot be used for tuning, calibration, threshold selection, or post-hoc model choice.
- Hourly windows are correlated; confidence intervals and paired tests use `subject_id` clusters.
- eICU is frozen external validation and cannot be used for primary model fitting or recalibration.
- Clinical concordance is alignment with a predefined rubric, not independent clinician adjudication.
