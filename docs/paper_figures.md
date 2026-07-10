# 論文圖表清單

所有正式圖表同時保留 300 dpi PNG 與向量 PDF。

主文圖表以 6 h primary outcome 為準；12/24 h outcome figures 僅放 secondary 或 supplementary material。

## 已產生

| 圖表 | 位置 |
|---|---|
| Optuna optimization history、parameter association | `outputs/explicit_temporal_fnn_tuning_6h/figures/` |
| Full-cohort training history | `outputs/explicit_temporal_fnn_formal_6h/seed_42/figures/` |
| Frozen final-model ROC、PR、calibration、DCA、risk strata | `outputs/final_test_evaluation_6h/advanced/figures/` |
| 正式消融 predictive performance、calibration、component effects、rule quality | `outputs/fnn_ablation_6h_equal_sample/figures/` |
| Observation-window sensitivity、explicit temporal coefficients | `outputs/explicit_temporal_observation_sensitivity_6h/figures/` |
| Baseline ROC、PR、calibration、DCA、risk strata | `outputs/advanced_evaluation_6h_equal_sample/figures/` |
| Rule membership before/after、TP/FP/FN timelines | `outputs/rule_evaluation_6h/figures/` |
| Final eICU external ROC/PR、calibration、risk strata | `outputs/eicu_external_validation/final_frozen_model_evaluation/figures/` |
| Figure 1 cohort flow、Figure 2 architecture、Figure 3 calibration、Figure 4 DCA、Figure 5 timelines | `outputs/manuscript_tables_figures_6h/figures/` |
| Table 1–5 publication CSV 與 Markdown | `outputs/manuscript_tables_figures_6h/`, `adult_cohort_manuscript_artifacts.md` |

## Primary Evaluation 尚待產生

- Fixed 90%/95% specificity operating-point figure；數值與 CI 已完成於 `fixed_specificity_metrics.csv` 與 final report。
- Full-cohort versus equal-sample sample-size sensitivity figure。

## 消融與規則圖

- 四組消融 AUROC/AUPRC 與 rule-quality comparison已產生。
- Membership functions before/after training已產生。
- Temporal coefficient heatmap。
- Top-K rule stability across seeds。
- True-positive、false-positive、false-negative patient timelines已產生。

## Cohort 與 External Validation

- MIMIC/eICU cohort flow diagram已產生。
- MIMIC versus eICU missingness/domain-shift heatmap。
- Age、sex、ethnicity、ICU type subgroup forest plot。
- Hospital-level external performance distribution。
- Internal/external decision curve已產生；lead-time distribution 尚待整理。

圖表數值必須來自固定 predictions 與正式 evaluation CSV，不可從圖片手動抄錄。
