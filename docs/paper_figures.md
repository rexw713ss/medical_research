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
| eICU external ROC/PR、calibration、risk strata | `outputs/eicu_external_validation/evaluation/figures/` |

## Primary Evaluation 尚待產生

- Fixed 90%/95% specificity operating-point figure；數值與 CI 已完成於 `fixed_specificity_metrics.csv` 與 final report。
- Full-cohort versus equal-sample sample-size sensitivity figure。

## 消融與規則圖

- 四組消融 AUROC/AUPRC 與 rule-quality comparison已產生。
- Membership functions before/after training。
- Temporal coefficient heatmap。
- Top-K rule stability across seeds。
- True-positive、false-positive、false-negative patient timelines。

## Cohort 與 External Validation

- MIMIC/eICU cohort flow diagram。
- MIMIC versus eICU missingness/domain-shift heatmap。
- Age、sex、ethnicity、ICU type subgroup forest plot。
- Hospital-level external performance distribution。
- Decision curve 與 lead-time distribution。

圖表數值必須來自固定 predictions 與正式 evaluation CSV，不可從圖片手動抄錄。
