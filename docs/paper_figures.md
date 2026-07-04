# 論文圖表清單

所有正式圖表同時保留 300 dpi PNG 與向量 PDF。

## 已產生

| 圖表 | 位置 |
|---|---|
| Optuna optimization history、parameter association | `outputs/explicit_temporal_fnn_tuning_6h/figures/` |
| Observation-window sensitivity、explicit temporal coefficients | `outputs/explicit_temporal_observation_sensitivity_6h/figures/` |
| Baseline ROC、PR、calibration、DCA、risk strata | `outputs/advanced_evaluation_6h_equal_sample/figures/` |
| eICU external ROC/PR、calibration、risk strata | `outputs/eicu_external_validation/evaluation/figures/` |

## 本次 Training 完成後

- Full-cohort training loss、validation AUROC、AUPRC curves。
- 新版 explicit FNN ROC、PR 與 calibration curve。
- Validation-only calibrated calibration plot。
- Fixed 90%/95% specificity operating-point figure。
- Full-cohort versus equal-sample sample-size sensitivity figure。

## 消融與規則圖

- 四組消融 AUROC/AUPRC 與 rule-quality comparison。
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
