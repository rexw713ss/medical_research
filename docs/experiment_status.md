# 實驗完成度

更新日期：2026-07-04

## 已完成

| 實驗 | 正式輸出 |
|---|---|
| Leakage-free MIMIC hourly SOFA 與 6/12/24 h labels | `sofa_scores_hourly.csv` |
| MIMIC v3 hourly、missingness、time-since features | `model_hourly_features_v3.csv` |
| Patient-level split 與 equal-sample protocol | `patient_split.csv`, `comparison_protocol.json` |
| 6 h baseline benchmark 與共同 predictions | `outputs/fair_comparison_6h_equal_sample/` |
| Independent test、500 次 subject bootstrap、paired tests、DCA、lead time、calibration | `outputs/advanced_evaluation_6h_equal_sample/` |
| Explicit temporal features 與 4/6/12/24 h observation sensitivity | `outputs/explicit_temporal_observation_sensitivity_6h/` |
| Explicit-temporal FNN 專屬 Optuna，30 trials、8 epochs | `outputs/explicit_temporal_fnn_tuning_6h/` |
| eICU harmonization、future SOFA labels 與 frozen-checkpoint external validation | `outputs/eicu_external_validation/` |

## 下一步

1. **Full-cohort 6 h training。** 執行 `outputs/explicit_temporal_fnn_tuning_6h/train_with_best_params.ps1`，使用 early stopping 選定新版 checkpoint，之後才評估一次 test。
2. **新版統一比較。** 匯出 explicit FNN validation/test predictions，加入現有 LR、EBM、GRU 等模型，重跑 patient-clustered paired comparison。
3. **新版正式消融。** 比較 random initialization、guideline static、explicit temporal without consistency、full explicit temporal，完整 cohort、3–5 seeds。
4. **Prediction horizons。** 將新版模型延伸至未來 12/24 h outcome；目前新版 sensitivity 只預測未來 6 h。
5. **Rule quality。** 補新版 IF-THEN table、complexity、stability、concordance、drift、activated rules、membership plots 與 TP/FP/FN timelines。
6. **最終 external validation。** 使用定案後的 full-cohort checkpoint 原封不動重跑 eICU，並補 comparator transport、site heterogeneity、domain shift 與 SOFA mapping sensitivity。

## 論文報告待補

- MIMIC/eICU cohort flow diagram。
- Development、internal test、external test Table 1 與 missingness table。
- TRIPOD+AI 與 PROBAST+AI checklist。
- 主要/次要 outcome 與多重比較策略。

## 重要限制

- Tuning AUROC 0.6515 是 validation 結果，不是 test performance。
- 最佳 `explicit_temporal_scale=1.9896` 接近搜尋上界 2.0；完整訓練後需檢查 calibration、rule drift 與 double counting，並補局部 sensitivity analysis。
- 目前 eICU AUROC 0.6034 使用較早但完全 frozen 的 explicit checkpoint；它是有效 external validation，但不是新 full-cohort 模型的最終 external result。
- Hourly windows 並不獨立；信賴區間與模型差異必須以 `subject_id` 為 cluster，site sensitivity 可再以 hospital 為 cluster。
