# 實驗完成度與下一步

更新日期：2026-07-10

## Outcome Priority

- Primary：未來 6 小時 SOFA increase >= 2。
- Secondary：未來 12/24 小時 SOFA increase >= 2。
- 12/24 h 不阻擋 primary manuscript；詳見 [analysis_plan.md](analysis_plan.md)。

## 已完成

| 實驗 | 正式輸出 |
|---|---|
| Leakage-free 6/12/24 h labels；6 h 為 primary | `sofa_scores_hourly.csv` |
| MIMIC v3 hourly、missingness、time-since features | `model_hourly_features_v3.csv` |
| Patient-level split 與 equal-sample protocol | `patient_split.csv`, `comparison_protocol.json` |
| 6 h baseline benchmark 與共同 predictions | `outputs/fair_comparison_6h_equal_sample/` |
| 6 h independent test、subject bootstrap、paired tests、DCA、lead time、calibration | `outputs/advanced_evaluation_6h_equal_sample/` |
| 4/6/12/24 h observation-window sensitivity，outcome 固定 6 h | `outputs/explicit_temporal_observation_sensitivity_6h/` |
| Explicit-temporal Optuna，30 trials、8 epochs | `outputs/explicit_temporal_fnn_tuning_6h/` |
| Explicit-temporal full-cohort 6 h training，best epoch 15 | `outputs/explicit_temporal_fnn_formal_6h/seed_42/` |
| Frozen final-model test evaluation、validation-only calibration、1,000 次 subject bootstrap | `outputs/final_test_evaluation_6h/` |
| 正式 FNN 消融；4 variants、3 seeds、完整 test | `outputs/fnn_ablation_6h_equal_sample/` |
| Frozen-model temporal fuzzy rule extraction；24 條 supported rules | `outputs/temporal_rule_extraction_6h/` |
| Rule Evaluation Framework；5-seed stability、drift、activated rules、TP/FP/FN timelines | `outputs/rule_evaluation_6h/` |
| eICU final frozen-checkpoint external validation；500 次 subject bootstrap | `outputs/eicu_external_validation/final_frozen_model_evaluation/` |
| Adult `age >= 18` eligibility audit；MIMIC split byte-identical | `outputs/manuscript_tables_figures_6h/adult_eligibility_audit.json` |
| Figure 1–5、Table 1–5、MIMIC/eICU missingness | `outputs/manuscript_tables_figures_6h/`, `docs/adult_cohort_manuscript_artifacts.md` |

## Primary 6 h 尚缺

1. **新版 paired comparison。** Equal-sample explicit FNN checkpoints 與 predictions 已由正式消融產生；需整合 LR、EBM、XGBoost、GRU predictions，重跑至少 500 次 subject-clustered bootstrap、fixed specificity、DCA、lead time、risk strata 與 calibration。
2. **Scale sensitivity。** 評估 explicit scale 1.0、1.5、2.0、2.5，並檢查 rule scale/double counting。
3. **External supplementary analyses。** Final frozen transport 已完成；尚可補 comparator transport、hospital-level heterogeneity 與 domain-shift analyses。

## Secondary Analysis

- 12 h outcome experiment。
- 24 h outcome experiment。
- 兩者可在 primary 6 h manuscript 完成後補入 supplement，不阻擋目前工作。

## 論文仍需補

- Age、sex、ethnicity、ICU type subgroup performance。
- Hospital-clustered external sensitivity analysis。
- TRIPOD+AI 與 PROBAST+AI checklist。
- 多重比較與模型選擇規則。

## 重要原則

- Tuning AUROC 0.6515 是 validation performance；full-cohort test AUROC 為 0.6559。
- Final test 已鎖定 checkpoint SHA-256 `158427a5...9125688`；正式 calibrated Brier 為 0.0521、ECE 為 0.0012。
- 消融顯示 temporal design 是主要效能來源（paired AUROC +0.0510）；consistency loss 未改善 AUROC，僅 Rule Stability 呈現方向性提升。
- Rule Evaluation：Top-10 complexity 1.44 antecedents、5-seed Jaccard 0.720；正負 windows 平均 activated rules 為 1.833 與 1.819。
- Final eICU：AUROC 0.6221、AUPRC 0.0922；MIMIC thresholds 未維持目標 specificity，應列為 transportability limitation。
- Full-cohort FNN 與 equal-sample baselines 不可作 primary 公平效能宣稱。
- Hourly windows 彼此相關；CI 與模型差異必須以 `subject_id` 為 bootstrap cluster。
- eICU 不可用於 hyperparameter selection 或 primary recalibration。
- 成人條件沒有排除任何 MIMIC 病人，重建 split 的 SHA-256 與原檔完全相同；因此既有 fitted-model artifacts 已是相同成人 cohort，不因這項 eligibility clarification 重訓或解鎖 final test。
