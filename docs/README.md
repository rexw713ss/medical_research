# 文件索引

## 專案狀態

- [專案入口](../README.md)
- [實驗完成度與下一步](experiment_status.md)
- [Primary/secondary analysis plan](analysis_plan.md)
- [Full-cohort 6 h result](full_cohort_training_6h.md)
- [Frozen final-model 6 h test report](../outputs/final_test_evaluation_6h/final_test_report.md)
- [正式 6 h FNN 消融實驗](formal_ablation_study_6h.md)
- [實際模型萃取的 temporal fuzzy rules](extracted_temporal_rules_6h.md)
- [Rule Evaluation Framework 實驗結果](rule_evaluation_framework_6h.md)
- [eICU final external validation](eicu_final_external_validation_6h.md)
- [成人 cohort、Table 1–5 與 Figure 1–5](adult_cohort_manuscript_artifacts.md)
- [投稿底稿實驗與數學審查](manuscript_experiment_audit.md)
- [TRIPOD+AI reporting checklist](TRIPOD_AI_checklist.md)
- [PROBAST+AI risk-of-bias self-assessment](PROBAST_AI_checklist.md)
- [正式輸出索引](../outputs/README.md)

## 方法

- [Preprocessing 與 leakage-free labels](preprocessing_v2_method.md)
- [Patient split 與公平比較 protocol](fair_comparison_protocol.md)
- [Explicit-temporal FNN tuning](explicit_temporal_fnn_tuning_6h.md)
- [Explicit temporal features 與 observation windows](explicit_temporal_observation_sensitivity.md)

## 結果

- [MIMIC-IV 6 h baseline benchmark](mimic_iv_6h_evaluation_report.md)
- [Feature-matched GRU/tree baseline report](../outputs/feature_matched_baselines_6h_equal_sample/feature_matched_baseline_report.md)
- [Clinical sensitivity analyses](../outputs/clinical_sensitivity_analyses_6h/clinical_sensitivity_report.md)
- [eICU hospital-clustered sensitivity](../outputs/eicu_hospital_sensitivity_6h/eicu_hospital_sensitivity_report.md)
- [Expanded cohort/SOFA/calibration/alarm/site reporting](../outputs/expanded_experiment_reporting_6h/expanded_experiment_reporting.md)
- [Raw rule firing sensitivity](../outputs/raw_rule_firing_6h/raw_rule_firing_report.md)
- [論文圖表清單](paper_figures.md)
- [eICU external validation report](../outputs/eicu_external_validation/final_frozen_model_evaluation/eicu_external_validation_report.md)

## 研究資料

- `研究計畫二版.pdf`
- `literature review/`
- `citi/`

文件中的指標必須標明 split。Tuning 結果一律寫成 validation performance；模型定案後的結果才可寫成 test performance。
