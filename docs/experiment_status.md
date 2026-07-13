# 實驗狀態

更新日期：2026-07-13

## 研究主軸

- Primary outcome：未來 6 小時 SOFA increase >= 2。
- Primary observation history：24 小時。
- Secondary analyses：12/24 小時 outcome，以及 4/6/12/24 小時 observation-window sensitivity。

## 已完成實驗

| 實驗 | 狀態 | 主要產物 |
|---|---|---|
| Leakage-free SOFA labels、explicit temporal features | 完成 | `sofa_scores_hourly.csv`, `model_hourly_features_v3.csv` |
| Patient-level split、equal-sample protocol、cohort fingerprints | 完成 | `patient_split.csv`, `comparison_protocol.json` |
| Explicit KG-TFNN Optuna tuning 與 full-cohort final model | 完成 | `outputs/explicit_temporal_fnn_tuning_6h/`, `outputs/explicit_temporal_fnn_formal_6h/` |
| Frozen one-time internal test evaluation | 完成 | `outputs/final_test_evaluation_6h/` |
| Equal-sample paired pipeline comparison | 完成；1,000 次 patient-cluster bootstrap | `outputs/explicit_kg_tfnn_paired_comparison_6h/` |
| Four-component FNN ablation | 完成；4 variants x 3 seeds | `outputs/fnn_ablation_6h_equal_sample/` |
| Missingness-only / no-missingness ablation | 完成；3-seed ensemble + 1,000 次 patient-cluster bootstrap | `outputs/missingness_ablation_6h_equal_sample/evaluation/` |
| SOFA outcome-definition sensitivity | 完成 | `outputs/clinical_sensitivity_analyses_6h/` |
| Event-level alarm burden、lead time、false-alert burden | 完成 | `outputs/clinical_sensitivity_analyses_6h/` |
| Age、sex、ethnicity、ICU type、current SOFA subgroup | 完成 | `outputs/clinical_sensitivity_analyses_6h/` |
| Rule extraction 與 Rule Evaluation Framework | 完成 | `outputs/temporal_rule_extraction_6h/`, `outputs/rule_evaluation_6h/` |
| Frozen eICU external validation | 完成；無 retraining/recalibration | `outputs/eicu_external_validation/final_frozen_model_evaluation/` |
| eICU hospital-clustered sensitivity | 完成；205 hospitals | `outputs/eicu_hospital_sensitivity_6h/` |
| Feature-matched GRU、XGBoost、LightGBM | 完成；相同資訊、sample 與 test windows | `outputs/feature_matched_baselines_6h_equal_sample/` |
| Raw rule firing / activation threshold sensitivity | 完成 | `outputs/raw_rule_firing_6h/` |
| Cohort、SOFA、calibration、alarm/site reporting audit | 完成 | `outputs/expanded_experiment_reporting_6h/` |
| LightGBM/XGBoost + TreeSHAP、EBM、KG-TFNN explanation comparison | 探索性完成；1,000-case sample，正式 full-test-cohort 待重跑 | `outputs/posthoc_explainability_comparison_6h/` |
| Clinical-consistency regularization behavior audit | 探索性完成；3 seeds、1,000-case sample，正式 full-test-cohort 待重跑 | `outputs/clinical_consistency_regularization_6h/` |
| SOFA documentation-availability sensitivity 與 organ contribution | 完成；500 次 patient-cluster bootstrap | `outputs/sofa_documentation_bias_6h/` |
| Main Figures 1--7 / Supplementary Tables S1--S13 / Figures S1--S7 | 完成；membership 與 SOFA documentation 圖已移入主文 | `paper/TSP_template.pdf`, `paper/Supplementary_Material.pdf` |
| 投稿底稿整合與數學審查 | 完成第一輪；7 tables、5 figures、pseudocode | `paper/TSP_template.tex`, `paper/TSP_template_review.pdf` |
| TRIPOD+AI / PROBAST+AI 自評 | 完成初稿 | `docs/TRIPOD_AI_checklist.md`, `docs/PROBAST_AI_checklist.md` |

## 已鎖定結果

- Equal-sample explicit KG-TFNN：AUROC 0.6448（95% CI 0.6379-0.6515），AUPRC 0.1236（0.1177-0.1297）。
- Feature-matched GRU：AUROC 0.6587，AUPRC 0.1272；KG-TFNN paired AUROC 差分 -0.0139（-0.0192 至 -0.0088），AUPRC 差分 -0.0036（-0.0085 至 0.0008）。
- Feature-matched LightGBM：AUROC 0.6904，AUPRC 0.1710；XGBoost：AUROC 0.6870，AUPRC 0.1665。結果不支持 architecture superiority。
- Full-cohort MIMIC-IV：AUROC 0.6559，AUPRC 0.1309。
- Frozen eICU：AUROC 0.6221，AUPRC 0.0922。
- SOFA complete-case sensitivity：176,130 windows，AUROC 0.6237，AUPRC 0.0923。
- Pairwise common-component SOFA sensitivity：830,609 windows，AUROC 0.6097，AUPRC 0.0662；保留 67.1% primary positives。
- KG-TFNN explanation stability / nearest-neighbor consistency：1.000 / 0.979；LightGBM + TreeSHAP：0.965 / 0.408。這是 structural benchmark，不是 clinician validation。
- Consistency loss 的 three-seed rule stability 由 0.587 增至 0.674；violation、risk reversal、drift 與 guideline-risk correlation 未呈現一致改善。
- Event-level 90% specificity：sensitivity 0.3434，48.23 false alerts/100 patient-days，median lead time 3 hours。
- Event-level 95% specificity：sensitivity 0.2380，24.30 false alerts/100 patient-days，median lead time 2 hours。

## 投稿前仍需處理

1. 將 explanation-quality comparison 與 consistency behavioral audit 改為分批 full-test-cohort 執行；目前 1,000-case 結果只能標示 exploratory。
2. 在最終排版後補 TRIPOD+AI 頁碼，並請臨床與方法學專家獨立審查 PROBAST+AI、rules 與 case timelines。
3. 公開版本需封存 package versions、configs、split/protocol hashes 與 checkpoint hash。
4. 完成期刊格式、作者貢獻、補充資料與英文語言校閱。
5. 若資源允許，進行 prospective workflow/alarm usability study；這不是目前 retrospective manuscript 的完成條件。

## 報告規則

- Full-cohort 與 equal-sample estimates 必須分開標示。
- Test outcomes 不可用於 tuning、calibration、threshold selection 或 checkpoint selection。
- Hourly windows 彼此相關；CI 與 paired tests 以 `subject_id` 為 cluster。
- eICU 為 frozen external validation，不可用於主模型 fitting 或 recalibration。
- Guideline-direction alignment 只代表 prespecified NEWS2/SOFA direction alignment，不是獨立臨床驗證或 clinician-validated interpretability。
- Smoke-test 或臨時 sample 結果不得進入正式表圖；正式 test 結果必須使用完整 eligible test windows。
