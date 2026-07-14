# 實驗狀態

更新日期：2026-07-14

## 研究主軸

- Primary outcome：未來 6 小時 SOFA increase >= 2。
- Primary observation history：24 小時。
- Secondary analyses：4/6/12/24 小時 observation-window sensitivity、frozen eICU comparator transport 與 explanation complexity。
- Scope amendment（2026-07-14）：12/24 小時 outcome labels 保留，但本研究不再執行或報告 12/24 小時 prediction-horizon experiments。

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
| LightGBM/XGBoost + TreeSHAP、EBM、KG-TFNN explanation comparison | 正式全量完成；830,839 MIMIC + 6,215,890 eICU windows | `outputs/posthoc_explainability_comparison_6h/` |
| 跨模型統一 explanation complexity | 正式全量完成；共同單位為 13 個 clinical variables、80% attribution mass | `outputs/posthoc_explainability_comparison_6h/unified_explanation_complexity_report.md` |
| Frozen eICU equal-sample comparator transport | 正式全量完成；5 models、6,215,890 windows、200 次 patient bootstrap、無 eICU fitting | `outputs/eicu_frozen_baseline_validation_6h/` |
| Clinical-consistency regularization behavior audit | 正式全量完成；3 seeds x 2 variants x 830,839 windows | `outputs/clinical_consistency_regularization_6h/` |
| Formal data-scope audit | 完成；116 checks passed、0 failed | `outputs/formal_data_scope_audit_6h/` |
| SOFA documentation-availability sensitivity 與 organ contribution | 完成；500 次 patient-cluster bootstrap | `outputs/sofa_documentation_bias_6h/` |
| Main Figures 1--7 / Supplementary Tables S1--S13 / Figures S1--S7 | 完成；主文已納入 frozen eICU comparator transport 與 unified complexity | `paper/TSP_template.pdf`, `paper/Supplementary_Material.pdf` |
| 投稿底稿整合、數學與實驗一致性審查 | 完成；主文已依 prediction--interpretability trade-off 收斂，修改前原稿已備份 | `paper/TSP_template.tex`, `paper/backups/20260714_210917_before_tradeoff_reframing/` |
| TRIPOD+AI / PROBAST+AI 自評 | 完成初稿 | `docs/TRIPOD_AI_checklist.md`, `docs/PROBAST_AI_checklist.md` |

## 已鎖定結果

- Equal-sample explicit KG-TFNN：AUROC 0.6448（95% CI 0.6379-0.6515），AUPRC 0.1236（0.1177-0.1297）。
- Feature-matched GRU：AUROC 0.6587，AUPRC 0.1272；KG-TFNN paired AUROC 差分 -0.0139（-0.0192 至 -0.0088），AUPRC 差分 -0.0036（-0.0085 至 0.0008）。
- Feature-matched LightGBM：AUROC 0.6904，AUPRC 0.1710；XGBoost：AUROC 0.6870，AUPRC 0.1665。結果不支持 architecture superiority。
- Full-cohort MIMIC-IV：AUROC 0.6559，AUPRC 0.1309。
- Frozen eICU：AUROC 0.6221，AUPRC 0.0922。
- Equal-sample frozen eICU：KG-TFNN AUROC/AUPRC 0.6100/0.0862、LightGBM 0.6247/0.0949、XGBoost 0.6323/0.0999、GRU 0.6036/0.0721、current-state EBM 0.5869/0.0695。所有模型使用相同 6,215,890 windows；結果不支持 external architecture superiority。
- SOFA complete-case sensitivity：176,130 windows，AUROC 0.6237，AUPRC 0.0923。
- Pairwise common-component SOFA sensitivity：830,609 windows，AUROC 0.6097，AUPRC 0.0662；保留 67.1% primary positives。
- 完整 MIMIC test cohort 的 KG-TFNN explanation stability / within-stay trajectory continuity：1.000 / 0.998；LightGBM + TreeSHAP：0.965 / 0.914。這是 structural benchmark，不是 clinician validation。
- 統一 explanation complexity（80% absolute attribution mass 所需 clinical variables）：KG-TFNN 5（IQR 4–5）、LightGBM 6（6–7）、XGBoost 7（6–7）、current-state EBM 6（6–7）。KG-TFNN Top-10 mean antecedents 1.44 為模型專屬補充，不與 SHAP term 數混為同一指標。
- 完整 MIMIC test cohort上，consistency loss 的 three-seed rule stability 由 0.587 增至 0.674；violation 0.3689 至 0.3677，risk reversal 0.0886 至 0.0905，drift 0.2221 至 0.2244，guideline-risk correlation 0.5362 至 0.5281，未呈現一致改善。
- Event-level 90% specificity：sensitivity 0.3434，48.23 false alerts/100 patient-days，median lead time 3 hours。
- Event-level 95% specificity：sensitivity 0.2380，24.30 false alerts/100 patient-days，median lead time 2 hours。

## 投稿前仍需處理

1. 跨模型 complexity 已完成；若要做更強的 architecture-matched comparison，仍可補 24-hour feature-matched EBM。現有 EBM 必須標為 current-state comparator。
2. 在最終排版後補 TRIPOD+AI 頁碼，並請臨床與方法學專家獨立審查 PROBAST+AI、rules 與 case timelines。
3. 完成期刊格式、作者貢獻、資料擷取日期、protocol registration/PPI statement 與英文語言校閱。
4. 若資源允許，進行 prospective workflow/alarm usability study；這不是目前 retrospective manuscript 的完成條件。

## 報告規則

- Full-cohort 與 equal-sample estimates 必須分開標示。
- Test outcomes 不可用於 tuning、calibration、threshold selection 或 checkpoint selection。
- Hourly windows 彼此相關；CI 與 paired tests 以 `subject_id` 為 cluster。
- eICU 為 frozen external validation，不可用於主模型 fitting 或 recalibration。
- Guideline-direction alignment 只代表 prespecified NEWS2/SOFA direction alignment，不是獨立臨床驗證或 clinician-validated interpretability。
- Smoke-test 或臨時 sample 結果不得進入正式表圖；正式 test 結果必須使用完整 eligible test windows。
- `equal_sample` 是預先鎖定的公平比較敏感度分析，不可寫成 full-cohort training；其 test evaluation 仍為完整 830,839 windows。
