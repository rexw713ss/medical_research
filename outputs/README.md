# 實驗輸出索引

更新日期：2026-07-14

本索引是 `outputs/` 的唯一狀態登錄。資料夾名稱維持不動，避免破壞 scripts、checkpoint configs、README 與論文引用的相對路徑。正式、探索性與 superseded 結果不得混用。

## A. Primary Formal Results

| 資料夾 | 內容 | 使用規則 |
|---|---|---|
| `explicit_temporal_fnn_tuning_6h/` | 30-trial validation-only Optuna tuning | 只報 validation；test 未參與 tuning |
| `explicit_temporal_fnn_formal_6h/` | Explicit-temporal FNN full-cohort training | Final checkpoint 的 training provenance |
| `final_test_evaluation_6h/` | Frozen checkpoint、validation-only calibration、1,000 次 patient bootstrap | Canonical one-time MIMIC internal test |
| `explicit_kg_tfnn_paired_comparison_6h/` | KG-TFNN 與 matched comparators 的相同 test-window paired comparison | Canonical equal-sample model comparison |
| `feature_matched_baselines_6h_equal_sample/` | 24 x 39 GRU 與同源 temporal-summary XGBoost/LightGBM | Canonical architecture comparator evidence |
| `fnn_ablation_6h_equal_sample/` | 4 FNN variants x 3 seeds | Canonical component ablation |
| `missingness_ablation_6h_equal_sample/` | Full、no-missingness、missingness-only x 3 seeds | Canonical missingness evidence；test 為全部 830,839 windows |
| `temporal_rule_extraction_6h/` | Frozen-model supported temporal fuzzy rules | Canonical rule examples |
| `rule_evaluation_6h/` | Complexity、5-seed stability、alignment、drift、activated rules、case timelines | Canonical Rule Evaluation Framework |
| `rule_evaluation_full_fnn_extra_seeds/` | Seeds 72/82 checkpoints 與 rule inventories | Rule-stability supporting artifacts |
| `eicu_external_validation/final_frozen_model_evaluation/` | Frozen MIMIC model transported to eICU without fitting/recalibration | Canonical external validation |

## B. Formal Sensitivity And Reporting

| 資料夾 | 內容 | 使用規則 |
|---|---|---|
| `explicit_temporal_observation_sensitivity_6h/` | 4/6/12/24-hour observation windows x 3 seeds | Observation-window sensitivity |
| `clinical_sensitivity_analyses_6h/` | SOFA definitions、alarm burden、lead time、MIMIC subgroups | Frozen full-test-cohort sensitivity |
| `sofa_documentation_bias_6h/` | Common-component、same/stable-mask labels、organ contributions | Formal 830,839-window documentation sensitivity |
| `eicu_hospital_sensitivity_6h/` | 205-hospital performance 與 hospital-cluster bootstrap | External site heterogeneity |
| `raw_rule_firing_6h/` | Product-t-norm firing 與 activation-threshold sensitivity | Rule activation sensitivity |
| `expanded_experiment_reporting_6h/` | Cohort exclusions、SOFA harmonization、raw/calibrated results、alarm/site definitions | Methods/reporting audit |
| `posthoc_explainability_comparison_6h/` | Full-data TreeSHAP/current-state EBM/KG-TFNN structural benchmark | 830,839 MIMIC + 6,215,890 eICU windows；不是 clinician validation |
| `clinical_consistency_regularization_6h/` | Full-test violation/reversal stress test | 3 seeds x 2 variants；每模型 830,839 windows |
| `formal_data_scope_audit_6h/` | Canonical experiment counts 與 runtime-limit audit | 94 checks passed、0 failed |
| `manuscript_tables_figures_6h/` | Cohort flow、Table 1--5 與 manuscript figures | Canonical publication artifacts |
| `supplementary_material/` | Supplementary Tables S1--S13 與 Figures S1--S7 | Canonical supplement source tables/figures |
| `reproducibility_6h/` | Package versions、hashes、test-lock policy | Canonical reproducibility manifest |
| `eicu_external_validation/eicu_hourly_features.pkl` | Harmonized eICU hourly model table | Canonical external input；約 3.51 GiB |

## C. Exploratory Results

目前沒有 sampled/smoke analysis 登錄為 canonical evidence。`equal_sample` 資料夾不是 smoke test：它們使用預先鎖定的 200,000 train / 50,000 validation windows，並在完整 830,839 test windows 評估；因此必須標示為 equal-sample sensitivity，而非 full-cohort training。

完整完成度與可宣稱範圍見 `../docs/manuscript_experiment_audit.md`。

## D. Superseded But Retained

| 資料夾 | 狀態 | 保留理由 |
|---|---|---|
| `fair_comparison_6h_equal_sample/` | 早期共同 protocol pipeline | 保留歷史重現；不得取代 feature-matched primary comparison |
| `advanced_evaluation_6h_equal_sample/` | 早期 non-feature-matched evaluation | 保留舊報告與通用評估產物；architecture claim 以 A 區結果為準 |

## Locked Results

- Full-cohort MIMIC-IV：AUROC 0.6559，AUPRC 0.1309。
- Equal-sample KG-TFNN：AUROC 0.6448，AUPRC 0.1236。
- Feature-matched LightGBM：AUROC 0.6904，AUPRC 0.1710。
- Feature-matched XGBoost：AUROC 0.6870，AUPRC 0.1665。
- Frozen eICU：AUROC 0.6221，AUPRC 0.0922。
- eICU cohort：80,239 patients、99,262 stays、6,215,890 windows。
- Full-data XAI：KG-TFNN stability 1.000、within-stay continuity 0.998；LightGBM + TreeSHAP 為 0.965、0.914。
- Full-data consistency audit：rule stability 0.587 至 0.674，但 violation/reversal/drift/correlation 未一致改善。

## Retention Rules

- CSV/JSON 是數值 source of truth；圖片不可作為抄錄來源。
- `best_model.pt`、split/protocol、calibration/threshold provenance 與 prediction keys 不可刪除。
- `smoke test`、臨時抽樣與舊 preprocessing 結果不得進入正式表圖。
- Root-level transient logs、`__pycache__/` 與 LaTeX auxiliary files不是研究產物，可安全清除。
