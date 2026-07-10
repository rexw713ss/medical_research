# 正式輸出索引

本資料夾只保留目前最新版正式輸出。

| 資料夾 | 內容 | 論文角色 |
|---|---|---|
| `explicit_temporal_fnn_formal_6h/` | 新版 explicit-temporal FNN full-cohort 6 h training | 完成；test AUROC 0.6559、AUPRC 0.1309 |
| `final_test_evaluation_6h/` | Frozen checkpoint、validation-only calibration、1,000 次 patient bootstrap | 最終 internal test；完整且已鎖定 |
| `fnn_ablation_6h_equal_sample/` | 4 種 FNN variants、3 seeds、calibrated metrics 與 rule quality | 正式 6 h 消融 |
| `temporal_rule_extraction_6h/` | Frozen full-cohort model 的 temporal fuzzy rules、support 與 event rate | 主文規則範例 |
| `rule_evaluation_6h/` | 5-seed stability、complexity、concordance、drift、activated rules、case timelines | Rule Evaluation Framework |
| `rule_evaluation_full_fnn_extra_seeds/` | Full FNN seeds 72/82 checkpoints 與 rule inventories | 5-seed stability 支援資料 |
| `explicit_temporal_fnn_tuning_6h/` | Explicit-temporal FNN，30-trial validation-only Optuna | 新版模型 tuning |
| `explicit_temporal_observation_sensitivity_6h/` | 4/6/12/24 h observation windows、3 seeds | Observation-window sensitivity |
| `fair_comparison_6h_equal_sample/` | 共同 predictors、split、test windows 的模型與 predictions | 6 h benchmark 支援資料 |
| `advanced_evaluation_6h_equal_sample/` | 500 次 patient bootstrap、paired comparison、DCA、lead time、calibration | 6 h benchmark 結果 |
| `eicu_readiness/` | eICU 原始資料與 cohort readiness audit | 外部資料稽核 |
| `eicu_external_validation/final_frozen_model_evaluation/` | Final checkpoint、完整 MIMIC validation transfer、500 次 clustered bootstrap | 最終外部驗證 |
| `manuscript_tables_figures_6h/` | 成人 eligibility audit、Table 1–5、Figure 1–5 | 主文 cohort 與整合圖表 |
| `model_evaluation/` | 2026-07-05 產生的 clinical-score 6/12/24 h diagnostic report | 6 h 可供 primary 參考；12/24 h 僅屬 secondary，且尚未包含新版 FNN |

## Tuning 結果

- 最佳 validation AUROC：0.6515。
- 最佳 validation AUPRC：0.1232。
- 最佳 trial：24；21 complete、9 pruned。
- Train/validation：200,000/50,000 固定 windows。
- Test set 未參與 tuning。
- 重現 full-cohort training 的入口：`explicit_temporal_fnn_tuning_6h/train_with_best_params.ps1`。

## Outcome Hierarchy

- Primary：未來 6 小時 SOFA increase >= 2。
- Secondary：未來 12/24 小時 SOFA increase >= 2。
- 未指定 `--horizons` 時，comparison 與 evaluation 只執行 primary 6 h。

## Final External Validation

- eICU patients：80,239。
- ICU stays：99,262。
- Windows：6,215,890。
- AUROC：0.6221（95% CI 0.6192–0.6249）。
- AUPRC：0.0922（95% CI 0.0902–0.0942）。
- Primary analysis 沒有使用 eICU outcome fitting 或 recalibration。

資料來源逐表數量與用途另存於 `manuscript_tables_figures_6h/data_source_inventory.csv`；MIMIC-IV 共使用 7 張表、608,690,476 raw rows，eICU 共使用 8 張表、399,686,522 raw rows。

使用 final full-cohort checkpoint，沒有 eICU fitting 或 recalibration；MIMIC 90%/95% specificity thresholds 在 eICU 的 observed specificity 為 79.0%/88.2%。
