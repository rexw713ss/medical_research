# Explicit-Temporal FNN 正式 Tuning

執行日期：2026-07-01

## 設計

- Outcome：未來 6 小時 SOFA 增加至少 2 分。
- Observation history：24 小時。
- 輸入：13 個 raw features、13 個 missingness indicators、13 個 time-since-last-measurement，共 39 維。
- Cohort：固定 equal-sample train 200,000、validation 50,000 windows。
- Patient split：`subject_id` 層級固定切分。
- Objective：最大化 validation AUROC。
- Search：Optuna TPE，30 trials，每個 trial 最多 8 epochs。
- Pruning：MedianPruner，5 個 startup trials、2 個 warmup epochs。
- Test isolation：tuning 程式未建立或評估 test dataset。
- 結果：21 trials complete、9 trials pruned。

Train cohort fingerprint：`322baea356ba6995a4a5146cc8a1916397e441dbdea5e2934e3512f8b99f173a`

Validation cohort fingerprint：`efd4a21c2cd57ac6fc6f620bdc02a3ae1b8193d22ea2a7703986a8f6f62ede0d`

## 最佳結果

最佳 trial 為 24：

| Metric | Validation result |
|---|---:|
| AUROC | 0.6515 |
| AUPRC | 0.1232 |
| Total loss | 1.2759 |

舊 sequence-only 超參數套用至 explicit model 時，validation AUROC 約為 0.6401；新版專屬 tuning 提升約 0.0114。

## 最佳參數

| Parameter | Value |
|---|---:|
| learning rate | 0.00489068 |
| weight decay | 0.00028051 |
| batch size | 256 |
| rule score scale | 0.418917 |
| threshold | 8.64150 |
| attention hidden | 16 |
| lambda consistency | 0.322766 |
| lambda sparsity | 0.00088134 |
| lambda drift | 0.00087918 |
| lambda nonnegative | 0.0782171 |
| gradient clipping | 5.0 |
| explicit temporal scale | 1.98962 |

## 解讀限制

`explicit_temporal_scale=1.98962` 接近本次搜尋上界 2.0；`rule_score_scale=0.4189` 亦偏高。因此目前結果證明較強 temporal contribution 有效，但不能推論 1.98962 是穩定最佳值。Full-cohort 訓練後應檢查 calibration、rule drift、clinical consistency loss 與 double counting，並補 explicit scale 的局部 sensitivity analysis。

本次 tuning 僅用 validation 選模，沒有 test performance；論文不可把 0.6515 寫成 test AUROC。

## Artifacts

- `outputs/explicit_temporal_fnn_tuning_6h/best_params.json`
- `outputs/explicit_temporal_fnn_tuning_6h/optuna_study.db`
- `outputs/explicit_temporal_fnn_tuning_6h/optuna_trials.csv`
- `outputs/explicit_temporal_fnn_tuning_6h/trial_metrics.csv`
- `outputs/explicit_temporal_fnn_tuning_6h/cohort_audit.json`
- `outputs/explicit_temporal_fnn_tuning_6h/figures/`
- `outputs/explicit_temporal_fnn_tuning_6h/train_with_best_params.ps1`

產生的 PowerShell script 已於 2026-07-04 啟動 full-cohort early-stopping training。進度與完成後評估流程見 `docs/full_cohort_training_6h.md`。
