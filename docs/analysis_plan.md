# Analysis Plan

鎖定日期：2026-07-05

## Primary Analysis

- Primary outcome：**未來 6 小時內 SOFA score increase >= 2**。
- Outcome column：`label_sofa_increase_ge2_6h`。
- Index windows：至少具有 24 小時 history 的 hourly windows。
- Primary model：Knowledge-Guided Explicit-Temporal FNN。
- Primary comparison：相同 patient split、200,000 train windows、50,000 validation windows 與完整 test windows 的 equal-sample comparison。
- Primary metric：AUROC。
- Key secondary metrics：AUPRC、Brier、ECE、calibration intercept/slope、90%/95% specificity sensitivity、DCA 與 lead time。
- Uncertainty：`subject_id` clustered bootstrap 95% CI 與 paired model differences。

## Secondary Analyses

- 未來 12 小時 SOFA increase >= 2。
- 未來 24 小時 SOFA increase >= 2。
- 4/6/12/24 小時 observation-window sensitivity；這是 lookback length，不是 prediction horizon。
- Full-cohort 6 h training 作為 sample-size/deployment sensitivity。
- eICU frozen-checkpoint external validation。

12/24 h 不阻擋 6 h primary manuscript。除非命令明確傳入 `--horizons 12,24`，正式 comparison 與 evaluation 預設只執行 6 h。

## Model Selection Policy

- Hyperparameters、checkpoint、threshold 與 calibration 只使用 train/validation。
- Test set 只在模型與流程固定後使用。
- eICU outcome 不得用於 primary hyperparameter selection 或 primary recalibration。
- Full-cohort 與 equal-sample 結果分開呈現，不混作主要公平比較。

## Reporting Order

1. Primary equal-sample 6 h internal test comparison。
2. Full-cohort 6 h sample-size sensitivity。
3. eICU external validation。
4. 12/24 h secondary outcome analyses。
