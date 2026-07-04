# Explicit-Temporal FNN Full-Cohort Training

更新日期：2026-07-04

## 狀態

**進行中。** 正式輸出位於 `outputs/explicit_temporal_fnn_formal_6h/seed_42/`。

2026-07-04 檢查時已完成 epoch 12；當時 validation AUROC 為 0.6614、AUPRC 為 0.1312。這些是訓練中的 validation 指標，不是最終 test performance。Early stopping 最多訓練 20 epochs，minimum 10 epochs、patience 5、minimum AUROC delta 0.0001。

## Cohort Audit

| Split | Windows | Positive | Prevalence | SHA-256 |
|---|---:|---:|---:|---|
| Train | 3,843,400 | 217,650 | 5.663% | `01c84dbfa19a680e2645006e5cc5e56a6f747636b3e5ad2e9bf93968beb85062` |
| Validation | 819,573 | 47,638 | 5.813% | `87ba42c5c12be7b6b1113d8b347fb31711f1b2ad2100d03564e35f287ec1cd9a` |
| Test | 830,839 | 47,292 | 5.692% | `5a3afc9059e6a5dbf7f7e3da3f0297c07889e7dec19a65fe77122c2e157168db` |

Patient split 與 cohort fingerprint 已通過 `comparison_protocol.json` 稽核。

## 完成判定

訓練程序結束後必須同時存在：

- `best_model.pt`
- `last_model.pt`
- `training_summary.json`
- `test_metrics.csv`
- `metrics.csv`
- `figures/training_history.png` 與 PDF

在上述檔案齊全前，不應將 full-cohort training 標示為完成。

## 完成後立即執行

匯出固定 validation/test windows 的 prediction-level 檔案，供 calibration 與 patient-clustered evaluation 使用：

```powershell
.\env\Scripts\python.exe model_evaluation_report.py --sources fnn --fnn-run-dirs outputs\explicit_temporal_fnn_formal_6h\seed_42 --comparison-mode full --horizons 6 --save-predictions --prediction-output-dir outputs\explicit_temporal_fnn_formal_6h\seed_42\predictions --output-dir outputs\explicit_temporal_fnn_formal_6h\seed_42\evaluation --device cuda
```

這次 full-cohort 模型屬於 sample-size/deployment analysis，不能直接與只用 200,000 training windows 的 baseline 當作主要公平比較。

## 主要公平比較仍需補跑

使用同一組最佳參數，在固定 200,000/50,000 equal-sample cohort 重訓 explicit model：

```powershell
.\env\Scripts\python.exe train_fnn.py --comparison-mode equal_sample --best-params-json outputs\explicit_temporal_fnn_tuning_6h\best_params.json --explicit-temporal-features --epochs 20 --early-stopping-patience 5 --early-stopping-min-epochs 10 --output-dir outputs\explicit_temporal_fnn_equal_sample_6h\seed_42 --device cuda
```

完成後需匯出 predictions，加入 `outputs/fair_comparison_6h_equal_sample/`，再重跑 500 次 subject-clustered bootstrap 與 paired model comparison。

## 必查項目

- 最佳 epoch 是否由 validation AUROC 選出。
- Test 是否只在 checkpoint 固定後評估。
- Raw 與 validation-only calibrated Brier、ECE、intercept、slope。
- Sensitivity at 90%/95% specificity。
- `explicit_temporal_scale=1.9896` 的局部 sensitivity。
- 高 rule scale 與 explicit scale 是否造成 double counting、rule drift 或 calibration 惡化。
