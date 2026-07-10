# Explicit-Temporal FNN Full-Cohort 6 h Training

更新日期：2026-07-05

## 狀態

**完成。** Primary outcome 為未來 6 小時 SOFA increase >= 2。最佳 checkpoint 由 validation AUROC 選定於 epoch 15；訓練在 epoch 20 依 patience 5 early stopping。

## Cohort

| Split | Windows | Positive | Prevalence | SHA-256 |
|---|---:|---:|---:|---|
| Train | 3,843,400 | 217,650 | 5.663% | `01c84dbfa19a680e2645006e5cc5e56a6f747636b3e5ad2e9bf93968beb85062` |
| Validation | 819,573 | 47,638 | 5.813% | `87ba42c5c12be7b6b1113d8b347fb31711f1b2ad2100d03564e35f287ec1cd9a` |
| Test | 830,839 | 47,292 | 5.692% | `5a3afc9059e6a5dbf7f7e3da3f0297c07889e7dec19a65fe77122c2e157168db` |

## Frozen Final Test Results

| Metric | Estimate | Patient-clustered 95% CI |
|---|---:|---:|
| AUROC | 0.6559 | 0.6492–0.6628 |
| AUPRC | 0.1309 | 0.1250–0.1375 |
| Brier, validation-only calibrated | 0.0521 | 0.0507–0.0534 |
| ECE, validation-only calibrated | 0.0012 | 0.0006–0.0026 |
| Calibration intercept | -0.020 | - |
| Calibration slope | 0.997 | - |

在 90% specificity 下 sensitivity 為 0.2667、PPV 0.1394、NPV 0.9532；在 95% specificity 下 sensitivity 為 0.1755、PPV 0.1767、NPV 0.9503。完整信賴區間見 [final test report](../outputs/final_test_evaluation_6h/final_test_report.md)。

Raw probabilities 來自 class-weighted BCE，raw Brier 為 0.1962、raw ECE 為 0.3700，不可直接當床邊絕對風險。正式結果採 validation-only Platt calibration，參數原封不動套用 test。

## Artifacts

- `outputs/explicit_temporal_fnn_formal_6h/seed_42/best_model.pt`
- `outputs/explicit_temporal_fnn_formal_6h/seed_42/training_summary.json`
- `outputs/explicit_temporal_fnn_formal_6h/seed_42/test_metrics.csv`
- `outputs/explicit_temporal_fnn_formal_6h/seed_42/metrics.csv`
- `outputs/explicit_temporal_fnn_formal_6h/seed_42/figures/`
- `outputs/final_test_evaluation_6h/FINAL_TEST_LOCK.json`
- `outputs/final_test_evaluation_6h/final_test_report.md`
- `outputs/final_test_evaluation_6h/advanced/`
- `outputs/temporal_rule_extraction_6h/extracted_temporal_rules.csv`
- `docs/extracted_temporal_rules_6h.md`

## 解讀

Checkpoint 由 validation AUROC 選定，SHA-256 為 `158427a5c358016f35b435b1ab5f75c7194a3ff3f9b6c9d68c5190a8a9125688`。一次性 final test evaluation 已完成並鎖定。此 full-cohort model 是 6 h sample-size/deployment sensitivity；primary model comparison 亦已在固定 200,000/50,000 equal-sample cohort 完成，並以相同 test windows 對 baseline 執行 1,000 次 patient-clustered paired bootstrap，結果見 `outputs/explicit_kg_tfnn_paired_comparison_6h/evaluation/`。
