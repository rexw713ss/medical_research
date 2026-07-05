# 正式 6 小時 FNN 消融實驗

更新日期：2026-07-05

## 設計

- Primary outcome：未來 6 小時 SOFA increase >= 2。
- 所有變體使用相同 patient split、200,000 個 train windows、50,000 個 validation windows，以及完整 830,839 個 test windows。
- Random seeds：42、52、62。
- Checkpoint 僅依 validation AUROC 選定；Brier 與 ECE 使用 validation-only Platt calibration。
- 除被消融元件外，四組共用 explicit-temporal Optuna 最佳超參數。

## 結果

數值為三個 random seeds 的 mean +/- SD。Rule Stability 為 Top-10 規則在三組 seed pair 的平均 Jaccard similarity；Rule Drift 為相對初始化尺度正規化後的參數 RMSE。

| 模型 | AUROC | AUPRC | Brier | ECE | Rule Concordance | Rule Stability | Rule Drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| Randomly initialized FNN | 0.6395 +/- 0.0008 | 0.1183 +/- 0.0024 | 0.05244 +/- 0.00006 | 0.00117 +/- 0.00012 | 0.496 +/- 0.127 | 0.207 | 0.165 +/- 0.010 |
| Guideline-guided FNN without temporal features | 0.5949 +/- 0.0004 | 0.0837 +/- 0.0006 | 0.05330 +/- 0.00001 | 0.00121 +/- 0.00007 | 0.647 +/- 0.021 | 0.818 | 0.245 +/- 0.003 |
| Temporal FNN without clinical consistency regularization | **0.6459 +/- 0.0019** | 0.1230 +/- 0.0004 | 0.05231 +/- 0.00001 | 0.00125 +/- 0.00008 | 0.536 +/- 0.119 | 0.587 | 0.222 +/- 0.007 |
| Full Knowledge-Guided Temporal FNN | 0.6456 +/- 0.0012 | **0.1230 +/- 0.0006** | **0.05230 +/- 0.00001** | **0.00123 +/- 0.00007** | 0.528 +/- 0.123 | 0.674 | 0.224 +/- 0.011 |

## 元件貢獻

| 元件 | Paired AUROC difference | 95% CI | Paired t-test p | 解讀 |
|---|---:|---:|---:|---|
| Expert knowledge initialization | +0.0061 | +0.0026 to +0.0096 | 0.017 | 小幅提升辨識效能；僅三個 seeds，推論需保守。 |
| Temporal feature design | +0.0510 | +0.0457 to +0.0563 | 0.0006 | 為最主要的效能來源。 |
| Clinical consistency regularization | -0.0003 | -0.0019 to +0.0013 | 0.530 | 未改善 AUROC；AUPRC 差異亦近乎為零。 |

Consistency regularization 沒有改善預測效能或 Rule Concordance，但 Rule Stability 由 0.587 提升至 0.674。這只能解讀為規則重現性可能改善，不能宣稱它提升整體預測表現。若投稿篇幅有限，可將 consistency loss 定位為 interpretability regularizer，並把其權重敏感度分析放入 supplement。

## 指標定義

- **Rule Concordance**：訓練後 static fuzzy risk 與 frozen NEWS2/SOFA-guided reference risk 在 validation windows 上的 Spearman correlation。
- **Rule Stability**：不同 seeds 的 Top-10 feature/cross-rule importance 集合之 pairwise Jaccard similarity。
- **Rule Drift**：membership centers、sigmas、feature-rule weights 與 cross-rule weights，相對各自初始化 RMS 尺度的平均 normalized RMSE。

## 正式產物

- `outputs/fnn_ablation_6h_equal_sample/ablation_summary.csv`
- `outputs/fnn_ablation_6h_equal_sample/ablation_aggregate.csv`
- `outputs/fnn_ablation_6h_equal_sample/ablation_publication_table.csv`
- `outputs/fnn_ablation_6h_equal_sample/paired_component_effects.csv`
- `outputs/fnn_ablation_6h_equal_sample/rule_stability_pairwise.csv`
- `outputs/fnn_ablation_6h_equal_sample/figures/`
