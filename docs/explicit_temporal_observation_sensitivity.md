# 明確時序特徵與觀察窗敏感度分析

實驗日期：2026-06-30

## 研究目的

本實驗在 Knowledge-Guided Temporal FNN 中加入可明確解釋的時序特徵，並以 4、6、12、24 小時觀察窗進行敏感度分析。主要預測結果為未來 6 小時內 SOFA 分數增加至少 2 分（`label_sofa_increase_ge2_6h`）。

## 明確時序特徵

對每個生理變項，模型先以模糊 membership functions 與規則權重得到每小時特徵風險分數，再於指定觀察窗內計算下列 12 個訊號：

| 程式欄位 | 定義 |
|---|---|
| `mean_risk` | 觀察窗內平均風險 |
| `min_risk` | 觀察窗內最低風險 |
| `max_risk` | 觀察窗內最高風險 |
| `risk_std` | 觀察窗內風險標準差 |
| `risk_slope` | 以等距小時估計的線性風險斜率，經 `tanh` 限幅 |
| `short_term_change` | 當前小時減前一小時的風險，經 `tanh` 限幅 |
| `window_change` | 當前小時減觀察窗第一小時的風險，經 `tanh` 限幅 |
| `abnormal_duration` | 觀察窗內可微分異常機率的平均值 |
| `abnormal_frequency` | 僅在實際量測小時中計算異常機率平均值，不將 LOCF 小時視為新量測 |
| `missing_fraction` | 觀察窗內缺失比例 |
| `current_missing` | 當前小時是否缺失 |
| `time_since_last_measurement` | 距最近一次實際量測時間，截斷於 168 小時並以 `log1p(x) / log(169)` 正規化 |

異常機率定義為 `sigmoid((feature_risk - 0.5) / 0.20)`。每個特徵與時序訊號均有可訓練係數；其加權貢獻加總後除以 `sqrt(13)`，再與 attention-based temporal risk 相加。模型同時對這些係數施加 sparsity、initialization drift 與 non-negative regularization。

## 比較條件

- Patient-level split：依 `subject_id` 固定分派 train、validation、test，病患不跨 split。
- Cohort eligibility：所有模型皆要求至少 24 小時病史；4、6、12 小時模型只取最後對應長度，因此觀察窗差異不會改變納入樣本。
- 相同 outcome、predictors 定義與 test windows。
- Equal-sample train/validation：200,000 / 50,000 windows。
- 獨立 test set：830,839 windows，其中 47,292 個陽性，盛行率 5.69%。
- 三個 random seeds：42、52、62。
- 每組最多 20 epochs，使用 validation AUROC 選擇 checkpoint。
- 另設 24 小時 sequence-only control；此控制組保留 attention，但不使用 12 個明確時序訊號。
- 15 組執行的 train、validation、test cohort SHA-256 完全一致。

## 主要結果

表中 95% CI 是三個 seeds 平均值的 t interval，反映訓練隨機性；不是 patient-clustered bootstrap CI。

| 模型 | AUROC, mean (95% CI) | AUPRC, mean (95% CI) |
|---|---:|---:|
| Explicit temporal FNN, 4 h | 0.6205 (0.6201–0.6210) | 0.0942 (0.0933–0.0952) |
| Explicit temporal FNN, 6 h | 0.6216 (0.6202–0.6229) | 0.0941 (0.0929–0.0953) |
| Explicit temporal FNN, 12 h | 0.6220 (0.6193–0.6248) | 0.0943 (0.0918–0.0969) |
| Explicit temporal FNN, 24 h | **0.6331 (0.6298–0.6364)** | **0.1060 (0.1038–0.1081)** |
| Sequence-only FNN, 24 h | 0.5780 (0.5755–0.5805) | 0.0748 (0.0741–0.0755) |

以相同 seed 配對後，24 小時 explicit temporal FNN 相較 sequence-only control：

- AUROC 平均增加 0.0551（seed-level 95% CI 0.0518–0.0584）。
- AUPRC 平均增加 0.0312（seed-level 95% CI 0.0296–0.0327）。
- 24 小時模型相較 4、6、12 小時模型，AUROC 分別增加 0.0126、0.0116、0.0111；對應配對 CI 均未跨 0。

## 解讀

在固定 cohort 與相同 test windows 下，24 小時觀察窗表現最佳，顯示較長病程資訊對未來 6 小時 SOFA 惡化預測有額外價值。24 小時模型的平均絕對係數以 `risk_slope`、`window_change`、`abnormal_frequency`、`time_since_last_measurement` 與 `short_term_change` 較大，支持趨勢、整窗變化、異常量測型態及資訊新鮮度具有預測價值。

係數大小仍受訊號尺度及正則化影響，因此應解讀為 coefficient profile，而不是直接等同 permutation importance 或因果重要性。正式模型比較仍應以 prediction-level patient-clustered bootstrap 與 paired test 為主。

## 產出檔案

- 完整執行結果：`outputs/explicit_temporal_observation_sensitivity_6h/observation_window_runs.csv`
- 彙總結果：`outputs/explicit_temporal_observation_sensitivity_6h/observation_window_aggregate.csv`
- 配對差異：`outputs/explicit_temporal_observation_sensitivity_6h/paired_sensitivity_differences.csv`
- 時序係數：`outputs/explicit_temporal_observation_sensitivity_6h/explicit_temporal_weights.csv`
- 係數摘要：`outputs/explicit_temporal_observation_sensitivity_6h/explicit_temporal_weight_summary.csv`
- 觀察窗圖：`outputs/explicit_temporal_observation_sensitivity_6h/figures/observation_window_sensitivity.png` 與 `.pdf`
- 時序係數圖：`outputs/explicit_temporal_observation_sensitivity_6h/figures/explicit_temporal_coefficients.png` 與 `.pdf`

## 目前限制

本次沿用先前 sequence-only FNN 的超參數，以避免對新模型額外最佳化造成不公平；明確時序分支尚未獨立 tuning。Brier score 與 ECE 顯示 raw probabilities 仍需在 validation set 進行 calibration。另因本敏感度分析只使用三個 seeds，seed-level CI 不應取代既有的 subject-level clustered bootstrap 評估。
