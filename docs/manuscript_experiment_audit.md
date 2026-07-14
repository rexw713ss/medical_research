# 投稿底稿與教授建議完成度稽核

更新日期：2026-07-14

稽核範圍：Post-hoc XAI、Clinical Consistency Regularization、SOFA outcome sensitivity、missingness，以及目前主文可支持的宣稱。

本次已逐項比對 `paper/TSP_template.tex`、`paper/Supplementary_Material.tex`、正式實驗輸出與核心模型程式。先前一致性修訂備份位於 `paper/backups/20260714_151000_before_consistency_alignment/`；本輪 trade-off 收斂前的主文與 PDF 另完整備份至 `paper/backups/20260714_210917_before_tradeoff_reframing/`。本輪只修改主文，Supplementary 未變動。

## 狀態定義

- **正式完成**：使用預先鎖定 patient split；equal-sample 僅限 train/validation，test 使用全部 eligible windows；CI 以 patient 為 cluster。
- **探索性完成**：方法與結果已產生，但使用預先抽樣案例，不能當作 full-cohort confirmatory evidence。
- **部分完成**：已有部分指標或文字，但比較條件、證據層級或外部驗證尚未完整。
- **尚未完成**：目前沒有可支持該宣稱的實驗結果。

## 總結判定

| 教授建議 | 判定 | 核心理由 |
|---|---|---|
| LightGBM/XGBoost/EBM/KG-TFNN 預測比較 | 正式完成 | Feature-matched tree/GRU 與 KG-TFNN 使用相同 split、train/validation sampling 與完整 test windows |
| Post-hoc 與 intrinsic explanation quality 比較 | 正式全量完成 | 830,839 MIMIC 與 6,215,890 eICU prediction windows；`formal_full_data=true` |
| Frozen baseline eICU transport comparison | 正式全量完成 | 5 models、相同 6,215,890 external windows、200 次 patient-cluster bootstrap；無 eICU fitting |
| 跨模型統一 explanation complexity | 正式全量完成 | 所有 attribution 聚合為相同 13 個 clinical variables，以 80% attribution mass 定義 complexity |
| Clinical consistency regularization 的預測與 rule stability ablation | 正式完成 | 4 variants x 3 seeds；full 與 no-consistency 可直接配對 |
| Consistency violation / risk reversal behavioral audit | 正式全量完成 | 3 seeds x 2 variants，每個模型使用完整 830,839 MIMIC test windows |
| SOFA outcome-definition / documentation-bias sensitivity | 正式完成 | 完整 830,839 test windows、500 次 patient-cluster bootstrap、common-component/mask analyses |
| Missingness-only / no-missingness ablation | 正式完成 | 3-seed ensemble、完整 test windows、1,000 次 patient-cluster bootstrap |
| Missingness Discussion 四種機制 | 完成 | 已明確區分 clinical workflow、physician attention、disease severity 與 monitoring frequency，並避免 causal biomarker 宣稱 |
| Clinician-validated understandability | 尚未完成 | 沒有 blinded clinician reader study；現有結果只能支持 structural inspectability |

## 1. KG-TFNN 與 Post-hoc XAI

### 已完成的預測比較

Feature-matched LightGBM、XGBoost 與 GRU 已使用相同 39 hourly channels 的同源資訊、相同 200,000/50,000 train/validation windows，以及完整 MIMIC test windows。正式結果顯示 boosting models 的 discrimination 高於 equal-sample KG-TFNN：

| Model | AUROC | AUPRC | 證據層級 |
|---|---:|---:|---|
| LightGBM | 0.6904 | 0.1710 | 正式 feature-matched comparison |
| XGBoost | 0.6870 | 0.1665 | 正式 feature-matched comparison |
| KG-TFNN | 0.6448 | 0.1236 | 正式 equal-sample comparison |

正式來源：`outputs/feature_matched_baselines_6h_equal_sample/`、`outputs/explicit_kg_tfnn_paired_comparison_6h/`。

### 正式全量 explanation benchmark

`outputs/posthoc_explainability_comparison_6h/analysis_config.json` 明確記錄：

- `formal_full_data=true`
- 完整 830,839 個 MIMIC test windows 與 6,215,890 個 eICU external windows
- 每個 prediction-key window 均由 hourly source 精確重建；patient/stay/label counts 與 SHA-256 均通過稽核
- MIMIC 每個 window 執行 3 次、1% training-SD perturbation，共 2,492,517 perturbation pairs
- Explanation continuity 比較同一 ICU stay 內相鄰的 eligible windows，而非抽樣 nearest neighbors
- EBM 是 13-feature current-state comparator，並非 24-hour feature-matched model

| Model | Stability cosine | Within-stay continuity cosine | 80% attribution 所需變數 | Temporal attribution mass | MIMIC-eICU rank rho |
|---|---:|---:|---:|---:|---:|
| LightGBM + TreeSHAP | 0.965 | 0.914 | 6 | 0.865 | 0.885 |
| XGBoost + TreeSHAP | 0.950 | 0.928 | 7 | 0.852 | 0.967 |
| EBM, current state | 0.989 | 0.887 | 6 | 0.000 | 0.819 |
| KG-TFNN | 1.000 | 0.998 | 5 | 0.190 | 0.962 |

共同 complexity 指標已正式鎖定為：每個 local explanation 先聚合至相同 13 個 harmonized clinical variables，取絕對 attribution 並正規化，再計算累積 80% attribution mass 所需的最少變數數量。中位數（IQR）為 KG-TFNN 5（4–5）、LightGBM 6（6–7）、XGBoost 7（6–7）、current-state EBM 6（6–7）。KG-TFNN Top-10 mean antecedents 1.44 仍是模型專屬結構指標，不與 SHAP/EBM terms 當成同義的 rule complexity。

### Frozen eICU comparator transport

所有 equal-sample models 直接套用於相同 80,239 patients、99,262 stays、6,215,890 windows。Calibration 與 fixed-specificity thresholds 僅由 MIMIC validation 決定，且五個 source prediction reproduction checks 的最大絕對機率差均小於 `3e-08`。

| Model | eICU AUROC | eICU AUPRC | Brier | ECE |
|---|---:|---:|---:|---:|
| KG-TFNN, equal-sample | 0.6100 | 0.0862 | 0.0456 | 0.0222 |
| LightGBM, feature-matched | 0.6247 | 0.0949 | 0.0460 | 0.0203 |
| XGBoost, feature-matched | 0.6323 | 0.0999 | 0.0455 | 0.0180 |
| GRU, feature-matched | 0.6036 | 0.0721 | 0.0470 | 0.0309 |
| EBM, current state | 0.5869 | 0.0695 | 0.0452 | 0.0120 |

來源：`outputs/eicu_frozen_baseline_validation_6h/`。這是 equal-sample transport comparison；prespecified formal full-cohort KG-TFNN 的 eICU AUROC/AUPRC 0.6221/0.0922 只能另列背景，不能與前表混作 architecture-matched inference。

目前可以支持的敘述：KG-TFNN 在這個**full-data structural benchmark** 中，對小擾動較穩定、同一 stay 的連續 explanations 較一致、attribution 較稀疏，且輸出可讀 temporal fuzzy rules。Prediction 不是最佳，因此研究定位應是 predictive-performance / intrinsic-inspectability trade-off。

目前不能支持的敘述：

- KG-TFNN 在所有 cross-dataset 指標都最好；XGBoost 的 rank rho 0.967 高於 KG-TFNN 0.962，且 XGBoost/EBM 的 Top-5 Jaccard 為 1.000。
- Temporal attribution mass 越高必然越可解釋；不同 explanation form 的數值語意不等價。
- KG-TFNN 已證明比 SHAP 更容易被臨床人員理解。
- Current-state EBM 與 24-hour KG-TFNN 已達完整 architecture matching。

### 尚缺項目

1. 建立 24-hour feature-matched EBM；目前 EBM 只能作為 current-state additive comparator。
2. 若要使用「clinically understandable」而非「human-readable/inspectable」，需要 blinded clinician reader study，評估正確性、理解時間、信心與 inter-rater agreement。

## 2. Clinical Consistency Regularization

### 正式完成部分

正式 3-seed ablation 已比較 temporal FNN without consistency 與 full KG-TFNN。Consistency loss 沒有改善 AUROC/AUPRC，但 Top-10 rule stability 由 0.587 提升為 0.674。

| 指標 | No consistency | Full | Full - no consistency |
|---|---:|---:|---:|
| AUROC | 0.6459 | 0.6456 | -0.0003 |
| AUPRC | 0.1230 | 0.1230 | +0.0001 |
| Rule stability | 0.587 | 0.674 | +0.087 |
| Normalized membership drift | 0.222 | 0.224 | +0.002，未改善 |
| Guideline-risk correlation | 0.536 | 0.528 | -0.008，未改善 |

來源：`outputs/fnn_ablation_6h_equal_sample/ablation_publication_table.csv` 與 `paired_component_effects.csv`。

### 正式 full-test-cohort behavioral audit

| 指標 | No consistency | Full | 判讀 |
|---|---:|---:|---|
| Violation rate given worsening | 0.3689 | 0.3677 | -0.0012，幾乎不變 |
| Consistency penalty | 0.000382 | 0.000332 | 有下降 |
| Risk reversal frequency | 0.0886 | 0.0905 | +0.0019，略增 |
| Reversal magnitude median | 0.0889 | 0.0732 | 下降 |

來源：`outputs/clinical_consistency_regularization_6h/`；3 seeds x 2 variants，每個模型均處理完整 830,839 個 MIMIC test windows，總計 4,985,034 次 model-window evaluations，`formal_full_data=true`。

Guideline-direction alignment 在兩組都是 1.0，原因是兩組共用同一套固定 antecedent directions；它不能證明 consistency loss 的效果。現有證據只適合把 consistency regularization 定位為**可能改善跨 seed 規則重現性**的 regularizer，不能宣稱它已降低所有不合理規則、risk reversals 或 membership drift。

模型與 directional perturbation definitions 均維持 frozen。全量結果確認 consistency regularization 的主要正向訊號是 cross-seed rule stability，而非全面降低 behavioral violations。

## 3. SOFA Outcome Definition

本項已正式完成，而且結果顯示 documentation bias **存在且具實質影響**，不是已被排除。

已完成分析：

- Primary：index 與未來 6 小時 SOFA increase >= 2，至少 4 個可觀測 components。
- Missing-as-normal sensitivity。
- Six-component complete-case sensitivity。
- Pairwise common components，要求 index/future 至少 4 個相同可觀測 components。
- Index 與 primary future maximum 使用相同 component mask。
- 六個 future hours 全部使用 stable component mask。
- Respiratory、coagulation、liver、cardiovascular、neurological、renal 的 positive-point contribution。

| Definition | Windows | AUROC | AUPRC |
|---|---:|---:|---:|
| Primary | 830,839 | 0.6559 | 0.1309 |
| Missing components assumed normal | 830,839 | 0.6557 | 0.1314 |
| Six-component complete case | 176,130 | 0.6237 | 0.0923 |
| Pairwise common components | 830,609 | 0.6097 | 0.0662 |
| Same mask at future maximum | 788,385 | 0.6013 | 0.0564 |
| Stable mask over six future hours | 679,878 | 0.6027 | 0.0594 |

47,292 個 primary positive windows 中，18,426 個（39.0%）在 future maximum 新增了 index hour 未觀測的 component；31,740 個（67.1%）在 pairwise-common label 下仍為 positive。Positive component-point increases 以 renal 46.3% 最高，其次為 neurological 19.1%、cardiovascular 16.4%、respiratory 12.9%、coagulation 3.8%、liver 1.5%。

來源：`outputs/clinical_sensitivity_analyses_6h/` 與 `outputs/sofa_documentation_bias_6h/`。

## 4. Missingness

### 正式實驗

Missingness ablation 已使用相同 split、200,000/50,000 train/validation windows、完整 830,839 test windows、3 seeds 與 1,000 次 patient-cluster bootstrap：

| Model | AUROC | AUPRC | Brier | ECE |
|---|---:|---:|---:|---:|
| Full KG-TFNN ensemble | 0.6475 | 0.1244 | 0.0523 | 0.0014 |
| Without missingness channels | 0.6042 | 0.0879 | 0.0532 | 0.0011 |
| Missingness-only | 0.5954 | 0.0904 | 0.0532 | 0.0019 |

Full minus no-missingness 的 AUROC 差為 0.0435（95% CI 0.0391--0.0478）；full minus missingness-only 為 0.0518（0.0458--0.0576）。Missingness 本身帶有預測訊號，但 clinical trajectories 與 missingness 結合才有最佳表現。

### Discussion 完整度

主文 Discussion 與 Limitations 已明確指出 missingness 可能同時反映 **clinical workflow、physician attention、disease severity、monitoring frequency** 與病人生理狀態，並將其定位為 context-dependent care-process information，而不是 causal biomarker。此敘述與 missingness-only、no-missingness ablation 及跨資料庫 calibration shift 一致。

## 投稿可用結論

目前主文的保守定位是合理的：boosting models 預測較好；KG-TFNN 提供較穩定、稀疏且可直接檢視的 intrinsic fuzzy-rule representation；consistency loss 主要與 rule stability 相關；SOFA label 與 missingness 都含有 care-process/documentation signal。

大型 full-data 重跑、frozen baseline external transport 與統一 complexity 已完成。投稿前仍有兩個可選的解釋性證據層級補強：

1. Feature-matched temporal EBM；目前跨模型 complexity 已有共同定義，但 EBM 仍是 current-state comparator。
2. 若要宣稱「clinically understandable」，需 clinician reader study；現有結果只能稱 structural stability、continuity、sparsity 與 intrinsic inspectability。

正式 full-data explanation、consistency、frozen baseline external transport 與 unified complexity 均已寫入主文；完整細節仍由 Supplementary S10--S11/S6--S7 與正式輸出支援。稿件已明確避免「prediction best」與「clinician-validated interpretability」宣稱。

## 本輪稿件一致性修正

- 模型輸入由含糊的 `24 x P` 改為實際的 `24 x 3P`，其中 `P=13`、每小時共 39 channels。
- Temporal descriptor 定義與 `anfis_model.py` 對齊，包括 current missingness、縮放、soft abnormal duration 與 time-since transform。
- Explanation benchmark 改為完整 830,839 MIMIC 與 6,215,890 eICU windows；continuity 定義改為同一 stay 的連續 windows。
- Clinical-consistency behavior 改為每個 seed/variant 全部 830,839 test windows 的正式數字。
- 補清 1,000-replicate 與 500-replicate clustered bootstrap 的使用範圍。
- Frozen-test 敘述改為鎖模後產生 prediction files，再重用於 post-lock analyses，且不回饋 tuning/calibration。
- Table 2 新增 frozen equal-sample eICU model-by-model transport，Figure 4 納入 external ROC/PR；Table 3 Panel B 明確使用共同 13-variable、80\% attribution-mass complexity 定義。
- Abstract、Introduction、Contributions、Discussion 與新增 Conclusion 已統一定位為 predictive-performance / intrinsic-interpretability trade-off，並明列 boosting discrimination 較高與 XGBoost cross-dataset rank stability 較高。
- 主文 `paper/TSP_template.pdf` 成功編譯為 35 頁，Supplement 維持 20 頁；無 undefined reference、undefined citation、missing figure 或 oversized float。既有兩個 1.77 pt table-cell overfull warnings 與本輪新增內容無關。
