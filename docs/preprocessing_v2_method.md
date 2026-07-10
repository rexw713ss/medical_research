# Preprocessing v2 方法紀錄

更新日期：2026-07-10

## 成人 cohort

- 納入條件固定為 ICU 入住時 `age >= 18`。
- MIMIC-IV 年齡計算為 `anchor_age + ICU admission year - anchor_year`；原始 ICU cohort 最小年齡即為 18，因此排除 0 位未成年病人。
- eICU 將 `> 89` 編碼為 90 後套用相同門檻；原 preprocessing 已排除 437 位未成年病人、530 個 stays。
- 成人條件重建的 `patient_split.csv` 與原檔 byte-identical，split assignment 差異為 0；完整證據見 `outputs/manuscript_tables_figures_6h/adult_eligibility_audit.json`。

## 時間對齊與補值

- 以 ICU `intime` 建立每個 `stay_id` 的逐小時網格。
- 所有 chart/lab events 僅對齊到事件發生時所屬的 ICU stay。
- 缺值只在同一 `stay_id` 內 forward-fill，不使用 backward-fill。
- temporal features 使用過去 4、6、12 小時的 mean、min、max、standard deviation 與 slope。

## 單位與編碼

- FiO2 統一為比例值 0.21-1.00；FNN membership functions 與預設值使用相同尺度。
- Temperature Fahrenheit 轉換為 Celsius 後，與 Temperature Celsius 合併。
- Mechanical ventilation 由原始文字狀態轉成 0/1，不使用 ventilator mode 的數字代碼。
- PaO2/FiO2 ratio 使用比例形式的 FiO2 計算。

## 異常值處理

- 所有生命徵象、GCS、實驗室數值與升壓劑劑量先套用寬鬆的臨床合理範圍。
- 範圍外數值設為缺值，不做 clipping，並在之後依相同 stay 的過去紀錄 forward-fill。
- 實際範圍定義於 `clinical_data_quality.py`。

## SOFA outcome

- SOFA 使用過去 24 小時 worst value 計算六個器官 component。
- `sofa_score_assume_normal`：缺失 component 視為 0，供敏感度分析。
- `sofa_score_complete`：僅六個 component 全部可觀測時提供分數。
- `sofa_score`：主要分析分數，至少需要 4 個可觀測 component。
- 未來 6、12、24 小時標籤只在目前與未來均有有效 SOFA、且具有完整預測 horizon 時建立。
- 6 小時 label 為 primary outcome；12/24 小時 labels 僅供 secondary analyses。

## 正式輸出

- `sofa_scores_hourly.csv`：8,275,274 rows、94,444 ICU stays。
- `model_hourly_features_v3.csv`：8,275,274 rows，並包含 explicit temporal FNN 所需的 missingness 與 time-since channels。
- `sofa_scores_hourly_quality.json`：component completeness 與 outcome prevalence。
- `model_hourly_features_v3_quality.json`：各特徵覆蓋率、最小值、最大值與 outcome prevalence。
- `outputs/manuscript_tables_figures_6h/table_2_feature_missingness.csv`：LOCF 前 current-hour raw missingness，分 MIMIC train/validation/test 與 eICU 回報。

正式 outcome prevalence：

- 6 小時：6.46%
- 12 小時：11.90%
- 24 小時：20.48%

## Patient-level 資料切分

- 使用 `subject_id` 作為唯一切分單位，同一病人的所有 ICU stays 僅能位於同一組。
- 先把 6、12、24 小時標籤聚合為 patient-level ever-event，再依三個 outcome 的組合分層。
- 固定 random seed 42，比例為 train 70%、validation 15%、test 15%。
- validation 僅用於超參數、threshold 與 checkpoint 選擇；test 僅在模型定案後評估一次。
- 資料切分鍵為 `subject_id`；逐小時事件與 SOFA 標籤仍使用 `stay_id + sofa_hour` 對齊。
- 固定名單存於 `patient_split.csv`，統計與事件率存於 `patient_split_summary.json`。

正式病人數：

- Train：45,757 patients、66,090 ICU stays。
- Validation：9,807 patients、14,183 ICU stays。
- Test：9,802 patients、14,185 ICU stays。

## 公平模型比較

- 完整定義見 `docs/fair_comparison_protocol.md`。
- `comparison_protocol.json` 固定 predictors、24h eligibility、outcomes 與完整 cohort fingerprints。
- `equal_sample_windows.csv.gz` 固定每個 horizon 的 200,000 train 與 50,000 validation windows。
- Full-cohort 與 equal-sample comparison 使用相同完整 test windows。
