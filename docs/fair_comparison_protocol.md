# 公平模型比較 Protocol

更新日期：2026-06-28

## 固定條件

- Patient split：所有模型使用 `patient_split.csv` 的 70%/15%/15% 分組。
- Window key：以 `stay_id + sofa_hour` 唯一識別一個 prediction target。
- Lookback：所有正式比較只納入至少具有 24 小時歷史的 target windows。
- Primary outcome：未來 6 小時 `SOFA increase >= 2`。
- Secondary outcomes：未來 12 與 24 小時 `SOFA increase >= 2`；必須明確指定後才執行。
- Predictors：所有 machine-learning models 固定使用 `FEATURE_ORDER` 的 13 個來源變數。
- Imputation：只在同一 ICU stay 內 forward-fill，再補固定臨床預設值；不使用 backward-fill。
- Test policy：test set 一律使用完整 eligible windows，不允許再抽樣。

13 個 predictors 為 heart rate、respiratory rate、SpO2、FiO2、temperature、SBP、GCS、MAP、PaO2/FiO2、platelets、bilirubin、creatinine 與 lactate。

Tabular models 使用相同來源變數的 index-time values；FNN、LSTM 與 GRU 使用相同變數的 24 小時 sequence。兩者的差異屬於待比較的 temporal representation，不額外加入 SOFA 或其他 predictors。NEWS2 與 SOFA 為預先定義的 clinical-score comparators，因此不列入「相同 ML predictors」限制，但仍使用完全相同的 patient split、target windows 與 outcomes。

## Full-cohort Comparison

| Horizon | Train | Validation | Test |
|---|---:|---:|---:|
| 6h | 3,843,400 | 819,573 | 830,839 |
| 12h | 3,602,603 | 768,058 | 779,240 |
| 24h | 3,184,689 | 678,998 | 689,904 |

## Equal-sample Comparison

- 每個 horizon 固定抽取 200,000 train windows。
- 每個 horizon 固定抽取 50,000 validation windows。
- 依 outcome 分層抽樣並固定 seed 42。
- Test windows 與 full-cohort comparison 完全相同。
- 固定名單為 `equal_sample_windows.csv.gz`。

## 自動稽核

每支模型在訓練前會輸出 `cohort_audit.json`，記錄各 split 的 window 數、正負類數、prevalence 與 cohort SHA-256。任一模型的 window 或 outcome 與 `comparison_protocol.json` 不一致時，程式會直接停止。

執行 equal-sample comparison：

```powershell
.\env\Scripts\python.exe .\run_fair_comparison.py --mode equal_sample
```

上述預設只執行 6 h primary outcome。Secondary analyses 使用
`--horizons 12,24`，不得與 primary 表格混為同一項主要假設檢定。

另跑 full-cohort comparison：

```powershell
.\env\Scripts\python.exe .\run_fair_comparison.py --mode full
```
