# 舊實驗結果狀態

`outputs/` 內原有 checkpoints、metrics 與 figures 是使用舊 preprocessing 與 stay-level 80/20 split 產生。

正式資料已修正以下項目：

- FiO2 單位
- Mechanical ventilation 二元編碼
- Fahrenheit temperature
- 臨床異常值
- SOFA component completeness 與 labels

固定 70%/15%/15% patient-level split 已建立於 `patient_split.csv`，公平比較條件已建立於 `comparison_protocol.json`。因此，既有結果只能作為流程與程式驗證，不可直接作為論文最終結果；仍需分別重跑 full-cohort 與 equal-sample comparisons。

例外：`outputs/formal_fnn_v2_20260628/` 已使用修正後資料、固定 patient split 與 comparison protocol 正式重跑，可作為目前 FNN 正式結果。

`outputs/formal_fnn_ablation_v1_20260628/` 已使用完整 6h cohort 與三個 random seeds 正式重跑，可作為正式消融結果。舊的 `outputs/fnn_ablation/fnn_ablation_20260626_032349/` 仍屬 stale，請勿引用。
