# Knowledge-Guided Temporal FNN

本專案以 MIMIC-IV 建立 ICU 動態惡化預測模型，主要 outcome 為未來 6 小時 SOFA 增加至少 2 分，並以 eICU-CRD 進行外部驗證。目前僅保留最新版資料、程式與正式輸出；共用路徑定義於 `project_config.py`。

**Primary analysis 已鎖定為未來 6 小時 SOFA increase >= 2。** 12/24 小時 outcomes 為 secondary analyses，不阻擋 6 小時主分析與投稿。完整規則見 `docs/analysis_plan.md`。

## 目前狀態

| 項目 | 狀態 | 正式位置 |
|---|---|---|
| MIMIC leakage-free SOFA labels | 6 h primary 完成；12/24 h secondary labels 已備妥 | `sofa_scores_hourly.csv` |
| MIMIC v3 hourly、missingness 與 time-since features | 完成 | `model_hourly_features_v3.csv` |
| Patient-level split 與 equal-sample protocol | 完成 | `patient_split.csv`, `comparison_protocol.json` |
| 6 h baseline、獨立 test、clustered CI 與 paired comparison | 完成 | `outputs/advanced_evaluation_6h_equal_sample/` |
| Explicit temporal features 與 4/6/12/24 h observation sensitivity | 完成 | `outputs/explicit_temporal_observation_sensitivity_6h/` |
| Explicit-temporal FNN 專屬 Optuna tuning | 完成 | `outputs/explicit_temporal_fnn_tuning_6h/` |
| eICU harmonization、SOFA labels 與 frozen-checkpoint external validation | 完成 | `outputs/eicu_external_validation/` |
| 新版 FNN full-cohort 6 h training | 完成；test AUROC 0.6559 | `outputs/explicit_temporal_fnn_formal_6h/seed_42/` |
| Frozen final-model 6 h test evaluation | 完成；1,000 次 patient-clustered bootstrap | `outputs/final_test_evaluation_6h/` |
| 正式 6 h FNN 消融，4 variants x 3 seeds | 完成 | `outputs/fnn_ablation_6h_equal_sample/` |
| Frozen-model temporal fuzzy rule extraction | 完成；24 條 supported rules | `outputs/temporal_rule_extraction_6h/` |
| Baseline paired comparison、scale sensitivity 與最終 external validation | 待完成 | `docs/experiment_status.md` |

## 正式資料

- MIMIC-IV：`dataset/MIMIC-IV/`
- eICU-CRD：`dataset/e-ICU/`
- MIMIC hourly features：`model_hourly_features_v3.csv`
- MIMIC hourly SOFA：`sofa_scores_hourly.csv`
- Patient split：`patient_split.csv`
- Comparison protocol：`comparison_protocol.json`
- Equal-sample windows：`equal_sample_windows.csv.gz`
- eICU hourly features：`outputs/eicu_external_validation/eicu_hourly_features.pkl`

## 程式對照

| 階段 | 程式 |
|---|---|
| MIMIC SOFA 與 outcome | `sofa_score.py` |
| MIMIC hourly 與 temporal preprocessing | `preprocessing.py` |
| Patient split 與公平比較 cohort | `patient_split.py`, `comparison_protocol.py` |
| FNN 架構、訓練與 tuning | `anfis_model.py`, `train_fnn.py`, `tune_fnn_optuna.py` |
| Observation-window sensitivity | `run_observation_window_sensitivity.py` |
| 共用消融元件 | `ablation_fnn_experiments.py` |
| Temporal fuzzy rule extraction | `extract_temporal_fuzzy_rules.py` |
| Interpretable baselines | `interpretable_baselines.py` |
| Black-box baselines | `blackbox_baselines.py` |
| Clinical scores | `clinical_score_baselines.py`, `news2_score.py` |
| 公平比較與統一評估 | `run_fair_comparison.py`, `model_evaluation_report.py`, `advanced_model_evaluation.py` |
| 一次性 frozen final test evaluation | `final_test_evaluation.py` |
| 論文圖表 | `paper_figures.py` |
| eICU audit、preprocessing 與 external validation | `eicu_data_audit.py`, `eicu_preprocessing.py`, `eicu_external_validation.py` |

## MIMIC 執行流程

所有命令由專案根目錄執行：

```powershell
# 1. Leakage-free SOFA 與 outcomes
.\env\Scripts\python.exe sofa_score.py

# 2. v3 hourly 與 explicit temporal input channels
.\env\Scripts\python.exe preprocessing.py

# 3. 固定 patient split 與公平比較 cohort
.\env\Scripts\python.exe patient_split.py
.\env\Scripts\python.exe comparison_protocol.py

# 4. 重新執行 explicit-temporal tuning 時使用
.\env\Scripts\python.exe tune_fnn_optuna.py --explicit-temporal-features --comparison-mode equal_sample --n-trials 30 --trial-epochs 8 --study-name explicit_temporal_fnn_6h_v1 --device cuda --output-dir outputs\explicit_temporal_fnn_tuning_6h

# 5. 以最佳 validation 參數進行 full-cohort training
& .\outputs\explicit_temporal_fnn_tuning_6h\train_with_best_params.ps1

# 6. Frozen final test evaluation（已完成並鎖定，不可重跑挑選結果）
.\env\Scripts\python.exe final_test_evaluation.py --bootstrap-reps 1000 --device cuda

# 7. Observation-window sensitivity
.\env\Scripts\python.exe run_observation_window_sensitivity.py

# 8. Baselines 與統一評估
.\env\Scripts\python.exe run_fair_comparison.py --mode equal_sample --horizons 6
.\env\Scripts\python.exe advanced_model_evaluation.py
```

所有模型必須使用相同 `subject_id` split、test windows、predictors 與 outcome。Checkpoint、threshold、calibration 與 hyperparameters 只能由 train/validation 決定；test 僅能在模型定案後使用一次。

## eICU 外部驗證

```powershell
.\env\Scripts\python.exe eicu_data_audit.py
.\env\Scripts\python.exe eicu_preprocessing.py --write-csv
.\env\Scripts\python.exe eicu_external_validation.py --bootstrap-reps 200
```

目前 external test 包含 80,239 位病人、99,262 次 ICU stay、6,215,890 個視窗與 205 家醫院。Frozen 24 h explicit-temporal checkpoint 的 AUROC 為 0.6034（patient-clustered 95% CI 0.6006–0.6059），AUPRC 為 0.0762（0.0748–0.0778）；沒有使用 eICU outcome fitting。

這個 external result 使用 full-cohort 新模型完成前的 frozen checkpoint。新版 full-cohort checkpoint 定案後，需原封不動重跑一次 eICU 才能成為最終論文結果。

## 最新結果索引

- 文件總覽：`docs/README.md`
- Analysis plan：`docs/analysis_plan.md`
- 實驗完成度：`docs/experiment_status.md`
- Full-cohort training：`docs/full_cohort_training_6h.md`
- Frozen final test report：`outputs/final_test_evaluation_6h/final_test_report.md`
- 正式消融報告：`docs/formal_ablation_study_6h.md`
- 實際 temporal fuzzy rules：`docs/extracted_temporal_rules_6h.md`
- Tuning 報告：`docs/explicit_temporal_fnn_tuning_6h.md`
- Observation sensitivity：`docs/explicit_temporal_observation_sensitivity.md`
- MIMIC 6 h baseline report：`docs/mimic_iv_6h_evaluation_report.md`
- 輸出索引：`outputs/README.md`

## 環境

```powershell
.\env\Scripts\python.exe -m pip install -r requirements.txt
```
