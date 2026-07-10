# Knowledge-Guided Temporal FNN

本專案以 MIMIC-IV 建立 ICU 動態惡化預測模型，主要 outcome 為未來 6 小時 SOFA 增加至少 2 分，並以 eICU-CRD 進行外部驗證。目前僅保留最新版資料、程式與正式輸出；共用路徑定義於 `project_config.py`。

**Primary analysis 已鎖定為未來 6 小時 SOFA increase >= 2。** 12/24 小時 outcomes 為 secondary analyses，不阻擋 6 小時主分析與投稿。完整規則見 `docs/analysis_plan.md`。

**成人 cohort 已鎖定為 ICU 入住時 `age >= 18`。** MIMIC-IV 原始 ICU cohort 最小年齡即為 18 歲，成人條件排除 0 位病人；重建後 `patient_split.csv` 與原檔 byte-identical，因此既有模型 cohort 與結果不變。eICU 原 preprocessing 已套用相同條件。稽核與 Table 1–5、Figure 1–5 見 `docs/adult_cohort_manuscript_artifacts.md`。

## 目前狀態

| 項目 | 狀態 | 正式位置 |
|---|---|---|
| MIMIC leakage-free SOFA labels | 6 h primary 完成；12/24 h secondary labels 已備妥 | `sofa_scores_hourly.csv` |
| MIMIC v3 hourly、missingness 與 time-since features | 完成 | `model_hourly_features_v3.csv` |
| Patient-level split 與 equal-sample protocol | 完成 | `patient_split.csv`, `comparison_protocol.json` |
| 6 h equal-sample baseline benchmark、獨立 test 與 clustered CI | 完成 | `outputs/advanced_evaluation_6h_equal_sample/` |
| Explicit temporal features 與 4/6/12/24 h observation sensitivity | 完成 | `outputs/explicit_temporal_observation_sensitivity_6h/` |
| Explicit-temporal FNN 專屬 Optuna tuning | 完成 | `outputs/explicit_temporal_fnn_tuning_6h/` |
| eICU harmonization、SOFA labels 與 final frozen-checkpoint external validation | 完成；AUROC 0.6221 | `outputs/eicu_external_validation/final_frozen_model_evaluation/` |
| 新版 FNN full-cohort 6 h training | 完成；test AUROC 0.6559 | `outputs/explicit_temporal_fnn_formal_6h/seed_42/` |
| Frozen final-model 6 h test evaluation | 完成；1,000 次 patient-clustered bootstrap | `outputs/final_test_evaluation_6h/` |
| 正式 6 h FNN 消融，4 variants x 3 seeds | 完成 | `outputs/fnn_ablation_6h_equal_sample/` |
| Frozen-model temporal fuzzy rule extraction | 完成；24 條 supported rules | `outputs/temporal_rule_extraction_6h/` |
| Rule Evaluation Framework 與 TP/FP/FN timelines | 完成；5 seeds | `outputs/rule_evaluation_6h/` |
| 成人 eligibility audit、cohort flow、Table 1–5、Figure 1–5 | 完成 | `outputs/manuscript_tables_figures_6h/` |
| 新版 explicit-FNN paired comparison、scale sensitivity、external supplementary analyses | 待完成 | `docs/experiment_status.md` |

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
| Rule Evaluation Framework | `rule_evaluation_framework.py` |
| Interpretable baselines | `interpretable_baselines.py` |
| Black-box baselines | `blackbox_baselines.py` |
| Clinical scores | `clinical_score_baselines.py`, `news2_score.py` |
| 公平比較與統一評估 | `run_fair_comparison.py`, `model_evaluation_report.py`, `advanced_model_evaluation.py` |
| 一次性 frozen final test evaluation | `final_test_evaluation.py` |
| 論文圖表 | `paper_figures.py` |
| Cohort flow、Table 1–5 與 Figure 1–5 | `cohort_tables_figures.py` |
| eICU audit、preprocessing 與 external validation | `eicu_data_audit.py`, `eicu_preprocessing.py`, `eicu_external_validation.py` |

## MIMIC 執行流程

所有命令由專案根目錄執行：

```powershell
# 1. Leakage-free SOFA 與 outcomes；成人定義固定為 age >= 18
.\env\Scripts\python.exe sofa_score.py --min-age 18

# 2. v3 hourly 與 explicit temporal input channels
.\env\Scripts\python.exe preprocessing.py --min-age 18

# 3. 固定 patient split 與公平比較 cohort
.\env\Scripts\python.exe patient_split.py --min-age 18
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
.\env\Scripts\python.exe eicu_external_validation.py --bootstrap-reps 500 --output-dir outputs\eicu_external_validation\final_frozen_model_evaluation
```

Final external test 包含 80,239 位病人、99,262 次 ICU stay、6,215,890 個視窗與 205 家醫院。Frozen final checkpoint 的 AUROC 為 0.6221（patient-clustered 95% CI 0.6192–0.6249），AUPRC 為 0.0922（0.0902–0.0942）；沒有使用 eICU outcome fitting 或 recalibration。

## 最新結果索引

- 文件總覽：`docs/README.md`
- Analysis plan：`docs/analysis_plan.md`
- 實驗完成度：`docs/experiment_status.md`
- Full-cohort training：`docs/full_cohort_training_6h.md`
- Frozen final test report：`outputs/final_test_evaluation_6h/final_test_report.md`
- 正式消融報告：`docs/formal_ablation_study_6h.md`
- 實際 temporal fuzzy rules：`docs/extracted_temporal_rules_6h.md`
- Rule Evaluation Framework：`docs/rule_evaluation_framework_6h.md`
- Final eICU external validation：`docs/eicu_final_external_validation_6h.md`
- Tuning 報告：`docs/explicit_temporal_fnn_tuning_6h.md`
- Observation sensitivity：`docs/explicit_temporal_observation_sensitivity.md`
- MIMIC 6 h baseline report：`docs/mimic_iv_6h_evaluation_report.md`
- 輸出索引：`outputs/README.md`

## 環境

```powershell
.\env\Scripts\python.exe -m pip install -r requirements.txt
```

## 論文寫作底稿

以下內容固定本研究目前正式版本的資料來源、cohort、predictors、outcome、實驗與主要結果。撰寫論文時應以此區與正式 CSV/JSON 為準，不要從圖檔手動抄數字。

### 1. 研究設計

- 研究類型：多中心 ICU retrospective cohort study。
- Development database：MIMIC-IV。
- External validation database：eICU-CRD。
- Primary outcome：目前時間點起未來 6 小時內，SOFA 相對目前增加至少 2 分。
- Secondary outcomes：未來 12 與 24 小時 SOFA increase >= 2。
- Primary observation window：過去 24 小時逐小時序列。
- Sensitivity observation windows：4、6、12、24 小時。
- 成人定義：ICU 入住時 `age >= 18`。
- 分析單位：每個 eligible ICU stay-hour prediction window；信賴區間以 `subject_id` clustered bootstrap 計算。

### 2. 資料庫數量總覽

| 統計單位 | MIMIC-IV development/internal | eICU-CRD external |
|---|---:|---:|
| 實際使用 raw tables | 7 | 8 |
| 實際讀取 raw rows | 608,690,476 | 399,686,522 |
| ICU source patients | 65,366 | 139,367 |
| ICU source stays | 94,458 | 200,859 |
| Adult valid-time patients | 65,355 | 138,868 |
| Adult valid-time stays | 94,444 | 200,232 |
| Harmonized hourly states | 8,275,274 | 12,994,585 |
| Valid 6 h SOFA labels | 6,938,122 | 7,821,053 |
| 24 h history prediction windows | 5,493,812 | 6,215,890 |
| Final evaluation patients | 7,287 internal-test patients | 80,239 external-test patients |
| Final evaluation stays | 9,894 internal-test stays | 99,262 external-test stays |
| Hospitals represented in final evaluation | 1 | 205 |

Raw rows、hourly states、valid labels 與 prediction windows 是不同分析層級，論文中不可把它們統稱為「樣本數」。模型效能的正式 N 應使用 prediction windows，cohort characteristics 則使用 patients 或 ICU stays。

### 3. MIMIC-IV 實際使用資料

MIMIC-IV 原始資料中，只有下列 7 張表進入 cohort、predictor 或 SOFA 建構。Raw rows 是本機 `.csv.gz` 實際資料列數，不含 header。

| 原始表 | Raw rows | 本研究用途 | 主要抽取內容 |
|---|---:|---|---|
| `patients.csv.gz` | 364,627 | 成人條件與 Table 1 | `anchor_age`、`anchor_year`、gender |
| `admissions.csv.gz` | 546,028 | Table 1 | race |
| `icustays.csv.gz` | 94,458 | ICU cohort 與逐小時時間軸 | `subject_id`、`hadm_id`、`stay_id`、`intime`、`outtime` |
| `chartevents.csv.gz` | 432,997,491 | 生理 predictors 與 SOFA | heart rate、respiratory rate、SpO2、BP、temperature、FiO2、GCS、ventilation |
| `labevents.csv.gz` | 158,374,764 | laboratory predictors 與 SOFA | lactate、PaO2、bilirubin、creatinine、platelets |
| `inputevents.csv.gz` | 10,953,713 | cardiovascular SOFA | dopamine、dobutamine、epinephrine、norepinephrine |
| `outputevents.csv.gz` | 5,359,395 | renal SOFA | urine output |
| **合計** | **608,690,476** |  |  |

`diagnoses_icd.csv.gz`、`d_icd_diagnoses.csv.gz` 與 `d_items.csv.gz` 沒有進入目前模型。診斷碼、年齡、性別與 race/ethnicity 都不是 13 個模型 predictors；人口學資料只用於 eligibility、Table 1 與後續 subgroup analysis。

### 4. MIMIC-IV cohort 數量

| 階段 | Patients | ICU stays / windows | 說明 |
|---|---:|---:|---|
| MIMIC ICU source | 65,366 | 94,458 stays | `icustays.csv.gz` 中的 unique subjects |
| 成人條件排除後 | 65,366 | 94,458 stays | 最小年齡已為 18；未成年排除數為 0 |
| 有效 ICU 時間軸 | 65,355 | 94,444 stays | 排除 14 個缺失或無效 ICU time stays |
| Hourly feature/SOFA table | 65,355 | 8,275,274 stay-hours | `model_hourly_features_v3.csv` |
| 有效 6 h SOFA labels | - | 6,938,122 stay-hours | Event prevalence 6.46% |
| 24 h history eligible model cohort | - | 5,493,812 windows | Train、validation、test 合計 |
| Train | 45,746 | 3,843,400 windows | 217,650 positives；5.66% |
| Validation | 9,807 | 819,573 windows | 47,638 positives；5.81% |
| Internal test | 9,802 | 830,839 windows | 47,292 positives；5.69% |
| Final-model test evaluation | 7,287 | 830,839 windows | 僅計算具有完整 eligible windows 的 test patients |

`patient_split.csv` 本身包含 45,757/9,807/9,802 位 train/validation/test subjects；Table 1 的 train 為 45,746 位，是因為另排除沒有有效 ICU 時間軸的 subjects。Adult-filtered split 重建與原 manifest byte-identical，assignment 差異為 0。

### 5. eICU-CRD 實際使用資料

eICU 只使用下列 8 張表進行 feature harmonization 與 SOFA label construction。

| 原始表 | Raw rows | 本研究用途 | Relevant/eligible rows 或主要訊號 |
|---|---:|---|---|
| `patient.csv.gz` | 200,859 | patient/stay/hospital cohort | age、gender、ethnicity、hospital、ICU offset |
| `vitalPeriodic.csv.gz` | 146,671,642 | periodic vital signs | 145,736,206 eligible rows |
| `vitalAperiodic.csv.gz` | 25,075,074 | aperiodic vital signs | 24,935,625 eligible rows |
| `lab.csv.gz` | 39,132,531 | laboratory predictors 與 SOFA | 1,901,407 relevant rows |
| `nurseCharting.csv.gz` | 151,604,232 | GCS 與 nursing vital signs | 52,204,521 relevant rows |
| `respiratoryCharting.csv.gz` | 20,168,176 | FiO2 與 respiratory support | 4,572,294 relevant rows |
| `infusionDrug.csv.gz` | 4,803,719 | cardiovascular SOFA | 776,364 pressor rows；104,395 可換算 positive rows |
| `intakeOutput.csv.gz` | 12,030,289 | renal SOFA | 2,970,314 urinary rows |
| **合計** | **399,686,522** |  |  |

升壓劑只保留可換算為 mcg/kg/min 的紀錄；尿量只使用 urinary output，排除 drain、chest-tube 與 `outputtotal`。所有時間以 ICU admission-relative offset 對齊，不使用未來量測補值。

### 6. eICU-CRD cohort 數量

| 階段 | Patients | ICU stays / windows | Hospitals | 說明 |
|---|---:|---:|---:|---|
| eICU source | 139,367 | 200,859 stays | 208 | `patient.csv.gz` |
| 成人且有效 duration | 138,868 | 200,232 stays | 208 | 排除 95 個缺失年齡 stays、530 個未成年 stays、2 個無效 duration stays |
| Harmonized hourly table | 138,868 | 12,994,585 stay-hours | 208 | MIMIC-compatible schema |
| 有效 6 h SOFA labels | - | 7,821,053 stay-hours | - | 421,089 positives；5.38% |
| Frozen external test | 80,239 | 6,215,890 windows；99,262 stays | 205 | 294,949 positives；4.75% |

eICU 不參與 hyperparameter tuning、checkpoint selection、probability recalibration 或 threshold selection。MIMIC 訓練完成的 checkpoint、Platt calibration 與 operating thresholds 原封不動套用至 eICU。

### 7. 最終 13 個模型 predictors

| Predictor | MIMIC-IV source | eICU source | Hourly aggregation/derivation |
|---|---|---|---|
| Heart rate | `chartevents` | `vitalPeriodic`、`nurseCharting` | mean |
| Respiratory rate | `chartevents` | `vitalPeriodic`、`nurseCharting` | mean |
| SpO2 | `chartevents` | `vitalPeriodic`、`nurseCharting` | minimum |
| FiO2 | `chartevents` | `respiratoryCharting` | maximum；統一為 0.21–1.00 |
| Temperature | `chartevents` | periodic/aperiodic/nursing | mean；Fahrenheit 轉 Celsius |
| Systolic BP | `chartevents` | periodic/aperiodic | arterial 優先，否則 non-invasive；minimum |
| GCS total | `chartevents` | `nurseCharting` | eye + verbal + motor；worst value |
| MAP | `chartevents` | periodic/aperiodic | arterial 優先，否則 non-invasive；minimum |
| PaO2/FiO2 | `labevents` + FiO2 | `lab` + respiratory | 同一小時 PaO2 / FiO2 |
| Platelets | `labevents` | `lab` | minimum |
| Bilirubin | `labevents` | `lab` | maximum |
| Creatinine | `labevents` | `lab` | maximum |
| Lactate | `labevents` | `lab` | maximum |

機械通氣、PaO2、升壓劑與尿量另外用於 SOFA component 計算，但不是最終 13 個 predictors。

### 8. Preprocessing 與 temporal features

1. 以每個 ICU `intime` 或 admission-relative offset 建立逐小時 grid。
2. 先套用臨床合理範圍；異常格式或不合理數值轉為 missing，不做 outcome-aware clipping。
3. 同一小時多筆紀錄依臨床 worst direction 或 mean 聚合。
4. 在任何補值前建立 `is_missing` 與 `time_since_last_measurement`。
5. 只在同一 `stay_id` 內 forward-fill；不 backward-fill，也不跨 stay 補值。
6. 建立 current、mean、min、max、standard deviation、short-term change、window change、slope、abnormal duration 與 abnormal measurement frequency。
7. Temporal descriptors 分別以 4/6/12/24 h 建立；primary model 使用完整 24 h sequence。
8. 訓練資料缺失值最後使用固定 clinical defaults；defaults 與 scaling 只依既定規則或 train data 決定。

Table 2 的 missingness 是 LOCF 前的 current-hour raw missingness，不是 forward-fill 後仍為空值的比例。

### 9. Outcome construction

- SOFA 依過去 24 小時 worst values 計算 respiration、coagulation、liver、cardiovascular、CNS 與 renal 六個 components。
- Primary SOFA 至少需要 4 個可觀測 components；另保留 assume-normal 與 complete-case sensitivity scores。
- 每個 index hour 的 label 只查看 index hour 之後的 6 小時 SOFA；若未來最大 SOFA 相對目前增加至少 2 分則標記為 1。
- ICU stay 尾端不足完整 6 小時、目前或未來 SOFA 無效的 rows 不建立 label。
- Predictor 僅使用目前與過去資料；future SOFA 只用於 outcome，不回流 predictors。

### 10. Split、訓練與評估原則

- `subject_id` patient-level 70%/15%/15% train/validation/test split；同一病人的所有 stays 不跨 split。
- Primary fair comparison：200,000 train windows、50,000 validation windows、完整 830,839 test windows。
- Full-cohort final FNN：3,843,400 train 與 819,573 validation windows。
- Hyperparameters、early stopping、checkpoint、Platt calibration 與 fixed-specificity thresholds 只能使用 train/validation。
- Frozen final checkpoint 只在 internal test 評估一次，SHA-256 為 `158427a5c358016f35b435b1ab5f75c7194a3ff3f9b6c9d68c5190a8a9125688`。
- AUROC、AUPRC、Brier、ECE、fixed-specificity sensitivity 與 paired comparisons 的 95% CI 以 patient-clustered bootstrap 計算。

### 11. 已完成模型與實驗

| 類別 | 模型/分析 |
|---|---|
| Clinical scores | NEWS2、SOFA |
| Interpretable baselines | Logistic Regression、Decision Tree、GAM、EBM |
| Black-box baselines | Random Forest、XGBoost、LightGBM、LSTM、GRU |
| Proposed model | Full Knowledge-Guided Temporal FNN |
| Ablation | Random initialization、static guideline FNN、temporal without consistency、full model |
| Sensitivity | 4/6/12/24 h observation windows |
| Interpretability | Rule extraction、complexity、stability、concordance、drift、activated rules、TP/FP/FN timelines |
| Validation | MIMIC independent test、eICU frozen external validation |

### 12. 主要結果底稿

| Analysis | AUROC | AUPRC | Brier | ECE |
|---|---:|---:|---:|---:|
| MIMIC frozen internal test | 0.6559 (0.6492–0.6628) | 0.1309 (0.1250–0.1375) | 0.0521 (0.0507–0.0534) | 0.0012 |
| eICU frozen external test | 0.6221 (0.6192–0.6249) | 0.0922 (0.0902–0.0942) | 0.0459 (0.0455–0.0463) | 0.0267 |

- MIMIC 90% specificity threshold：sensitivity 0.2667、PPV 0.1394、NPV 0.9532。
- MIMIC 95% specificity threshold：sensitivity 0.1755、PPV 0.1767、NPV 0.9503。
- 相同 thresholds 套用 eICU 後，observed specificity 降至 79.0% 與 88.2%，表示 operating point 具有 transportability gap。
- Ablation 顯示 temporal design 是主要效能來源，paired AUROC difference +0.0510；clinical consistency loss 未提升 AUROC。
- Rule evaluation：Top-10 mean antecedents 1.44、five-seed Jaccard 0.720、guideline-rubric concordance 1.000、median membership-center drift 0.264 initial sigmas。
- Equal-sample benchmark 中 GRU AUROC 0.6238，高於同條件下的 sequence-only FNN；不能宣稱 proposed framework 在所有公平比較中具有最高 discrimination。

### 13. 論文表圖配置

| 編號 | 內容 | 正式位置 |
|---|---|---|
| Figure 1 | MIMIC/eICU cohort selection flow | `outputs/manuscript_tables_figures_6h/figures/figure_1_cohort_flow.pdf` |
| Table 1 | MIMIC train/validation/test 與 eICU characteristics | `outputs/manuscript_tables_figures_6h/table_1_patient_characteristics.csv` |
| Table 2 | Pre-LOCF feature missingness | `outputs/manuscript_tables_figures_6h/table_2_feature_missingness.csv` |
| Table 3 | Model performance | `outputs/manuscript_tables_figures_6h/table_3_model_performance.csv` |
| Table 4 | Ablation study | `outputs/manuscript_tables_figures_6h/table_4_ablation_study.csv` |
| Table 5 | Rule evaluation | `outputs/manuscript_tables_figures_6h/table_5_rule_evaluation.csv` |
| Figure 2 | System architecture | `outputs/manuscript_tables_figures_6h/figures/figure_2_system_architecture.pdf` |
| Figure 3 | Internal/external calibration | `outputs/manuscript_tables_figures_6h/figures/figure_3_calibration_curve.pdf` |
| Figure 4 | Decision curve analysis | `outputs/manuscript_tables_figures_6h/figures/figure_4_decision_curve_analysis.pdf` |
| Figure 5 | TP/FP/FN timelines and activated rules | `outputs/manuscript_tables_figures_6h/figures/figure_5_patient_timeline_activated_rules.pdf` |

完整 data-source inventory：`outputs/manuscript_tables_figures_6h/data_source_inventory.csv`。完整成人 cohort 與 Table 1–5 Markdown：`docs/adult_cohort_manuscript_artifacts.md`。

### 14. 可直接改寫進論文的方法段落

本研究使用 MIMIC-IV 建立成人 ICU 動態惡化預測 cohort，並以 eICU-CRD 進行 frozen external validation。納入 ICU 入住時年齡至少 18 歲的病人，以 ICU stay 為時間對齊邊界建立逐小時資料。生理訊號、實驗室檢驗、呼吸支持、升壓劑及尿量資料分別由兩資料庫中語意相對應的資料表抽取，經單位一致化、臨床合理範圍檢查與逐小時聚合後，建立 13 個共同 predictors。缺失狀態在補值前記錄，後續僅於同一 ICU stay 內使用 last observation carried forward，不使用 backward filling。

主要預測目標定義為未來 6 小時內 SOFA 分數相較目前增加至少 2 分。SOFA 依過去 24 小時六個器官系統的 worst values 計算，主要標籤要求至少 4 個可觀測 components，且 ICU stay 尾端不足完整預測 horizon 的時間點不納入。資料以 `subject_id` 分為 mutually exclusive train、validation 與 test cohorts，所有模型選擇、校正與 operating thresholds 僅使用 train/validation data；test set 僅供 frozen final-model evaluation。

Knowledge-Guided Temporal FNN 使用 NEWS2/SOFA-guided membership initialization、additive 與 cross-feature fuzzy rules、明確 temporal descriptors 及 temporal attention，並以 clinical consistency、sparsity、membership drift 與 non-negativity constraints 進行正規化。最終模型在 MIMIC-IV independent test set 進行 patient-clustered bootstrap 評估，完成後不重新訓練、不重新校正地套用至 eICU-CRD，以檢驗跨資料庫 transportability。

### 15. 撰寫時必須保留的限制

- Full-cohort FNN 與 equal-sample baselines 的訓練樣本數不同，不能直接宣稱 full-cohort FNN 優於所有 baselines。
- Clinical concordance 1.000 是與預先定義 guideline rubric 的一致性，不是獨立臨床專家盲審。
- Consistency regularization 未改善 AUROC；目前只能定位為可能提升規則穩定性的 interpretability regularizer。
- Positive 與 negative windows 的平均 activated rule 數接近，解釋時需同時考慮 rule type、activation strength、weight 與時間軸，而非只看規則數量。
- eICU calibration 與 fixed-specificity thresholds 明顯衰退，部署前需要 local validation；primary external result不可用 eICU recalibration 修飾。
- Hourly windows 並非獨立樣本，所有 CI 與模型比較必須以病人為 cluster。
