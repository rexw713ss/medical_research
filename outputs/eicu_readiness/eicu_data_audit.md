# eICU 外部驗證資料就緒稽核

產生時間：2026-06-30T15:46:28.259027+00:00

整體狀態：**READY**

## Cohort

- ICU stays：200,859
- Unique patients：139,367
- Adult stays with at least 24 h：132,611

## Predictor 與 outcome 來源

| 訊號 | 狀態 | 來源 | Sample matched rows |
|---|---:|---|---:|
| `heart_rate` | OK | vitalPeriodic.csv.gz:heartrate | 0 |
| `respiratory_rate` | OK | vitalPeriodic.csv.gz:respiration | 0 |
| `spo2` | OK | vitalPeriodic.csv.gz:sao2 | 0 |
| `temperature_c` | OK | vitalPeriodic.csv.gz:temperature | 0 |
| `sbp` | OK | vitalAperiodic.csv.gz:noninvasivesystolic | 0 |
| `map` | OK | vitalAperiodic.csv.gz:noninvasivemean | 0 |
| `pao2` | OK | lab.csv.gz:labname | 24,213 |
| `platelets` | OK | lab.csv.gz:labname | 59,563 |
| `bilirubin` | OK | lab.csv.gz:labname | 22,188 |
| `creatinine` | OK | lab.csv.gz:labname | 61,408 |
| `lactate` | OK | lab.csv.gz:labname | 5,226 |
| `gcs_total` | OK | nurseCharting.csv.gz:GCS labels | 47,044 |
| `fio2` | OK | respiratoryCharting.csv.gz:FiO2 labels | 317,245 |
| `pao2_fio2` | OK | derived from PaO2 / FiO2 | 24,213 |
| `urine_output` | OK | intakeOutput.csv.gz:urinary output labels | 550,876 |
| `dopamine` | OK | infusionDrug.csv.gz:drugname | 24,308 |
| `dobutamine` | OK | infusionDrug.csv.gz:drugname | 25,045 |
| `epinephrine` | OK | infusionDrug.csv.gz:drugname | 21,305 |
| `norepinephrine` | OK | infusionDrug.csv.gz:drugname | 239,467 |

## SOFA components

| Component | 狀態 |
|---|---:|
| respiration | OK |
| coagulation | OK |
| liver | OK |
| cardiovascular | OK |
| cns | OK |
| renal | OK |

## 外部驗證約束

- eICU 使用 `uniquepid` 作為 patient-level identifier、`patientunitstayid` 作為 stay identifier。
- 所有事件時間以 ICU admission-relative offset 對齊至整點，不使用未來量測補值。
- MIMIC-IV 訓練完成的模型與 calibration 必須原封不動套用至 eICU；不得以 eICU test outcome 重新調參。
- 升壓藥只採可換算為 mcg/kg/min 的紀錄；無法確認濃度的 mL/hr 不可直接當劑量。
- 尿量必須使用逐筆 urinary output 欄位，排除 `outputtotal` 與其他 drain/chest-tube output。
- 此稽核證明資料來源齊全，不代表 MIMIC/eICU 單位與定義已完成 harmonization。

Schema fingerprint：`bd0e788684f7136ffd3ff290bc0a81127cc779716d51c08137d80bc50039f4cf`
