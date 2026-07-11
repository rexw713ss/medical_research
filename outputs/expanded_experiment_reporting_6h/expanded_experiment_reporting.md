# Expanded Experiment Reporting (Primary 6-hour Task)

This report consolidates frozen analyses. It does not refit the model, calibration, or operating thresholds.

## Analytic Denominators

| database | analysis_set | patients | stays | windows | positive_windows | prevalence |
| --- | --- | --- | --- | --- | --- | --- |
| MIMIC-IV | train | 34185 | 46214 | 3843400 | 217650 | 0.0566 |
| MIMIC-IV | validation | 7301 | 9870 | 819573 | 47638 | 0.0581 |
| MIMIC-IV | test | 7287 | 9894 | 830839 | 47292 | 0.0569 |

Counts are mutually exclusive by `subject_id`. Final windows require a non-missing 6-hour outcome and 24 hourly observations ending at the index hour.

## Sequential Cohort Accounting

| database | stage_or_exclusion | patients | stays | stay_hours_or_windows |
| --- | --- | --- | --- | --- |
| MIMIC-IV | Source ICU cohort | 65366.0 | 94458.0 | - |
| MIMIC-IV | Excluded: invalid ICU time | - | 14.0 | - |
| MIMIC-IV | Excluded: missing age | - | 0.0 | - |
| MIMIC-IV | Excluded: age <18 | 0.0 | 0.0 | - |
| MIMIC-IV | Adult valid ICU cohort | 65355.0 | 94444.0 | 8275274.0 |
| MIMIC-IV | Excluded stay-hours: invalid/incomplete 6-h SOFA outcome | - | - | 1337152.0 |
| MIMIC-IV | Valid 6-h outcome hours | - | - | 6938122.0 |
| MIMIC-IV | Excluded valid-label hours: insufficient 24-h history | - | - | 1444310.0 |
| MIMIC-IV | Final analytic windows | 48773.0 | 65978.0 | 5493812.0 |
| eICU-CRD | Source ICU cohort | 139367.0 | 200859.0 | - |
| eICU-CRD | Excluded: missing age | - | 95.0 | - |
| eICU-CRD | Excluded: age <18 | 437.0 | 530.0 | - |
| eICU-CRD | Excluded: invalid ICU duration | - | 2.0 | - |
| eICU-CRD | Adult valid ICU cohort | 138868.0 | 200232.0 | 12994585.0 |
| eICU-CRD | Excluded stay-hours: invalid/incomplete 6-h SOFA outcome | - | - | 5173532.0 |
| eICU-CRD | Valid 6-h outcome hours | - | - | 7821053.0 |
| eICU-CRD | Excluded valid-label hours: insufficient 24-h history | - | - | 1605163.0 |
| eICU-CRD | Final external analytic windows | 80239.0 | 99262.0 | 6215890.0 |

Patient/stay exclusions and stay-hour exclusions are separate denominators and must not be subtracted from one another. The final MIMIC patient total is the sum of mutually exclusive split-specific patients; eICU is a single frozen external cohort.

## SOFA Construction and Harmonization

At each index hour, SOFA uses the worst value in the current and preceding 23 hours. The six standard components are scored 0-4 and summed. The primary score requires at least four observed components; missing-as-normal and all-six-component complete-case definitions are sensitivity analyses. FiO2 may be carried forward within the stay for SOFA scoring, but no future value or backward fill is used.

| component | 24h_worst_value | mimic_iv_source | eicu_source | harmonization_rule |
| --- | --- | --- | --- | --- |
| Respiratory | Worst P/F ratio; invasive support status | PaO2 lab + charted FiO2 + ventilation | PaO2 lab + respiratoryCharting FiO2/support | Same thresholds; database-specific source mapping |
| Coagulation | Minimum platelet count | Platelet lab | Platelet lab (10^3/uL harmonized) | Same thresholds |
| Liver | Maximum total bilirubin | Total bilirubin lab | Total bilirubin lab | Same thresholds |
| Cardiovascular | Minimum MAP and maximum vasopressor dose | MAP + inputevents pressors | MAP + infusionDrug pressors | Rates normalized to mcg/kg/min; unconvertible eICU doses excluded |
| Central nervous system | Minimum total GCS | Total GCS or eye+verbal+motor | nurseCharting total GCS | Same thresholds |
| Renal | Maximum creatinine and 24-h urine sum; worse score | Creatinine lab + outputevents urine | Creatinine lab + intakeOutput urine | Urine criterion available only after a complete trailing 24 h |

The outcome is positive when the maximum valid SOFA in `(t, t+6]` is at least two points above SOFA at `t`. A complete six-hour future horizon is required. The same scoring and label functions are called after database-specific source and unit mapping.

## Raw and Calibrated Calibration

| cohort | probability_scale | brier | ece | log_loss | calibration_intercept | calibration_slope |
| --- | --- | --- | --- | --- | --- | --- |
| MIMIC-IV test | Raw model output | 0.196245 | 0.37 | 0.582572 | -2.627532 | 1.06327 |
| MIMIC-IV test | Validation-only Platt calibrated | 0.052077 | 0.001241 | 0.208027 | -0.020024 | 0.99739 |
| eICU external | Raw model output | 0.244426 | 0.433987 | 0.686635 | -3.040978 | 0.780318 |
| eICU external | Transported MIMIC calibration | 0.045936 | 0.026685 | 0.192364 | -1.127368 | 0.731969 |

The raw output is retained because class-weighted training does not target calibrated absolute risk. Platt parameters are fitted on MIMIC validation predictions only, then frozen for MIMIC test and transported unchanged to eICU. External calibration is therefore a transport result, not eICU recalibration.

## Raw Rule Firing

Raw cross-rule firing is the product t-norm before normalization. A rule is counted as activated when raw firing is at least the prespecified threshold. Results are reported for the current index hour and the attention-selected hour across thresholds 0.01, 0.025, 0.05, 0.10, 0.20, 0.35, and 0.50.

| basis | threshold | outcome | windows | mean_activated_rules | median_activated_rules | windows_with_any_rule | fraction_with_any_rule |
| --- | --- | --- | --- | --- | --- | --- | --- |
| current_hour | 0.1 | 0 | 783547 | 0.529 | 0.0 | 350777 | 0.4477 |
| current_hour | 0.1 | 1 | 47292 | 0.5685 | 0.0 | 22284 | 0.4712 |

Rule agreement is named **Guideline-direction alignment**. It measures consistency with prespecified NEWS2/SOFA directions and is not clinician adjudication or clinician-validated interpretability.

## Event-Level Alarm Definition

- Event: the first hour in each ICU stay at which SOFA reaches at least the index-hour SOFA plus 2 within the next 6 hours. Only the first qualifying event per stay is evaluated.
- Alert: calibrated risk at or above a threshold selected on MIMIC validation data to target 90% or 95% window-level specificity.
- Refractory period: after a retained alert, further alerts in the same stay are suppressed for 6 hours.
- Detection: at least one retained alert in `[event hour - 6, event hour)`. Lead time is measured from the earliest matching retained alert.
- Burden denominator: pre-event analytic stay-hours divided by 24; alerts after the first event are excluded. A false alert is a retained alert outside a qualifying pre-event window.

| target_specificity | threshold | refractory_hours | first_events | detected_events | event_sensitivity | alerts | true_event_alerts | false_alerts | alert_ppv | patient_days_observed | alerts_per_100_patient_days | false_alerts_per_100_patient_days | patients_with_false_alert | patients_with_false_alert_fraction | lead_time_median_h | lead_time_q1_h | lead_time_q3_h |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.9 | 0.0924 | 6 | 3710 | 1274 | 0.3434 | 8151 | 1274 | 6877 | 0.1563 | 14258.1667 | 57.1672 | 48.232 | 2652 | 0.3639 | 3.0 | 1.0 | 4.0 |
| 0.95 | 0.1164 | 6 | 3710 | 883 | 0.238 | 4348 | 883 | 3465 | 0.2031 | 14258.1667 | 30.4948 | 24.3019 | 1698 | 0.233 | 2.0 | 1.0 | 4.0 |

Window-level specificity and event sensitivity are distinct estimands and must not be described interchangeably.

## Site-Level eICU Analysis

The pooled frozen external analysis includes 205 hospitals. Per-hospital estimates are reported only for hospitals with at least 100 patients, 50 positive windows, and 50 negative windows. Hospital-clustered uncertainty uses 500 bootstrap replicates that resample `hospital_id`; this is a sensitivity analysis and does not replace the prespecified patient-clustered primary external confidence intervals.

| metric | eligible_hospitals | macro_mean | median | q1 | q3 | minimum | maximum |
| --- | --- | --- | --- | --- | --- | --- | --- |
| auroc | 142 | 0.6224 | 0.6216 | 0.5915 | 0.6524 | 0.4919 | 0.7657 |
| auprc | 142 | 0.088 | 0.087 | 0.0719 | 0.1078 | 0.0107 | 0.1667 |
| brier | 142 | 0.0434 | 0.0437 | 0.0352 | 0.053 | 0.0094 | 0.0801 |
| ece | 142 | 0.0327 | 0.0322 | 0.0198 | 0.0436 | 0.003 | 0.0805 |
| prevalence | 142 | 0.0441 | 0.0438 | 0.0338 | 0.0555 | 0.0045 | 0.0868 |

No eICU outcome was used for retraining, checkpoint selection, calibration fitting, or threshold selection.
