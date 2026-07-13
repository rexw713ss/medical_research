# Supplementary Material: SOFA and Knowledge Mapping

## Supplementary Table S1. SOFA Reconstruction

| Organ system | Clinical variable | Score 0 | Score 1 | Score 2 | Score 3 | Score 4 | Hourly / 24-h aggregation | MIMIC-IV source | eICU source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Respiratory | PaO2/FiO2 + respiratory support | >=400 | <400 | <300 | <200 + support | <100 + support | Hourly minimum P/F and maximum support; trailing 24-h minimum P/F and maximum support | labevents PaO2 50821; chartevents FiO2 223835 and ventilation 223848 / 223849 / 229314 / 225792 / 225794 | lab PaO2; respiratoryCharting FiO2, PEEP, PEEP/CPAP, RT vent on/off |
| Coagulation | Platelets, x10^3/uL | >=150 | 100-149 | 50-99 | 20-49 | <20 | Hourly minimum; trailing 24-h minimum | labevents itemid 51265 | lab name 'platelets x 1000' |
| Liver | Total bilirubin, mg/dL | <1.2 | 1.2-1.9 | 2.0-5.9 | 6.0-11.9 | >=12.0 | Hourly maximum; trailing 24-h maximum | labevents itemid 50885 | lab name 'total bilirubin' |
| Cardiovascular | MAP / vasopressors, mcg/kg/min | MAP >=70 and no qualifying pressor | MAP <70 | dopamine <=5 or any dobutamine | dopamine >5 to 15 or epinephrine/norepinephrine <=0.1 | dopamine >15 or epinephrine/norepinephrine >0.1 | Hourly minimum MAP and maximum active dose; trailing 24-h minimum MAP and maximum dose; take worst criterion | chartevents MAP 220052/220181; inputevents dopamine 221662, dobutamine 221653, epinephrine 221289/229617, norepinephrine 221906 | periodic/aperiodic MAP; infusionDrug dopamine/dobutamine/epinephrine/norepinephrine |
| Neurological | GCS total | 15 | 13-14 | 10-12 | 6-9 | <6 | Hourly minimum total GCS; trailing 24-h minimum | chartevents eye 220739 + verbal 223900 + motor 223901 | nurseCharting GCS total |
| Renal | Creatinine, mg/dL / urine output, mL/24 h | creatinine <1.2 / urine >=500; worse observed criterion | creatinine 1.2-1.9 | creatinine 2.0-3.4 | creatinine 3.5-4.9 or urine <500 | creatinine >=5.0 or urine <200 | Hourly maximum creatinine and sum of all qualifying urine records; trailing 24-h maximum creatinine and urine sum; take higher score | labevents creatinine 50912; outputevents urinary itemids 226557-226567, 226584, 226627, 226631, 227489 | lab name 'creatinine'; intakeOutput urinary labels |

### S1 implementation notes

1. Events are aligned to ICU-admission-relative integer hours. Multiple records in the same hour use the clinically worse direction shown in S1; urine records are summed and vasopressor doses use the maximum active dose.
2. Every component uses the worst value in the current and preceding 23 hours. Renal urine output is therefore a 24-hour sum, not a 6-hour or 12-hour sum. Urine criteria are not used before 24 complete ICU hours.
3. Vasopressor rates are converted to mcg/kg/min. Weight-normalized rates are used directly; mg is converted to mcg, hourly rates to per-minute rates, and non-weight-normalized rates are divided by a valid recorded body weight. Unconvertible or implausible rates are excluded.
4. MIMIC respiratory support is positive for non-empty active invasive or non-invasive ventilation chart values after excluding off/standby/none states. eICU support is positive for PEEP or PEEP/CPAP >0 or RT vent states 'start'/'continued'. This is an operational respiratory-support indicator, not a perfectly harmonized invasive-ventilation phenotype.
5. FiO2 is normalized to a fraction and forward-filled within the ICU stay without backward fill. Missing respiratory-support records are treated as no documented support.
6. Creatinine and urine output are scored independently and the higher renal score is retained.
7. MIMIC total GCS is eye + verbal + motor when all components are present; eICU uses charted total GCS. No sedation correction, pre-sedation substitution, or special imputation of an intubated verbal component is performed. The resulting CNS score may reflect sedation and intubation documentation.
8. The primary SOFA is available when at least four of six components are observed. Missing components are assigned zero in that primary sum after the four-component criterion is met. Sensitivity analyses use missing-as-normal regardless of component count and a strict six-component complete-case score.
9. The index and future SOFA scores are reconstructed independently and may have different observed component sets. A positive primary label requires max SOFA in `(t,t+6]` minus SOFA at `t` >=2 and a complete six-hour future horizon.

## Supplementary Table S2. NEWS2-to-Fuzzy Initialization

| Variable | NEWS2 interval / operational proxy | NEWS2 points | KG-TFNN fuzzy term | Initial center | Initial width (sigma) | Initial risk weight |
| --- | --- | --- | --- | --- | --- | --- |
| Heart rate, beats/min | <=40 | 3 | very_low | 35.0 | 6.0 | 3.0 |
| Heart rate, beats/min | 41-50 | 1 | overlap: very_low + normal; no dedicated term | 35 / 70 | 6 / 15 | 3 / 0 |
| Heart rate, beats/min | 51-90 | 0 | normal | 70.0 | 15.0 | 0.0 |
| Heart rate, beats/min | 91-110 | 1 | mild_high | 100.0 | 8.0 | 1.0 |
| Heart rate, beats/min | 111-130 | 2 | high | 120.0 | 8.0 | 2.0 |
| Heart rate, beats/min | >=131 | 3 | critical_high | 140.0 | 10.0 | 3.0 |
| Respiratory rate, breaths/min | <=8 | 3 | very_low | 7.0 | 2.0 | 3.0 |
| Respiratory rate, breaths/min | 9-11 | 1 | mild_low | 10.0 | 1.5 | 1.0 |
| Respiratory rate, breaths/min | 12-20 | 0 | normal | 16.0 | 4.0 | 0.0 |
| Respiratory rate, breaths/min | 21-24 | 2 | high | 22.5 | 2.0 | 2.0 |
| Respiratory rate, breaths/min | >=25 | 3 | critical_high | 28.0 | 3.0 | 3.0 |
| SpO2, % | <=91 (Scale 1) | 3 | critical_low | 88.0 | 2.0 | 3.0 |
| SpO2, % | 92-93 (Scale 1) | 2 | low | 92.5 | 1.2 | 2.0 |
| SpO2, % | 94-95 (Scale 1) | 1 | mild_low | 94.5 | 1.0 | 1.0 |
| SpO2, % | >=96 (Scale 1) | 0 | normal | 97.0 | 2.5 | 0.0 |
| Supplemental oxygen / FiO2 fraction | FiO2 <=0.21 proxy | 0 | room_air | 0.21 | 0.03 | 0.0 |
| Supplemental oxygen / FiO2 fraction | FiO2 >0.21 proxy | 2 | supplemental_o2 | 0.4 | 0.12 | 2.0 |
| Supplemental oxygen / FiO2 fraction | FiO2 >0.21; higher support | 2 | high_support | 0.8 | 0.18 | 2.0 |
| Temperature, C | <=35.0 | 3 | very_low | 34.5 | 0.6 | 3.0 |
| Temperature, C | 35.1-36.0 | 1 | mild_low | 35.6 | 0.5 | 1.0 |
| Temperature, C | 36.1-38.0 | 0 | normal | 37.0 | 0.8 | 0.0 |
| Temperature, C | 38.1-39.0 | 1 | fever | 38.5 | 0.5 | 1.0 |
| Temperature, C | >=39.1 | 2 | high_fever | 39.5 | 0.6 | 2.0 |
| Systolic blood pressure, mmHg | <=90 | 3 | very_low | 85.0 | 8.0 | 3.0 |
| Systolic blood pressure, mmHg | 91-100 | 2 | low | 95.0 | 5.0 | 2.0 |
| Systolic blood pressure, mmHg | 101-110 | 1 | mild_low | 105.0 | 5.0 | 1.0 |
| Systolic blood pressure, mmHg | 111-219 | 0 | normal | 130.0 | 18.0 | 0.0 |
| Systolic blood pressure, mmHg | >=220 | 3 | very_high | 225.0 | 15.0 | 3.0 |
| Consciousness (GCS proxy) | <15 proxy for C/V/P/U | 3 | severely_altered | 5.0 | 2.0 | 4.0 |
| Consciousness (GCS proxy) | <15 proxy for C/V/P/U | 3 | altered | 12.0 | 2.0 | 3.0 |
| Consciousness (GCS proxy) | 15 proxy for Alert | 0 | normal | 15.0 | 0.8 | 0.0 |

### S2 implementation notes

1. The comparator implements NEWS2 (2017) SpO2 Scale 1. Scale 2 is not used because chronic hypercapnic respiratory failure cannot be identified reliably from the harmonized variables.
2. Supplemental oxygen is operationalized as charted FiO2 >21% and receives two points. This proxy may miss low-flow oxygen when a reliable FiO2 above room air is not charted.
3. NEWS2 ACVPU is approximated as GCS 15 = alert and GCS <15 = altered (three points). New confusion cannot be separated reliably, and sedation/intubation correction is not performed.
4. The baseline NEWS2 comparator uses the standard Scale 1 intervals after within-stay forward fill. A still-missing channel contributes no points, so it is not a strictly complete-case bedside NEWS2 implementation.
5. Gaussian centers are placed near clinically meaningful interval centers or representative abnormal values; sigma controls overlap and is not the interval width. Initial risk weights usually follow NEWS2 points but may encode SOFA-like severity for non-NEWS2 organ variables and severe GCS states.
6. NEWS2 heart rate 41-50 has no dedicated fuzzy term in the frozen model and is represented by overlap between the very-low and normal Gaussians. This is reported as an implementation limitation rather than retrospectively changing the checkpoint.
7. Centers, sigmas, and weights are trainable. Their initial values document how knowledge enters the model; they are not fixed scoring cutoffs after training.

## Supplementary Table S3. Complete equal-sample baseline performance on the independent MIMIC-IV test set.

| Model | AUROC (95% CI) | AUPRC (95% CI) | Brier (95% CI) | ECE |
| --- | --- | --- | --- | --- |
| Explicit KG-TFNN | 0.6448 (0.6379-0.6515) | 0.1236 (0.1177-0.1297) | 0.05229 (0.05094-0.05365) | 0.0013 |
| Feature-matched GRU | 0.6587 (0.6528-0.6653) | 0.1272 (0.1218-0.1328) | 0.05225 (0.05087-0.05359) | 0.0016 |
| Feature-matched XGBoost | 0.6870 (0.6808-0.6933) | 0.1665 (0.1598-0.1739) | 0.05104 (0.04972-0.05235) | 0.0015 |
| Feature-matched LightGBM | 0.6904 (0.6844-0.6968) | 0.1710 (0.1643-0.1790) | 0.05095 (0.04965-0.05227) | 0.0025 |
| EBM | 0.6072 (0.6008-0.6141) | 0.0891 (0.0853-0.0929) | 0.05316 (0.05175-0.05458) | 0.0012 |
| Logistic regression | 0.5795 (0.5718-0.5872) | 0.0794 (0.0760-0.0827) | 0.05341 (0.05199-0.05481) | 0.0013 |
| XGBoost (legacy features) | 0.6073 (0.6007-0.6141) | 0.0896 (0.0859-0.0934) | 0.05315 (0.05175-0.05456) | 0.0013 |

## Supplementary Table S4. Validation-derived operating points transported without test-set threshold selection.

| Cohort | Target specificity | Threshold | Observed specificity (95% CI) | Sensitivity (95% CI) | PPV (95% CI) | NPV (95% CI) | F1 (95% CI) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MIMIC-IV internal | 0.90 | 0.0924 | 0.901 (0.895-0.906) | 0.267 (0.253-0.281) | 0.139 (0.134-0.145) | 0.953 (0.952-0.955) | 0.183 (0.177-0.190) |
| MIMIC-IV internal | 0.95 | 0.1164 | 0.951 (0.947-0.954) | 0.175 (0.165-0.188) | 0.177 (0.169-0.185) | 0.950 (0.949-0.952) | 0.176 (0.168-0.185) |
| eICU external | 0.90 | 0.0924 | 0.790 (0.787-0.793) | 0.377 (0.372-0.383) | 0.082 (0.081-0.083) | 0.962 (0.962-0.963) | 0.135 (0.133-0.136) |
| eICU external | 0.95 | 0.1164 | 0.882 (0.880-0.884) | 0.262 (0.257-0.267) | 0.100 (0.098-0.101) | 0.960 (0.960-0.960) | 0.144 (0.142-0.147) |

## Supplementary Table S5. Event-level alarm burden after a six-hour refractory period.

| Target specificity | First evaluable events | Detected, n (%) | Alerts | Alert PPV | False alerts / 100 pre-event days | Lead time, median [IQR], h |
| --- | --- | --- | --- | --- | --- | --- |
| 0.90 | 3,710 | 1,274 (34.3%) | 8,151 | 0.1563 | 48.23 | 3 [1-4] |
| 0.95 | 3,710 | 883 (23.8%) | 4,348 | 0.2031 | 24.30 | 2 [1-4] |

## Supplementary Table S6. Hospital-level heterogeneity in eICU-CRD.

| Metric | Eligible hospitals | Median [IQR] | Range |
| --- | --- | --- | --- |
| AUROC | 142 | 0.6216 [0.5915-0.6524] | 0.4919-0.7657 |
| AUPRC | 142 | 0.0870 [0.0719-0.1078] | 0.0107-0.1667 |
| BRIER | 142 | 0.0437 [0.0352-0.0530] | 0.0094-0.0801 |
| ECE | 142 | 0.0322 [0.0198-0.0436] | 0.0030-0.0805 |
| PREVALENCE | 142 | 0.0438 [0.0338-0.0555] | 0.0045-0.0868 |

## Supplementary Table S7. Rule evaluation framework results.

| Analysis | Estimate | Interpretation |
| --- | --- | --- |
| Top-10 complexity | 1.50 antecedents/rule | Mean antecedent count among ranked rules |
| Five-seed top-10 stability | mean Jaccard 0.720 | All pairwise seed comparisons |
| Normalized center drift | mean 1.421 | Absolute center shift / initial sigma |
| Relative width drift | mean 0.658 | Absolute relative sigma change |
| Activated rules, negative windows | mean 1.819 | Frozen predefined activation rule |
| Activated rules, positive windows | mean 1.833 | Frozen predefined activation rule |

## Supplementary Table S8. Raw rule-firing activation-threshold sensitivity.

| Basis | Threshold | Outcome | Mean activated rules | Windows with any rule (%) |
| --- | --- | --- | --- | --- |
| current_hour | 0.01 | Negative | 1.395 | 87.0 |
| current_hour | 0.01 | Positive | 1.487 | 88.3 |
| current_hour | 0.05 | Negative | 0.806 | 62.2 |
| current_hour | 0.05 | Positive | 0.860 | 64.4 |
| current_hour | 0.10 | Negative | 0.529 | 44.8 |
| current_hour | 0.10 | Positive | 0.569 | 47.1 |
| current_hour | 0.20 | Negative | 0.151 | 14.3 |
| current_hour | 0.20 | Positive | 0.173 | 16.3 |
| current_hour | 0.35 | Negative | 0.011 | 1.1 |
| current_hour | 0.35 | Positive | 0.021 | 2.1 |
| current_hour | 0.50 | Negative | 0.003 | 0.3 |
| current_hour | 0.50 | Positive | 0.007 | 0.6 |
| attention_selected_hour | 0.01 | Negative | 1.663 | 87.8 |
| attention_selected_hour | 0.01 | Positive | 1.743 | 89.3 |
| attention_selected_hour | 0.05 | Negative | 1.065 | 69.9 |
| attention_selected_hour | 0.05 | Positive | 1.128 | 72.7 |
| attention_selected_hour | 0.10 | Negative | 0.764 | 56.7 |
| attention_selected_hour | 0.10 | Positive | 0.807 | 59.2 |
| attention_selected_hour | 0.20 | Negative | 0.291 | 25.7 |
| attention_selected_hour | 0.20 | Positive | 0.321 | 28.3 |
| attention_selected_hour | 0.35 | Negative | 0.071 | 6.9 |
| attention_selected_hour | 0.35 | Positive | 0.097 | 9.4 |
| attention_selected_hour | 0.50 | Negative | 0.024 | 2.4 |
| attention_selected_hour | 0.50 | Positive | 0.035 | 3.5 |

## Supplementary Table S9. Patient-specific current-hour cross-rule firing for the prespecified TP, FP, and FN cases.

| Case | Rank | Current-hour rule | Raw firing | Weight | Active at 0.10 |
| --- | --- | --- | --- | --- | --- |
| TP | 1 | IF pao2_fio2 IS very_low AND fio2 IS high_support THEN deterioration risk increases | 0.1868 | 0.060 | Yes |
| TP | 2 | IF lactate IS high AND map IS low THEN deterioration risk increases | 0.0155 | 0.316 | No |
| TP | 3 | IF creatinine IS high AND platelets IS low AND bilirubin IS high THEN deterioration risk increases | 0.0003 | 0.561 | No |
| FP | 1 | IF gcs_total IS altered AND spo2 IS low THEN deterioration risk increases | 0.0631 | 1.835 | No |
| FP | 2 | IF pao2_fio2 IS very_low AND fio2 IS high_support THEN deterioration risk increases | 0.2159 | 0.060 | Yes |
| FP | 3 | IF lactate IS high AND map IS low THEN deterioration risk increases | 0.0049 | 0.316 | No |
| FN | 1 | IF gcs_total IS altered AND spo2 IS low THEN deterioration risk increases | 0.0088 | 1.835 | No |
| FN | 2 | IF pao2_fio2 IS very_low AND fio2 IS high_support THEN deterioration risk increases | 0.1544 | 0.060 | Yes |
| FN | 3 | IF lactate IS high AND map IS low THEN deterioration risk increases | 0.0036 | 0.316 | No |

## Supplementary Table S10. Exploratory structural explanation-quality comparison on the prespecified 1,000-case sample.

| Model | AUROC / AUPRC | Stability cosine / top-5 | Neighbor consistency cosine / top-5 | Features for 80% mass, median [IQR] | Temporal mass | Cross-database rho / top-5 | Explanation output |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM + TreeSHAP | 0.6904 / 0.1710 | 0.965 / 0.776 | 0.408 / 0.407 | 6 [5-7] | 0.865 | 0.797 / 0.667 | Signed feature attributions |
| XGBoost + TreeSHAP | 0.6870 / 0.1665 | 0.944 / 0.733 | 0.467 / 0.421 | 7 [6-7] | 0.845 | 0.879 / 0.667 | Signed feature attributions |
| EBM (current state) | 0.6072 / 0.0891 | 0.982 / 0.905 | 0.484 / 0.482 | 6 [6-7] | 0.000 | 0.846 / 0.667 | Additive feature/interaction terms |
| KG-TFNN | 0.6448 / 0.1236 | 1.000 / 0.997 | 0.979 / 0.746 | 5 [5-5] | 0.188 | 0.978 / 0.667 | Temporal IF-THEN rules |

## Supplementary Table S11. Exploratory clinical-consistency behavior on the 1,000-case sample (mean across three seeds).

| Variant | Violation given worsening | Consistency penalty | Risk reversal | Reversal magnitude | Rule stability | Normalized drift | Guideline-risk rho | Direction alignment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Temporal FNN without consistency loss | 0.370 | 0.000411 | 0.104 | 0.107 | 0.587 | 0.222 | 0.536 | 1.000 |
| Full KG-TFNN | 0.370 | 0.000359 | 0.106 | 0.090 | 0.674 | 0.224 | 0.528 | 1.000 |

## Supplementary Table S12. SOFA outcome-definition and documentation-availability sensitivity with patient-clustered 95% confidence intervals.

| Outcome definition | Patients / windows | Positive, n (%) | AUROC (95% CI) | AUPRC (95% CI) | Brier (95% CI) |
| --- | --- | --- | --- | --- | --- |
| Primary: >=4 observed components | 7,287 / 830,839 | 47,292 (5.69) | 0.6559 (0.6492-0.6624) | 0.1309 (0.1250-0.1376) | 0.05208 (0.05075-0.05342) |
| Missing components assumed normal | 7,287 / 830,839 | 47,513 (5.72) | 0.6557 (0.6489-0.6620) | 0.1314 (0.1256-0.1380) | 0.05230 (0.05100-0.05364) |
| Six-component complete case | 2,346 / 176,130 | 8,804 (5.00) | 0.6237 (0.6112-0.6368) | 0.0923 (0.0849-0.1008) | 0.04701 (0.04470-0.04902) |
| Pairwise common components (>=4) | 7,287 / 830,609 | 32,244 (3.88) | 0.6097 (0.6018-0.6181) | 0.0662 (0.0628-0.0700) | 0.03784 (0.03666-0.03898) |
| Same component mask at primary future maximum | 7,278 / 788,385 | 28,275 (3.59) | 0.6013 (0.5923-0.6103) | 0.0564 (0.0535-0.0595) | 0.03528 (0.03417-0.03629) |
| Stable component mask across all six future hours | 7,143 / 679,878 | 25,499 (3.75) | 0.6027 (0.5938-0.6109) | 0.0594 (0.0562-0.0625) | 0.03665 (0.03531-0.03780) |

## Supplementary Table S13. Observed SOFA component contributions among primary positive windows.

| SOFA component | Comparable positive windows (%) | Positive windows with increase (%) | Positive SOFA points | Share of positive component-point increases (%) |
| --- | --- | --- | --- | --- |
| Renal | 99.9 | 36.7 | 41,727 | 46.3 |
| Neurological | 99.7 | 20.8 | 17,235 | 19.1 |
| Cardiovascular | 98.4 | 16.5 | 14,754 | 16.4 |
| Respiratory | 42.4 | 14.5 | 11,574 | 12.9 |
| Coagulation | 97.1 | 6.6 | 3,449 | 3.8 |
| Liver | 38.4 | 2.5 | 1,313 | 1.5 |

## Supplementary Figures

- Figure S1: raw rule-firing activation-threshold sensitivity.
- Figure S2: patient-clustered MIMIC-IV subgroup AUROC estimates.
- Figure S3: eICU hospital-level heterogeneity.
- Figure S4: selected true-positive, false-positive, and false-negative case timelines.
- Figure S5: patient-specific raw cross-rule firing over each 24-hour observation window.
- Figure S6: explanation stability, local consistency, sparsity, and cross-database rank stability.
- Figure S7: clinical-consistency behavior before and after consistency regularization.
- Membership functions and SOFA documentation sensitivity were promoted to the main manuscript.
