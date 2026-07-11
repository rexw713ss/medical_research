# Adult Cohort Manuscript Tables and Figures

更新日期：2026-07-10

成人納入條件固定為 ICU 入住時 `age >= 18`。MIMIC-IV 原始 ICU cohort 的最小年齡已為 18 歲，因此新增條件排除 0 位病人、0 個 stays；eICU preprocessing 原本即使用相同條件。13 個 predictors、6 小時 outcome、patient split 與 eligible windows 均未因成人條件改變。

## Adult-filter rerun audit

- MIMIC underage exclusions: 0 patients and 0 ICU stays.
- eICU underage exclusions: 437 patients and 530 ICU stays; this filter was already active in the external preprocessing.
- Adult-filtered patient split rebuild is byte-identical: `True`; assignment differences = 0.
- Because MIMIC subjects, split assignments, predictors, outcomes, and eligible windows are unchanged, existing fitted-model artifacts remain the same adult-cohort experiments. Model training was not repeated solely to reproduce an identical cohort; reporting tables and figures were regenerated. The frozen final test remains locked.

## Figure 1. Cohort selection flow diagram

- `outputs/manuscript_tables_figures_6h/figures/figure_1_cohort_flow.png`
- `outputs/manuscript_tables_figures_6h/figures/figure_1_cohort_flow.pdf`

## Table 1. Patient characteristics

| Characteristic | MIMIC train | MIMIC validation | MIMIC test | eICU external |
|---|---|---|---|---|
| Patients, n | 45,746 | 9,807 | 9,802 | 80,239 |
| ICU stays, n | 66,078 | 14,182 | 14,184 | 99,262 |
| Age, years, median [IQR] | 67.0 [55.0-78.0] | 66.0 [54.0-77.0] | 67.0 [54.0-78.0] | 66.0 [54.0-77.0] |
| Female, n (%) | 20,029 (43.8%) | 4,293 (43.8%) | 4,319 (44.1%) | 36,545 (45.5%) |
| White, n (%) | 30,118 (65.8%) | 6,449 (65.8%) | 6,443 (65.7%) | 61,943 (77.2%) |
| Black, n (%) | 4,240 (9.3%) | 899 (9.2%) | 885 (9.0%) | 9,069 (11.3%) |
| Asian, n (%) | 1,397 (3.1%) | 300 (3.1%) | 290 (3.0%) | 1,253 (1.6%) |
| Hispanic/Latino, n (%) | 1,584 (3.5%) | 359 (3.7%) | 355 (3.6%) | 2,945 (3.7%) |
| Other/Unknown race, n (%) | 8,407 (18.4%) | 1,800 (18.4%) | 1,829 (18.7%) | 5,029 (6.3%) |
| ICU length of stay, days, median [IQR] | 2.0 [1.1-3.9] | 2.0 [1.1-3.9] | 2.0 [1.1-3.9] | 2.8 [1.9-4.9] |
| Eligible 24-hour windows, n | 3,843,400 | 819,573 | 830,839 | 6,215,890 |
| Positive 6-hour windows, n (%) | 217,650 (5.7%) | 47,638 (5.8%) | 47,292 (5.7%) | 294,949 (4.7%) |

Age, sex, and race/ethnicity use each patient's first included ICU stay; ICU length of stay is stay-level. Window prevalence uses the complete eligible 24-hour prediction-window cohort.

## Table 2. Feature missingness before LOCF

| Feature | MIMIC overall missing, % | MIMIC train missing, % | MIMIC validation missing, % | MIMIC test missing, % | eICU overall missing, % |
|---|---|---|---|---|---|
| Heart rate | 6.1 | 6.1 | 6.0 | 6.1 | 3.5 |
| Respiratory rate | 7.4 | 7.4 | 7.3 | 7.4 | 9.9 |
| SpO2 | 8.0 | 8.0 | 7.9 | 8.1 | 7.8 |
| FiO2 | 86.7 | 86.7 | 86.7 | 86.9 | 83.0 |
| Temperature | 71.8 | 71.8 | 71.8 | 71.7 | 70.4 |
| Systolic blood pressure | 13.3 | 13.4 | 13.1 | 13.5 | 12.4 |
| GCS total | 74.0 | 74.1 | 73.7 | 73.5 | 80.4 |
| Mean arterial pressure | 13.3 | 13.3 | 13.0 | 13.4 | 12.2 |
| PaO2/FiO2 | 98.4 | 98.4 | 98.4 | 98.3 | 99.1 |
| Platelets | 93.7 | 93.7 | 93.7 | 93.6 | 95.6 |
| Bilirubin | 98.1 | 98.1 | 98.1 | 98.0 | 98.4 |
| Creatinine | 93.1 | 93.1 | 93.1 | 93.0 | 94.9 |
| Lactate | 96.3 | 96.3 | 96.3 | 96.2 | 99.0 |

Missingness is defined from the current-hour raw measurement indicator before forward filling. It is not the fraction remaining missing after LOCF.

## Table 3. Model performance

| Analysis | Model | AUROC (95% CI) | AUPRC (95% CI) | Brier (95% CI) | ECE |
|---|---|---|---|---|---|
| MIMIC equal-sample comparison | LightGBM | 0.6002 (0.5929-0.6068) | 0.0872 (0.0839-0.0908) | 0.0532 (0.0519-0.0546) | 0.0021 |
| MIMIC equal-sample comparison | Random Forest | 0.6038 (0.5973-0.6098) | 0.0866 (0.0831-0.0903) | 0.0532 (0.0518-0.0546) | 0.0022 |
| MIMIC equal-sample comparison | XGBoost | 0.6073 (0.6007-0.6141) | 0.0896 (0.0859-0.0934) | 0.0532 (0.0517-0.0546) | 0.0013 |
| MIMIC equal-sample comparison | GRU | 0.6238 (0.6170-0.6306) | 0.1037 (0.0992-0.1082) | 0.0529 (0.0514-0.0542) | 0.0014 |
| MIMIC equal-sample comparison | LSTM | 0.6156 (0.6090-0.6221) | 0.1002 (0.0958-0.1042) | 0.0529 (0.0516-0.0543) | 0.0023 |
| MIMIC equal-sample comparison | NEWS2 | 0.5699 (0.5635-0.5765) | 0.0736 (0.0710-0.0765) | 0.0535 (0.0521-0.0549) | 0.0016 |
| MIMIC equal-sample comparison | SOFA | 0.4978 (0.4895-0.5064) | 0.0558 (0.0540-0.0584) | 0.0537 (0.0523-0.0550) | 0.0021 |
| MIMIC equal-sample comparison | Explicit Knowledge-Guided Temporal FNN | 0.6448 (0.6379-0.6515) | 0.1236 (0.1177-0.1297) | 0.0523 (0.0509-0.0536) | 0.0013 |
| MIMIC equal-sample comparison | Decision Tree | 0.5741 (0.5677-0.5809) | 0.0754 (0.0728-0.0783) | 0.0535 (0.0521-0.0548) | 0.0030 |
| MIMIC equal-sample comparison | Explainable Boosting Machine | 0.6072 (0.6008-0.6141) | 0.0891 (0.0853-0.0929) | 0.0532 (0.0518-0.0546) | 0.0012 |
| MIMIC equal-sample comparison | Generalized Additive Model | 0.6003 (0.5933-0.6068) | 0.0878 (0.0841-0.0911) | 0.0532 (0.0518-0.0546) | 0.0018 |
| MIMIC equal-sample comparison | Logistic Regression | 0.5795 (0.5718-0.5872) | 0.0794 (0.0760-0.0827) | 0.0534 (0.0520-0.0548) | 0.0013 |
| MIMIC full-cohort frozen final model | Knowledge-Guided Temporal FNN | 0.6559 (0.6492-0.6628) | 0.1309 (0.1250-0.1375) | 0.0521 (0.0507-0.0534) | 0.0012 |
| eICU frozen external validation | Knowledge-Guided Temporal FNN | 0.6221 (0.6192-0.6249) | 0.0922 (0.0902-0.0942) | 0.0459 (0.0455-0.0463) | 0.0267 |

Equal-sample comparisons and the full-cohort frozen model are distinct analyses and must not be presented as if they used the same training sample size.

## Table 4. Ablation study

| Model | AUROC | AUPRC | Brier | ECE | Guideline-Direction Alignment | Rule Stability | Rule Drift | Seeds |
|---|---|---|---|---|---|---|---|---|
| Randomly initialized FNN | 0.6395 | 0.1183 | 0.0524 | 0.0012 | 0.4959 | 0.207 | 0.1653 | 42,52,62 |
| Guideline-guided FNN without temporal features | 0.5949 | 0.0837 | 0.0533 | 0.0012 | 0.6473 | 0.8182 | 0.2448 | 42,52,62 |
| Temporal FNN without clinical consistency regularization | 0.6459 | 0.123 | 0.0523 | 0.0012 | 0.5362 | 0.5873 | 0.2221 | 42,52,62 |
| Full Knowledge-Guided Temporal FNN | 0.6456 | 0.123 | 0.0523 | 0.0012 | 0.5281 | 0.6744 | 0.2244 | 42,52,62 |

## Table 5. Rule evaluation

| Analysis | Definition | Result |
|---|---|---|
| Rule Complexity | Top-10 mean antecedents | 1.44 |
| Rule Stability | Five-seed pairwise Top-10 Jaccard | 0.72 |
| Guideline-Direction Alignment | Prespecified NEWS2/SOFA direction alignment | 1.0 |
| Rule Drift | Median center shift, initial sigmas | 0.264 |
| Activated Rules | Mean activated rules: Negative windows | 1.819 |
| Activated Rules | Mean activated rules: Positive windows | 1.833 |

Guideline-direction alignment is an investigator-defined model diagnostic, not independent clinician adjudication or clinician-validated interpretability.

## Figures 2-5

| Figure | Artifact |
|---|---|
| Figure 2, system architecture | `outputs/manuscript_tables_figures_6h/figures/figure_2_system_architecture.pdf` |
| Figure 3, calibration curve | `outputs/manuscript_tables_figures_6h/figures/figure_3_calibration_curve.pdf` |
| Figure 4, decision curve analysis | `outputs/manuscript_tables_figures_6h/figures/figure_4_decision_curve_analysis.pdf` |
| Figure 5, example timelines and activated rules | `outputs/manuscript_tables_figures_6h/figures/figure_5_patient_timeline_activated_rules.pdf` |

Figure 3 and Figure 4 use the frozen final model. eICU probabilities and thresholds are transported from MIMIC without external refitting or recalibration.
