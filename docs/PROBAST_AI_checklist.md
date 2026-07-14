# PROBAST+AI Risk-of-Bias and Applicability Assessment

This is a project-side self-assessment of the primary 6-hour KG-TFNN study. It should be reviewed independently by a clinical and methodological reviewer before submission.

Official reference: Moons KGM, et al. PROBAST+AI. *BMJ*. 2025;388:e082505. <https://www.bmj.com/content/388/bmj-2024-082505>

Judgements use **Low**, **Some concerns**, or **High**. These are transparent working judgements, not claims of formal external certification.

## Domain Assessment

| Domain | Risk of bias | Applicability concern | Evidence and rationale | Remaining action |
|---|---|---|---|---|
| Participants and data sources | Low | Some concerns | Adult ICU cohorts are defined before modeling; patient-level splits prevent overlap; MIMIC-IV and multicenter eICU are reported separately | Discuss retrospective EHR selection, US-only data, and database-era differences |
| Predictors | Low | Some concerns | Thirteen protocol predictors are defined at or before the index hour; harmonization is documented; missingness/recency channels also encode care-process signals | Document unit checks and real-time availability assumptions; discuss workflow and monitoring-frequency transportability |
| Outcome | Some concerns | Some concerns | The primary label is future 6-hour SOFA increase >=2; complete-case, common-component, same-mask, and stable-mask analyses show material documentation sensitivity | Preserve the derived-surrogate limitation and do not claim the endpoint is free of documentation bias |
| Analysis | Low | Low | Large event count; patient-level train/validation/test split; validation-only tuning/calibration; frozen test; subject-clustered CIs; paired comparison; external validation; reproducibility manifest | Preserve one-time test lock and publish the manifest with the code release |
| Model performance and evaluation | Low | Some concerns | Discrimination, calibration, fixed-specificity sensitivity, DCA, alarm burden, lead time, subgroup performance, and eICU hospital-clustered sensitivity are reported | Avoid presenting window-level sensitivity as event-level clinical utility |
| Explainability and human factors | Some concerns | Some concerns | Actual supported rules, formal rule diagnostics, and a full-data TreeSHAP/EBM/KG-TFNN structural benchmark are available | Avoid treating operational structural metrics as clinician validation; obtain independent clinician review if feasible |
| Fairness | Some concerns | Some concerns | Age, sex, ethnicity, ICU type, and current-SOFA subgroup estimates with clustered CIs are available | Small and heterogeneous groups limit conclusions; do not infer absence of inequity from overlapping CIs |
| External transportability | Low | Some concerns | Frozen MIMIC checkpoint is applied to 205 eICU hospitals without retraining or recalibration; hospital-clustered CIs and site heterogeneity are reported | Discuss degraded calibration and specificity transport as deployment limitations |

## Key Signaling Questions

| Question | Answer | Evidence |
|---|---|---|
| Were eligibility criteria defined without knowledge of model predictions? | Yes | Preprocessing and cohort-flow audit |
| Were the development, validation, and test participants separated at patient level? | Yes | `patient_split.csv`, split audit |
| Were predictors measured before the predicted outcome period? | Yes | Leakage-free index-time construction |
| Were predictor definitions consistent between development and external data? | Mostly yes | eICU harmonization mapping; missingness/domain-shift tables |
| Was the outcome defined consistently and without model prediction input? | Yes | Rule-based SOFA construction in both databases |
| Was outcome missingness examined? | Yes | Assume-normal, complete-case, pairwise-common, same-mask, and stable-mask SOFA analyses |
| Was sample size adequate relative to events and model complexity? | Likely yes | 200,000 equal-sample training windows and full-cohort sensitivity; event counts reported |
| Were missing predictor values handled without test-data fitting? | Yes | Fixed clinical defaults, LOCF, missingness indicators, time-since channels |
| Were all modeling and tuning choices restricted to training/validation data? | Yes | Tuning and checkpoint-selection configs |
| Was the independent test set protected from repeated tuning? | Yes, by design | Frozen checkpoint and one-time test evaluation record |
| Were correlated hourly observations handled in uncertainty estimation? | Yes | `subject_id`-clustered bootstrap |
| Were model comparisons paired on identical test windows? | Yes | Explicit KG-TFNN paired-comparison audit |
| Were calibration and clinically relevant operating points reported? | Yes | Brier, ECE, calibration curve, intercept/slope, 90%/95% specificity |
| Was clinical utility assessed beyond window-level metrics? | Yes | DCA, lead time, event detection, alert and false-alert burden |
| Was external validation performed without local model fitting? | Yes | Frozen eICU transport; no retraining/recalibration |
| Were model explanations evaluated empirically? | Yes, with scope limits | Formal intrinsic-rule metrics plus a full-data TreeSHAP/current-state EBM structural benchmark; no clinician reader study |

## Overall Judgement

**Overall risk of bias: Some concerns.** The analysis design is strong for retrospective EHR model development, but the outcome is a derived SOFA surrogate with incomplete components and the guideline-direction alignment rubric is investigator-defined.

**Overall applicability: Some concerns.** The model targets adult ICU stay-hour risk stratification, but transportability varies across eICU hospitals and fixed MIMIC operating thresholds generate materially different external specificity. The manuscript should not frame the model as ready for autonomous bedside deployment.

## Priority Mitigations

1. Retain the labeled Discussion of retrospective selection, derived outcome, alarm burden, subgroup uncertainty, and external calibration shift; add the explicit missingness mechanisms after author approval.
2. Obtain an independent clinician review of the extracted rules and representative case timelines if feasible; do not describe the current guideline-direction rubric as clinician validation.
3. Archive the frozen split, comparison protocol, package environment, configurations, and checkpoint hashes with the submitted code release.
