# TRIPOD+AI Reporting Checklist

This working checklist maps the current project to the TRIPOD+AI reporting domains. It is intended for manuscript completion and does not replace the journal's official submission form. Page numbers must be added after the manuscript layout is frozen.

Official reference: Collins GS, et al. TRIPOD+AI statement. *BMJ*. 2024;385:e078378. <https://www.bmj.com/content/385/bmj-2023-078378>

Status definitions: **Complete** means that both analysis evidence and manuscript text are present; **Partial** means that evidence exists but reporting or page mapping is incomplete; **Pending** means that author input or new text is required.

| Reporting domain | Status | Current manuscript/artifact | Required final action |
|---|---|---|---|
| Identification as a prediction-model study | Complete | `paper/TSP_template.tex`, title and abstract | Add final page number |
| Structured summary of objectives, data, model, validation, and performance | Complete | Abstract | Recheck values after the final manuscript table lock |
| Clinical context and rationale | Complete | Introduction and Literature Review | Add final page numbers |
| Intended use, target population, prediction time, and horizon | Complete | Methodology: Study Design; Outcome Definition | State that this is risk stratification, not an autonomous alarm system |
| Data sources, setting, sites, and roles of each database | Complete | Data Sources and Cohort Construction; `README.md` | Add database versions and extraction dates to final manuscript |
| Eligibility and cohort selection | Complete | Cohort Eligibility; Figure 1; `adult_eligibility_audit.json` | Add final page/figure references |
| Participant flow and analysis counts | Complete | Figure 1, Table 1; `docs/adult_cohort_manuscript_artifacts.md` | Keep patients, stays, stay-hours, and windows explicitly separated |
| Outcome definition, timing, and ascertainment | Complete | Outcome Definition; `docs/preprocessing_v2_method.md` | Cite primary and sensitivity definitions together |
| Predictor definition, timing, availability, and harmonization | Complete | Predictors and Preprocessing; README predictor table | State that predictors are available at index time and future values are excluded |
| Data quality and missing-data handling | Complete | Database Harmonization and Missingness; Table 2; formal missingness ablation | Retain the care-process interpretation and avoid causal claims |
| Sample-size rationale and event counts | Partial | Equal-sample protocol and full-cohort counts are reported | Add a formal rationale based on events, parameters, and computational design |
| Data partitioning and leakage prevention | Complete | Patient-level split; `comparison_protocol.json`; cohort audits | Add split seed and immutable manifest hash to supplement |
| Preprocessing and transformations | Complete | `preprocessing.py`; Methodology | Distinguish train-fitted transforms from fixed clinical defaults |
| Model specification and reproducibility | Complete | KG-TFNN equations; `anfis_model.py`; tuning/training configs; `outputs/reproducibility_6h/analysis_manifest.json` | Publish the local manifest with the code release |
| Hyperparameter tuning and model selection | Complete | `docs/explicit_temporal_fnn_tuning_6h.md` | Report search space, trial count, pruning, early stopping, and selected values in supplement |
| Comparator specification and fair comparison | Complete | `docs/fair_comparison_protocol.md`; paired and feature-matched comparison audits | Keep superseded non-feature-matched results out of architecture claims |
| Performance measures and uncertainty | Complete | AUROC, AUPRC, Brier, ECE, fixed-specificity metrics; clustered bootstrap | State that the clustering unit is `subject_id` |
| Calibration and clinical utility | Complete | Calibration curves, intercept/slope, DCA, event-level alarm burden and lead time | Keep raw/calibrated and window/event-level estimates distinct |
| Model comparison and statistical testing | Complete | `outputs/explicit_kg_tfnn_paired_comparison_6h/evaluation/` | Report paired effect estimates and 95% CIs, not only P values |
| Subgroup and fairness evaluation | Complete | `outputs/clinical_sensitivity_analyses_6h/` | Discuss small or heterogeneous subgroups and avoid causal fairness claims |
| Internal validation | Complete | Frozen MIMIC-IV test evaluation | Make clear that the final test set was evaluated once |
| External validation and transportability | Complete | Frozen eICU evaluation and hospital-clustered sensitivity | Report no eICU retraining, outcome fitting, threshold fitting, or recalibration |
| Model output and interpretation | Complete | Temporal rule extraction and Rule Evaluation Framework | Include actual supported rules and TP/FP/FN cases |
| Results for all participants and analyses | Complete | Seven main tables/figures, Supplementary Tables S1--S13/Figures S1--S7, and `formal_data_scope_audit_6h` | Keep full-cohort and prespecified equal-sample estimates explicitly separated |
| Limitations and interpretation | Partial | Labeled Discussion, Contributions, Limitations, and Future Work are present | After author approval, explicitly name physician attention and disease severity in the missingness discussion |
| Protocol and registration | Pending | `docs/analysis_plan.md` exists but no public registration is reported | State whether a protocol was registered; if not, explicitly report retrospective analysis-plan finalization |
| Data and code availability | Partial | PhysioNet availability statement; local reproducible code | Add repository URL/release DOI and note that source EHR data cannot be redistributed |
| Ethics and consent | Complete | Ethics Approval statement; CITI records | Verify wording against institutional requirements |
| Funding, conflicts, and author contributions | Complete | Statements section | Correct encoding artifacts in author-contribution punctuation |
| Patient and public involvement | Pending | Not currently stated | Add a statement that patients/public were not involved, with rationale for secondary de-identified data |
| Use of AI tools in research/reporting | Pending | Not currently stated | Follow the target journal's policy and disclose applicable software/AI assistance |

## Submission Lock

Before submission, replace every artifact path with a manuscript page, table, figure, or supplement reference. The official TRIPOD+AI form should then be completed from this matrix and submitted with the frozen manuscript.
