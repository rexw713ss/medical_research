# 論文圖表清單

更新日期：2026-07-14

正式圖表同時保留 PNG 預覽與向量 PDF。數值必須來自 canonical CSV/JSON，不可從圖片手動抄錄。Primary outcome 固定為未來 6 小時 SOFA increase >= 2。

## Main Figures

| Figure | 內容 | Canonical source |
|---|---|---|
| Figure 1 | MIMIC-IV / eICU cohort flow | `outputs/manuscript_tables_figures_6h/figures/figure_1_cohort_flow.*` |
| Figure 2 | KG-TFNN system architecture | `outputs/manuscript_tables_figures_6h/figures/figure_2_system_architecture.*` |
| Figure 3 | Membership functions before/after training | `outputs/rule_evaluation_6h/figures/` |
| Figure 4 | Equal-sample ROC / PR curves | `outputs/explicit_kg_tfnn_paired_comparison_6h/evaluation/figures/` |
| Figure 5 | Frozen-model calibration / DCA | `outputs/final_test_evaluation_6h/advanced/figures/`、`outputs/eicu_external_validation/final_frozen_model_evaluation/figures/` |
| Figure 6 | SOFA documentation-availability sensitivity | `outputs/sofa_documentation_bias_6h/figures/sofa_documentation_bias_sensitivity.*` |
| Figure 7 | TP / FP / FN patient timelines | `outputs/manuscript_tables_figures_6h/figures/figure_5_patient_timeline_activated_rules.*` |

## Main Tables

| Table | 內容 | Canonical source |
|---|---|---|
| Table 1 | Cohort characteristics、windows、pre-LOCF missingness | `outputs/manuscript_tables_figures_6h/` |
| Table 2 | Primary predictive performance | `outputs/final_test_evaluation_6h/`、`outputs/explicit_kg_tfnn_paired_comparison_6h/` |
| Table 3 | Ablation 與 full-data structural explanation panel | `outputs/fnn_ablation_6h_equal_sample/`、`outputs/posthoc_explainability_comparison_6h/` |
| Table 4 | Observation-window / SOFA sensitivity | `outputs/explicit_temporal_observation_sensitivity_6h/`、`outputs/clinical_sensitivity_analyses_6h/` |
| Table 5 | Representative temporal fuzzy rules | `outputs/temporal_rule_extraction_6h/` |
| Table 6 | Rule Evaluation Framework | `outputs/rule_evaluation_6h/` |
| Table 7 | Supported frozen-model rule inventory | `outputs/temporal_rule_extraction_6h/` |

## Supplementary

- Tables S1--S13 與 Figures S1--S7 的索引：`outputs/supplementary_material/supplementary_material.md`。
- Raw firing：`outputs/raw_rule_firing_6h/figures/`。
- MIMIC subgroup：`outputs/clinical_sensitivity_analyses_6h/figures/`。
- eICU hospital heterogeneity：`outputs/eicu_hospital_sensitivity_6h/figures/`。
- Patient-specific raw firing：`outputs/rule_evaluation_6h/figures/`。
- Post-hoc XAI comparison：`outputs/posthoc_explainability_comparison_6h/figures/`；完整 830,839 MIMIC 與 6,215,890 eICU windows。
- Consistency behavioral audit：`outputs/clinical_consistency_regularization_6h/figures/`；3 seeds x 2 variants，每模型完整 830,839 MIMIC windows。

## Pending Figure-Level Evidence

- Figure S6 已完成 full-data structural benchmark，但 current-state EBM 並非 24-hour feature-matched comparator，且不同 explanation forms 不代表相同語意。
- Figure S7 已完成 full-test-cohort directional stress test；結果只支持 stability 改善，不支持全面 clinical-consistency 改善。
- Clinician-reader understandability figure/table 尚未完成，除非另行執行 blinded reader study。
