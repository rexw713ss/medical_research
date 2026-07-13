# Post-hoc Explainability Comparison

**Exploratory analysis only; `formal_full_data=false`.** All explanation-quality metrics use 1,000 prespecified one-window-per-stay samples per database and are not formal full-cohort results. MIMIC perturbation stability uses three 1% training-SD perturbations of raw physiological channels. Neighbor consistency compares each MIMIC case with its nearest trajectory neighbor. Cross-dataset stability compares mean absolute feature-level explanation rankings without eICU fitting.

| model | auroc | auprc | explanation_method | output_form | stability_cosine_mean | stability_top5_jaccard_mean | neighbor_consistency_cosine_mean | neighbor_consistency_top5_jaccard_mean | effective_features_80_median | effective_features_80_iqr_low | effective_features_80_iqr_high | temporal_attribution_mass_mean | cross_dataset_global_spearman | cross_dataset_top5_jaccard | human_readable_temporal_rule_output | clinician_validated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM + TreeSHAP | 0.6904 | 0.1710 | TreeSHAP on 24-h matched summaries | Signed feature attributions | 0.9647 | 0.7764 | 0.4079 | 0.4069 | 6.0000 | 5.0000 | 7.0000 | 0.8650 | 0.7967 | 0.6667 | False | False |
| XGBoost + TreeSHAP | 0.6870 | 0.1665 | TreeSHAP on 24-h matched summaries | Signed feature attributions | 0.9439 | 0.7327 | 0.4667 | 0.4209 | 7.0000 | 6.0000 | 7.0000 | 0.8451 | 0.8791 | 0.6667 | False | False |
| EBM (current state) | 0.6072 | 0.0891 | Additive term contributions; current state only | Additive feature/interaction terms | 0.9821 | 0.9048 | 0.4839 | 0.4817 | 6.0000 | 6.0000 | 7.0000 | 0.0000 | 0.8462 | 0.6667 | False | False |
| KG-TFNN | 0.6448 | 0.1236 | Model-intrinsic fuzzy contributions | Temporal IF-THEN rules | 1.0000 | 0.9967 | 0.9786 | 0.7464 | 5.0000 | 5.0000 | 5.0000 | 0.1878 | 0.9780 | 0.6667 | True | False |

Structural outputs are reported separately from clinician understandability. No explanation method in this analysis has been validated by a clinician reader study.
