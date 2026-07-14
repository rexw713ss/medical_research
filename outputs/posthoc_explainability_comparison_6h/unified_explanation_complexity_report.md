# Unified Cross-Model Explanation Complexity

## Common Definition

For every eligible MIMIC-IV test window, each method's signed local explanation was aggregated to the same 13 harmonized clinical variables. Absolute attribution was normalized to sum to one within each window. Complexity is the minimum number of variables required to explain 80% of that attribution mass; lower values indicate a more concentrated explanation.

Tree summary features were mapped back to their clinical variable, and EBM interaction contributions were divided equally among participating variables. The metric therefore compares explanation concentration, not native rule syntax.

| model | effective_features_80_median | effective_features_80_iqr_low | effective_features_80_iqr_high | normalized_effective_features_80_median | mimic_windows | eicu_windows |
| --- | --- | --- | --- | --- | --- | --- |
| LightGBM + TreeSHAP | 6.0000 | 6.0000 | 7.0000 | 0.4615 | 830839.0000 | 6215890.0000 |
| XGBoost + TreeSHAP | 7.0000 | 6.0000 | 7.0000 | 0.5385 | 830839.0000 | 6215890.0000 |
| EBM (current state) | 6.0000 | 6.0000 | 7.0000 | 0.4615 | 830839.0000 | 6215890.0000 |
| KG-TFNN | 5.0000 | 4.0000 | 5.0000 | 0.3846 | 830839.0000 | 6215890.0000 |

## Model-Specific Structural Complexity

KG-TFNN Top-10 rules contained a mean of 1.44 antecedents across the five-seed rule analysis. This rule-specific result is reported separately and is not treated as directly equivalent to SHAP or EBM terms.

## Interpretation Boundary

This structural analysis does not constitute clinician validation. The EBM uses current-state inputs and remains a non-architecture-matched comparator.
