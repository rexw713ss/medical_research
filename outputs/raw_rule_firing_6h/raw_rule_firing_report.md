# Raw Rule Firing and Activation-Threshold Sensitivity

Frozen checkpoint SHA-256: `158427a5c358016f35b435b1ab5f75c7194a3ff3f9b6c9d68c5190a8a9125688`.
Raw firing is the product t-norm before normalization across rules.

## Rule-Level Results at Raw Firing >= 0.10

| Rule | Support | Positive rate | Lift | Mean firing positive | Mean firing negative |
|---|---:|---:|---:|---:|---:|
| altered_consciousness_hypoxemia | 18,613 | 0.094 | 1.65 | 0.0174 | 0.0117 |
| hypoperfusion_pattern | 221,567 | 0.057 | 0.99 | 0.0626 | 0.0634 |
| oxygenation_failure_with_support | 188,747 | 0.061 | 1.07 | 0.0562 | 0.0516 |
| multi_organ_dysfunction_pattern | 8,000 | 0.080 | 1.40 | 0.0043 | 0.0031 |
| shock_pattern | 4,005 | 0.108 | 1.89 | 0.0043 | 0.0019 |
| respiratory_failure_pattern | 434 | 0.104 | 1.82 | 0.0003 | 0.0001 |

## Activated Rules

| Basis | Threshold | Outcome | Mean activated | Median activated | Any rule |
|---|---:|---:|---:|---:|---:|
| current_hour | 0.010 | 0 | 1.395 | 1.0 | 0.870 |
| current_hour | 0.010 | 1 | 1.487 | 1.0 | 0.883 |
| current_hour | 0.025 | 0 | 1.056 | 1.0 | 0.748 |
| current_hour | 0.025 | 1 | 1.131 | 1.0 | 0.767 |
| current_hour | 0.050 | 0 | 0.806 | 1.0 | 0.622 |
| current_hour | 0.050 | 1 | 0.860 | 1.0 | 0.644 |
| current_hour | 0.100 | 0 | 0.529 | 0.0 | 0.448 |
| current_hour | 0.100 | 1 | 0.569 | 0.0 | 0.471 |
| current_hour | 0.200 | 0 | 0.151 | 0.0 | 0.143 |
| current_hour | 0.200 | 1 | 0.173 | 0.0 | 0.163 |
| current_hour | 0.350 | 0 | 0.011 | 0.0 | 0.011 |
| current_hour | 0.350 | 1 | 0.021 | 0.0 | 0.021 |
| current_hour | 0.500 | 0 | 0.003 | 0.0 | 0.003 |
| current_hour | 0.500 | 1 | 0.007 | 0.0 | 0.006 |
| attention_selected_hour | 0.010 | 0 | 1.663 | 2.0 | 0.878 |
| attention_selected_hour | 0.010 | 1 | 1.743 | 2.0 | 0.893 |
| attention_selected_hour | 0.025 | 0 | 1.335 | 1.0 | 0.785 |
| attention_selected_hour | 0.025 | 1 | 1.412 | 1.0 | 0.808 |
| attention_selected_hour | 0.050 | 0 | 1.065 | 1.0 | 0.699 |
| attention_selected_hour | 0.050 | 1 | 1.128 | 1.0 | 0.727 |
| attention_selected_hour | 0.100 | 0 | 0.764 | 1.0 | 0.567 |
| attention_selected_hour | 0.100 | 1 | 0.807 | 1.0 | 0.592 |
| attention_selected_hour | 0.200 | 0 | 0.291 | 0.0 | 0.257 |
| attention_selected_hour | 0.200 | 1 | 0.321 | 0.0 | 0.283 |
| attention_selected_hour | 0.350 | 0 | 0.071 | 0.0 | 0.069 |
| attention_selected_hour | 0.350 | 1 | 0.097 | 0.0 | 0.094 |
| attention_selected_hour | 0.500 | 0 | 0.024 | 0.0 | 0.024 |
| attention_selected_hour | 0.500 | 1 | 0.035 | 0.0 | 0.035 |

## Top-K Threshold Stability

| Basis | Threshold | Reference | Top-K | Jaccard | Rules with support >=100 |
|---|---:|---:|---:|---:|---:|
| current_hour | 0.010 | 0.100 | 5 | 1.000 | 6 |
| current_hour | 0.025 | 0.100 | 5 | 1.000 | 6 |
| current_hour | 0.050 | 0.100 | 5 | 1.000 | 6 |
| current_hour | 0.100 | 0.100 | 5 | 1.000 | 6 |
| current_hour | 0.200 | 0.100 | 5 | 1.000 | 6 |
| current_hour | 0.350 | 0.100 | 5 | 0.667 | 4 |
| current_hour | 0.500 | 0.100 | 5 | 0.667 | 3 |
| attention_selected_hour | 0.010 | 0.100 | 5 | 1.000 | 6 |
| attention_selected_hour | 0.025 | 0.100 | 5 | 1.000 | 6 |
| attention_selected_hour | 0.050 | 0.100 | 5 | 1.000 | 6 |
| attention_selected_hour | 0.100 | 0.100 | 5 | 1.000 | 6 |
| attention_selected_hour | 0.200 | 0.100 | 5 | 1.000 | 6 |
| attention_selected_hour | 0.350 | 0.100 | 5 | 0.667 | 5 |
| attention_selected_hour | 0.500 | 0.100 | 5 | 0.667 | 3 |

These are model-internal firing diagnostics, not clinician validation or causal evidence.
