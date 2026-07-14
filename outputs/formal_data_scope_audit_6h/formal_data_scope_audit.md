# Formal Data-Scope Audit

- Status: **PASSED**
- Checks passed: 116
- Checks failed: 0
- Primary outcome: future 6-hour SOFA increase >=2

## Locked Data Scopes

| Analysis scope | Train | Validation | Test / external | Interpretation |
|---|---:|---:|---:|---|
| Primary full-cohort KG-TFNN | 3,843,400 | 819,573 | 830,839 | Every eligible MIMIC-IV window |
| Equal-sample sensitivity | 200,000 | 50,000 | 830,839 | Prespecified fair train/validation subset; complete test |
| Frozen eICU transport | NA | NA | 6,215,890 | Every eligible external window; no refitting/recalibration |
| Frozen eICU equal-sample comparator transport | 200,000 | 50,000 | 6,215,890 | Five models; complete external cohort and source-only calibration |
| Full post-hoc XAI | NA | NA | 830,839 MIMIC + 6,215,890 eICU | Every prediction-key window |
| Consistency behavior | NA | NA | 830,839 per model | 3 seeds x 2 variants = 4,985,034 model-window evaluations |

## Failed Checks

None. No smoke-test or runtime-truncated result is registered as canonical evidence.

## Interpretation

`equal_sample` is not a full-cohort training estimate and must remain labelled as a prespecified fairness sensitivity analysis. Its independent test evaluation is complete. Primary full-cohort and external estimates must be reported separately.
