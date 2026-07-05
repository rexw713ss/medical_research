# Extracted Temporal Fuzzy Rules

這些規則由 frozen full-cohort 6-hour FNN 與實際 MIMIC-IV test windows 萃取，並非假想規則。
候選規則與排名只使用模型參數及 support；positive rate 未參與規則挑選。

- Frozen checkpoint SHA-256: `158427a5c358016f35b435b1ab5f75c7194a3ff3f9b6c9d68c5190a8a9125688`
- Test windows: 830,839
- Overall deterioration rate: 5.69%
- Rule weight: model-derived weight normalized to the largest candidate rule.
- Clinical concordance: fraction of static, temporal and cross-rule directions aligned with guideline priors.

## Main-Text Cross-Feature Examples

| Rank | Extracted temporal fuzzy rule | Rule weight | Support | Positive rate | Clinical concordance |
|---:|---|---:|---:|---:|---:|
| 1 | IF GCS IS altered AND SpO2 IS low AND GCS-related fuzzy risk increased over the 24-hour window THEN deterioration risk IS high | 1.000 | n=393 | 15.0% | 1.00 |
| 2 | IF creatinine IS high AND platelet count IS low AND bilirubin IS high AND creatinine-related fuzzy risk increased over the 24-hour window THEN deterioration risk IS high | 0.643 | n=781 | 10.9% | 1.00 |
| 3 | IF PaO2/FiO2 ratio IS very low AND FiO2 requirement IS high support AND PaO2/FiO2 ratio-related fuzzy risk increased during the last hour THEN deterioration risk IS high | 0.258 | n=789 | 17.1% | 1.00 |
| 4 | IF SpO2 IS critical low AND respiratory rate IS critical high AND SpO2-related fuzzy risk increased over the 24-hour window THEN deterioration risk IS high | 0.272 | n=240 | 16.2% | 1.00 |

Cross-feature antecedents use fuzzy membership >= 0.35; single-feature rules use >= 0.50.

## Overall Top Model-Supported Rules

| Rank | Extracted temporal fuzzy rule | Rule weight | Support | Positive rate | Clinical concordance |
|---:|---|---:|---:|---:|---:|
| 1 | IF FiO2 requirement IS supplemental o2 AND FiO2 requirement measurements were frequently abnormal THEN deterioration risk IS high | 0.535 | n=529,833 | 6.3% | 1.00 |
| 2 | IF platelet count IS low AND platelet count abnormality persisted across the observation window THEN deterioration risk IS high | 0.399 | n=72,960 | 6.4% | 1.00 |
| 3 | IF creatinine IS very high AND creatinine-related fuzzy risk increased over the 24-hour window THEN deterioration risk IS high | 0.599 | n=19,768 | 6.9% | 1.00 |
| 4 | IF GCS IS severely altered AND GCS-related fuzzy risk increased over the 24-hour window THEN deterioration risk IS high | 0.723 | n=12,799 | 8.5% | 1.00 |
| 5 | IF systolic blood pressure IS very low AND systolic blood pressure-related fuzzy risk increased over the 24-hour window THEN deterioration risk IS high | 0.369 | n=30,046 | 9.2% | 1.00 |
| 6 | IF bilirubin IS critical high AND bilirubin-related fuzzy risk increased over the 24-hour window THEN deterioration risk IS high | 0.600 | n=9,853 | 8.2% | 1.00 |
| 7 | IF lactate IS severe AND lactate-related fuzzy risk increased over the 24-hour window THEN deterioration risk IS high | 0.809 | n=5,373 | 16.7% | 1.00 |
| 8 | IF respiratory rate IS critical high AND respiratory rate-related fuzzy risk increased over the 24-hour window THEN deterioration risk IS high | 0.254 | n=53,578 | 7.7% | 1.00 |
| 9 | IF platelet count IS low AND platelet count-related fuzzy risk increased over the 24-hour window THEN deterioration risk IS high | 0.435 | n=16,474 | 5.8% | 1.00 |
| 10 | IF SpO2 IS low AND SpO2-related fuzzy risk increased over the 24-hour window THEN deterioration risk IS high | 0.332 | n=19,273 | 9.0% | 1.00 |

## Method Note

`risk_slope`, `short_term_change`, and `window_change` describe changes in learned fuzzy risk, not raw-value slopes. Persistence and frequency are computed from the model's differentiable abnormality probability over the 24-hour observation window.

完整候選規則、support、event counts、lift 與 thresholds 見 `extracted_temporal_rules.csv`。
