# Missingness Ablation, Primary 6-Hour Outcome

All variants use the same patient split, 200,000 training windows, 50,000 validation windows,
830,839 independent test windows, predictors, outcome, optimizer settings, and random seeds.

## Seed-level Results

| Model | AUROC, mean +/- SD | AUPRC, mean +/- SD | Brier, mean +/- SD | ECE, mean +/- SD |
|---|---:|---:|---:|---:|
| Full Knowledge-Guided Temporal FNN | 0.6448 +/- nan | 0.1236 +/- nan | 0.0523 +/- nan | 0.0013 +/- nan |
| KG-TFNN without missingness channels | 0.6037 +/- nan | 0.0875 +/- nan | 0.0532 +/- nan | 0.0011 +/- nan |
| Missingness-only temporal FNN | 0.5949 +/- nan | 0.0902 +/- nan | 0.0532 +/- nan | 0.0019 +/- nan |

## Three-Seed Ensemble

| Model | AUROC | AUPRC | Brier | ECE |
|---|---:|---:|---:|---:|
| Full Knowledge-Guided Temporal FNN | 0.6448 | 0.1236 | 0.0523 | 0.0013 |
| KG-TFNN without missingness channels | 0.6037 | 0.0875 | 0.0532 | 0.0011 |
| Missingness-only temporal FNN | 0.5949 | 0.0902 | 0.0532 | 0.0019 |

## Paired Patient-Clustered Bootstrap

Difference is full KG-TFNN minus the comparator; 95% CIs are percentile intervals.

| Comparator | Metric | Mean difference | 95% CI | P value |
|---|---|---:|---:|---:|
| KG-TFNN without missingness channels | AUROC | 0.0417 | 0.0389 to 0.0459 | 0.0000 |
| KG-TFNN without missingness channels | AUPRC | 0.0364 | 0.0344 to 0.0402 | 0.0000 |
| KG-TFNN without missingness channels | BRIER | -0.0009 | -0.0010 to -0.0009 | 0.0000 |
| KG-TFNN without missingness channels | ECE | 0.0003 | -0.0001 to 0.0011 | 0.4000 |
| Missingness-only temporal FNN | AUROC | 0.0496 | 0.0467 to 0.0523 | 0.0000 |
| Missingness-only temporal FNN | AUPRC | 0.0340 | 0.0308 to 0.0380 | 0.0000 |
| Missingness-only temporal FNN | BRIER | -0.0009 | -0.0010 to -0.0008 | 0.0000 |
| Missingness-only temporal FNN | ECE | -0.0004 | -0.0010 to 0.0004 | 0.4000 |

## Audit

- Protocol SHA-256: `47629531beb32b3d2022e37471835ca592d7779197b7fe3696ee1cc8fc3ad184`
- Test subjects: 7,287
- Test windows: 830,839
- Test windows and outcomes were byte-order matched before analysis.
- Thresholds and calibration were determined without using test outcomes.
