# Missingness Ablation, Primary 6-Hour Outcome

All variants use the same patient split, 200,000 training windows, 50,000 validation windows,
830,839 independent test windows, predictors, outcome, optimizer settings, and random seeds.

## Seed-level Results

| Model | AUROC, mean +/- SD | AUPRC, mean +/- SD | Brier, mean +/- SD | ECE, mean +/- SD |
|---|---:|---:|---:|---:|
| Full Knowledge-Guided Temporal FNN | 0.6456 +/- 0.0012 | 0.1230 +/- 0.0006 | 0.0523 +/- 0.0000 | 0.0012 +/- 0.0001 |
| KG-TFNN without missingness channels | 0.6035 +/- 0.0003 | 0.0877 +/- 0.0004 | 0.0532 +/- 0.0000 | 0.0011 +/- 0.0000 |
| Missingness-only temporal FNN | 0.5953 +/- 0.0003 | 0.0903 +/- 0.0001 | 0.0532 +/- 0.0000 | 0.0019 +/- 0.0001 |

## 3-Seed Ensemble

| Model | AUROC | AUPRC | Brier | ECE |
|---|---:|---:|---:|---:|
| Full Knowledge-Guided Temporal FNN | 0.6475 | 0.1244 | 0.0523 | 0.0014 |
| KG-TFNN without missingness channels | 0.6042 | 0.0879 | 0.0532 | 0.0011 |
| Missingness-only temporal FNN | 0.5954 | 0.0904 | 0.0532 | 0.0019 |

## Paired Patient-Clustered Bootstrap

Difference is full KG-TFNN minus the comparator; 95% CIs are percentile intervals.

| Comparator | Metric | Mean difference | 95% CI | P value |
|---|---|---:|---:|---:|
| KG-TFNN without missingness channels | AUROC | 0.0435 | 0.0391 to 0.0478 | <0.002 |
| KG-TFNN without missingness channels | AUPRC | 0.0366 | 0.0322 to 0.0410 | <0.002 |
| KG-TFNN without missingness channels | BRIER | -0.0009 | -0.0011 to -0.0008 | <0.002 |
| KG-TFNN without missingness channels | ECE | 0.0003 | -0.0005 to 0.0009 | 0.3700 |
| Missingness-only temporal FNN | AUROC | 0.0518 | 0.0458 to 0.0576 | <0.002 |
| Missingness-only temporal FNN | AUPRC | 0.0339 | 0.0291 to 0.0390 | <0.002 |
| Missingness-only temporal FNN | BRIER | -0.0009 | -0.0010 to -0.0008 | <0.002 |
| Missingness-only temporal FNN | ECE | -0.0005 | -0.0013 to 0.0003 | 0.2520 |

## Audit

- Protocol SHA-256: `47629531beb32b3d2022e37471835ca592d7779197b7fe3696ee1cc8fc3ad184`
- Test subjects: 7,287
- Test windows: 830,839
- Test windows and outcomes were byte-order matched before analysis.
- Thresholds and calibration were determined without using test outcomes.
