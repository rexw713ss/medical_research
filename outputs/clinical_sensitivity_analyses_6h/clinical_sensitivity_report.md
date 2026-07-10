# Clinical Sensitivity Analyses

All analyses use the frozen full-cohort MIMIC final model. No checkpoint, calibration, or threshold was refit.

## SOFA Outcome Definition

| definition | patients | windows | prevalence | AUROC (95% CI) | AUPRC (95% CI) | BRIER (95% CI) |
|---|---|---|---|---|---|---|
| Primary: >=4 observed components | 7287 | 830839 | 0.0569207742810249 | 0.6559 (0.6492-0.6624) | 0.1309 (0.1250-0.1376) | 0.0521 (0.0508-0.0534) |
| Missing components assumed normal | 7287 | 830839 | 0.057186771184206 | 0.6557 (0.6489-0.6620) | 0.1314 (0.1256-0.1380) | 0.0523 (0.0510-0.0536) |
| Six-component complete case | 2346 | 176130 | 0.0499858073890209 | 0.6237 (0.6112-0.6368) | 0.0923 (0.0849-0.1008) | 0.0470 (0.0447-0.0490) |

## Event-Level Alarm Burden

| target_specificity | first_events | detected_events | event_sensitivity | alerts | alert_ppv | false_alerts_per_100_patient_days | patients_with_false_alert_fraction | lead_time_median_h | lead_time_q1_h | lead_time_q3_h |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.9 | 3710 | 1274 | 0.3433962264150943 | 8151 | 0.1562998405103668 | 48.23200738757905 | 0.3639357760395224 | 3.0 | 1.0 | 4.0 |
| 0.95 | 3710 | 883 | 0.2380053908355795 | 4348 | 0.203081876724931 | 24.30186209073163 | 0.2330177027583367 | 2.0 | 1.0 | 4.0 |

The event analysis uses the first SOFA deterioration event per ICU stay and a 6-hour alert refractory period.

## Subgroups

| factor | group | patients | windows | positive | prevalence | AUROC (95% CI) | AUPRC (95% CI) | BRIER (95% CI) |
|---|---|---|---|---|---|---|---|---|
| age_group | 18-44 | 896 | 100968 | 4946 | 0.0489858165383338 | 0.6596 (0.6388-0.6792) | 0.1238 (0.1015-0.1466) | 0.0452 (0.0413-0.0493) |
| age_group | 45-64 | 2358 | 288396 | 15767 | 0.0546713545918464 | 0.6747 (0.6640-0.6863) | 0.1488 (0.1378-0.1629) | 0.0496 (0.0476-0.0517) |
| age_group | 65-79 | 2614 | 302794 | 17638 | 0.0582508258521556 | 0.6508 (0.6409-0.6608) | 0.1279 (0.1198-0.1358) | 0.0534 (0.0513-0.0554) |
| age_group | >=80 | 1538 | 138681 | 8941 | 0.0644716992974281 | 0.6272 (0.6125-0.6406) | 0.1141 (0.1022-0.1247) | 0.0594 (0.0557-0.0629) |
| sex_group | Female | 3162 | 349854 | 21596 | 0.0617286078631877 | 0.6458 (0.6363-0.6550) | 0.1294 (0.1204-0.1394) | 0.0564 (0.0544-0.0585) |
| sex_group | Male | 4125 | 480985 | 25696 | 0.0534237027168273 | 0.6645 (0.6564-0.6733) | 0.1329 (0.1248-0.1414) | 0.0489 (0.0475-0.0507) |
| ethnicity_group | Asian | 219 | 22811 | 1115 | 0.0488799251616001 | 0.6624 (0.6221-0.7012) | 0.1137 (0.0876-0.1442) | 0.0453 (0.0382-0.0527) |
| ethnicity_group | Black | 674 | 90556 | 5280 | 0.0583064630627632 | 0.6591 (0.6381-0.6784) | 0.1274 (0.1125-0.1476) | 0.0534 (0.0490-0.0576) |
| ethnicity_group | Hispanic/Latino | 260 | 33618 | 1775 | 0.0527990944683551 | 0.6650 (0.6394-0.6915) | 0.1414 (0.1103-0.1702) | 0.0482 (0.0425-0.0533) |
| ethnicity_group | Other/Unknown | 1435 | 164348 | 9391 | 0.0571409463882446 | 0.6471 (0.6330-0.6597) | 0.1278 (0.1128-0.1418) | 0.0524 (0.0490-0.0555) |
| ethnicity_group | White | 4770 | 519506 | 29731 | 0.0572293661534786 | 0.6570 (0.6497-0.6658) | 0.1330 (0.1258-0.1411) | 0.0523 (0.0506-0.0540) |
| icu_type_group | Cardiac | 2356 | 202859 | 10051 | 0.049546729773283 | 0.6834 (0.6701-0.6982) | 0.1346 (0.1233-0.1488) | 0.0454 (0.0425-0.0481) |
| icu_type_group | Medical/Mixed | 1645 | 177831 | 11432 | 0.0642857551574707 | 0.6728 (0.6604-0.6862) | 0.1464 (0.1347-0.1598) | 0.0581 (0.0549-0.0610) |
| icu_type_group | Neuro | 907 | 104161 | 6026 | 0.0578527487814426 | 0.5643 (0.5494-0.5804) | 0.0783 (0.0712-0.0877) | 0.0544 (0.0512-0.0582) |
| icu_type_group | Surgical/Trauma | 3202 | 343216 | 19551 | 0.0569641292095184 | 0.6569 (0.6477-0.6667) | 0.1360 (0.1260-0.1458) | 0.0520 (0.0503-0.0538) |
| current_sofa_group | 0-3 | 5129 | 258838 | 14609 | 0.056440707296133 | 0.6330 (0.6219-0.6440) | 0.1073 (0.0999-0.1156) | 0.0523 (0.0500-0.0547) |
| current_sofa_group | 4-7 | 5067 | 292997 | 16623 | 0.0567343682050704 | 0.6606 (0.6508-0.6702) | 0.1319 (0.1227-0.1414) | 0.0519 (0.0499-0.0538) |
| current_sofa_group | >=8 | 3494 | 279004 | 16060 | 0.0575619004666805 | 0.6830 (0.6712-0.6924) | 0.1536 (0.1426-0.1650) | 0.0520 (0.0497-0.0543) |

Subgroup analyses are exploratory and use patient-clustered confidence intervals. They do not establish causal fairness or absence of performance disparities.
ICU type and current-SOFA strata are assigned at the stay/window level; a patient may therefore contribute to more than one stratum, while bootstrap resampling still uses the patient as the cluster.

## Figures

- `figures/sofa_outcome_sensitivity.pdf`
- `figures/event_level_alarm_burden.pdf`
- `figures/mimic_subgroup_auroc_forest.pdf`
