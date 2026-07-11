"""Build manuscript-ready cohort, SOFA, calibration, alarm, and site reports.

This script does not fit a model or inspect test outcomes for model selection. It
only consolidates prespecified definitions and frozen analysis outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "expanded_experiment_reporting_6h"
TARGET = "label_sofa_increase_ge2_6h"
SEQUENCE_LENGTH = 24


def markdown_table(frame: pd.DataFrame) -> str:
    headers = "| " + " | ".join(frame.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([headers, rule, *rows])


def analytic_mimic_counts() -> pd.DataFrame:
    cached = OUTPUT / "mimic_analytic_denominators.csv"
    sources = [ROOT / "model_hourly_features_v3.csv", ROOT / "patient_split.csv"]
    if cached.exists() and all(cached.stat().st_mtime >= path.stat().st_mtime for path in sources):
        return pd.read_csv(cached)
    split = pd.read_csv(ROOT / "patient_split.csv", usecols=["subject_id", "split"])
    split_map = split.set_index("subject_id")["split"]
    split_names = ("train", "validation", "test")
    accum = {name: {"patients": set(), "stays": set(), "windows": 0, "positive": 0} for name in split_names}
    columns = ["subject_id", "stay_id", "sofa_hour", TARGET]
    for chunk in pd.read_csv(ROOT / "model_hourly_features_v3.csv", usecols=columns, chunksize=500_000):
        chunk["split"] = chunk["subject_id"].map(split_map)
        eligible = chunk[TARGET].notna() & chunk["sofa_hour"].ge(SEQUENCE_LENGTH - 1) & chunk["split"].notna()
        for name, group in chunk.loc[eligible].groupby("split", sort=False):
            item = accum[str(name)]
            item["patients"].update(group["subject_id"].astype("int64").unique().tolist())
            item["stays"].update(group["stay_id"].astype("int64").unique().tolist())
            item["windows"] += len(group)
            item["positive"] += int(group[TARGET].sum())
    rows = []
    for name in split_names:
        item = accum[name]
        rows.append({
            "database": "MIMIC-IV",
            "analysis_set": name,
            "patients": len(item["patients"]),
            "stays": len(item["stays"]),
            "windows": item["windows"],
            "positive_windows": item["positive"],
            "prevalence": item["positive"] / item["windows"],
        })
    return pd.DataFrame(rows)


def cohort_exclusions(mimic: pd.DataFrame) -> pd.DataFrame:
    eligibility = json.loads((ROOT / "outputs/manuscript_tables_figures_6h/adult_eligibility_audit.json").read_text())
    eicu_quality = json.loads((ROOT / "outputs/eicu_external_validation/eicu_hourly_quality.json").read_text())
    eicu_final = json.loads((ROOT / "outputs/eicu_external_validation/final_frozen_model_evaluation/external_metrics.json").read_text())
    m = eligibility["mimic_iv"]
    e = eligibility["eicu_crd"]
    mimic_windows = int(mimic["windows"].sum())
    rows = [
        ["MIMIC-IV", "Source ICU cohort", m["source_patients"], m["source_icu_stays"], None],
        ["MIMIC-IV", "Excluded: invalid ICU time", None, m["excluded_missing_or_invalid_icu_time_stays"], None],
        ["MIMIC-IV", "Excluded: missing age", None, m["excluded_missing_age_stays"], None],
        ["MIMIC-IV", "Excluded: age <18", m["excluded_age_below_minimum_patients"], m["excluded_age_below_minimum_stays"], None],
        ["MIMIC-IV", "Adult valid ICU cohort", m["eligible_adult_patients"], m["eligible_adult_icu_stays"], 8_275_274],
        ["MIMIC-IV", "Excluded stay-hours: invalid/incomplete 6-h SOFA outcome", None, None, 8_275_274 - 6_938_122],
        ["MIMIC-IV", "Valid 6-h outcome hours", None, None, 6_938_122],
        ["MIMIC-IV", "Excluded valid-label hours: insufficient 24-h history", None, None, 6_938_122 - mimic_windows],
        ["MIMIC-IV", "Final analytic windows", int(mimic["patients"].sum()), int(mimic["stays"].sum()), mimic_windows],
        ["eICU-CRD", "Source ICU cohort", e["source_patients"], e["source_icu_stays"], None],
        ["eICU-CRD", "Excluded: missing age", None, e["excluded_missing_age_stays"], None],
        ["eICU-CRD", "Excluded: age <18", e["excluded_age_below_minimum_patients"], e["excluded_age_below_minimum_stays"], None],
        ["eICU-CRD", "Excluded: invalid ICU duration", None, e["excluded_missing_or_invalid_duration_stays"], None],
        ["eICU-CRD", "Adult valid ICU cohort", e["eligible_adult_patients"], e["eligible_adult_icu_stays"], eicu_quality["rows"]],
        ["eICU-CRD", "Excluded stay-hours: invalid/incomplete 6-h SOFA outcome", None, None, eicu_quality["rows"] - eicu_quality["labels"][TARGET]["valid_rows"]],
        ["eICU-CRD", "Valid 6-h outcome hours", None, None, eicu_quality["labels"][TARGET]["valid_rows"]],
        ["eICU-CRD", "Excluded valid-label hours: insufficient 24-h history", None, None, eicu_quality["labels"][TARGET]["valid_rows"] - eicu_final["windows"]],
        ["eICU-CRD", "Final external analytic windows", eicu_final["patients"], eicu_final["stays"], eicu_final["windows"]],
    ]
    return pd.DataFrame(rows, columns=["database", "stage_or_exclusion", "patients", "stays", "stay_hours_or_windows"])


def calibration_results() -> pd.DataFrame:
    mimic = pd.read_csv(ROOT / "outputs/final_test_evaluation_6h/advanced/advanced_metrics.csv").iloc[0]
    external = json.loads((ROOT / "outputs/eicu_external_validation/final_frozen_model_evaluation/external_metrics.json").read_text())
    rows = [
        ["MIMIC-IV test", "Raw model output", mimic.raw_brier, mimic.raw_ece, mimic.raw_log_loss, mimic.raw_calibration_intercept, mimic.raw_calibration_slope],
        ["MIMIC-IV test", "Validation-only Platt calibrated", mimic.brier, mimic.ece, mimic.log_loss, mimic.calibration_intercept, mimic.calibration_slope],
        ["eICU external", "Raw model output", external["raw"]["brier"], external["raw"]["ece"], external["raw"]["log_loss"], external["raw"]["calibration_intercept"], external["raw"]["calibration_slope"]],
        ["eICU external", "Transported MIMIC calibration", external["mimic_calibrated"]["brier"], external["mimic_calibrated"]["ece"], external["mimic_calibrated"]["log_loss"], external["mimic_calibrated"]["calibration_intercept"], external["mimic_calibrated"]["calibration_slope"]],
    ]
    return pd.DataFrame(rows, columns=["cohort", "probability_scale", "brier", "ece", "log_loss", "calibration_intercept", "calibration_slope"])


def sofa_harmonization() -> pd.DataFrame:
    rows = [
        ["Respiratory", "Worst P/F ratio; invasive support status", "PaO2 lab + charted FiO2 + ventilation", "PaO2 lab + respiratoryCharting FiO2/support", "Same thresholds; database-specific source mapping"],
        ["Coagulation", "Minimum platelet count", "Platelet lab", "Platelet lab (10^3/uL harmonized)", "Same thresholds"],
        ["Liver", "Maximum total bilirubin", "Total bilirubin lab", "Total bilirubin lab", "Same thresholds"],
        ["Cardiovascular", "Minimum MAP and maximum vasopressor dose", "MAP + inputevents pressors", "MAP + infusionDrug pressors", "Rates normalized to mcg/kg/min; unconvertible eICU doses excluded"],
        ["Central nervous system", "Minimum total GCS", "Total GCS or eye+verbal+motor", "nurseCharting total GCS", "Same thresholds"],
        ["Renal", "Maximum creatinine and 24-h urine sum; worse score", "Creatinine lab + outputevents urine", "Creatinine lab + intakeOutput urine", "Urine criterion available only after a complete trailing 24 h"],
    ]
    return pd.DataFrame(rows, columns=["component", "24h_worst_value", "mimic_iv_source", "eicu_source", "harmonization_rule"])


def write_report(mimic: pd.DataFrame, exclusions: pd.DataFrame, calibration: pd.DataFrame, sofa: pd.DataFrame) -> None:
    alarm = pd.read_csv(ROOT / "outputs/clinical_sensitivity_analyses_6h/event_level_alarm_burden.csv")
    sites = pd.read_csv(ROOT / "outputs/eicu_hospital_sensitivity_6h/hospital_heterogeneity_summary.csv")
    raw_rule = pd.read_csv(ROOT / "outputs/raw_rule_firing_6h/activation_threshold_summary.csv")
    threshold_10 = raw_rule[(raw_rule["basis"] == "current_hour") & raw_rule["threshold"].eq(0.10)]
    report = [
        "# Expanded Experiment Reporting (Primary 6-hour Task)", "",
        "This report consolidates frozen analyses. It does not refit the model, calibration, or operating thresholds.", "",
        "## Analytic Denominators", "", markdown_table(mimic.assign(prevalence=lambda x: x.prevalence.map(lambda v: f"{v:.4f}"))), "",
        "Counts are mutually exclusive by `subject_id`. Final windows require a non-missing 6-hour outcome and 24 hourly observations ending at the index hour.", "",
        "## Sequential Cohort Accounting", "", markdown_table(exclusions.fillna("-")), "",
        "Patient/stay exclusions and stay-hour exclusions are separate denominators and must not be subtracted from one another. The final MIMIC patient total is the sum of mutually exclusive split-specific patients; eICU is a single frozen external cohort.", "",
        "## SOFA Construction and Harmonization", "",
        "At each index hour, SOFA uses the worst value in the current and preceding 23 hours. The six standard components are scored 0-4 and summed. The primary score requires at least four observed components; missing-as-normal and all-six-component complete-case definitions are sensitivity analyses. FiO2 may be carried forward within the stay for SOFA scoring, but no future value or backward fill is used.", "",
        markdown_table(sofa), "",
        "The outcome is positive when the maximum valid SOFA in `(t, t+6]` is at least two points above SOFA at `t`. A complete six-hour future horizon is required. The same scoring and label functions are called after database-specific source and unit mapping.", "",
        "## Raw and Calibrated Calibration", "", markdown_table(calibration.round(6)), "",
        "The raw output is retained because class-weighted training does not target calibrated absolute risk. Platt parameters are fitted on MIMIC validation predictions only, then frozen for MIMIC test and transported unchanged to eICU. External calibration is therefore a transport result, not eICU recalibration.", "",
        "## Raw Rule Firing", "",
        "Raw cross-rule firing is the product t-norm before normalization. A rule is counted as activated when raw firing is at least the prespecified threshold. Results are reported for the current index hour and the attention-selected hour across thresholds 0.01, 0.025, 0.05, 0.10, 0.20, 0.35, and 0.50.", "",
        markdown_table(threshold_10.round(4)), "",
        "Rule agreement is named **Guideline-direction alignment**. It measures consistency with prespecified NEWS2/SOFA directions and is not clinician adjudication or clinician-validated interpretability.", "",
        "## Event-Level Alarm Definition", "",
        "- Event: the first hour in each ICU stay at which SOFA reaches at least the index-hour SOFA plus 2 within the next 6 hours. Only the first qualifying event per stay is evaluated.",
        "- Alert: calibrated risk at or above a threshold selected on MIMIC validation data to target 90% or 95% window-level specificity.",
        "- Refractory period: after a retained alert, further alerts in the same stay are suppressed for 6 hours.",
        "- Detection: at least one retained alert in `[event hour - 6, event hour)`. Lead time is measured from the earliest matching retained alert.",
        "- Burden denominator: pre-event analytic stay-hours divided by 24; alerts after the first event are excluded. A false alert is a retained alert outside a qualifying pre-event window.", "",
        markdown_table(alarm.round(4)), "",
        "Window-level specificity and event sensitivity are distinct estimands and must not be described interchangeably.", "",
        "## Site-Level eICU Analysis", "",
        "The pooled frozen external analysis includes 205 hospitals. Per-hospital estimates are reported only for hospitals with at least 100 patients, 50 positive windows, and 50 negative windows. Hospital-clustered uncertainty uses 500 bootstrap replicates that resample `hospital_id`; this is a sensitivity analysis and does not replace the prespecified patient-clustered primary external confidence intervals.", "",
        markdown_table(sites.round(4)), "",
        "No eICU outcome was used for retraining, checkpoint selection, calibration fitting, or threshold selection.",
    ]
    (OUTPUT / "expanded_experiment_reporting.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    mimic = analytic_mimic_counts()
    exclusions = cohort_exclusions(mimic)
    calibration = calibration_results()
    sofa = sofa_harmonization()
    mimic.to_csv(OUTPUT / "mimic_analytic_denominators.csv", index=False)
    exclusions.to_csv(OUTPUT / "analytic_cohort_exclusions.csv", index=False)
    calibration.to_csv(OUTPUT / "raw_calibrated_calibration.csv", index=False)
    sofa.to_csv(OUTPUT / "sofa_cross_database_harmonization.csv", index=False)
    write_report(mimic, exclusions, calibration, sofa)
    config = {
        "primary_outcome": "future 6-hour SOFA increase >=2",
        "observation_window_hours": SEQUENCE_LENGTH,
        "patient_split": "subject_id",
        "reporting_only": True,
        "model_refit": False,
        "external_recalibration": False,
    }
    (OUTPUT / "reporting_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Wrote reporting artifacts to {OUTPUT}")


if __name__ == "__main__":
    main()
