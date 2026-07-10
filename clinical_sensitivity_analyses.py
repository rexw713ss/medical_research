"""Frozen-model outcome, alarm-burden, and subgroup sensitivity analyses."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from advanced_model_evaluation import (
    apply_platt_calibration,
    operating_metrics,
    patient_bootstrap,
    percentile_ci,
)
from cohort_tables_figures import broad_race, manuscript_markdown
from model_evaluation_report import binary_metrics


OUTPUT = Path("outputs/clinical_sensitivity_analyses_6h")
PREDICTIONS = Path("outputs/final_test_evaluation_6h/predictions/test_predictions.csv.gz")
METRICS = Path("outputs/final_test_evaluation_6h/advanced/advanced_metrics.csv")
SOFA = Path("sofa_scores_hourly.csv")
BOOTSTRAP_REPS = 500
SEED = 42
KEYS = ["stay_id", "sofa_hour"]


def load_predictions() -> tuple[pd.DataFrame, dict[float, float]]:
    predictions = pd.read_csv(PREDICTIONS)
    metrics = pd.read_csv(METRICS).iloc[0]
    predictions["y_prob_raw"] = pd.to_numeric(predictions["y_prob"], errors="raise")
    predictions["y_prob"] = apply_platt_calibration(
        predictions["y_prob_raw"].to_numpy(),
        float(metrics["validation_platt_intercept"]),
        float(metrics["validation_platt_slope"]),
    )
    thresholds = {
        0.90: float(metrics["threshold_spec_90"]),
        0.95: float(metrics["threshold_spec_95"]),
    }
    return predictions.sort_values(["stay_id", "sofa_hour"]).reset_index(drop=True), thresholds


def load_test_sofa(stay_ids: set[int]) -> pd.DataFrame:
    columns = [
        "subject_id",
        "stay_id",
        "sofa_hour",
        "sofa_score",
        "sofa_score_assume_normal",
        "sofa_score_complete",
        "sofa_component_count",
    ]
    parts = []
    for chunk in pd.read_csv(SOFA, usecols=columns, chunksize=500_000):
        part = chunk[chunk["stay_id"].isin(stay_ids)]
        if not part.empty:
            parts.append(part)
    if not parts:
        raise ValueError("No test SOFA rows found")
    return pd.concat(parts, ignore_index=True).sort_values(KEYS).reset_index(drop=True)


def add_alternative_label(frame: pd.DataFrame, score_col: str, label_col: str) -> pd.DataFrame:
    frame = frame.copy()
    frame[score_col] = pd.to_numeric(frame[score_col], errors="coerce")
    future = (
        frame.groupby("stay_id", group_keys=False)[score_col]
        .apply(lambda series: series.shift(-1).iloc[::-1].rolling(6, min_periods=1).max().iloc[::-1])
    )
    remaining = frame.groupby("stay_id")["sofa_hour"].transform("max") - frame["sofa_hour"]
    label = ((future - frame[score_col]) >= 2).astype("float64")
    label[(remaining < 6) | frame[score_col].isna() | future.isna()] = np.nan
    frame[label_col] = label
    return frame


def evaluate_frame(frame: pd.DataFrame, label: str, thresholds: dict[float, float]) -> tuple[dict, pd.DataFrame]:
    frame = frame.dropna(subset=["y_true", "y_prob"]).copy()
    frame["y_true"] = frame["y_true"].astype("int8")
    point = binary_metrics(frame["y_true"].to_numpy(), frame["y_prob"].to_numpy())
    bootstrap = patient_bootstrap(frame, thresholds, BOOTSTRAP_REPS, SEED)
    row = {
        "definition": label,
        "patients": int(frame["subject_id"].nunique()),
        "stays": int(frame["stay_id"].nunique()),
        "windows": int(len(frame)),
        "positive": int(frame["y_true"].sum()),
        "prevalence": float(frame["y_true"].mean()),
        **point,
    }
    for metric in [
        "auroc", "auprc", "brier", "ece",
        "sensitivity_at_spec_90", "specificity_at_spec_90",
        "sensitivity_at_spec_95", "specificity_at_spec_95",
    ]:
        if metric in bootstrap:
            low, high = percentile_ci(bootstrap[metric])
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
    return row, bootstrap.assign(definition=label)


def outcome_sensitivity(predictions: pd.DataFrame, sofa: pd.DataFrame, thresholds: dict[float, float]) -> None:
    assume = add_alternative_label(sofa, "sofa_score_assume_normal", "label_assume_normal")
    complete = add_alternative_label(sofa, "sofa_score_complete", "label_complete_case")
    definitions = {
        "Primary: >=4 observed components": predictions[[*KEYS, "subject_id", "y_true", "y_prob"]],
        "Missing components assumed normal": predictions[[*KEYS, "subject_id", "y_prob"]].merge(
            assume[[*KEYS, "label_assume_normal"]], on=KEYS, how="left"
        ).rename(columns={"label_assume_normal": "y_true"}),
        "Six-component complete case": predictions[[*KEYS, "subject_id", "y_prob"]].merge(
            complete[[*KEYS, "label_complete_case"]], on=KEYS, how="left"
        ).rename(columns={"label_complete_case": "y_true"}),
    }
    rows = []
    bootstraps = []
    for label, frame in definitions.items():
        row, bootstrap = evaluate_frame(frame, label, thresholds)
        rows.append(row)
        bootstraps.append(bootstrap)
        print(f"Outcome sensitivity: {label}: {row['windows']:,} windows")
    pd.DataFrame(rows).to_csv(OUTPUT / "sofa_outcome_definition_sensitivity.csv", index=False)
    pd.concat(bootstraps, ignore_index=True).to_csv(
        OUTPUT / "sofa_outcome_patient_bootstrap.csv.gz", index=False, compression="gzip"
    )


def build_first_events(predictions: pd.DataFrame, sofa: pd.DataFrame) -> pd.DataFrame:
    score_lookup = {
        stay: group.set_index("sofa_hour")["sofa_score"]
        for stay, group in sofa.groupby("stay_id", sort=False)
    }
    rows = []
    positive = predictions[predictions["y_true"].eq(1)]
    for stay_id, group in positive.groupby("stay_id", sort=False):
        index_hour = int(group["sofa_hour"].min())
        scores = score_lookup.get(stay_id)
        if scores is None or index_hour not in scores.index or pd.isna(scores.loc[index_hour]):
            continue
        baseline = float(scores.loc[index_hour])
        future = scores.loc[(scores.index > index_hour) & (scores.index <= index_hour + 6)]
        hit = future[future >= baseline + 2]
        if hit.empty:
            continue
        first = group.iloc[0]
        rows.append(
            {
                "subject_id": int(first["subject_id"]),
                "stay_id": int(stay_id),
                "event_hour": int(hit.index[0]),
                "baseline_window_hour": index_hour,
                "baseline_sofa": baseline,
                "event_sofa": float(hit.iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def refractory_alerts(frame: pd.DataFrame, threshold: float, refractory_h: int = 6) -> pd.DataFrame:
    alerts = frame[frame["y_prob"] >= threshold].sort_values(["stay_id", "sofa_hour"])
    kept = []
    for _, group in alerts.groupby("stay_id", sort=False):
        last = -10**9
        for row in group.itertuples(index=False):
            hour = int(row.sofa_hour)
            if hour - last >= refractory_h:
                kept.append(row)
                last = hour
    return pd.DataFrame(kept, columns=alerts.columns) if kept else alerts.iloc[:0].copy()


def alarm_burden(predictions: pd.DataFrame, sofa: pd.DataFrame, thresholds: dict[float, float]) -> None:
    events = build_first_events(predictions, sofa)
    events.to_csv(OUTPUT / "first_deterioration_events.csv", index=False)
    event_by_stay = events.set_index("stay_id") if not events.empty else pd.DataFrame()
    rows = []
    detail_parts = []
    for target_specificity, threshold in thresholds.items():
        analysis = predictions.copy()
        if not events.empty:
            event_hour = analysis["stay_id"].map(events.set_index("stay_id")["event_hour"])
            analysis = analysis[event_hour.isna() | (analysis["sofa_hour"] < event_hour)].copy()
        alerts = refractory_alerts(analysis, threshold, refractory_h=6)
        alerts["target_specificity"] = target_specificity
        alerts["true_event_alert"] = 0
        alerts["lead_time_h"] = np.nan
        for stay_id, event in event_by_stay.iterrows() if not events.empty else []:
            mask = (
                alerts["stay_id"].eq(stay_id)
                & alerts["sofa_hour"].lt(event["event_hour"])
                & alerts["sofa_hour"].ge(event["event_hour"] - 6)
            )
            alerts.loc[mask, "true_event_alert"] = 1
            alerts.loc[mask, "lead_time_h"] = event["event_hour"] - alerts.loc[mask, "sofa_hour"]
        event_details = []
        for event in events.itertuples(index=False):
            matching = alerts[
                alerts["stay_id"].eq(event.stay_id)
                & alerts["sofa_hour"].lt(event.event_hour)
                & alerts["sofa_hour"].ge(event.event_hour - 6)
            ]
            event_details.append(
                {
                    "target_specificity": target_specificity,
                    "subject_id": event.subject_id,
                    "stay_id": event.stay_id,
                    "event_hour": event.event_hour,
                    "detected": int(not matching.empty),
                    "lead_time_h": float(event.event_hour - matching["sofa_hour"].min()) if not matching.empty else np.nan,
                }
            )
        event_detail = pd.DataFrame(event_details)
        patient_days = len(analysis) / 24.0
        false_alerts = alerts[alerts["true_event_alert"].eq(0)]
        patients_with_false = false_alerts["subject_id"].nunique()
        detected = int(event_detail["detected"].sum()) if not event_detail.empty else 0
        true_alerts = int(alerts["true_event_alert"].sum())
        rows.append(
            {
                "target_specificity": target_specificity,
                "threshold": threshold,
                "refractory_hours": 6,
                "first_events": len(events),
                "detected_events": detected,
                "event_sensitivity": detected / len(events) if len(events) else np.nan,
                "alerts": len(alerts),
                "true_event_alerts": true_alerts,
                "false_alerts": len(false_alerts),
                "alert_ppv": true_alerts / len(alerts) if len(alerts) else np.nan,
                "patient_days_observed": patient_days,
                "alerts_per_100_patient_days": 100 * len(alerts) / patient_days,
                "false_alerts_per_100_patient_days": 100 * len(false_alerts) / patient_days,
                "patients_with_false_alert": patients_with_false,
                "patients_with_false_alert_fraction": patients_with_false / analysis["subject_id"].nunique(),
                "lead_time_median_h": float(event_detail["lead_time_h"].median()) if detected else np.nan,
                "lead_time_q1_h": float(event_detail["lead_time_h"].quantile(0.25)) if detected else np.nan,
                "lead_time_q3_h": float(event_detail["lead_time_h"].quantile(0.75)) if detected else np.nan,
            }
        )
        detail_parts.append(event_detail)
        alerts.to_csv(OUTPUT / f"alarm_details_spec_{int(target_specificity * 100)}.csv.gz", index=False, compression="gzip")
    pd.DataFrame(rows).to_csv(OUTPUT / "event_level_alarm_burden.csv", index=False)
    pd.concat(detail_parts, ignore_index=True).to_csv(OUTPUT / "event_detection_details.csv", index=False)


def icu_category(value: object) -> str:
    text = str(value).upper()
    if "NEURO" in text:
        return "Neuro"
    if "CARDIAC" in text or "CORONARY" in text or "CVICU" in text or "CCU" in text:
        return "Cardiac"
    if "SURG" in text or "TRAUMA" in text or "PACU" in text:
        return "Surgical/Trauma"
    if "MEDICAL" in text or "MEDICINE" in text or "MICU" in text:
        return "Medical/Mixed"
    return "Other"


def add_demographics(predictions: pd.DataFrame, sofa: pd.DataFrame) -> pd.DataFrame:
    icu = pd.read_csv(
        "dataset/MIMIC-IV/icustays.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "first_careunit", "intime"],
        parse_dates=["intime"],
    )
    patients = pd.read_csv(
        "dataset/MIMIC-IV/patients.csv.gz",
        usecols=["subject_id", "gender", "anchor_age", "anchor_year"],
    )
    admissions = pd.read_csv("dataset/MIMIC-IV/admissions.csv.gz", usecols=["hadm_id", "race"])
    demo = icu.merge(patients, on="subject_id", validate="many_to_one").merge(
        admissions, on="hadm_id", validate="many_to_one"
    )
    demo["age"] = demo["anchor_age"] + demo["intime"].dt.year - demo["anchor_year"]
    demo["age_group"] = pd.cut(
        demo["age"], [17, 44, 64, 79, np.inf], labels=["18-44", "45-64", "65-79", ">=80"]
    ).astype(str)
    demo["sex_group"] = demo["gender"].map({"F": "Female", "M": "Male"}).fillna("Unknown")
    demo["ethnicity_group"] = broad_race(demo["race"])
    demo["icu_type_group"] = demo["first_careunit"].map(icu_category)
    current = sofa[["stay_id", "sofa_hour", "sofa_score"]].drop_duplicates(KEYS)
    frame = predictions.merge(
        demo[["stay_id", "age_group", "sex_group", "ethnicity_group", "icu_type_group"]],
        on="stay_id", how="left", validate="many_to_one",
    ).merge(current, on=KEYS, how="left", validate="many_to_one")
    frame["current_sofa_group"] = pd.cut(
        frame["sofa_score"], [-np.inf, 3, 7, np.inf], labels=["0-3", "4-7", ">=8"]
    ).astype(str).replace("nan", "Unknown")
    return frame


def subgroup_analysis(predictions: pd.DataFrame, sofa: pd.DataFrame, thresholds: dict[float, float]) -> None:
    frame = add_demographics(predictions, sofa)
    factors = ["age_group", "sex_group", "ethnicity_group", "icu_type_group", "current_sofa_group"]
    rows = []
    bootstrap_parts = []
    for factor in factors:
        for group, part in frame.groupby(factor, dropna=False, sort=True):
            part = part.dropna(subset=["y_true", "y_prob"])
            positive = int(part["y_true"].sum())
            negative = int(len(part) - positive)
            if part["subject_id"].nunique() < 100 or positive < 20 or negative < 20:
                continue
            point = binary_metrics(part["y_true"].to_numpy(), part["y_prob"].to_numpy())
            operating90 = operating_metrics(part["y_true"].to_numpy(), part["y_prob"].to_numpy(), thresholds[0.90])
            bootstrap = patient_bootstrap(part, {0.90: thresholds[0.90]}, BOOTSTRAP_REPS, SEED)
            row = {
                "factor": factor,
                "group": str(group),
                "patients": int(part["subject_id"].nunique()),
                "stays": int(part["stay_id"].nunique()),
                "windows": len(part),
                "positive": positive,
                "prevalence": positive / len(part),
                **point,
                "sensitivity_at_global_spec90_threshold": operating90["sensitivity"],
                "specificity_at_global_spec90_threshold": operating90["specificity"],
            }
            for metric in ["auroc", "auprc", "brier", "sensitivity_at_spec_90", "specificity_at_spec_90"]:
                low, high = percentile_ci(bootstrap[metric])
                row[f"{metric}_ci95_low"] = low
                row[f"{metric}_ci95_high"] = high
            rows.append(row)
            bootstrap.insert(0, "group", str(group))
            bootstrap.insert(0, "factor", factor)
            bootstrap_parts.append(bootstrap)
            print(f"Subgroup {factor}/{group}: {len(part):,} windows")
    pd.DataFrame(rows).to_csv(OUTPUT / "mimic_subgroup_performance.csv", index=False)
    pd.concat(bootstrap_parts, ignore_index=True).to_csv(
        OUTPUT / "mimic_subgroup_patient_bootstrap.csv.gz", index=False, compression="gzip"
    )


def save_figures() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = OUTPUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outcome = pd.read_csv(OUTPUT / "sofa_outcome_definition_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].barh(outcome["definition"], outcome["auroc"], color="#4E79A7")
    axes[1].barh(outcome["definition"], outcome["auprc"], color="#F28E2B")
    axes[0].set(xlabel="AUROC", xlim=(0.5, max(0.7, outcome["auroc"].max() + 0.03)))
    axes[1].set(xlabel="AUPRC", xlim=(0, outcome["auprc"].max() + 0.03))
    fig.suptitle("SOFA Outcome-Definition Sensitivity")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"sofa_outcome_sensitivity.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    alarm = pd.read_csv(OUTPUT / "event_level_alarm_burden.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    labels = [f"{value:.0%} specificity" for value in alarm["target_specificity"]]
    axes[0].bar(labels, alarm["event_sensitivity"], color="#59A14F")
    axes[0].set(ylabel="Event sensitivity", ylim=(0, 1))
    axes[1].bar(labels, alarm["false_alerts_per_100_patient_days"], color="#E15759")
    axes[1].set(ylabel="False alerts per 100 patient-days")
    fig.suptitle("Event-Level Alarm Burden")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"event_level_alarm_burden.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    subgroup = pd.read_csv(OUTPUT / "mimic_subgroup_performance.csv")
    subgroup = subgroup.sort_values(["factor", "auroc"], ascending=[True, True]).reset_index(drop=True)
    labels = [f"{factor.replace('_group', '')}: {group}" for factor, group in zip(subgroup["factor"], subgroup["group"])]
    y_positions = np.arange(len(subgroup))
    lower = subgroup["auroc"] - subgroup["auroc_ci95_low"]
    upper = subgroup["auroc_ci95_high"] - subgroup["auroc"]
    palette = {
        factor: color
        for factor, color in zip(
            subgroup["factor"].drop_duplicates(),
            ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1"],
        )
    }
    colors = subgroup["factor"].map(palette)
    fig_height = max(6.5, 0.38 * len(subgroup))
    fig, axis = plt.subplots(figsize=(9.5, fig_height))
    axis.errorbar(
        subgroup["auroc"],
        y_positions,
        xerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="#777777",
        elinewidth=1.2,
        capsize=2.5,
        zorder=1,
    )
    axis.scatter(subgroup["auroc"], y_positions, c=colors, s=34, zorder=2)
    axis.axvline(0.5, color="#555555", linestyle="--", linewidth=1)
    axis.set(
        yticks=y_positions,
        yticklabels=labels,
        xlabel="AUROC with patient-clustered 95% CI",
        title="MIMIC-IV Subgroup Performance",
    )
    axis.set_xlim(min(0.45, subgroup["auroc_ci95_low"].min() - 0.02), min(1.0, subgroup["auroc_ci95_high"].max() + 0.03))
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.7)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"mimic_subgroup_auroc_forest.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report() -> None:
    outcome = pd.read_csv(OUTPUT / "sofa_outcome_definition_sensitivity.csv")
    alarm = pd.read_csv(OUTPUT / "event_level_alarm_burden.csv")
    subgroup = pd.read_csv(OUTPUT / "mimic_subgroup_performance.csv")
    outcome_report = outcome[["definition", "patients", "windows", "prevalence"]].copy()
    for metric in ["auroc", "auprc", "brier"]:
        outcome_report[f"{metric.upper()} (95% CI)"] = outcome.apply(
            lambda row: (
                f"{row[metric]:.4f} "
                f"({row[f'{metric}_ci95_low']:.4f}-{row[f'{metric}_ci95_high']:.4f})"
            ),
            axis=1,
        )
    alarm_report = alarm[
        [
            "target_specificity",
            "first_events",
            "detected_events",
            "event_sensitivity",
            "alerts",
            "alert_ppv",
            "false_alerts_per_100_patient_days",
            "patients_with_false_alert_fraction",
            "lead_time_median_h",
            "lead_time_q1_h",
            "lead_time_q3_h",
        ]
    ].copy()
    subgroup_report = subgroup[["factor", "group", "patients", "windows", "positive", "prevalence"]].copy()
    for metric in ["auroc", "auprc", "brier"]:
        subgroup_report[f"{metric.upper()} (95% CI)"] = subgroup.apply(
            lambda row: (
                f"{row[metric]:.4f} "
                f"({row[f'{metric}_ci95_low']:.4f}-{row[f'{metric}_ci95_high']:.4f})"
            ),
            axis=1,
        )
    lines = [
        "# Clinical Sensitivity Analyses",
        "",
        "All analyses use the frozen full-cohort MIMIC final model. No checkpoint, calibration, or threshold was refit.",
        "",
        "## SOFA Outcome Definition",
        "",
        manuscript_markdown(outcome_report),
        "",
        "## Event-Level Alarm Burden",
        "",
        manuscript_markdown(alarm_report),
        "",
        "The event analysis uses the first SOFA deterioration event per ICU stay and a 6-hour alert refractory period.",
        "",
        "## Subgroups",
        "",
        manuscript_markdown(subgroup_report),
        "",
        "Subgroup analyses are exploratory and use patient-clustered confidence intervals. They do not establish causal fairness or absence of performance disparities.",
        "ICU type and current-SOFA strata are assigned at the stay/window level; a patient may therefore contribute to more than one stratum, while bootstrap resampling still uses the patient as the cluster.",
        "",
        "## Figures",
        "",
        "- `figures/sofa_outcome_sensitivity.pdf`",
        "- `figures/event_level_alarm_burden.pdf`",
        "- `figures/mimic_subgroup_auroc_forest.pdf`",
    ]
    (OUTPUT / "clinical_sensitivity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions, thresholds = load_predictions()
    print("Loading test-stay SOFA history...")
    sofa = load_test_sofa(set(predictions["stay_id"].unique()))
    outcome_sensitivity(predictions, sofa, thresholds)
    alarm_burden(predictions, sofa, thresholds)
    subgroup_analysis(predictions, sofa, thresholds)
    save_figures()
    write_report()
    config = {
        "model": "frozen full-cohort explicit KG-TFNN",
        "prediction_file": str(PREDICTIONS),
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_unit": "subject_id",
        "alarm_refractory_hours": 6,
        "event_definition": "first SOFA increase >=2 event per ICU stay",
        "thresholds": thresholds,
        "no_model_refitting": True,
    }
    (OUTPUT / "analysis_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
