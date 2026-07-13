"""Generate manuscript Supplementary Tables S1-S2 from implementation constants."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from anfis_model import expert_feature_config


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs" / "supplementary_material"
PAPER = ROOT / "paper"


SOFA_ROWS = [
    {
        "Organ system": "Respiratory",
        "Clinical variable": "PaO2/FiO2 + respiratory support",
        "Score 0": ">=400",
        "Score 1": "<400",
        "Score 2": "<300",
        "Score 3": "<200 + support",
        "Score 4": "<100 + support",
        "Hourly / 24-h aggregation": "Hourly minimum P/F and maximum support; trailing 24-h minimum P/F and maximum support",
        "MIMIC-IV source": "labevents PaO2 50821; chartevents FiO2 223835 and ventilation 223848 / 223849 / 229314 / 225792 / 225794",
        "eICU source": "lab PaO2; respiratoryCharting FiO2, PEEP, PEEP/CPAP, RT vent on/off",
    },
    {
        "Organ system": "Coagulation",
        "Clinical variable": "Platelets, x10^3/uL",
        "Score 0": ">=150",
        "Score 1": "100-149",
        "Score 2": "50-99",
        "Score 3": "20-49",
        "Score 4": "<20",
        "Hourly / 24-h aggregation": "Hourly minimum; trailing 24-h minimum",
        "MIMIC-IV source": "labevents itemid 51265",
        "eICU source": "lab name 'platelets x 1000'",
    },
    {
        "Organ system": "Liver",
        "Clinical variable": "Total bilirubin, mg/dL",
        "Score 0": "<1.2",
        "Score 1": "1.2-1.9",
        "Score 2": "2.0-5.9",
        "Score 3": "6.0-11.9",
        "Score 4": ">=12.0",
        "Hourly / 24-h aggregation": "Hourly maximum; trailing 24-h maximum",
        "MIMIC-IV source": "labevents itemid 50885",
        "eICU source": "lab name 'total bilirubin'",
    },
    {
        "Organ system": "Cardiovascular",
        "Clinical variable": "MAP / vasopressors, mcg/kg/min",
        "Score 0": "MAP >=70 and no qualifying pressor",
        "Score 1": "MAP <70",
        "Score 2": "dopamine <=5 or any dobutamine",
        "Score 3": "dopamine >5 to 15 or epinephrine/norepinephrine <=0.1",
        "Score 4": "dopamine >15 or epinephrine/norepinephrine >0.1",
        "Hourly / 24-h aggregation": "Hourly minimum MAP and maximum active dose; trailing 24-h minimum MAP and maximum dose; take worst criterion",
        "MIMIC-IV source": "chartevents MAP 220052/220181; inputevents dopamine 221662, dobutamine 221653, epinephrine 221289/229617, norepinephrine 221906",
        "eICU source": "periodic/aperiodic MAP; infusionDrug dopamine/dobutamine/epinephrine/norepinephrine",
    },
    {
        "Organ system": "Neurological",
        "Clinical variable": "GCS total",
        "Score 0": "15",
        "Score 1": "13-14",
        "Score 2": "10-12",
        "Score 3": "6-9",
        "Score 4": "<6",
        "Hourly / 24-h aggregation": "Hourly minimum total GCS; trailing 24-h minimum",
        "MIMIC-IV source": "chartevents eye 220739 + verbal 223900 + motor 223901",
        "eICU source": "nurseCharting GCS total",
    },
    {
        "Organ system": "Renal",
        "Clinical variable": "Creatinine, mg/dL / urine output, mL/24 h",
        "Score 0": "creatinine <1.2 / urine >=500; worse observed criterion",
        "Score 1": "creatinine 1.2-1.9",
        "Score 2": "creatinine 2.0-3.4",
        "Score 3": "creatinine 3.5-4.9 or urine <500",
        "Score 4": "creatinine >=5.0 or urine <200",
        "Hourly / 24-h aggregation": "Hourly maximum creatinine and sum of all qualifying urine records; trailing 24-h maximum creatinine and urine sum; take higher score",
        "MIMIC-IV source": "labevents creatinine 50912; outputevents urinary itemids 226557-226567, 226584, 226627, 226631, 227489",
        "eICU source": "lab name 'creatinine'; intakeOutput urinary labels",
    },
]


NEWS2_INTERVALS = {
    "heart_rate": {
        "very_low": ("<=40", "3"), "low_overlap": ("41-50", "1"),
        "normal": ("51-90", "0"),
        "mild_high": ("91-110", "1"), "high": ("111-130", "2"),
        "critical_high": (">=131", "3"),
    },
    "respiratory_rate": {
        "very_low": ("<=8", "3"), "mild_low": ("9-11", "1"),
        "normal": ("12-20", "0"), "high": ("21-24", "2"),
        "critical_high": (">=25", "3"),
    },
    "spo2": {
        "critical_low": ("<=91 (Scale 1)", "3"), "low": ("92-93 (Scale 1)", "2"),
        "mild_low": ("94-95 (Scale 1)", "1"), "normal": (">=96 (Scale 1)", "0"),
    },
    "fio2": {
        "room_air": ("FiO2 <=0.21 proxy", "0"),
        "supplemental_o2": ("FiO2 >0.21 proxy", "2"),
        "high_support": ("FiO2 >0.21; higher support", "2"),
    },
    "temperature_c": {
        "very_low": ("<=35.0", "3"), "mild_low": ("35.1-36.0", "1"),
        "normal": ("36.1-38.0", "0"), "fever": ("38.1-39.0", "1"),
        "high_fever": (">=39.1", "2"),
    },
    "sbp": {
        "very_low": ("<=90", "3"), "low": ("91-100", "2"),
        "mild_low": ("101-110", "1"), "normal": ("111-219", "0"),
        "very_high": (">=220", "3"),
    },
    "gcs_total": {
        "severely_altered": ("<15 proxy for C/V/P/U", "3"),
        "altered": ("<15 proxy for C/V/P/U", "3"),
        "normal": ("15 proxy for Alert", "0"),
    },
}


DISPLAY_NAMES = {
    "heart_rate": "Heart rate, beats/min",
    "respiratory_rate": "Respiratory rate, breaths/min",
    "spo2": "SpO2, %",
    "fio2": "Supplemental oxygen / FiO2 fraction",
    "temperature_c": "Temperature, C",
    "sbp": "Systolic blood pressure, mmHg",
    "gcs_total": "Consciousness (GCS proxy)",
}


def news2_fuzzy_mapping() -> pd.DataFrame:
    rows = []
    for feature, mappings in NEWS2_INTERVALS.items():
        by_name = {entry["name"]: entry for entry in expert_feature_config[feature]}
        for term, (interval, points) in mappings.items():
            if feature == "heart_rate" and term == "low_overlap":
                fuzzy_term = "overlap: very_low + normal; no dedicated term"
                center: object = "35 / 70"
                sigma: object = "6 / 15"
                weight: object = "3 / 0"
            else:
                config = by_name[term]
                fuzzy_term = term
                center = config["center"]
                sigma = config["sigma"]
                weight = config["weight"]
            rows.append({
                "Variable": DISPLAY_NAMES[feature],
                "NEWS2 interval / operational proxy": interval,
                "NEWS2 points": points,
                "KG-TFNN fuzzy term": fuzzy_term,
                "Initial center": center,
                "Initial width (sigma)": sigma,
                "Initial risk weight": weight,
            })
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    display.columns = [str(column) for column in display.columns]
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    body = ["| " + " | ".join(str(value).replace("|", "/") for value in row) + " |" for row in display.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *body])


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "^": r"\textasciicircum{}",
        ">=": r"$\geq$", "<=": r"$\leq$",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("/", r"/\allowbreak{}")
    for word, broken in {
        "dopamine": r"dopa\-mine",
        "dobutamine": r"dobu\-tamine",
        "epinephrine": r"epineph\-rine",
        "norepinephrine": r"norepineph\-rine",
        "creatinine": r"creati\-nine",
    }.items():
        text = text.replace(word, broken)
    return text


def latex_longtable(frame: pd.DataFrame, widths: list[str], caption: str, label: str) -> str:
    columns = "".join(f"p{{{width}}}" for width in widths)
    header = " & ".join(latex_escape(column) for column in frame.columns) + r" \\"
    rows = [" & ".join(latex_escape(value) for value in row) + r" \\" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([
        rf"\begin{{longtable}}{{{columns}}}",
        rf"\caption{{{latex_escape(caption)}}}\label{{{label}}}\\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        rf"\multicolumn{{{len(widths)}}}{{c}}{{\tablename\ \thetable\ (continued)}}\\",
        r"\toprule", header, r"\midrule", r"\endhead",
        r"\bottomrule", r"\endfoot",
        *rows, r"\end{longtable}",
    ])


def fmt(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return "--"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return str(value)


def supplementary_result_tables() -> list[tuple[pd.DataFrame, list[str], str, str]]:
    """Build compact result tables directly from frozen experiment artifacts."""
    metrics = pd.read_csv(
        ROOT / "outputs/explicit_kg_tfnn_paired_comparison_6h/evaluation/advanced_metrics.csv"
    )
    order = [
        "explicit_kg_tfnn", "gru_matched", "xgboost_matched", "lightgbm_matched",
        "ebm", "logistic_regression", "decision_tree", "random_forest", "xgboost",
    ]
    labels = {
        "explicit_kg_tfnn": "Explicit KG-TFNN", "gru_matched": "Feature-matched GRU",
        "xgboost_matched": "Feature-matched XGBoost", "lightgbm_matched": "Feature-matched LightGBM",
        "ebm": "EBM", "logistic_regression": "Logistic regression",
        "decision_tree": "Decision tree", "random_forest": "Random forest", "xgboost": "XGBoost (legacy features)",
    }
    rows = []
    indexed = metrics.set_index("model")
    for model in order:
        if model not in indexed.index:
            continue
        row = indexed.loc[model]
        rows.append({
            "Model": labels[model],
            "AUROC (95% CI)": f"{row.auroc:.4f} ({row.auroc_ci95_low:.4f}-{row.auroc_ci95_high:.4f})",
            "AUPRC (95% CI)": f"{row.auprc:.4f} ({row.auprc_ci95_low:.4f}-{row.auprc_ci95_high:.4f})",
            "Brier (95% CI)": f"{row.brier:.5f} ({row.brier_ci95_low:.5f}-{row.brier_ci95_high:.5f})",
            "ECE": fmt(row.ece),
        })
    baseline = pd.DataFrame(rows)

    internal_long = pd.read_csv(ROOT / "outputs/final_test_evaluation_6h/advanced/fixed_specificity_metrics.csv")
    internal = internal_long.pivot(index="target_specificity", columns="metric", values="value").reset_index()
    internal_metric_row = pd.read_csv(
        ROOT / "outputs/final_test_evaluation_6h/advanced/advanced_metrics.csv"
    ).iloc[0]
    external = pd.read_csv(
        ROOT / "outputs/eicu_external_validation/final_frozen_model_evaluation/external_fixed_specificity.csv"
    )
    external_ci = json.loads(
        (ROOT / "outputs/eicu_external_validation/final_frozen_model_evaluation/external_metrics.json").read_text(encoding="utf-8")
    )["fixed_specificity_ci95"]
    op_rows = []
    for cohort, frame in [("MIMIC-IV internal", internal), ("eICU external", external)]:
        for row in frame.itertuples(index=False):
            tag = int(round(row.target_specificity * 100))
            if cohort == "MIMIC-IV internal":
                intervals = {
                    metric: (
                        internal_metric_row[f"{metric}_at_spec_{tag}_ci95_low"]
                        if metric == "sensitivity" else internal_metric_row[f"{metric}_at_spec_{tag}_ci95_low"],
                        internal_metric_row[f"{metric}_at_spec_{tag}_ci95_high"]
                        if metric == "sensitivity" else internal_metric_row[f"{metric}_at_spec_{tag}_ci95_high"],
                    )
                    for metric in ("sensitivity", "specificity", "ppv", "npv", "f1")
                }
            else:
                intervals = {metric: external_ci[str(tag)][metric] for metric in ("sensitivity", "specificity", "ppv", "npv", "f1")}
            def estimate_ci(metric: str) -> str:
                low, high = intervals[metric]
                return f"{getattr(row, metric):.3f} ({low:.3f}-{high:.3f})"
            op_rows.append({
                "Cohort": cohort,
                "Target specificity": fmt(row.target_specificity, 2),
                "Threshold": fmt(row.threshold), "Observed specificity (95% CI)": estimate_ci("specificity"),
                "Sensitivity (95% CI)": estimate_ci("sensitivity"), "PPV (95% CI)": estimate_ci("ppv"),
                "NPV (95% CI)": estimate_ci("npv"), "F1 (95% CI)": estimate_ci("f1"),
            })
    operating = pd.DataFrame(op_rows)

    alarm = pd.read_csv(ROOT / "outputs/clinical_sensitivity_analyses_6h/event_level_alarm_burden.csv")
    alarm_table = pd.DataFrame({
        "Target specificity": alarm.target_specificity.map(lambda x: fmt(x, 2)),
        "First evaluable events": alarm.first_events.map(fmt),
        "Detected, n (%)": [f"{fmt(n)} ({100*r:.1f}%)" for n, r in zip(alarm.detected_events, alarm.event_sensitivity)],
        "Alerts": alarm.alerts.map(fmt),
        "Alert PPV": alarm.alert_ppv.map(lambda x: fmt(x)),
        "False alerts / 100 pre-event days": alarm.false_alerts_per_100_patient_days.map(lambda x: fmt(x, 2)),
        "Lead time, median [IQR], h": [f"{m:.0f} [{q1:.0f}-{q3:.0f}]" for m, q1, q3 in zip(alarm.lead_time_median_h, alarm.lead_time_q1_h, alarm.lead_time_q3_h)],
    })

    hospital = pd.read_csv(ROOT / "outputs/eicu_hospital_sensitivity_6h/hospital_heterogeneity_summary.csv")
    hospital_table = pd.DataFrame({
        "Metric": hospital.metric.str.upper(), "Eligible hospitals": hospital.eligible_hospitals.map(fmt),
        "Median [IQR]": [f"{m:.4f} [{q1:.4f}-{q3:.4f}]" for m, q1, q3 in zip(hospital["median"], hospital.q1, hospital.q3)],
        "Range": [f"{lo:.4f}-{hi:.4f}" for lo, hi in zip(hospital.minimum, hospital.maximum)],
    })

    stability = pd.read_csv(ROOT / "outputs/rule_evaluation_6h/five_seed_rule_stability.csv")
    activated = pd.read_csv(ROOT / "outputs/rule_evaluation_6h/activated_rule_summary.csv")
    complexity = pd.read_csv(ROOT / "outputs/rule_evaluation_6h/top_k_rule_complexity.csv")
    drift = pd.read_csv(ROOT / "outputs/rule_evaluation_6h/membership_parameter_drift.csv")
    rule_summary = pd.DataFrame([
        {"Analysis": "Top-10 complexity", "Estimate": f"{complexity.head(10).antecedent_count.mean():.2f} antecedents/rule", "Interpretation": "Mean antecedent count among ranked rules"},
        {"Analysis": "Five-seed top-10 stability", "Estimate": f"mean Jaccard {stability.jaccard.mean():.3f}", "Interpretation": "All pairwise seed comparisons"},
        {"Analysis": "Normalized center drift", "Estimate": f"mean {drift.center_shift_in_initial_sigma.mean():.3f}", "Interpretation": "Absolute center shift / initial sigma"},
        {"Analysis": "Relative width drift", "Estimate": f"mean {drift.relative_sigma_shift.abs().mean():.3f}", "Interpretation": "Absolute relative sigma change"},
        {"Analysis": "Activated rules, negative windows", "Estimate": f"mean {activated.loc[activated.outcome == 0, 'mean_activated_rules'].iloc[0]:.3f}", "Interpretation": "Frozen predefined activation rule"},
        {"Analysis": "Activated rules, positive windows", "Estimate": f"mean {activated.loc[activated.outcome == 1, 'mean_activated_rules'].iloc[0]:.3f}", "Interpretation": "Frozen predefined activation rule"},
    ])

    threshold = pd.read_csv(ROOT / "outputs/raw_rule_firing_6h/activation_threshold_summary.csv")
    threshold = threshold[threshold.threshold.isin([0.01, 0.05, 0.10, 0.20, 0.35, 0.50])].copy()
    threshold_table = pd.DataFrame({
        "Basis": threshold.basis, "Threshold": threshold.threshold.map(lambda x: fmt(x, 2)),
        "Outcome": threshold.outcome.map({0: "Negative", 1: "Positive"}),
        "Mean activated rules": threshold.mean_activated_rules.map(lambda x: fmt(x, 3)),
        "Windows with any rule (%)": threshold.fraction_with_any_rule.map(lambda x: f"{100*x:.1f}"),
    })

    case_rules = pd.read_csv(
        ROOT / "outputs/rule_evaluation_6h/patient_case_rules/patient_specific_activated_rules.csv"
    )
    case_table = pd.DataFrame({
        "Case": case_rules.case_type, "Rank": case_rules["rank"].map(fmt),
        "Current-hour rule": case_rules.rule,
        "Raw firing": case_rules.raw_firing.map(lambda x: fmt(x, 4)),
        "Weight": case_rules.trained_rule_weight.map(lambda x: fmt(x, 3)),
        "Active at 0.10": case_rules["active_at_0.10"].map({True: "Yes", False: "No"}),
    })

    explanations = pd.read_csv(
        ROOT / "outputs/posthoc_explainability_comparison_6h/explanation_quality_comparison.csv"
    )
    explanation_table = pd.DataFrame({
        "Model": explanations.model,
        "AUROC / AUPRC": [f"{a:.4f} / {p:.4f}" for a, p in zip(explanations.auroc, explanations.auprc)],
        "Stability cosine / top-5": [
            f"{c:.3f} / {j:.3f}"
            for c, j in zip(explanations.stability_cosine_mean, explanations.stability_top5_jaccard_mean)
        ],
        "Neighbor consistency cosine / top-5": [
            f"{c:.3f} / {j:.3f}"
            for c, j in zip(
                explanations.neighbor_consistency_cosine_mean,
                explanations.neighbor_consistency_top5_jaccard_mean,
            )
        ],
        "Features for 80% mass, median [IQR]": [
            f"{m:.0f} [{q1:.0f}-{q3:.0f}]"
            for m, q1, q3 in zip(
                explanations.effective_features_80_median,
                explanations.effective_features_80_iqr_low,
                explanations.effective_features_80_iqr_high,
            )
        ],
        "Temporal mass": explanations.temporal_attribution_mass_mean.map(lambda x: fmt(x, 3)),
        "Cross-database rho / top-5": [
            f"{r:.3f} / {j:.3f}"
            for r, j in zip(explanations.cross_dataset_global_spearman, explanations.cross_dataset_top5_jaccard)
        ],
        "Explanation output": explanations.output_form,
    })

    consistency = pd.read_csv(
        ROOT / "outputs/clinical_consistency_regularization_6h/consistency_metrics_by_seed.csv"
    )
    consistency_means = consistency.groupby("variant", sort=False).mean(numeric_only=True)
    consistency_labels = {
        "no_consistency": "Temporal FNN without consistency loss",
        "full": "Full KG-TFNN",
    }
    consistency_rows = []
    for variant in ("no_consistency", "full"):
        row = consistency_means.loc[variant]
        consistency_rows.append({
            "Variant": consistency_labels[variant],
            "Violation given worsening": fmt(row.feature_consistency_violation_rate_given_worsening, 3),
            "Consistency penalty": fmt(row.clinical_consistency_penalty_mean, 6),
            "Risk reversal": fmt(row.risk_reversal_frequency_given_additive_worsening, 3),
            "Reversal magnitude": fmt(row.risk_reversal_magnitude_median, 3),
            "Rule stability": fmt(row.rule_stability, 3),
            "Normalized drift": fmt(row.normalized_rule_drift, 3),
            "Guideline-risk rho": fmt(row.guideline_risk_correlation, 3),
            "Direction alignment": fmt(row.guideline_direction_alignment, 3),
        })
    consistency_table = pd.DataFrame(consistency_rows)

    sofa_sensitivity = pd.read_csv(
        ROOT / "outputs/sofa_documentation_bias_6h/complete_sofa_outcome_sensitivity.csv"
    )
    sofa_sensitivity_table = pd.DataFrame({
        "Outcome definition": sofa_sensitivity.definition,
        "Patients / windows": [
            f"{int(p):,} / {int(w):,}" for p, w in zip(sofa_sensitivity.patients, sofa_sensitivity.windows)
        ],
        "Positive, n (%)": [
            f"{int(n):,} ({100*r:.2f})" for n, r in zip(sofa_sensitivity.positive, sofa_sensitivity.prevalence)
        ],
        "AUROC (95% CI)": [
            f"{v:.4f} ({lo:.4f}-{hi:.4f})"
            for v, lo, hi in zip(
                sofa_sensitivity.auroc,
                sofa_sensitivity.auroc_ci95_low,
                sofa_sensitivity.auroc_ci95_high,
            )
        ],
        "AUPRC (95% CI)": [
            f"{v:.4f} ({lo:.4f}-{hi:.4f})"
            for v, lo, hi in zip(
                sofa_sensitivity.auprc,
                sofa_sensitivity.auprc_ci95_low,
                sofa_sensitivity.auprc_ci95_high,
            )
        ],
        "Brier (95% CI)": [
            f"{v:.5f} ({lo:.5f}-{hi:.5f})"
            for v, lo, hi in zip(
                sofa_sensitivity.brier,
                sofa_sensitivity.brier_ci95_low,
                sofa_sensitivity.brier_ci95_high,
            )
        ],
    })

    organ = pd.read_csv(ROOT / "outputs/sofa_documentation_bias_6h/organ_component_contributions.csv")
    organ_table = pd.DataFrame({
        "SOFA component": organ.component,
        "Comparable positive windows (%)": organ.comparable_fraction.map(lambda x: f"{100*x:.1f}"),
        "Positive windows with increase (%)": organ.fraction_positive_windows_with_increase.map(lambda x: f"{100*x:.1f}"),
        "Positive SOFA points": organ.positive_sofa_points.map(lambda x: f"{int(x):,}"),
        "Share of positive component-point increases (%)": organ.share_of_positive_component_point_increases.map(lambda x: f"{100*x:.1f}"),
    })

    return [
        (baseline, ["3.2cm", "4.0cm", "4.0cm", "4.0cm", "1.5cm"], "Complete equal-sample baseline performance on the independent MIMIC-IV test set.", "tab:s3_baselines"),
        (operating, ["3.0cm", "2.2cm", "2.0cm", "2.8cm", "2.2cm", "2.0cm", "2.0cm", "1.8cm"], "Validation-derived operating points transported without test-set threshold selection.", "tab:s4_operating"),
        (alarm_table, ["2.2cm", "2.7cm", "3.0cm", "1.8cm", "2.0cm", "3.7cm", "3.5cm"], "Event-level alarm burden after a six-hour refractory period.", "tab:s5_alarm"),
        (hospital_table, ["3.0cm", "3.0cm", "5.0cm", "4.5cm"], "Hospital-level heterogeneity in eICU-CRD.", "tab:s6_hospital"),
        (rule_summary, ["5.0cm", "4.0cm", "8.0cm"], "Rule evaluation framework results.", "tab:s7_rule_eval"),
        (threshold_table, ["3.0cm", "2.5cm", "2.5cm", "4.0cm", "4.5cm"], "Raw rule-firing activation-threshold sensitivity.", "tab:s8_activation"),
        (case_table, ["1.4cm", "1.2cm", "11.0cm", "2.0cm", "1.8cm", "2.2cm"], "Patient-specific current-hour cross-rule firing for the prespecified TP, FP, and FN cases.", "tab:s9_case_rules"),
        (explanation_table, ["3.0cm", "2.4cm", "3.1cm", "3.4cm", "3.3cm", "2.0cm", "2.9cm", "3.7cm"], "Exploratory structural explanation-quality comparison on the prespecified 1,000-case sample.", "tab:s10_explanations"),
        (consistency_table, ["4.0cm", "2.8cm", "2.5cm", "2.2cm", "2.4cm", "2.2cm", "2.4cm", "2.5cm", "2.4cm"], "Exploratory clinical-consistency behavior on the 1,000-case sample (mean across three seeds).", "tab:s11_consistency"),
        (sofa_sensitivity_table, ["5.1cm", "3.1cm", "3.0cm", "4.0cm", "4.0cm", "4.2cm"], "SOFA outcome-definition and documentation-availability sensitivity with patient-clustered 95% confidence intervals.", "tab:s12_sofa_sensitivity"),
        (organ_table, ["3.4cm", "4.5cm", "4.5cm", "3.5cm", "5.0cm"], "Observed SOFA component contributions among primary positive windows.", "tab:s13_organ_contributions"),
    ]


def write_markdown(sofa: pd.DataFrame, mapping: pd.DataFrame) -> None:
    notes = [
        "# Supplementary Material: SOFA and Knowledge Mapping", "",
        "## Supplementary Table S1. SOFA Reconstruction", "", markdown_table(sofa), "",
        "### S1 implementation notes", "",
        "1. Events are aligned to ICU-admission-relative integer hours. Multiple records in the same hour use the clinically worse direction shown in S1; urine records are summed and vasopressor doses use the maximum active dose.",
        "2. Every component uses the worst value in the current and preceding 23 hours. Renal urine output is therefore a 24-hour sum, not a 6-hour or 12-hour sum. Urine criteria are not used before 24 complete ICU hours.",
        "3. Vasopressor rates are converted to mcg/kg/min. Weight-normalized rates are used directly; mg is converted to mcg, hourly rates to per-minute rates, and non-weight-normalized rates are divided by a valid recorded body weight. Unconvertible or implausible rates are excluded.",
        "4. MIMIC respiratory support is positive for non-empty active invasive or non-invasive ventilation chart values after excluding off/standby/none states. eICU support is positive for PEEP or PEEP/CPAP >0 or RT vent states 'start'/'continued'. This is an operational respiratory-support indicator, not a perfectly harmonized invasive-ventilation phenotype.",
        "5. FiO2 is normalized to a fraction and forward-filled within the ICU stay without backward fill. Missing respiratory-support records are treated as no documented support.",
        "6. Creatinine and urine output are scored independently and the higher renal score is retained.",
        "7. MIMIC total GCS is eye + verbal + motor when all components are present; eICU uses charted total GCS. No sedation correction, pre-sedation substitution, or special imputation of an intubated verbal component is performed. The resulting CNS score may reflect sedation and intubation documentation.",
        "8. The primary SOFA is available when at least four of six components are observed. Missing components are assigned zero in that primary sum after the four-component criterion is met. Sensitivity analyses use missing-as-normal regardless of component count and a strict six-component complete-case score.",
        "9. The index and future SOFA scores are reconstructed independently and may have different observed component sets. A positive primary label requires max SOFA in `(t,t+6]` minus SOFA at `t` >=2 and a complete six-hour future horizon.", "",
        "## Supplementary Table S2. NEWS2-to-Fuzzy Initialization", "", markdown_table(mapping), "",
        "### S2 implementation notes", "",
        "1. The comparator implements NEWS2 (2017) SpO2 Scale 1. Scale 2 is not used because chronic hypercapnic respiratory failure cannot be identified reliably from the harmonized variables.",
        "2. Supplemental oxygen is operationalized as charted FiO2 >21% and receives two points. This proxy may miss low-flow oxygen when a reliable FiO2 above room air is not charted.",
        "3. NEWS2 ACVPU is approximated as GCS 15 = alert and GCS <15 = altered (three points). New confusion cannot be separated reliably, and sedation/intubation correction is not performed.",
        "4. The baseline NEWS2 comparator uses the standard Scale 1 intervals after within-stay forward fill. A still-missing channel contributes no points, so it is not a strictly complete-case bedside NEWS2 implementation.",
        "5. Gaussian centers are placed near clinically meaningful interval centers or representative abnormal values; sigma controls overlap and is not the interval width. Initial risk weights usually follow NEWS2 points but may encode SOFA-like severity for non-NEWS2 organ variables and severe GCS states.",
        "6. NEWS2 heart rate 41-50 has no dedicated fuzzy term in the frozen model and is represented by overlap between the very-low and normal Gaussians. This is reported as an implementation limitation rather than retrospectively changing the checkpoint.",
        "7. Centers, sigmas, and weights are trainable. Their initial values document how knowledge enters the model; they are not fixed scoring cutoffs after training.",
    ]
    for index, (frame, _, caption, _) in enumerate(supplementary_result_tables(), start=3):
        notes.extend(["", f"## Supplementary Table S{index}. {caption}", "", markdown_table(frame)])
    notes.extend([
        "", "## Supplementary Figures", "",
        "- Figure S1: raw rule-firing activation-threshold sensitivity.",
        "- Figure S2: patient-clustered MIMIC-IV subgroup AUROC estimates.",
        "- Figure S3: eICU hospital-level heterogeneity.",
        "- Figure S4: selected true-positive, false-positive, and false-negative case timelines.",
        "- Figure S5: patient-specific raw cross-rule firing over each 24-hour observation window.",
        "- Figure S6: explanation stability, local consistency, sparsity, and cross-database rank stability.",
        "- Figure S7: clinical-consistency behavior before and after consistency regularization.",
        "- Membership functions and SOFA documentation sensitivity were promoted to the main manuscript.",
    ])
    (OUTPUT / "supplementary_material.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def write_latex(sofa: pd.DataFrame, mapping: pd.DataFrame) -> None:
    preamble = r"""\documentclass[10pt]{article}
\usepackage[margin=1.2cm,landscape]{geometry}
\usepackage{booktabs,longtable,array,ragged2e,graphicx,float}
\newcolumntype{P}[1]{>{\RaggedRight\arraybackslash}p{#1}}
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.15}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thefigure}{S\arabic{figure}}
\begin{document}
\begin{center}\Large\textbf{Supplementary Material}\end{center}
\footnotesize
"""
    sofa_table = latex_longtable(
        sofa,
        ["2.2cm", "2.2cm", "1.25cm", "1.2cm", "1.2cm", "1.55cm", "1.65cm", "3.5cm", "4.1cm", "3.4cm"],
        "SOFA reconstruction and cross-database source harmonization.", "tab:s1_sofa",
    )
    mapping_table = latex_longtable(
        mapping,
        ["3.2cm", "4.0cm", "1.5cm", "3.0cm", "2.0cm", "2.2cm", "2.0cm"],
        "NEWS2-to-KG-TFNN fuzzy membership initialization.", "tab:s2_news2_fuzzy",
    )
    result_tables = "\n\\clearpage\n".join(
        latex_longtable(frame, widths, caption, label)
        for frame, widths, caption, label in supplementary_result_tables()
    )
    figure_text = r"""
\section*{Supplementary Figures}
\begin{figure}[H]\centering
\includegraphics[width=0.88\textwidth]{../outputs/raw_rule_firing_6h/figures/raw_rule_firing_threshold_sensitivity.pdf}
\caption{Raw rule-firing activation-threshold sensitivity.}\label{fig:s1_raw_firing}
\end{figure}
\begin{figure}[H]\centering
\includegraphics[width=0.90\textwidth]{../outputs/clinical_sensitivity_analyses_6h/figures/mimic_subgroup_auroc_forest.pdf}
\caption{Exploratory MIMIC-IV subgroup AUROC with patient-clustered confidence intervals.}\label{fig:s2_subgroups}
\end{figure}
\begin{figure}[H]\centering
\includegraphics[width=0.90\textwidth]{../outputs/eicu_hospital_sensitivity_6h/figures/eicu_hospital_heterogeneity.pdf}
\caption{Hospital-level performance heterogeneity in eICU-CRD.}\label{fig:s3_hospital}
\end{figure}
\begin{figure}[H]\centering
\includegraphics[width=0.94\textwidth]{../outputs/rule_evaluation_6h/figures/tp_fp_fn_case_timelines.pdf}
\caption{Selected true-positive, false-positive, and false-negative patient timelines. Physiologic annotations summarize the largest changes and are not a patient-specific list of activated fuzzy rules.}\label{fig:s4_cases}
\end{figure}
\begin{figure}[H]\centering
\includegraphics[width=0.94\textwidth]{../outputs/rule_evaluation_6h/patient_case_rules/figures/patient_specific_rule_firing.pdf}
\caption{Patient-specific raw cross-rule firing over the 24-hour observation window. Vertical dotted lines identify the attention-selected hour; the dashed horizontal line is the prespecified raw-firing threshold of 0.10.}\label{fig:s5_case_rules}
\end{figure}
\begin{figure}[H]\centering
\includegraphics[width=0.94\textwidth]{../outputs/posthoc_explainability_comparison_6h/figures/explanation_quality_comparison.pdf}
\caption{Exploratory structural explanation-quality comparison. Stability uses three independent 1\% training-standard-deviation perturbations; local consistency compares nearest trajectory neighbors. The balanced 1,000-case sample is not a formal full-cohort analysis and is not used to estimate population-level predictive performance.}\label{fig:s6_explanations}
\end{figure}
\begin{figure}[H]\centering
\includegraphics[width=0.94\textwidth]{../outputs/clinical_consistency_regularization_6h/figures/consistency_regularization_effects.pdf}
\caption{Exploratory clinical-consistency behavior on the 1,000-case sample with and without consistency regularization across three random seeds. Lower is better for violation, reversal, and drift; higher is better for stability and alignment.}\label{fig:s7_consistency}
\end{figure}
"""
    note_text = r"""
\subsection*{Supplementary implementation notes}
\begin{enumerate}
\item Multiple records in an hour use the worst clinical direction; urine is summed and active vasopressor dose is maximized. All component values then use a trailing 24-hour window.
\item Vasopressor doses are normalized to mcg/kg/min. Unconvertible and implausible rates are excluded. The renal urine criterion is a 24-hour sum and is unavailable before a complete 24-hour history; creatinine and urine scores are combined by retaining the higher score.
\item Respiratory support is an operational cross-database indicator. MIMIC includes active invasive/non-invasive ventilation charting; eICU includes positive PEEP/PEEP-CPAP or RT vent-on states.
\item Primary SOFA requires at least four observed components; remaining missing components contribute zero. Missing-as-normal and strict six-component complete-case definitions are sensitivity analyses.
\item GCS uses observed charting without sedation correction, pre-sedation substitution, or special handling of an intubated verbal component. This is a limitation of the reconstructed CNS component.
\item The outcome compares SOFA at the index hour with the maximum independently reconstructed SOFA in the strictly future interval $(t,t+6]$ and requires a complete six-hour horizon.
\item NEWS2 uses the 2017 Scale 1 oxygen saturation thresholds. Scale 2 is not used. Supplemental oxygen is proxied by FiO2 above 21 percent, and GCS below 15 proxies altered ACVPU status. Missing channels after within-stay forward fill contribute no points.
\item NEWS2 heart rate 41--50 has no dedicated fuzzy term in the frozen model; it is represented by overlap between the very-low and normal Gaussian memberships.
\item Fuzzy centers and sigmas initialize overlapping Gaussian memberships. Sigma is not a hard interval width, and all membership parameters remain trainable.
\item Post-hoc and intrinsic explanations are aggregated to the same 13 clinical-variable groups. LightGBM and XGBoost use feature-matched 24-hour summaries; the available EBM is a current-state comparator and is therefore not architecture matched. S10--S11 and Figures S6--S7 use exploratory 1,000-case samples, not formal full-cohort analyses or clinician-reader validation.
\item Clinical-consistency violation and risk-reversal rates are directional stress tests. In this experiment, regularization improved cross-seed rule stability but did not uniformly improve reversal frequency, membership drift, or guideline-risk correlation.
\item Among primary positive windows, 39.0 percent had a component first observed at the future maximum; 67.1 percent remained positive under the pairwise common-component definition. This supports reporting documentation sensitivity rather than claiming that the reconstructed endpoint is free of documentation bias.
\end{enumerate}
\end{document}
"""
    (PAPER / "Supplementary_Material.tex").write_text(
        preamble + sofa_table + "\n\\clearpage\n" + mapping_table
        + "\n\\clearpage\n" + result_tables + figure_text + note_text,
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sofa = pd.DataFrame(SOFA_ROWS)
    mapping = news2_fuzzy_mapping()
    sofa.to_csv(OUTPUT / "table_s1_sofa_reconstruction.csv", index=False)
    mapping.to_csv(OUTPUT / "table_s2_news2_fuzzy_mapping.csv", index=False)
    write_markdown(sofa, mapping)
    write_latex(sofa, mapping)
    print(f"Wrote {OUTPUT} and {PAPER / 'Supplementary_Material.tex'}")


if __name__ == "__main__":
    main()
