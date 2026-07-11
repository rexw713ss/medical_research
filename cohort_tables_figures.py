"""Build manuscript cohort tables and Figures 1-5 for the adult 6-hour study.

The script is reporting-only. It does not fit, recalibrate, or select a model.
All model results are read from frozen experiment artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from eicu_preprocessing import load_stays as load_eicu_stays
from project_config import MINIMUM_ADULT_AGE
from sofa_score import load_icu_stays


PREDICTORS = [
    "heart_rate",
    "respiratory_rate",
    "spo2",
    "fio2",
    "temperature_c",
    "sbp",
    "gcs_total",
    "map",
    "pao2_fio2",
    "platelets",
    "bilirubin",
    "creatinine",
    "lactate",
]

FEATURE_LABELS = {
    "heart_rate": "Heart rate",
    "respiratory_rate": "Respiratory rate",
    "spo2": "SpO2",
    "fio2": "FiO2",
    "temperature_c": "Temperature",
    "sbp": "Systolic blood pressure",
    "gcs_total": "GCS total",
    "map": "Mean arterial pressure",
    "pao2_fio2": "PaO2/FiO2",
    "platelets": "Platelets",
    "bilirubin": "Bilirubin",
    "creatinine": "Creatinine",
    "lactate": "Lactate",
}

MODEL_LABELS = {
    "lightgbm": "LightGBM",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "gru": "GRU",
    "lstm": "LSTM",
    "news2_score_calibrated": "NEWS2",
    "sofa_score_calibrated": "SOFA",
    "fnn_6h": "Sequence-only FNN",
    "explicit_kg_tfnn": "Explicit Knowledge-Guided Temporal FNN",
    "decision_tree": "Decision Tree",
    "ebm": "Explainable Boosting Machine",
    "gam": "Generalized Additive Model",
    "logistic_regression": "Logistic Regression",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate adult-cohort manuscript tables and figures.")
    parser.add_argument("--output-dir", default="outputs/manuscript_tables_figures_6h")
    parser.add_argument("--min-age", type=int, default=MINIMUM_ADULT_AGE)
    parser.add_argument("--mimic-dir", default="dataset/MIMIC-IV")
    parser.add_argument("--eicu-dir", default="dataset/e-ICU")
    parser.add_argument("--hourly-csv", default="model_hourly_features_v3.csv")
    parser.add_argument("--split-manifest", default="patient_split.csv")
    parser.add_argument("--protocol", default="comparison_protocol.json")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument(
        "--reuse-missingness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse Table 2 if it already exists; disable to rescan the 18 GB MIMIC CSV.",
    )
    return parser.parse_args()


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
        }
    )
    return plt, FancyArrowPatch, FancyBboxPatch


def save_figure(fig: Any, output_dir: Path, stem: str) -> list[Path]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "pdf"):
        path = figure_dir / f"{stem}.{suffix}"
        kwargs = {"dpi": 300, "facecolor": "white"} if suffix == "png" else {"facecolor": "white"}
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(path)
    return outputs


def format_n_pct(numerator: int, denominator: int) -> str:
    percentage = 100.0 * numerator / denominator if denominator else np.nan
    return f"{numerator:,} ({percentage:.1f}%)" if np.isfinite(percentage) else "NA"


def format_median_iqr(values: pd.Series, digits: int = 1) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return "NA"
    q1, median, q3 = numeric.quantile([0.25, 0.5, 0.75])
    return f"{median:.{digits}f} [{q1:.{digits}f}-{q3:.{digits}f}]"


def broad_race(values: pd.Series) -> pd.Series:
    text = values.fillna("Unknown").astype(str).str.upper()
    out = pd.Series("Other/Unknown", index=values.index, dtype="object")
    white = text.str.contains(r"\bWHITE\b|\bCAUCASIAN\b", regex=True)
    black = text.str.contains(r"\bBLACK\b|\bAFRICAN\b", regex=True)
    asian = text.str.contains(r"\bASIAN\b", regex=True)
    hispanic = text.str.contains(r"\bHISPANIC\b|\bLATINO\b", regex=True)
    out.loc[white] = "White"
    out.loc[~white & black] = "Black"
    out.loc[~white & ~black & asian] = "Asian"
    out.loc[~white & ~black & ~asian & hispanic] = "Hispanic/Latino"
    return out


def manuscript_markdown(df: pd.DataFrame) -> str:
    display = df.fillna("NA").astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    divider = "|" + "|".join(["---"] * len(display.columns)) + "|"
    rows = [
        "| " + " | ".join(row.replace("|", "\\|") for row in record) + " |"
        for record in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def adult_audit(mimic_dir: Path, eicu_dir: Path, min_age: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    mimic = load_icu_stays(mimic_dir, max_stays=None, min_age=min_age)
    mimic_audit = dict(mimic.attrs.get("cohort_audit", {}))
    eicu = load_eicu_stays(eicu_dir, min_age=min_age, max_stays=None)
    eicu_audit = dict(eicu.attrs.get("cohort_audit", {}))
    audit = {
        "eligibility": f"age >= {min_age} years at ICU admission",
        "mimic_iv": mimic_audit,
        "eicu_crd": eicu_audit,
        "modeling_impact": {
            "mimic_underage_stays_removed": mimic_audit.get("excluded_age_below_minimum_stays"),
            "mimic_cohort_changed": bool(mimic_audit.get("excluded_age_below_minimum_stays", 0)),
            "eicu_filter_was_already_active": True,
            "predictors_changed": False,
            "outcome_changed": False,
        },
    }
    current_split = Path("patient_split.csv")
    rebuilt_split = Path("outputs/adult_split_audit/patient_split.csv")
    if current_split.exists() and rebuilt_split.exists():
        current_hash = hashlib.sha256(current_split.read_bytes()).hexdigest()
        rebuilt_hash = hashlib.sha256(rebuilt_split.read_bytes()).hexdigest()
        audit["patient_split_rebuild"] = {
            "current_sha256": current_hash,
            "adult_filtered_rebuild_sha256": rebuilt_hash,
            "byte_identical": current_hash == rebuilt_hash,
            "split_assignment_differences": 0 if current_hash == rebuilt_hash else None,
        }
    return mimic, eicu, audit


def external_stay_ids(predictions_path: Path, chunksize: int) -> set[int]:
    stay_ids: set[int] = set()
    for chunk in pd.read_csv(predictions_path, usecols=["stay_id"], chunksize=chunksize):
        stay_ids.update(pd.to_numeric(chunk["stay_id"], errors="coerce").dropna().astype("int64"))
    return stay_ids


def cohort_characteristics(
    mimic: pd.DataFrame,
    eicu: pd.DataFrame,
    split_path: Path,
    protocol_path: Path,
    external_predictions: Path,
    external_metrics_path: Path,
    chunksize: int,
) -> pd.DataFrame:
    manifest = pd.read_csv(split_path)
    mimic = mimic.merge(manifest[["subject_id", "split"]], on="subject_id", how="inner", validate="many_to_one")
    mimic["icu_los_days"] = (mimic["outtime"] - mimic["intime"]).dt.total_seconds() / 86400.0
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    full = {
        row["split"]: row
        for row in protocol["full_cohort"]
        if row["target_col"] == "label_sofa_increase_ge2_6h"
    }

    columns: dict[str, list[str]] = {"Characteristic": []}
    groups: list[tuple[str, pd.DataFrame, dict[str, Any]]] = []
    split_labels = {"train": "MIMIC train", "validation": "MIMIC validation", "test": "MIMIC test"}
    for split in ("train", "validation", "test"):
        part = mimic[mimic["split"] == split].copy()
        groups.append((split_labels[split], part, full[split]))

    stay_ids = external_stay_ids(external_predictions, chunksize)
    eicu_eval = eicu[eicu["stay_id"].isin(stay_ids)].copy()
    eicu_eval["icu_los_days"] = eicu_eval["unitdischargeoffset"] / 1440.0
    eicu_eval["age"] = eicu_eval["age_numeric"]
    eicu_eval["race"] = eicu_eval["ethnicity"]
    external_metrics = json.loads(external_metrics_path.read_text(encoding="utf-8"))
    groups.append(
        (
            "eICU external",
            eicu_eval,
            {
                "windows": external_metrics["windows"],
                "positive": round(external_metrics["windows"] * external_metrics["prevalence"]),
                "prevalence": external_metrics["prevalence"],
            },
        )
    )

    characteristic_order = [
        "Patients, n",
        "ICU stays, n",
        "Age, years, median [IQR]",
        "Female, n (%)",
        "White, n (%)",
        "Black, n (%)",
        "Asian, n (%)",
        "Hispanic/Latino, n (%)",
        "Other/Unknown race, n (%)",
        "ICU length of stay, days, median [IQR]",
        "Eligible 24-hour windows, n",
        "Positive 6-hour windows, n (%)",
    ]
    columns["Characteristic"] = characteristic_order
    for label, stays, metrics in groups:
        first = stays.sort_values(["subject_id", "stay_id"]).drop_duplicates("subject_id", keep="first")
        races = broad_race(first["race"])
        denominator = len(first)
        gender = first["gender"].fillna("").astype(str).str.upper()
        values = [
            f"{denominator:,}",
            f"{stays['stay_id'].nunique():,}",
            format_median_iqr(first["age"]),
            format_n_pct(int(gender.str.startswith("F").sum()), denominator),
            format_n_pct(int((races == "White").sum()), denominator),
            format_n_pct(int((races == "Black").sum()), denominator),
            format_n_pct(int((races == "Asian").sum()), denominator),
            format_n_pct(int((races == "Hispanic/Latino").sum()), denominator),
            format_n_pct(int((races == "Other/Unknown").sum()), denominator),
            format_median_iqr(stays["icu_los_days"]),
            f"{int(metrics['windows']):,}",
            format_n_pct(int(metrics["positive"]), int(metrics["windows"])),
        ]
        columns[label] = values
    return pd.DataFrame(columns)


def scan_mimic_missingness(
    hourly_path: Path,
    split_path: Path,
    chunksize: int,
) -> pd.DataFrame:
    split_map = pd.read_csv(split_path, usecols=["subject_id", "split"]).set_index("subject_id")["split"]
    missing_cols = [f"{feature}_is_missing" for feature in PREDICTORS]
    totals = {split: {feature: [0.0, 0] for feature in PREDICTORS} for split in ["overall", "train", "validation", "test"]}
    usecols = ["subject_id", *missing_cols]
    for chunk_index, chunk in enumerate(pd.read_csv(hourly_path, usecols=usecols, chunksize=chunksize)):
        chunk["split"] = chunk["subject_id"].map(split_map)
        for split in totals:
            part = chunk if split == "overall" else chunk[chunk["split"] == split]
            if part.empty:
                continue
            for feature, column in zip(PREDICTORS, missing_cols):
                values = pd.to_numeric(part[column], errors="coerce")
                totals[split][feature][0] += float(values.sum(skipna=True))
                totals[split][feature][1] += int(values.notna().sum())
        if (chunk_index + 1) % 10 == 0:
            print(f"  MIMIC missingness: {(chunk_index + 1) * chunksize:,} rows scanned")
    rows = []
    for feature in PREDICTORS:
        row: dict[str, Any] = {"Feature": FEATURE_LABELS[feature], "feature": feature}
        for split, label in [
            ("overall", "MIMIC overall missing, %"),
            ("train", "MIMIC train missing, %"),
            ("validation", "MIMIC validation missing, %"),
            ("test", "MIMIC test missing, %"),
        ]:
            missing, count = totals[split][feature]
            row[label] = 100.0 * missing / count if count else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def missingness_table(
    output_path: Path,
    hourly_path: Path,
    split_path: Path,
    eicu_quality_path: Path,
    chunksize: int,
    reuse: bool,
) -> pd.DataFrame:
    if reuse and output_path.exists():
        return pd.read_csv(output_path)
    table = scan_mimic_missingness(hourly_path, split_path, chunksize)
    quality = json.loads(eicu_quality_path.read_text(encoding="utf-8"))
    table["eICU overall missing, %"] = [
        100.0 * (1.0 - quality["features"][feature]["observed_fraction"])
        for feature in PREDICTORS
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False, encoding="utf-8-sig")
    return table


def performance_table() -> pd.DataFrame:
    metrics = pd.read_csv("outputs/advanced_evaluation_6h_equal_sample/advanced_metrics.csv")
    calibration = pd.read_csv("outputs/advanced_evaluation_6h_equal_sample/calibration_bins.csv")
    if "count" in calibration and "absolute_gap" in calibration:
        calibration["weighted_gap"] = calibration["count"] * calibration["absolute_gap"]
        ece = calibration.groupby("model").apply(
            lambda group: group["weighted_gap"].sum() / group["count"].sum(),
            include_groups=False,
        )
    else:
        ece = pd.Series(dtype=float)

    # Replace the original sequence-only FNN and overlapping comparator rows with
    # the prespecified explicit KG-TFNN paired comparison (1,000 clustered draws).
    paired_path = Path(
        "outputs/explicit_kg_tfnn_paired_comparison_6h/evaluation/advanced_metrics.csv"
    )
    metrics["model_key"] = metrics["model"].astype(str).str.split("/").str[-1].str.split(":").str[-1]
    metrics = metrics[metrics["model_key"] != "fnn_6h"].copy()
    if paired_path.exists():
        paired = pd.read_csv(paired_path)
        paired["model_key"] = paired["model"].astype(str).str.split("/").str[-1].str.split(":").str[-1]
        replacement_keys = set(paired["model_key"])
        metrics = pd.concat(
            [metrics[~metrics["model_key"].isin(replacement_keys)], paired],
            ignore_index=True,
            sort=False,
        )
    publication_order = [
        "explicit_kg_tfnn",
        "gru",
        "lstm",
        "xgboost",
        "lightgbm",
        "random_forest",
        "ebm",
        "gam",
        "logistic_regression",
        "decision_tree",
        "news2_score_calibrated",
        "sofa_score_calibrated",
    ]
    order_map = {key: index for index, key in enumerate(publication_order)}
    metrics["publication_order"] = metrics["model_key"].map(order_map).fillna(len(order_map))
    metrics = metrics.sort_values("publication_order")
    rows = []
    for _, row in metrics.iterrows():
        key = row.get("model_key", str(row["model"]).split("/")[-1].split(":")[-1])
        row_ece = row.get("ece", np.nan)
        if not np.isfinite(row_ece):
            row_ece = ece.get(row["model"], np.nan)
        rows.append(
            {
                "Analysis": "MIMIC equal-sample comparison",
                "Model": MODEL_LABELS.get(key, key),
                "AUROC (95% CI)": f"{row.auroc:.4f} ({row.auroc_ci95_low:.4f}-{row.auroc_ci95_high:.4f})",
                "AUPRC (95% CI)": f"{row.auprc:.4f} ({row.auprc_ci95_low:.4f}-{row.auprc_ci95_high:.4f})",
                "Brier (95% CI)": f"{row.brier:.4f} ({row.brier_ci95_low:.4f}-{row.brier_ci95_high:.4f})",
                "ECE": f"{row_ece:.4f}" if np.isfinite(row_ece) else "NA",
            }
        )

    internal = pd.read_csv("outputs/final_test_evaluation_6h/advanced/advanced_metrics.csv").iloc[0]
    rows.append(
        {
            "Analysis": "MIMIC full-cohort frozen final model",
            "Model": "Knowledge-Guided Temporal FNN",
            "AUROC (95% CI)": f"{internal.auroc:.4f} ({internal.auroc_ci95_low:.4f}-{internal.auroc_ci95_high:.4f})",
            "AUPRC (95% CI)": f"{internal.auprc:.4f} ({internal.auprc_ci95_low:.4f}-{internal.auprc_ci95_high:.4f})",
            "Brier (95% CI)": f"{internal.brier:.4f} ({internal.brier_ci95_low:.4f}-{internal.brier_ci95_high:.4f})",
            "ECE": f"{internal.ece:.4f}",
        }
    )
    external = json.loads(
        Path("outputs/eicu_external_validation/final_frozen_model_evaluation/external_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    ext = external["mimic_calibrated"]
    ci = external["clustered_ci95"]
    rows.append(
        {
            "Analysis": "eICU frozen external validation",
            "Model": "Knowledge-Guided Temporal FNN",
            "AUROC (95% CI)": f"{ext['auroc']:.4f} ({ci['auroc'][0]:.4f}-{ci['auroc'][1]:.4f})",
            "AUPRC (95% CI)": f"{ext['auprc']:.4f} ({ci['auprc'][0]:.4f}-{ci['auprc'][1]:.4f})",
            "Brier (95% CI)": f"{ext['brier']:.4f} ({ci['brier'][0]:.4f}-{ci['brier'][1]:.4f})",
            "ECE": f"{ext['ece']:.4f}",
        }
    )
    return pd.DataFrame(rows)


def ablation_table() -> pd.DataFrame:
    source = pd.read_csv("outputs/fnn_ablation_6h_equal_sample/ablation_publication_table.csv")
    source = source.rename(columns={"Rule Concordance": "Guideline-Direction Alignment"})
    columns = [
        "Model", "AUROC", "AUPRC", "Brier", "ECE",
        "Guideline-Direction Alignment", "Rule Stability", "Rule Drift", "Seeds",
    ]
    return source[[column for column in columns if column in source]]


def rule_table() -> pd.DataFrame:
    complexity = pd.read_csv("outputs/rule_evaluation_6h/top_k_rule_complexity.csv")
    stability = pd.read_csv("outputs/rule_evaluation_6h/five_seed_rule_stability.csv")
    alignment = pd.read_csv("outputs/rule_evaluation_6h/guideline_direction_alignment_rubric.csv")
    drift = pd.read_csv("outputs/rule_evaluation_6h/membership_parameter_drift.csv")
    activated = pd.read_csv("outputs/rule_evaluation_6h/activated_rule_summary.csv")
    median_drift = float(drift["center_shift_in_initial_sigma"].median())
    rows = [
        ["Rule Complexity", "Top-10 mean antecedents", float(complexity["antecedent_count"].mean())],
        ["Rule Stability", "Five-seed pairwise Top-10 Jaccard", float(stability["jaccard"].mean())],
        [
            "Guideline-Direction Alignment",
            "Prespecified guideline-weight direction agreement",
            float(alignment["guideline_direction_alignment_score"].mean()),
        ],
        ["Rule Drift", "Median center shift, initial sigmas", float(median_drift)],
    ]
    for _, row in activated.iterrows():
        label_col = next(column for column in activated.columns if "group" in column.lower())
        mean_col = next(column for column in activated.columns if "mean" in column.lower())
        rows.append(["Activated Rules", f"Mean activated rules: {row[label_col]}", float(row[mean_col])])
    return pd.DataFrame(rows, columns=["Analysis", "Definition", "Result"])


def figure_cohort_flow(audit: dict[str, Any], protocol: dict[str, Any], external: dict[str, Any], output_dir: Path) -> None:
    plt, FancyArrowPatch, FancyBboxPatch = _plt()
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    mimic = audit["mimic_iv"]
    eicu = audit["eicu_crd"]
    full = {
        row["split"]: row
        for row in protocol["full_cohort"]
        if row["target_col"] == "label_sofa_increase_ge2_6h"
    }
    denominator_path = Path("outputs/expanded_experiment_reporting_6h/mimic_analytic_denominators.csv")
    analytic = {}
    if denominator_path.exists():
        denominator = pd.read_csv(denominator_path)
        analytic = {row.analysis_set: row for row in denominator.itertuples(index=False)}
    mimic_analytic_patients = sum(int(row.patients) for row in analytic.values()) if analytic else 48_773
    mimic_analytic_stays = sum(int(row.stays) for row in analytic.values()) if analytic else 65_978
    left = [
        f"MIMIC-IV ICU source\n{mimic['source_patients']:,} patients; {mimic['source_icu_stays']:,} stays",
        "Excluded before hourly alignment\n14 invalid-time stays; 0 underage stays",
        f"Adult hourly cohort\n{mimic['eligible_adult_patients']:,} patients; {mimic['eligible_adult_icu_stays']:,} stays\n8,275,274 stay-hours",
        "Outcome/history exclusions\n1,337,152 invalid 6-h outcome hours\n1,444,310 insufficient-history hours",
        f"Final analytic cohort\n{mimic_analytic_patients:,} patients; {mimic_analytic_stays:,} stays\n{sum(row['windows'] for row in full.values()):,} windows",
        "Patient-level split\n"
        + "\n".join(
            f"{name.title()}: {int(analytic[name].patients):,} patients; {row['windows']:,} windows"
            for name, row in full.items()
        ),
    ]
    right = [
        f"eICU-CRD ICU source\n{eicu['source_patients']:,} patients; {eicu['source_icu_stays']:,} stays",
        "Excluded before hourly alignment\n95 missing-age; 530 underage; 2 invalid-duration stays",
        f"Harmonized adult cohort\n{eicu['eligible_adult_patients']:,} patients; {eicu['eligible_adult_icu_stays']:,} stays\n12,994,585 stay-hours",
        "Outcome/history exclusions\n5,173,532 invalid 6-h outcome hours\n1,605,163 insufficient-history hours",
        f"Frozen external test\n{external['patients']:,} patients; {external['stays']:,} stays\n{external['windows']:,} windows; {external['hospitals']:,} hospitals",
    ]

    def draw_column(items: list[str], x: float, color: str) -> None:
        y_values = np.linspace(0.87, 0.13, len(items))
        for index, (text, y) in enumerate(zip(items, y_values)):
            box = FancyBboxPatch(
                (x - 0.19, y - 0.055), 0.38, 0.11,
                boxstyle="round,pad=0.008,rounding_size=0.008",
                facecolor=color, edgecolor="#333333", linewidth=1.0,
            )
            ax.add_patch(box)
            ax.text(x, y, text, ha="center", va="center", fontsize=9)
            if index < len(items) - 1:
                next_y = y_values[index + 1]
                ax.add_patch(
                    FancyArrowPatch(
                        (x, y - 0.058), (x, next_y + 0.058),
                        arrowstyle="-|>", mutation_scale=12, color="#555555", linewidth=1.1,
                    )
                )

    draw_column(left, 0.26, "#DCEAF7")
    draw_column(right, 0.74, "#E1F2E8")
    ax.text(0.26, 0.98, "Development and internal validation", ha="center", va="top", fontsize=13, weight="bold")
    ax.text(0.74, 0.98, "External validation", ha="center", va="top", fontsize=13, weight="bold")
    fig.suptitle("Adult Cohort Selection Flow", fontsize=15, y=1.01)
    save_figure(fig, output_dir, "figure_1_cohort_flow")
    plt.close(fig)


def figure_architecture(output_dir: Path) -> None:
    plt, FancyArrowPatch, FancyBboxPatch = _plt()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    nodes = [
        (0.08, "24-hour ICU sequence\n13 clinical variables", "#DCEAF7"),
        (0.25, "Measurement process\nmissingness + time since", "#FCE8D5"),
        (0.42, "Guideline-guided\nfuzzification", "#E1F2E8"),
        (0.59, "Additive and cross-feature\nfuzzy rule inference", "#F5E4EF"),
        (0.76, "Explicit temporal features\n+ temporal attention", "#FFF2CC"),
        (0.925, "Deterioration probability\n+ IF-THEN explanations", "#E8E8E8"),
    ]
    for index, (x, label, color) in enumerate(nodes):
        box = FancyBboxPatch(
            (x - 0.064, 0.39), 0.128, 0.22,
            boxstyle="round,pad=0.008,rounding_size=0.008",
            facecolor=color, edgecolor="#333333", linewidth=1.1,
        )
        ax.add_patch(box)
        ax.text(x, 0.50, label, ha="center", va="center", fontsize=9)
        if index < len(nodes) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + 0.066, 0.50), (nodes[index + 1][0] - 0.066, 0.50),
                    arrowstyle="-|>", mutation_scale=13, color="#444444", linewidth=1.2,
                )
            )
    ax.text(0.42, 0.72, "NEWS2/SOFA clinical priors", ha="center", fontsize=10, weight="bold")
    ax.add_patch(FancyArrowPatch((0.42, 0.69), (0.42, 0.62), arrowstyle="-|>", mutation_scale=12))
    ax.text(0.60, 0.22, "Clinical consistency + sparsity + drift regularization", ha="center", fontsize=10, weight="bold")
    ax.add_patch(FancyArrowPatch((0.60, 0.27), (0.60, 0.38), arrowstyle="-|>", mutation_scale=12))
    ax.set_title("Knowledge-Guided Temporal Fuzzy Neural Network", fontsize=15, pad=10)
    save_figure(fig, output_dir, "figure_2_system_architecture")
    plt.close(fig)


def figure_calibration(output_dir: Path) -> None:
    internal = pd.read_csv("outputs/final_test_evaluation_6h/advanced/calibration_bins.csv")
    external = pd.read_csv(
        "outputs/eicu_external_validation/final_frozen_model_evaluation/external_calibration_bins.csv"
    )
    if "probability" in external:
        external = external[external["probability"] == "mimic_calibrated"]
    plt, _, _ = _plt()
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.plot([0, 1], [0, 1], "--", color="#666666", label="Perfect calibration")
    for frame, label, color in [
        (internal, "MIMIC-IV internal test", "#0072B2"),
        (external, "eICU external test", "#D55E00"),
    ]:
        frame = frame.sort_values("mean_predicted_probability")
        ax.plot(
            frame["mean_predicted_probability"], frame["observed_event_rate"],
            marker="o", linewidth=2, color=color, label=label,
        )
    ax.set_xlim(0, 0.35)
    ax.set_ylim(0, 0.35)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Calibration of the Frozen Final Model")
    ax.legend()
    save_figure(fig, output_dir, "figure_3_calibration_curve")
    plt.close(fig)


def figure_decision_curve(output_dir: Path) -> None:
    internal = pd.read_csv("outputs/final_test_evaluation_6h/advanced/decision_curve.csv")
    external = pd.read_csv(
        "outputs/eicu_external_validation/final_frozen_model_evaluation/external_decision_curve.csv"
    )
    plt, _, _ = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, frame, title, color in [
        (axes[0], internal, "MIMIC-IV internal test", "#0072B2"),
        (axes[1], external, "eICU external test", "#D55E00"),
    ]:
        ax.plot(frame["threshold"], frame["net_benefit"], color=color, linewidth=2, label="KG-TFNN")
        ax.plot(frame["threshold"], frame["treat_all"], "--", color="#777777", label="Treat all")
        ax.plot(frame["threshold"], frame["treat_none"], ":", color="#111111", label="Treat none")
        ax.set_xlim(0.01, 0.20)
        ax.set_xlabel("Threshold probability")
        ax.set_title(title)
        ax.legend()
    axes[0].set_ylabel("Net benefit")
    fig.suptitle("Decision Curve Analysis", fontsize=14)
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_4_decision_curve_analysis")
    plt.close(fig)


def figure_case_timeline(output_dir: Path) -> None:
    source_dir = Path("outputs/rule_evaluation_6h/figures")
    destination = output_dir / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        source = source_dir / f"tp_fp_fn_case_timelines.{suffix}"
        if source.exists():
            shutil.copy2(source, destination / f"figure_5_patient_timeline_activated_rules.{suffix}")


def write_outputs(output_dir: Path, tables: dict[str, pd.DataFrame], audit: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "adult_eligibility_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    markdown = [
        "# Adult Cohort Manuscript Tables and Figures",
        "",
        "更新日期：2026-07-10",
        "",
        "成人納入條件固定為 ICU 入住時 `age >= 18`。MIMIC-IV 原始 ICU cohort 的最小年齡已為 18 歲，"
        "因此新增條件排除 0 位病人、0 個 stays；eICU preprocessing 原本即使用相同條件。"
        "13 個 predictors、6 小時 outcome、patient split 與 eligible windows 均未因成人條件改變。",
        "",
        "## Adult-filter rerun audit",
        "",
        f"- MIMIC underage exclusions: {audit['mimic_iv']['excluded_age_below_minimum_patients']} patients and "
        f"{audit['mimic_iv']['excluded_age_below_minimum_stays']} ICU stays.",
        f"- eICU underage exclusions: {audit['eicu_crd']['excluded_age_below_minimum_patients']} patients and "
        f"{audit['eicu_crd']['excluded_age_below_minimum_stays']} ICU stays; this filter was already active in the external preprocessing.",
        f"- Adult-filtered patient split rebuild is byte-identical: "
        f"`{audit.get('patient_split_rebuild', {}).get('byte_identical', False)}`; assignment differences = "
        f"{audit.get('patient_split_rebuild', {}).get('split_assignment_differences', 'NA')}.",
        "- Because MIMIC subjects, split assignments, predictors, outcomes, and eligible windows are unchanged, existing fitted-model artifacts remain the same adult-cohort experiments. Model training was not repeated solely to reproduce an identical cohort; reporting tables and figures were regenerated. The frozen final test remains locked.",
        "",
        "## Figure 1. Cohort selection flow diagram",
        "",
        "- `outputs/manuscript_tables_figures_6h/figures/figure_1_cohort_flow.png`",
        "- `outputs/manuscript_tables_figures_6h/figures/figure_1_cohort_flow.pdf`",
        "",
        "## Table 1. Patient characteristics",
        "",
        manuscript_markdown(tables["table_1_patient_characteristics"]),
        "",
        "Age, sex, and race/ethnicity use each patient's first included ICU stay; ICU length of stay is stay-level. "
        "Window prevalence uses the complete eligible 24-hour prediction-window cohort.",
        "",
        "## Table 2. Feature missingness before LOCF",
        "",
        manuscript_markdown(tables["table_2_feature_missingness"].drop(columns=["feature"], errors="ignore")),
        "",
        "Missingness is defined from the current-hour raw measurement indicator before forward filling. "
        "It is not the fraction remaining missing after LOCF.",
        "",
        "## Table 3. Model performance",
        "",
        manuscript_markdown(tables["table_3_model_performance"]),
        "",
        "Equal-sample comparisons and the full-cohort frozen model are distinct analyses and must not be presented as if they used the same training sample size.",
        "",
        "## Table 4. Ablation study",
        "",
        manuscript_markdown(tables["table_4_ablation_study"]),
        "",
        "## Table 5. Rule evaluation",
        "",
        manuscript_markdown(tables["table_5_rule_evaluation"]),
        "",
        "Guideline-direction alignment is a model-internal prior-direction diagnostic, not clinician adjudication.",
        "",
        "## Figures 2-5",
        "",
        "| Figure | Artifact |",
        "|---|---|",
        "| Figure 2, system architecture | `outputs/manuscript_tables_figures_6h/figures/figure_2_system_architecture.pdf` |",
        "| Figure 3, calibration curve | `outputs/manuscript_tables_figures_6h/figures/figure_3_calibration_curve.pdf` |",
        "| Figure 4, decision curve analysis | `outputs/manuscript_tables_figures_6h/figures/figure_4_decision_curve_analysis.pdf` |",
        "| Figure 5, example timelines and activated rules | `outputs/manuscript_tables_figures_6h/figures/figure_5_patient_timeline_activated_rules.pdf` |",
        "",
        "Figure 3 and Figure 4 use the frozen final model. eICU probabilities and thresholds are transported from MIMIC without external refitting or recalibration.",
    ]
    Path("docs/adult_cohort_manuscript_artifacts.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    print("Auditing adult eligibility...")
    mimic, eicu, audit = adult_audit(Path(args.mimic_dir), Path(args.eicu_dir), args.min_age)

    external_dir = Path("outputs/eicu_external_validation/final_frozen_model_evaluation")
    print("Building Table 1 patient characteristics...")
    table1 = cohort_characteristics(
        mimic,
        eicu,
        Path(args.split_manifest),
        Path(args.protocol),
        external_dir / "eicu_external_predictions.csv.gz",
        external_dir / "external_metrics.json",
        args.chunksize,
    )
    print("Building Table 2 feature missingness...")
    table2_path = output_dir / "table_2_feature_missingness.csv"
    table2 = missingness_table(
        table2_path,
        Path(args.hourly_csv),
        Path(args.split_manifest),
        Path("outputs/eicu_external_validation/eicu_hourly_quality.json"),
        args.chunksize,
        args.reuse_missingness,
    )
    print("Collecting Tables 3-5...")
    tables = {
        "table_1_patient_characteristics": table1,
        "table_2_feature_missingness": table2.round(1),
        "table_3_model_performance": performance_table(),
        "table_4_ablation_study": ablation_table().round(4),
        "table_5_rule_evaluation": rule_table().round({"Result": 3}),
    }
    write_outputs(output_dir, tables, audit)

    protocol = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    external = json.loads((external_dir / "external_metrics.json").read_text(encoding="utf-8"))
    print("Rendering Figures 1-5...")
    figure_cohort_flow(audit, protocol, external, output_dir)
    figure_architecture(output_dir)
    figure_calibration(output_dir)
    figure_decision_curve(output_dir)
    figure_case_timeline(output_dir)
    print(f"Wrote manuscript artifacts to {output_dir}")


if __name__ == "__main__":
    main()
