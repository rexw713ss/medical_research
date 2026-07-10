"""Hospital-clustered and site-heterogeneity analysis for frozen eICU validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from advanced_model_evaluation import percentile_ci
from eicu_external_validation import optimized_cluster_bootstrap
from model_evaluation_report import binary_metrics, calibration_bins


SOURCE = Path("outputs/eicu_external_validation/final_frozen_model_evaluation")
OUTPUT = Path("outputs/eicu_hospital_sensitivity_6h")
BOOTSTRAP_REPS = 500
SEED = 42


def load_predictions() -> pd.DataFrame:
    columns = ["subject_id", "stay_id", "hospital_id", "y_true", "y_prob"]
    frame = pd.read_csv(SOURCE / "eicu_external_predictions.csv.gz", usecols=columns)
    frame["hospital_id"] = pd.to_numeric(frame["hospital_id"], errors="raise").astype("int64")
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="raise").astype("int8")
    frame["y_prob"] = pd.to_numeric(frame["y_prob"], errors="raise")
    return frame


def hospital_performance(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hospital, group in frame.groupby("hospital_id", sort=True):
        positive = int(group["y_true"].sum())
        negative = int(len(group) - positive)
        if group["subject_id"].nunique() < 100 or positive < 50 or negative < 50:
            continue
        point = binary_metrics(group["y_true"].to_numpy(), group["y_prob"].to_numpy())
        _, calibration = calibration_bins(
            group["y_true"].to_numpy(),
            group["y_prob"].to_numpy(),
            n_bins=10,
            model_key={"hospital_id": int(hospital)},
        )
        rows.append(
            {
                "hospital_id": int(hospital),
                "patients": int(group["subject_id"].nunique()),
                "stays": int(group["stay_id"].nunique()),
                "windows": len(group),
                "positive": positive,
                "prevalence": positive / len(group),
                **point,
                "ece": calibration["ece"],
            }
        )
    return pd.DataFrame(rows).sort_values("auroc", ascending=False)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("Loading frozen eICU predictions...")
    frame = load_predictions()
    fixed = pd.read_csv(SOURCE / "external_fixed_specificity.csv")
    thresholds = {
        float(row.target_specificity): float(row.threshold)
        for row in fixed.itertuples(index=False)
    }

    sites = hospital_performance(frame)
    sites.to_csv(OUTPUT / "per_hospital_performance.csv", index=False)
    summary_rows = []
    for metric in ["auroc", "auprc", "brier", "ece", "prevalence"]:
        summary_rows.append(
            {
                "metric": metric,
                "eligible_hospitals": len(sites),
                "macro_mean": float(sites[metric].mean()),
                "median": float(sites[metric].median()),
                "q1": float(sites[metric].quantile(0.25)),
                "q3": float(sites[metric].quantile(0.75)),
                "minimum": float(sites[metric].min()),
                "maximum": float(sites[metric].max()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(OUTPUT / "hospital_heterogeneity_summary.csv", index=False)

    print(f"Running {BOOTSTRAP_REPS} hospital-clustered replicates...")
    hospital_frame = frame.copy()
    hospital_frame["subject_id"] = hospital_frame["hospital_id"].astype(str)
    bootstrap = optimized_cluster_bootstrap(
        hospital_frame[["subject_id", "y_true", "y_prob"]],
        thresholds,
        BOOTSTRAP_REPS,
        SEED,
    )
    bootstrap.to_csv(OUTPUT / "hospital_cluster_bootstrap.csv.gz", index=False, compression="gzip")
    ci_rows = []
    for metric in bootstrap.columns:
        if metric == "replicate":
            continue
        low, high = percentile_ci(bootstrap[metric])
        ci_rows.append({"metric": metric, "ci95_low": low, "ci95_high": high})
    pd.DataFrame(ci_rows).to_csv(OUTPUT / "hospital_cluster_ci.csv", index=False)

    external = json.loads((SOURCE / "external_metrics.json").read_text(encoding="utf-8"))
    report = [
        "# eICU Hospital-Clustered Sensitivity",
        "",
        f"- Hospitals in pooled external evaluation: {frame['hospital_id'].nunique()}.",
        f"- Hospitals meeting per-site reporting threshold: {len(sites)} (>=100 patients and >=50 positive/negative windows).",
        f"- Hospital-clustered bootstrap replicates: {BOOTSTRAP_REPS}.",
        "- Frozen MIMIC model, calibration and thresholds; no eICU fitting.",
        "",
        "## Clustered Confidence Intervals",
        "",
        pd.DataFrame(ci_rows).to_csv(index=False),
        "",
        "## Site Heterogeneity",
        "",
        pd.DataFrame(summary_rows).to_csv(index=False),
        "",
        f"Subject-clustered external AUROC was {external['mimic_calibrated']['auroc']:.4f}; hospital-clustered uncertainty is reported separately and does not replace the prespecified subject-clustered primary analysis.",
    ]
    (OUTPUT / "eicu_hospital_sensitivity_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figures = OUTPUT / "figures"
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(sites["auroc"], bins=18, color="#4E79A7", edgecolor="white")
    axes[0].axvline(external["mimic_calibrated"]["auroc"], color="#E15759", linestyle="--", label="Pooled")
    axes[0].set(xlabel="Hospital AUROC", ylabel="Hospitals", title="External Site Discrimination")
    axes[0].legend()
    axes[1].scatter(sites["positive"], sites["auroc"], s=20, alpha=0.7, color="#59A14F")
    axes[1].set_xscale("log")
    axes[1].set(xlabel="Positive windows (log scale)", ylabel="AUROC", title="Hospital Size and Performance")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"eicu_hospital_heterogeneity.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    config = {
        "bootstrap_unit": "hospital_id",
        "bootstrap_reps": BOOTSTRAP_REPS,
        "seed": SEED,
        "per_hospital_minimum_patients": 100,
        "per_hospital_minimum_positive_windows": 50,
        "no_eicu_fitting": True,
        "thresholds_from_mimic_validation": thresholds,
    }
    (OUTPUT / "analysis_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
