"""Summarize the formal missingness ablation against the matched full KG-TFNN.

The script validates identical patient windows and outcomes, reports per-seed and
ensemble metrics, and uses subject-clustered paired bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from advanced_model_evaluation import patient_bootstrap, percentile_ci, threshold_at_specificity
from model_evaluation_report import binary_metrics, calibration_bins


KEY_COLS = ["subject_id", "stay_id", "sofa_hour"]
VARIANTS = {
    "full": "Full Knowledge-Guided Temporal FNN",
    "no_missingness": "KG-TFNN without missingness channels",
    "missingness_only": "Missingness-only temporal FNN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize missingness ablation results.")
    parser.add_argument("--full-root", default="outputs/fnn_ablation_6h_equal_sample")
    parser.add_argument("--ablation-root", default="outputs/missingness_ablation_6h_equal_sample")
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260710)
    parser.add_argument("--output-dir", default="outputs/missingness_ablation_6h_equal_sample/evaluation")
    return parser.parse_args()


def result_path(root: Path, seed: int, variant: str) -> Path:
    return root / f"seed_{seed}" / variant / "result.json"


def prediction_path(root: Path, seed: int, variant: str, split: str) -> Path:
    filename = "validation_predictions.csv.gz" if split == "validation" else "test_predictions.csv.gz"
    return root / f"seed_{seed}" / variant / filename


def load_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=[*KEY_COLS, "y_true", "y_prob"])
    frame = frame.sort_values(KEY_COLS).reset_index(drop=True)
    if frame.duplicated(KEY_COLS).any():
        raise ValueError(f"Duplicate prediction windows: {path}")
    return frame


def variant_root(full_root: Path, ablation_root: Path, variant: str) -> Path:
    return full_root if variant == "full" else ablation_root


def validate_and_load(
    full_root: Path,
    ablation_root: Path,
    seeds: list[int],
) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]], dict]:
    run_rows = []
    predictions: dict[str, dict[str, list[pd.DataFrame]]] = {
        name: {"validation": [], "test": []} for name in VARIANTS
    }
    reference: dict[str, pd.DataFrame] = {}
    protocol_hashes = set()

    for variant, display_name in VARIANTS.items():
        root = variant_root(full_root, ablation_root, variant)
        for seed in seeds:
            path = result_path(root, seed, variant)
            if not path.exists():
                raise FileNotFoundError(f"Incomplete experiment: {path}")
            result = json.loads(path.read_text(encoding="utf-8"))
            protocol_hashes.add(result.get("protocol_sha256"))
            run_rows.append(
                {
                    "variant": variant,
                    "model": display_name,
                    "seed": seed,
                    "auroc": result["test_auroc"],
                    "auprc": result["test_auprc"],
                    "brier": result["test_brier"],
                    "ece": result["test_ece"],
                    "best_epoch": result["best_epoch"],
                    "actual_epochs": result["actual_epochs"],
                    "train_windows": result["train_windows"],
                    "validation_windows": result["val_windows"],
                    "test_windows": result["test_windows"],
                }
            )
            for split in ["validation", "test"]:
                frame = load_prediction(prediction_path(root, seed, variant, split))
                if split not in reference:
                    reference[split] = frame[[*KEY_COLS, "y_true"]].copy()
                elif not frame[[*KEY_COLS, "y_true"]].equals(reference[split]):
                    raise ValueError(f"Window or outcome mismatch: {variant}, seed {seed}, {split}")
                predictions[variant][split].append(frame)

    if len(protocol_hashes) != 1:
        raise ValueError(f"Comparison protocol mismatch: {sorted(protocol_hashes)}")

    ensembles: dict[str, dict[str, pd.DataFrame]] = {}
    for variant in VARIANTS:
        ensembles[variant] = {}
        for split in ["validation", "test"]:
            ensemble = reference[split].copy()
            probability_matrix = np.column_stack(
                [frame["y_prob"].to_numpy(dtype=float) for frame in predictions[variant][split]]
            )
            ensemble["y_prob"] = probability_matrix.mean(axis=1)
            ensembles[variant][split] = ensemble

    audit = {
        "status": "passed",
        "seeds": seeds,
        "protocol_sha256": next(iter(protocol_hashes)),
        "validation_windows": len(reference["validation"]),
        "test_windows": len(reference["test"]),
        "test_subjects": int(reference["test"]["subject_id"].nunique()),
        "identical_windows_and_outcomes": True,
        "ensemble_definition": "Arithmetic mean of validation-calibrated probabilities across seeds",
    }
    return pd.DataFrame(run_rows), ensembles, audit


def summarize_runs(run_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, model), group in run_table.groupby(["variant", "model"], sort=False):
        row = {"variant": variant, "model": model, "seeds": len(group)}
        for metric in ["auroc", "auprc", "brier", "ece"]:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_ensembles(
    ensembles: dict[str, dict[str, pd.DataFrame]],
    reps: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    bootstrap_parts = []
    thresholds_by_variant = {}
    for variant, parts in ensembles.items():
        val = parts["validation"]
        test = parts["test"]
        y_val = val["y_true"].to_numpy(dtype=np.int8)
        p_val = val["y_prob"].to_numpy(dtype=float)
        y_test = test["y_true"].to_numpy(dtype=np.int8)
        p_test = test["y_prob"].to_numpy(dtype=float)
        thresholds = {
            specificity: threshold_at_specificity(y_val, p_val, specificity)
            for specificity in [0.90, 0.95]
        }
        thresholds_by_variant[variant] = thresholds
        metrics = binary_metrics(y_test, p_test)
        _, calibration = calibration_bins(y_test, p_test, 10, {})
        metric_rows.append(
            {
                "variant": variant,
                "model": VARIANTS[variant],
                **metrics,
                "ece": calibration["ece"],
                "threshold_at_spec_90": thresholds[0.90],
                "threshold_at_spec_95": thresholds[0.95],
            }
        )
        bootstrap = patient_bootstrap(test, thresholds, reps=reps, seed=seed)
        bootstrap.insert(0, "variant", variant)
        bootstrap_parts.append(bootstrap)

    metrics_table = pd.DataFrame(metric_rows)
    bootstrap_table = pd.concat(bootstrap_parts, ignore_index=True)
    ci_rows = []
    for variant, group in bootstrap_table.groupby("variant", sort=False):
        for metric in [column for column in group.columns if column not in {"variant", "replicate"}]:
            low, high = percentile_ci(group[metric])
            ci_rows.append({"variant": variant, "metric": metric, "ci_low": low, "ci_high": high})
    return metrics_table, bootstrap_table, pd.DataFrame(ci_rows)


def paired_differences(bootstrap: pd.DataFrame) -> pd.DataFrame:
    wide = bootstrap.pivot(index="replicate", columns="variant")
    rows = []
    for comparator in ["no_missingness", "missingness_only"]:
        for metric in ["auroc", "auprc", "brier", "ece"]:
            difference = wide[metric]["full"] - wide[metric][comparator]
            low, high = np.quantile(difference, [0.025, 0.975])
            p_value = min(1.0, 2 * min(float(np.mean(difference <= 0)), float(np.mean(difference >= 0))))
            rows.append(
                {
                    "reference": "full",
                    "comparator": comparator,
                    "metric": metric,
                    "mean_difference": float(difference.mean()),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "bootstrap_p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def save_figure(metrics: pd.DataFrame, ci_table: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_data = metrics[["variant", "model", "auroc", "auprc", "brier", "ece"]].copy()
    order = list(VARIANTS)
    figure_data["variant"] = pd.Categorical(figure_data["variant"], order, ordered=True)
    figure_data = figure_data.sort_values("variant")
    colors = ["#4E79A7", "#F28E2B", "#59A14F"]
    short_labels = ["Full KG-TFNN", "No missingness", "Missingness only"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for axis, metric, title in zip(
        axes.flat,
        ["auroc", "auprc", "brier", "ece"],
        ["AUROC", "AUPRC", "Brier score", "Expected calibration error"],
    ):
        point = figure_data[metric].to_numpy(dtype=float)
        ci = (
            ci_table[ci_table["metric"] == metric]
            .set_index("variant")
            .reindex(order)
        )
        errors = np.vstack(
            [
                point - ci["ci_low"].to_numpy(dtype=float),
                ci["ci_high"].to_numpy(dtype=float) - point,
            ]
        )
        x = np.arange(len(order))
        axis.bar(x, point, color=colors, width=0.68)
        axis.errorbar(x, point, yerr=errors, fmt="none", ecolor="#333333", capsize=3)
        axis.set_xticks(x, short_labels, rotation=12, ha="right")
        axis.set_title(title)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.7)
        if metric == "auroc":
            axis.set_ylim(0.5, min(1.0, max(ci["ci_high"].max() + 0.03, 0.7)))
        else:
            axis.set_ylim(0, max(ci["ci_high"].max() * 1.2, point.max() * 1.15))
    fig.suptitle("Missingness Feature Ablation, Seed-Ensemble Evaluation")
    fig.tight_layout()
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf"]:
        fig.savefig(figures / f"missingness_ablation_performance.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fmt(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.4f}"


def write_report(
    path: Path,
    run_summary: pd.DataFrame,
    ensemble_metrics: pd.DataFrame,
    paired: pd.DataFrame,
    audit: dict,
    bootstrap_reps: int,
) -> None:
    lines = [
        "# Missingness Ablation, Primary 6-Hour Outcome",
        "",
        "All variants use the same patient split, 200,000 training windows, 50,000 validation windows,",
        "830,839 independent test windows, predictors, outcome, optimizer settings, and random seeds.",
        "",
        "## Seed-level Results",
        "",
        "| Model | AUROC, mean +/- SD | AUPRC, mean +/- SD | Brier, mean +/- SD | ECE, mean +/- SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in run_summary.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.auroc_mean:.4f} +/- {row.auroc_sd:.4f} | "
            f"{row.auprc_mean:.4f} +/- {row.auprc_sd:.4f} | "
            f"{row.brier_mean:.4f} +/- {row.brier_sd:.4f} | "
            f"{row.ece_mean:.4f} +/- {row.ece_sd:.4f} |"
        )
    lines.extend(
        [
            "",
            f"## {len(audit['seeds'])}-Seed Ensemble",
            "",
            "| Model | AUROC | AUPRC | Brier | ECE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in ensemble_metrics.itertuples(index=False):
        lines.append(
            f"| {row.model} | {fmt(row.auroc)} | {fmt(row.auprc)} | {fmt(row.brier)} | {fmt(row.ece)} |"
        )
    lines.extend(
        [
            "",
            "## Paired Patient-Clustered Bootstrap",
            "",
            "Difference is full KG-TFNN minus the comparator; 95% CIs are percentile intervals.",
            "",
            "| Comparator | Metric | Mean difference | 95% CI | P value |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in paired.itertuples(index=False):
        p_value = (
            f"<{2 / bootstrap_reps:.3f}"
            if row.bootstrap_p_value == 0
            else f"{row.bootstrap_p_value:.4f}"
        )
        lines.append(
            f"| {VARIANTS[row.comparator]} | {row.metric.upper()} | {row.mean_difference:.4f} | "
            f"{row.ci_low:.4f} to {row.ci_high:.4f} | {p_value} |"
        )
    lines.extend(
        [
            "",
            "## Audit",
            "",
            f"- Protocol SHA-256: `{audit['protocol_sha256']}`",
            f"- Test subjects: {audit['test_subjects']:,}",
            f"- Test windows: {audit['test_windows']:,}",
            "- Test windows and outcomes were byte-order matched before analysis.",
            "- Thresholds and calibration were determined without using test outcomes.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_table, ensembles, audit = validate_and_load(
        Path(args.full_root), Path(args.ablation_root), seeds
    )
    run_summary = summarize_runs(run_table)
    ensemble_metrics, bootstrap, ci_table = evaluate_ensembles(
        ensembles, args.bootstrap_reps, args.bootstrap_seed
    )
    paired = paired_differences(bootstrap)

    run_table.to_csv(output_dir / "seed_level_results.csv", index=False)
    run_summary.to_csv(output_dir / "seed_aggregate_results.csv", index=False)
    ensemble_metrics.to_csv(output_dir / "ensemble_metrics.csv", index=False)
    bootstrap.to_csv(output_dir / "patient_cluster_bootstrap.csv.gz", index=False)
    ci_table.to_csv(output_dir / "ensemble_metric_ci.csv", index=False)
    paired.to_csv(output_dir / "paired_missingness_effects.csv", index=False)
    (output_dir / "comparison_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    write_report(
        output_dir / "missingness_ablation_report.md",
        run_summary,
        ensemble_metrics,
        paired,
        audit,
        args.bootstrap_reps,
    )
    save_figure(ensemble_metrics, ci_table, output_dir)
    print(f"Missingness ablation summary written to {output_dir}")


if __name__ == "__main__":
    main()
