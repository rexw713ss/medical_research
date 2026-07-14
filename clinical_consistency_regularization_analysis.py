"""Full-data behavioral audit of clinical consistency regularization."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from anfis_model import TemporalAttentionFNN, clinical_rule_priors, expert_feature_config
from full_data_window_utils import FormalWindowData, iter_window_batches, load_formal_window_data


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs/clinical_consistency_regularization_6h"
ABLATION = ROOT / "outputs/fnn_ablation_6h_equal_sample"
MIMIC_HOURLY = ROOT / "model_hourly_features_v3.csv"
MIMIC_PREDICTIONS = ROOT / "outputs/explicit_kg_tfnn_paired_comparison_6h/inputs/explicit_kg_tfnn/test_predictions.csv.gz"
TARGET = "label_sofa_increase_ge2_6h"
TIME = "sofa_hour"
SPLIT = "subject_id"
SEEDS = (42, 52, 62)
VARIANTS = ("no_consistency", "full")
EPSILON = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate consistency regularization on every eligible MIMIC test window."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> TemporalAttentionFNN:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    model = TemporalAttentionFNN(
        feature_configs=expert_feature_config,
        rule_configs=clinical_rule_priors,
        seq_length=args["seq_length"],
        attention_hidden=args["attention_hidden"],
        threshold=args["threshold"],
        rule_score_scale=args["rule_score_scale"],
        use_explicit_temporal_features=True,
        explicit_temporal_scale=args["explicit_temporal_scale"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def behavioral_metrics(
    model: TemporalAttentionFNN,
    data: FormalWindowData,
    device: torch.device,
    batch_size: int,
    progress_every: int,
    run_label: str,
) -> dict[str, float | int]:
    feature_worsening = 0
    feature_violations = 0
    all_feature_transitions = 0
    additive_worsening = 0
    risk_reversals = 0
    all_hour_transitions = 0
    consistency_sum = 0.0
    consistency_terms = 0
    reversal_magnitude_chunks: list[np.ndarray] = []
    processed_windows = 0
    next_progress = progress_every
    started = perf_counter()

    for sequences, _, _ in iter_window_batches(data, batch_size):
        x = torch.from_numpy(sequences).to(device)
        with torch.inference_mode():
            output = model(x)
        feature = output.feature_risks.detach().cpu().numpy()
        total = output.hourly_risk_scores.detach().cpu().numpy()
        delta_feature = feature[:, 1:] - feature[:, :-1]
        delta_total = total[:, 1:] - total[:, :-1]
        worsening = delta_feature > EPSILON
        reversal = worsening & (delta_total[:, :, None] < -EPSILON)
        feature_worsening += int(worsening.sum())
        feature_violations += int(reversal.sum())
        all_feature_transitions += int(delta_feature.size)
        penalty = np.maximum(delta_feature, 0) * np.maximum(
            -delta_total[:, :, None], 0
        )
        consistency_sum += float(penalty.sum())
        consistency_terms += int(penalty.size)

        delta_additive = delta_feature.sum(2)
        additive_mask = delta_additive > EPSILON
        additive_reversal = additive_mask & (delta_total < -EPSILON)
        additive_worsening += int(additive_mask.sum())
        risk_reversals += int(additive_reversal.sum())
        all_hour_transitions += int(delta_total.size)
        if np.any(additive_reversal):
            reversal_magnitude_chunks.append(
                (-delta_total[additive_reversal]).astype(np.float32, copy=False)
            )

        processed_windows += len(sequences)
        if processed_windows >= next_progress:
            elapsed = perf_counter() - started
            rate = processed_windows / max(elapsed, 1e-9)
            print(
                f"[{run_label}] {processed_windows:,}/{data.expected_windows:,} "
                f"windows ({rate:,.0f} windows/s)",
                flush=True,
            )
            next_progress += progress_every

    if processed_windows != data.expected_windows:
        raise ValueError(
            f"{run_label} processed {processed_windows:,} of "
            f"{data.expected_windows:,} formal windows."
        )
    reversal_magnitudes = (
        np.concatenate(reversal_magnitude_chunks)
        if reversal_magnitude_chunks
        else np.empty(0, dtype=np.float32)
    )
    weights = model.static_fnn.cross_rule_weights.detach().cpu().numpy()
    return {
        "processed_windows": processed_windows,
        "feature_consistency_violation_rate_given_worsening": (
            feature_violations / max(feature_worsening, 1)
        ),
        "feature_consistency_violation_rate_all_transitions": (
            feature_violations / max(all_feature_transitions, 1)
        ),
        "clinical_consistency_penalty_mean": (
            consistency_sum / max(consistency_terms, 1)
        ),
        "risk_reversal_frequency_given_additive_worsening": (
            risk_reversals / max(additive_worsening, 1)
        ),
        "risk_reversal_frequency_all_hour_transitions": (
            risk_reversals / max(all_hour_transitions, 1)
        ),
        "risk_reversal_magnitude_median": (
            float(np.median(reversal_magnitudes)) if len(reversal_magnitudes) else 0.0
        ),
        "risk_reversal_magnitude_iqr_low": (
            float(np.quantile(reversal_magnitudes, 0.25))
            if len(reversal_magnitudes)
            else 0.0
        ),
        "risk_reversal_magnitude_iqr_high": (
            float(np.quantile(reversal_magnitudes, 0.75))
            if len(reversal_magnitudes)
            else 0.0
        ),
        "negative_cross_rule_fraction": float(np.mean(weights < 0)),
        "feature_worsening_transitions": feature_worsening,
        "feature_violation_transitions": feature_violations,
        "additive_worsening_transitions": additive_worsening,
        "risk_reversal_transitions": risk_reversals,
        "all_feature_transitions": all_feature_transitions,
        "all_hour_transitions": all_hour_transitions,
    }


def paired_summary(seed_frame: pd.DataFrame) -> pd.DataFrame:
    excluded = {"seed", "variant", "lambda_cons"}
    metric_columns = [
        column
        for column in seed_frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(seed_frame[column])
    ]
    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        group = seed_frame[seed_frame.variant == variant]
        row: dict[str, object] = {"variant": variant, "n_seeds": len(group)}
        for metric in metric_columns:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        rows.append(row)

    wide = seed_frame.pivot(index="seed", columns="variant")
    difference: dict[str, object] = {
        "variant": "full_minus_no_consistency",
        "n_seeds": len(SEEDS),
    }
    for metric in metric_columns:
        values = wide[metric]["full"] - wide[metric]["no_consistency"]
        difference[f"{metric}_mean"] = values.mean()
        difference[f"{metric}_std"] = values.std(ddof=1)
    rows.append(difference)
    return pd.DataFrame(rows)


def write_report(output: Path, summary: pd.DataFrame, expected_windows: int) -> None:
    report = [
        "# Full-Data Clinical Consistency Regularization Analysis",
        "",
        (
            f"Formal evaluation used all {expected_windows:,} eligible MIMIC-IV test "
            f"windows for seeds {SEEDS} and both variants. A feature-level violation occurs "
            f"when one learned feature-risk contribution increases by more than {EPSILON} "
            "while total hourly fuzzy risk decreases. A risk reversal occurs when summed "
            "feature risk increases while total hourly fuzzy risk decreases."
        ),
        "",
        summary.to_csv(index=False, lineterminator="\n").strip(),
        "",
        (
            "Guideline-direction alignment is 1.0 in both variants because antecedent "
            "directions are fixed by the same rule inventory; it cannot demonstrate an "
            "effect of the consistency loss. Guideline-risk correlation and normalized "
            "membership drift are empirical trained-parameter diagnostics."
        ),
    ]
    (output / "clinical_consistency_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)

    print("Loading all formal MIMIC-IV test windows...", flush=True)
    data = load_formal_window_data(
        database="MIMIC-IV",
        hourly_path=MIMIC_HOURLY,
        prediction_path=MIMIC_PREDICTIONS,
        target_col=TARGET,
        time_col=TIME,
        split_col=SPLIT,
        seq_length=24,
    )
    cohort_audit = data.audit_record()
    print(json.dumps(cohort_audit, indent=2), flush=True)
    device = choose_device(args.device)
    aggregate = pd.read_csv(ABLATION / "ablation_aggregate.csv").set_index("variant")

    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for variant in VARIANTS:
            checkpoint_path = ABLATION / f"seed_{seed}" / variant / "best_model.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = load_model(checkpoint_path, device)
            metrics = behavioral_metrics(
                model,
                data,
                device,
                args.batch_size,
                args.progress_every,
                f"seed={seed}, variant={variant}",
            )
            quality = checkpoint["rule_quality"]
            rows.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "lambda_cons": (
                        float(checkpoint["args"]["lambda_cons"])
                        if variant == "full"
                        else 0.0
                    ),
                    **metrics,
                    "guideline_risk_correlation": quality["rule_concordance"],
                    "normalized_rule_drift": quality["rule_drift_normalized"],
                    "guideline_direction_alignment": 1.0,
                    "rule_stability": aggregate.loc[variant, "rule_stability"],
                }
            )
            del model, checkpoint
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    seed_frame = pd.DataFrame(rows)
    seed_frame.to_csv(output / "consistency_metrics_by_seed.csv", index=False)
    summary = paired_summary(seed_frame)
    summary.to_csv(output / "consistency_regularization_summary.csv", index=False)
    (output / "formal_cohort_audit.json").write_text(
        json.dumps(cohort_audit, indent=2), encoding="utf-8"
    )

    means = seed_frame.groupby("variant", sort=False).mean(numeric_only=True)
    plot_metrics = [
        "feature_consistency_violation_rate_given_worsening",
        "risk_reversal_frequency_given_additive_worsening",
        "normalized_rule_drift",
        "guideline_risk_correlation",
        "rule_stability",
    ]
    labels = [
        "Violation rate",
        "Risk reversal",
        "Normalized drift",
        "Guideline-risk rho",
        "Top-10 stability",
    ]
    figure, axes = plt.subplots(1, len(plot_metrics), figsize=(15, 4))
    colors = ["#D55E00", "#0072B2"]
    for axis, metric, label in zip(axes, plot_metrics, labels):
        values = means.loc[list(VARIANTS), metric]
        axis.bar([0, 1], values, color=colors)
        axis.set_xticks([0, 1], ["No consistency", "Full"], rotation=25, ha="right")
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Full-data clinical consistency regularization analysis")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            output / f"figures/consistency_regularization_effects.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)

    write_report(output, summary, data.expected_windows)
    config = {
        "formal_full_data": True,
        "analysis_scope": "all_eligible_mimic_test_windows",
        "target": TARGET,
        "sample_windows": None,
        "processed_windows_per_model": data.expected_windows,
        "models_evaluated": len(SEEDS) * len(VARIANTS),
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "batch_size": args.batch_size,
        "device": str(device),
        "epsilon": EPSILON,
        "violation_definition": (
            "delta feature risk > epsilon and delta total hourly risk < -epsilon"
        ),
        "risk_reversal_definition": (
            "delta summed feature risk > epsilon and delta total hourly risk < -epsilon"
        ),
    }
    (output / "analysis_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(seed_frame.groupby("variant").mean(numeric_only=True)[plot_metrics].to_string())


if __name__ == "__main__":
    main()
