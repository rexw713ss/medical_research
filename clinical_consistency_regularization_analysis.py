"""Evaluate the behavioral effect of clinical consistency regularization."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from anfis_model import TemporalAttentionFNN, clinical_rule_priors, expert_feature_config
from posthoc_explainability_comparison import OUTPUT as EXPLANATION_OUTPUT
from posthoc_explainability_comparison import build_sequences


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/clinical_consistency_regularization_6h"
ABLATION = ROOT / "outputs/fnn_ablation_6h_equal_sample"
SEEDS = (42, 52, 62)
VARIANTS = ("no_consistency", "full")
EPSILON = 1e-4


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


def behavioral_metrics(model: TemporalAttentionFNN, sequences: np.ndarray, device: torch.device) -> dict[str, float]:
    feature_worsening = 0
    feature_violations = 0
    all_feature_transitions = 0
    additive_worsening = 0
    risk_reversals = 0
    all_hour_transitions = 0
    consistency_sum = 0.0
    consistency_terms = 0
    hourly_reversal_magnitudes = []

    for start in range(0, len(sequences), 256):
        x = torch.from_numpy(sequences[start : start + 256]).to(device)
        with torch.no_grad():
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
        penalty = np.maximum(delta_feature, 0) * np.maximum(-delta_total[:, :, None], 0)
        consistency_sum += float(penalty.sum())
        consistency_terms += int(penalty.size)

        delta_additive = delta_feature.sum(2)
        additive_mask = delta_additive > EPSILON
        additive_reversal = additive_mask & (delta_total < -EPSILON)
        additive_worsening += int(additive_mask.sum())
        risk_reversals += int(additive_reversal.sum())
        all_hour_transitions += int(delta_total.size)
        hourly_reversal_magnitudes.extend((-delta_total[additive_reversal]).tolist())

    weights = model.static_fnn.cross_rule_weights.detach().cpu().numpy()
    return {
        "feature_consistency_violation_rate_given_worsening": feature_violations / max(feature_worsening, 1),
        "feature_consistency_violation_rate_all_transitions": feature_violations / max(all_feature_transitions, 1),
        "clinical_consistency_penalty_mean": consistency_sum / max(consistency_terms, 1),
        "risk_reversal_frequency_given_additive_worsening": risk_reversals / max(additive_worsening, 1),
        "risk_reversal_frequency_all_hour_transitions": risk_reversals / max(all_hour_transitions, 1),
        "risk_reversal_magnitude_median": float(np.median(hourly_reversal_magnitudes)) if hourly_reversal_magnitudes else 0.0,
        "negative_cross_rule_fraction": float(np.mean(weights < 0)),
        "feature_worsening_transitions": feature_worsening,
        "additive_worsening_transitions": additive_worsening,
    }


def paired_summary(seed_frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column for column in seed_frame.columns
        if column not in {"seed", "variant", "lambda_cons", "rule_stability", "guideline_risk_correlation", "normalized_rule_drift", "guideline_direction_alignment"}
        and pd.api.types.is_numeric_dtype(seed_frame[column])
    ]
    rows = []
    for variant in VARIANTS:
        group = seed_frame[seed_frame.variant == variant]
        row = {"variant": variant, "n_seeds": len(group)}
        for metric in metric_columns:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        rows.append(row)
    wide = seed_frame.pivot(index="seed", columns="variant")
    difference = {"variant": "full_minus_no_consistency", "n_seeds": len(SEEDS)}
    for metric in metric_columns:
        values = wide[metric]["full"] - wide[metric]["no_consistency"]
        difference[f"{metric}_mean"] = values.mean()
        difference[f"{metric}_std"] = values.std(ddof=1)
    rows.append(difference)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "figures").mkdir(exist_ok=True)
    keys = pd.read_csv(EXPLANATION_OUTPUT / "mimic_explanation_sample.csv")
    hourly = pd.read_pickle(EXPLANATION_OUTPUT / "mimic-iv_selected_hourly_rows.pkl")
    sequences, metadata = build_sequences(hourly, keys)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aggregate = pd.read_csv(ABLATION / "ablation_aggregate.csv").set_index("variant")

    rows = []
    for seed in SEEDS:
        for variant in VARIANTS:
            checkpoint_path = ABLATION / f"seed_{seed}" / variant / "best_model.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model = load_model(checkpoint_path, device)
            metrics = behavioral_metrics(model, sequences, device)
            quality = checkpoint["rule_quality"]
            rows.append(
                {
                    "seed": seed, "variant": variant,
                    "lambda_cons": float(checkpoint["args"]["lambda_cons"]) if variant == "full" else 0.0,
                    **metrics,
                    "guideline_risk_correlation": quality["rule_concordance"],
                    "normalized_rule_drift": quality["rule_drift_normalized"],
                    "guideline_direction_alignment": 1.0,
                    "rule_stability": aggregate.loc[variant, "rule_stability"],
                }
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    seed_frame = pd.DataFrame(rows)
    seed_frame.to_csv(OUTPUT / "consistency_metrics_by_seed.csv", index=False)
    summary = paired_summary(seed_frame)
    summary.to_csv(OUTPUT / "consistency_regularization_summary.csv", index=False)

    means = seed_frame.groupby("variant", sort=False).mean(numeric_only=True)
    plot_metrics = [
        "feature_consistency_violation_rate_given_worsening",
        "risk_reversal_frequency_given_additive_worsening",
        "normalized_rule_drift",
        "guideline_risk_correlation",
        "rule_stability",
    ]
    labels = ["Violation rate", "Risk reversal", "Normalized drift", "Guideline-risk rho", "Top-10 stability"]
    figure, axes = plt.subplots(1, len(plot_metrics), figsize=(15, 4))
    colors = ["#D55E00", "#0072B2"]
    for axis, metric, label in zip(axes, plot_metrics, labels):
        values = means.loc[list(VARIANTS), metric]
        axis.bar([0, 1], values, color=colors)
        axis.set_xticks([0, 1], ["No consistency", "Full"], rotation=25, ha="right")
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle("Clinical consistency regularization: behavioral and rule-quality effects")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(OUTPUT / f"figures/consistency_regularization_effects.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(figure)

    report = [
        "# Clinical Consistency Regularization Analysis", "",
        f"Exploratory analysis only: evaluation used {len(metadata):,} fixed one-window-per-stay MIMIC test cases and seeds {SEEDS}; it is not a formal full-cohort result. A feature-level violation occurs when one learned feature-risk contribution increases by more than {EPSILON} while total hourly fuzzy risk decreases. A risk reversal occurs when summed feature risk increases while total hourly fuzzy risk decreases.", "",
        summary.to_csv(index=False), "",
        "Guideline-direction alignment is 1.0 in both variants because antecedent directions are fixed by the same rule inventory; it cannot demonstrate an effect of the consistency loss. Guideline-risk correlation and normalized membership drift are empirical trained-parameter diagnostics.",
    ]
    (OUTPUT / "clinical_consistency_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUTPUT / "analysis_config.json").write_text(
        json.dumps({"formal_full_data": False, "analysis_scope": "exploratory_sample", "sample_windows": len(metadata), "one_window_per_stay": True, "seeds": SEEDS, "epsilon": EPSILON,
                    "violation_definition": "delta feature risk > epsilon and delta total hourly risk < -epsilon",
                    "risk_reversal_definition": "delta summed feature risk > epsilon and delta total hourly risk < -epsilon"}, indent=2),
        encoding="utf-8",
    )
    print(seed_frame.groupby("variant").mean(numeric_only=True)[plot_metrics].to_string())


if __name__ == "__main__":
    main()
