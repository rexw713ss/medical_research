"""Full-data comparison of KG-TFNN intrinsic and post-hoc explanations."""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
import torch
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from anfis_model import (
    FEATURE_ORDER,
    TemporalAttentionFNN,
    clinical_rule_priors,
    explicit_temporal_input_order,
    expert_feature_config,
)
from blackbox_baselines import matched_tree_feature_names
from full_data_window_utils import (
    FormalWindowData,
    iter_window_batches,
    load_formal_window_data,
    sha256_file,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs/posthoc_explainability_comparison_6h"
TARGET = "label_sofa_increase_ge2_6h"
TIME = "sofa_hour"
SPLIT = "subject_id"
SEQ_LENGTH = 24
TOP_K = 5
SEED = 42

MIMIC_PREDICTIONS = ROOT / "outputs/explicit_kg_tfnn_paired_comparison_6h/inputs/lightgbm_matched/test_predictions.csv.gz"
EICU_PREDICTIONS = ROOT / "outputs/eicu_external_validation/final_frozen_model_evaluation/eicu_external_predictions.csv.gz"
MIMIC_HOURLY = ROOT / "model_hourly_features_v3.csv"
EICU_HOURLY = ROOT / "outputs/eicu_external_validation/eicu_hourly_features.pkl"
BASELINE_ROOT = ROOT / "outputs/feature_matched_baselines_6h_equal_sample/sequence_matched_summary"
EBM_PATH = ROOT / "outputs/fair_comparison_6h_equal_sample/interpretable_6h/protocol/ebm/model.joblib"
FNN_CHECKPOINT = ROOT / "outputs/fnn_ablation_6h_equal_sample/seed_42/full/best_model.pt"
SCALING_JSON = ROOT / "outputs/feature_matched_baselines_6h_equal_sample/sequence_matched/gru/sequence_scaling.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the formal explanation comparison on every eligible MIMIC/eICU window."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--fnn-batch-size", type=int, default=512)
    parser.add_argument("--perturbation-repeats", type=int, default=3)
    parser.add_argument("--perturbation-sd-fraction", type=float, default=0.01)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--complexity-only",
        action="store_true",
        help="Build the unified complexity artifacts from an audited completed full-data run.",
    )
    return parser.parse_args()


def summarize_for_trees(sequences: np.ndarray) -> np.ndarray:
    raw = sequences[:, :, : len(FEATURE_ORDER)]
    missing = sequences[:, :, len(FEATURE_ORDER) : 2 * len(FEATURE_ORDER)]
    hours = np.arange(SEQ_LENGTH, dtype=np.float32)
    centered = hours - hours.mean()
    slope = np.einsum("btf,t->bf", raw, centered, optimize=True) / float(np.sum(centered**2))
    return np.concatenate(
        [
            sequences[:, -1, :],
            raw.mean(1),
            raw.min(1),
            raw.max(1),
            raw.std(1),
            slope,
            raw[:, -1] - raw[:, -2],
            raw[:, -1] - raw[:, 0],
            missing.mean(1),
        ],
        axis=1,
    ).astype(np.float32)


def aggregate_feature_names(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    mapping = np.empty(len(names), dtype=np.int64)
    temporal = np.zeros(len(names), dtype=bool)
    for index, name in enumerate(names):
        prefix, value = name.split("::", 1)
        matches = [
            i
            for i, feature in enumerate(FEATURE_ORDER)
            if value == feature or value.startswith(feature + "_")
        ]
        if len(matches) != 1:
            raise ValueError(f"Cannot map summary feature {name!r} to one clinical variable.")
        mapping[index] = matches[0]
        temporal[index] = prefix != "current" or "time_since_last_measurement" in value
    return mapping, temporal


def aggregate_columns(values: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    result = np.zeros((len(values), len(FEATURE_ORDER)), dtype=np.float64)
    for source, target in enumerate(mapping):
        result[:, target] += values[:, source]
    return result


def tree_shap_function(
    model: object,
    mapping: np.ndarray,
    temporal_mask: np.ndarray,
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    explainer = shap.TreeExplainer(model)

    def explain(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        try:
            values = explainer.shap_values(x, check_additivity=False)
        except TypeError:
            values = explainer.shap_values(x)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        aggregated = aggregate_columns(values, mapping)
        temporal_mass = np.abs(values[:, temporal_mask]).sum(1) / (
            np.abs(values).sum(1) + 1e-12
        )
        return aggregated, temporal_mass

    return explain


def ebm_function(model: object) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    term_features = model.term_features_

    def explain(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        terms = np.asarray(model.eval_terms(x))
        result = np.zeros((len(x), len(FEATURE_ORDER)), dtype=np.float64)
        for term_index, feature_indices in enumerate(term_features):
            share = terms[:, term_index] / len(feature_indices)
            for feature_index in feature_indices:
                result[:, feature_index] += share
        return result, np.zeros(len(x), dtype=np.float64)

    return explain


def load_fnn(device: torch.device) -> TemporalAttentionFNN:
    checkpoint = torch.load(FNN_CHECKPOINT, map_location=device, weights_only=False)
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


def fnn_explain(
    model: TemporalAttentionFNN,
    sequences: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    temporal_rows: list[np.ndarray] = []
    rule_antecedents = [
        [FEATURE_ORDER.index(feature) for feature, _ in rule["antecedents"]]
        for rule in model.static_fnn.rule_configs
    ]
    weights = model.static_fnn.cross_rule_weights.detach().cpu().numpy()
    for start in range(0, len(sequences), batch_size):
        x = torch.from_numpy(sequences[start : start + batch_size]).to(device)
        with torch.inference_mode():
            output = model(x)
        attention = output.attention_weights.detach().cpu().numpy()
        feature_risk = output.feature_risks.detach().cpu().numpy()
        rule_activation = output.rule_activations.detach().cpu().numpy()
        base = np.einsum("bt,btf->bf", attention, feature_risk)
        cross = np.zeros_like(base)
        for rule_index, antecedents in enumerate(rule_antecedents):
            contribution = (
                np.einsum("bt,bt->b", attention, rule_activation[:, :, rule_index])
                * weights[rule_index]
                * model.static_fnn.rule_score_scale
                / len(antecedents)
            )
            for feature_index in antecedents:
                cross[:, feature_index] += contribution
        explicit = output.explicit_temporal_contributions.detach().cpu().numpy()
        explicit = (
            explicit.sum(2)
            * model.explicit_temporal_scale
            / math.sqrt(len(FEATURE_ORDER))
        )
        total = base + cross + explicit
        denominator = (
            np.abs(base).sum(1)
            + np.abs(cross).sum(1)
            + np.abs(explicit).sum(1)
            + 1e-12
        )
        rows.append(total)
        temporal_rows.append(np.abs(explicit).sum(1) / denominator)
    return np.concatenate(rows), np.concatenate(temporal_rows)


def normalize_explanations(values: np.ndarray) -> np.ndarray:
    return values / (np.abs(values).sum(1, keepdims=True) + 1e-12)


def cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def top_k_jaccard_rows(left: np.ndarray, right: np.ndarray, k: int) -> np.ndarray:
    left_top = np.argpartition(-np.abs(left), kth=k - 1, axis=1)[:, :k]
    right_top = np.argpartition(-np.abs(right), kth=k - 1, axis=1)[:, :k]
    intersection = (
        left_top[:, :, None] == right_top[:, None, :]
    ).any(axis=2).sum(axis=1)
    return intersection / (2 * k - intersection)


def effective_features(values: np.ndarray, mass: float = 0.80) -> np.ndarray:
    ordered = np.sort(np.abs(values), axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) / (ordered.sum(1, keepdims=True) + 1e-12)
    return (cumulative < mass).sum(1) + 1


def perturb_sequences(
    sequences: np.ndarray,
    raw_sd: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    perturbed = sequences.copy()
    noise = rng.normal(
        0.0,
        raw_sd * fraction,
        size=perturbed[:, :, : len(FEATURE_ORDER)].shape,
    )
    perturbed[:, :, : len(FEATURE_ORDER)] += noise.astype(np.float32)
    return perturbed


@dataclass
class RunningMoments:
    count: int = 0
    total: float = 0.0
    total_squared: float = 0.0

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Explanation metric contains non-finite values.")
        self.count += int(values.size)
        self.total += float(values.sum())
        self.total_squared += float(np.square(values).sum())

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else math.nan

    @property
    def std(self) -> float:
        if self.count < 2:
            return math.nan
        variance = (self.total_squared - self.total**2 / self.count) / (self.count - 1)
        return math.sqrt(max(variance, 0.0))


def histogram_quantile(histogram: np.ndarray, probability: float) -> float:
    count = int(histogram.sum())
    if count == 0:
        return math.nan

    def order_value(position: int) -> float:
        cumulative = np.cumsum(histogram)
        return float(np.searchsorted(cumulative, position + 1))

    rank = probability * (count - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    weight = rank - lower
    return order_value(lower) * (1.0 - weight) + order_value(upper) * weight


@dataclass
class ExplanationAccumulator:
    n_features: int
    windows: int = 0
    absolute_sum: np.ndarray = field(init=False)
    effective_histogram: np.ndarray = field(init=False)
    temporal_mass: RunningMoments = field(default_factory=RunningMoments)
    stability_cosine: RunningMoments = field(default_factory=RunningMoments)
    stability_jaccard: RunningMoments = field(default_factory=RunningMoments)
    consistency_cosine: RunningMoments = field(default_factory=RunningMoments)
    consistency_jaccard: RunningMoments = field(default_factory=RunningMoments)
    previous_normalized: np.ndarray | None = None
    previous_stay: int | None = None
    previous_hour: int | None = None

    def __post_init__(self) -> None:
        self.absolute_sum = np.zeros(self.n_features, dtype=np.float64)
        self.effective_histogram = np.zeros(self.n_features + 1, dtype=np.int64)

    def update_baseline(
        self,
        values: np.ndarray,
        temporal: np.ndarray,
        stays: np.ndarray,
        hours: np.ndarray,
    ) -> None:
        if not np.isfinite(values).all():
            raise ValueError("Model explanation contains non-finite values.")
        self.windows += len(values)
        self.absolute_sum += np.abs(values).sum(axis=0)
        self.temporal_mass.update(temporal)
        counts = effective_features(values)
        self.effective_histogram += np.bincount(
            counts, minlength=self.n_features + 1
        )[: self.n_features + 1]

        normalized = normalize_explanations(values)
        if self.previous_normalized is not None:
            previous_values = np.vstack([self.previous_normalized, normalized[:-1]])
            previous_stays = np.concatenate([[self.previous_stay], stays[:-1]])
            previous_hours = np.concatenate([[self.previous_hour], hours[:-1]])
        else:
            previous_values = normalized[:-1]
            previous_stays = stays[:-1]
            previous_hours = hours[:-1]
            normalized = normalized[1:]
            stays = stays[1:]
            hours = hours[1:]

        if len(normalized):
            adjacent = (stays == previous_stays) & (hours == previous_hours + 1)
            if np.any(adjacent):
                self.consistency_cosine.update(
                    cosine_rows(normalized[adjacent], previous_values[adjacent])
                )
                self.consistency_jaccard.update(
                    top_k_jaccard_rows(
                        normalized[adjacent], previous_values[adjacent], TOP_K
                    )
                )

        self.previous_normalized = normalize_explanations(values[-1:])[0]
        self.previous_stay = int(stays[-1])
        self.previous_hour = int(hours[-1])

    def update_stability(self, baseline: np.ndarray, perturbed: np.ndarray) -> None:
        baseline = normalize_explanations(baseline)
        perturbed = normalize_explanations(perturbed)
        self.stability_cosine.update(cosine_rows(baseline, perturbed))
        self.stability_jaccard.update(top_k_jaccard_rows(baseline, perturbed, TOP_K))

    def mean_absolute(self) -> np.ndarray:
        return self.absolute_sum / max(self.windows, 1)

    def summary(self) -> dict[str, float | int]:
        return {
            "windows": self.windows,
            "stability_pairs": self.stability_cosine.count,
            "stability_cosine_mean": self.stability_cosine.mean,
            "stability_cosine_std": self.stability_cosine.std,
            "stability_top5_jaccard_mean": self.stability_jaccard.mean,
            "stability_top5_jaccard_std": self.stability_jaccard.std,
            "trajectory_consistency_pairs": self.consistency_cosine.count,
            "trajectory_consistency_cosine_mean": self.consistency_cosine.mean,
            "trajectory_consistency_cosine_std": self.consistency_cosine.std,
            "trajectory_consistency_top5_jaccard_mean": self.consistency_jaccard.mean,
            "trajectory_consistency_top5_jaccard_std": self.consistency_jaccard.std,
            "effective_features_80_median": histogram_quantile(
                self.effective_histogram, 0.50
            ),
            "effective_features_80_iqr_low": histogram_quantile(
                self.effective_histogram, 0.25
            ),
            "effective_features_80_iqr_high": histogram_quantile(
                self.effective_histogram, 0.75
            ),
            "temporal_attribution_mass_mean": self.temporal_mass.mean,
            "temporal_attribution_mass_std": self.temporal_mass.std,
        }


@dataclass
class ModelAdapter:
    name: str
    input_kind: str
    explain: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
    explanation_method: str
    output_form: str


def transformed_input(
    sequences: np.ndarray,
    input_kind: str,
    scaling_mean: np.ndarray,
    scaling_sd: np.ndarray,
) -> np.ndarray:
    if input_kind == "tree":
        normalized = (sequences - scaling_mean) / scaling_sd
        return summarize_for_trees(normalized)
    if input_kind == "ebm":
        return sequences[:, -1, : len(FEATURE_ORDER)]
    if input_kind == "fnn":
        return sequences
    raise ValueError(f"Unknown explanation input kind: {input_kind}")


def evaluate_model(
    *,
    data: FormalWindowData,
    adapter: ModelAdapter,
    scaling_mean: np.ndarray,
    scaling_sd: np.ndarray,
    raw_sd: np.ndarray,
    batch_size: int,
    perturbation_repeats: int,
    perturbation_fraction: float,
    progress_every: int,
) -> ExplanationAccumulator:
    accumulator = ExplanationAccumulator(len(FEATURE_ORDER))
    rng = np.random.default_rng(SEED)
    next_progress = progress_every
    started = perf_counter()
    for sequences, stays, hours in iter_window_batches(data, batch_size):
        model_input = transformed_input(
            sequences, adapter.input_kind, scaling_mean, scaling_sd
        )
        values, temporal = adapter.explain(model_input)
        accumulator.update_baseline(values, temporal, stays, hours)
        for _ in range(perturbation_repeats):
            perturbed_sequences = perturb_sequences(
                sequences, raw_sd, perturbation_fraction, rng
            )
            perturbed_input = transformed_input(
                perturbed_sequences, adapter.input_kind, scaling_mean, scaling_sd
            )
            perturbed_values, _ = adapter.explain(perturbed_input)
            accumulator.update_stability(values, perturbed_values)

        if accumulator.windows >= next_progress:
            elapsed = perf_counter() - started
            rate = accumulator.windows / max(elapsed, 1e-9)
            print(
                f"[{data.database}] {adapter.name}: {accumulator.windows:,}/"
                f"{data.expected_windows:,} windows ({rate:,.0f} windows/s)",
                flush=True,
            )
            next_progress += progress_every
    if accumulator.windows != data.expected_windows:
        raise ValueError(
            f"{data.database} {adapter.name} processed {accumulator.windows:,} of "
            f"{data.expected_windows:,} formal windows."
        )
    return accumulator


def performance_lookup() -> dict[str, tuple[float, float]]:
    metrics = pd.read_csv(
        ROOT / "outputs/explicit_kg_tfnn_paired_comparison_6h/evaluation/advanced_metrics.csv"
    ).set_index("model")
    return {
        "KG-TFNN": (
            metrics.loc["explicit_kg_tfnn", "auroc"],
            metrics.loc["explicit_kg_tfnn", "auprc"],
        ),
        "LightGBM + TreeSHAP": (
            metrics.loc["lightgbm_matched", "auroc"],
            metrics.loc["lightgbm_matched", "auprc"],
        ),
        "XGBoost + TreeSHAP": (
            metrics.loc["xgboost_matched", "auroc"],
            metrics.loc["xgboost_matched", "auprc"],
        ),
        "EBM (current state)": (
            metrics.loc["ebm", "auroc"],
            metrics.loc["ebm", "auprc"],
        ),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=["number"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("|", "/") for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def write_unified_explanation_complexity(
    result: pd.DataFrame,
    output: Path,
) -> pd.DataFrame:
    """Report one attribution-complexity definition shared by all explanation methods."""

    expected_models = {
        "KG-TFNN",
        "LightGBM + TreeSHAP",
        "XGBoost + TreeSHAP",
        "EBM (current state)",
    }
    required = {
        "model",
        "mimic_windows",
        "eicu_windows",
        "effective_features_80_median",
        "effective_features_80_iqr_low",
        "effective_features_80_iqr_high",
        "explanation_method",
        "output_form",
    }
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"Explanation results lack complexity fields: {sorted(missing)}")
    if set(result["model"]) != expected_models:
        raise ValueError("Unified complexity requires the locked four-model explanation set.")

    audit_path = output / "formal_cohort_audit.json"
    if not audit_path.exists():
        raise FileNotFoundError(audit_path)
    cohort_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    for database in ("MIMIC-IV", "eICU-CRD"):
        record = cohort_audit[database]
        if not record.get("all_prediction_windows_reconstructed", False):
            raise ValueError(f"{database} full-data explanation audit did not pass.")
        if record["processed_windows"] != record["expected_windows"]:
            raise ValueError(f"{database} explanation analysis was not full cohort.")

    complexity = result[
        [
            "model",
            "explanation_method",
            "output_form",
            "mimic_windows",
            "eicu_windows",
            "effective_features_80_median",
            "effective_features_80_iqr_low",
            "effective_features_80_iqr_high",
        ]
    ].copy()
    complexity.insert(1, "common_explanation_unit", "harmonized clinical variable")
    complexity.insert(2, "available_variables", len(FEATURE_ORDER))
    complexity.insert(3, "attribution_mass_threshold", 0.80)
    complexity["normalized_effective_features_80_median"] = (
        complexity["effective_features_80_median"] / len(FEATURE_ORDER)
    )
    complexity["lower_means_more_concentrated"] = True
    complexity["architecture_matched"] = complexity["model"] != "EBM (current state)"

    rule_path = ROOT / "outputs/rule_evaluation_6h/top_k_rule_complexity.csv"
    rule_specific_mean = math.nan
    if rule_path.exists():
        rules = pd.read_csv(rule_path)
        top_rules = rules[rules["rank"] <= 10]
        if not top_rules.empty:
            rule_specific_mean = float(top_rules["antecedent_count"].mean())
    complexity["kg_tfnn_top10_mean_rule_antecedents_model_specific"] = np.where(
        complexity["model"] == "KG-TFNN", rule_specific_mean, np.nan
    )
    complexity.to_csv(output / "unified_explanation_complexity.csv", index=False)

    plot = complexity.sort_values("effective_features_80_median", ascending=True)
    center = plot["effective_features_80_median"].to_numpy(dtype=float)
    lower = center - plot["effective_features_80_iqr_low"].to_numpy(dtype=float)
    upper = plot["effective_features_80_iqr_high"].to_numpy(dtype=float) - center
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.errorbar(
        center,
        np.arange(len(plot)),
        xerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=4,
        color="#0072B2",
    )
    axis.set_yticks(np.arange(len(plot)), plot["model"])
    axis.set_xlim(0, len(FEATURE_ORDER))
    axis.set(
        xlabel="Clinical variables required for 80% absolute attribution mass",
        title="Unified Local Explanation Complexity",
    )
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            output / f"figures/unified_explanation_complexity.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)

    report_columns = [
        "model",
        "effective_features_80_median",
        "effective_features_80_iqr_low",
        "effective_features_80_iqr_high",
        "normalized_effective_features_80_median",
        "mimic_windows",
        "eicu_windows",
    ]
    report = [
        "# Unified Cross-Model Explanation Complexity",
        "",
        "## Common Definition",
        "",
        (
            "For every eligible MIMIC-IV test window, each method's signed local "
            "explanation was aggregated to the same 13 harmonized clinical variables. "
            "Absolute attribution was normalized to sum to one within each window. "
            "Complexity is the minimum number of variables required to explain 80% of "
            "that attribution mass; lower values indicate a more concentrated explanation."
        ),
        "",
        (
            "Tree summary features were mapped back to their clinical variable, and EBM "
            "interaction contributions were divided equally among participating variables. "
            "The metric therefore compares explanation concentration, not native rule syntax."
        ),
        "",
        markdown_table(complexity[report_columns]),
        "",
        "## Model-Specific Structural Complexity",
        "",
        (
            f"KG-TFNN Top-10 rules contained a mean of {rule_specific_mean:.2f} antecedents "
            "across the five-seed rule analysis. This rule-specific result is reported "
            "separately and is not treated as directly equivalent to SHAP or EBM terms."
        ),
        "",
        "## Interpretation Boundary",
        "",
        (
            "This structural analysis does not constitute clinician validation. The EBM "
            "uses current-state inputs and remains a non-architecture-matched comparator."
        ),
        "",
    ]
    (output / "unified_explanation_complexity_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    complexity_audit = {
        "status": "passed",
        "source_results": str(output / "explanation_quality_comparison.csv"),
        "source_results_sha256": sha256_file(
            output / "explanation_quality_comparison.csv"
        ),
        "source_cohort_audit": str(audit_path),
        "source_cohort_audit_sha256": sha256_file(audit_path),
        "formal_full_data": True,
        "mimic_windows": int(cohort_audit["MIMIC-IV"]["processed_windows"]),
        "eicu_windows": int(cohort_audit["eICU-CRD"]["processed_windows"]),
        "common_explanation_unit": "13 harmonized clinical variables",
        "mass_threshold": 0.80,
        "lower_means_more_concentrated": True,
        "rule_antecedent_complexity_is_model_specific": True,
        "clinician_validated": False,
    }
    (output / "unified_explanation_complexity_audit.json").write_text(
        json.dumps(complexity_audit, indent=2), encoding="utf-8"
    )
    return complexity


def choose_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.fnn_batch_size <= 0:
        raise ValueError("Batch sizes must be positive.")
    if args.perturbation_repeats <= 0:
        raise ValueError("Formal stability analysis requires at least one perturbation repeat.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)

    if args.complexity_only:
        result_path = output / "explanation_quality_comparison.csv"
        if not result_path.exists():
            raise FileNotFoundError(
                "A completed full-data explanation_quality_comparison.csv is required."
            )
        complexity = write_unified_explanation_complexity(
            pd.read_csv(result_path), output
        )
        print(complexity.to_string(index=False), flush=True)
        return

    scaling = json.loads(SCALING_JSON.read_text(encoding="utf-8"))["scaling"]
    scaling_mean = np.asarray(scaling["mean"], dtype=np.float32)[None, None, :]
    scaling_sd = np.asarray(scaling["std"], dtype=np.float32)
    scaling_sd = np.maximum(scaling_sd, 1e-6)
    scaling_sd_3d = scaling_sd[None, None, :]
    summary_names = matched_tree_feature_names(explicit_temporal_input_order(FEATURE_ORDER))
    mapping, temporal_mask = aggregate_feature_names(summary_names)

    device = choose_device(args.device)
    fnn = load_fnn(device)
    adapters = [
        ModelAdapter(
            "LightGBM + TreeSHAP",
            "tree",
            tree_shap_function(
                joblib.load(BASELINE_ROOT / "lightgbm/model.joblib"),
                mapping,
                temporal_mask,
            ),
            "TreeSHAP on 24-hour matched summaries",
            "Signed feature attributions",
        ),
        ModelAdapter(
            "XGBoost + TreeSHAP",
            "tree",
            tree_shap_function(
                joblib.load(BASELINE_ROOT / "xgboost/model.joblib"),
                mapping,
                temporal_mask,
            ),
            "TreeSHAP on 24-hour matched summaries",
            "Signed feature attributions",
        ),
        ModelAdapter(
            "EBM (current state)",
            "ebm",
            ebm_function(joblib.load(EBM_PATH)),
            "Additive term contributions; current state only",
            "Additive feature/interaction terms",
        ),
        ModelAdapter(
            "KG-TFNN",
            "fnn",
            lambda x: fnn_explain(fnn, x, device, args.fnn_batch_size),
            "Model-intrinsic fuzzy contributions",
            "Temporal IF-THEN fuzzy rules",
        ),
    ]

    dataset_specs = [
        ("MIMIC-IV", MIMIC_HOURLY, MIMIC_PREDICTIONS, args.perturbation_repeats),
        ("eICU-CRD", EICU_HOURLY, EICU_PREDICTIONS, 0),
    ]
    database_results: dict[str, dict[str, ExplanationAccumulator]] = {}
    cohort_audits: dict[str, dict[str, object]] = {}
    for database, hourly_path, prediction_path, repeats in dataset_specs:
        print(f"Loading all formal {database} windows...", flush=True)
        data = load_formal_window_data(
            database=database,
            hourly_path=hourly_path,
            prediction_path=prediction_path,
            target_col=TARGET,
            time_col=TIME,
            split_col=SPLIT,
            seq_length=SEQ_LENGTH,
        )
        cohort_audits[database] = data.audit_record()
        print(json.dumps(cohort_audits[database], indent=2), flush=True)
        database_results[database] = {}
        for adapter in adapters:
            database_results[database][adapter.name] = evaluate_model(
                data=data,
                adapter=adapter,
                scaling_mean=scaling_mean,
                scaling_sd=scaling_sd_3d,
                raw_sd=scaling_sd[: len(FEATURE_ORDER)],
                batch_size=args.batch_size,
                perturbation_repeats=repeats,
                perturbation_fraction=args.perturbation_sd_fraction,
                progress_every=args.progress_every,
            )
        del data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    performance = performance_lookup()
    adapters_by_name = {adapter.name: adapter for adapter in adapters}
    rows: list[dict[str, object]] = []
    global_rows: list[dict[str, object]] = []
    for model_name, adapter in adapters_by_name.items():
        mimic = database_results["MIMIC-IV"][model_name]
        eicu = database_results["eICU-CRD"][model_name]
        mimic_importance = mimic.mean_absolute()
        eicu_importance = eicu.mean_absolute()
        correlation = float(spearmanr(mimic_importance, eicu_importance).statistic)
        mimic_top = set(np.argsort(-mimic_importance)[:TOP_K])
        eicu_top = set(np.argsort(-eicu_importance)[:TOP_K])
        transport_jaccard = len(mimic_top & eicu_top) / len(mimic_top | eicu_top)
        auroc, auprc = performance[model_name]
        rows.append(
            {
                "model": model_name,
                "auroc": auroc,
                "auprc": auprc,
                "explanation_method": adapter.explanation_method,
                "output_form": adapter.output_form,
                "mimic_windows": mimic.windows,
                "eicu_windows": eicu.windows,
                **mimic.summary(),
                "eicu_trajectory_consistency_pairs": eicu.consistency_cosine.count,
                "eicu_trajectory_consistency_cosine_mean": eicu.consistency_cosine.mean,
                "eicu_trajectory_consistency_top5_jaccard_mean": eicu.consistency_jaccard.mean,
                "cross_dataset_global_spearman": correlation,
                "cross_dataset_top5_jaccard": transport_jaccard,
                "human_readable_temporal_rule_output": model_name == "KG-TFNN",
                "clinician_validated": False,
            }
        )
        for feature, internal, external in zip(
            FEATURE_ORDER, mimic_importance, eicu_importance
        ):
            global_rows.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "mimic_mean_abs": internal,
                    "eicu_mean_abs": external,
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(output / "explanation_quality_comparison.csv", index=False)
    pd.DataFrame(global_rows).to_csv(
        output / "cross_dataset_global_explanations.csv", index=False
    )
    (output / "formal_cohort_audit.json").write_text(
        json.dumps(cohort_audits, indent=2), encoding="utf-8"
    )
    write_unified_explanation_complexity(result, output)

    plot = result.set_index("model")
    metrics = [
        "stability_cosine_mean",
        "trajectory_consistency_cosine_mean",
        "cross_dataset_global_spearman",
        "cross_dataset_top5_jaccard",
    ]
    labels = [
        "Perturbation\nstability",
        "Within-stay\ncontinuity",
        "Cross-dataset\nrank correlation",
        "Cross-dataset\nTop-5 Jaccard",
    ]
    figure, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for axis, metric, label in zip(axes, metrics, labels):
        values = plot[metric]
        axis.barh(np.arange(len(values)), values, color=colors)
        axis.set_title(label)
        axis.set_xlim(-0.05 if "spearman" in metric else 0, 1.02)
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(np.arange(len(plot)), plot.index)
    figure.suptitle("Full-data explanation stability and consistency")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            output / f"figures/explanation_quality_comparison.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)

    report = [
        "# Full-Data Explainability Comparison",
        "",
        (
            "Formal analysis on every eligible frozen prediction window: "
            f"{cohort_audits['MIMIC-IV']['processed_windows']:,} MIMIC-IV windows and "
            f"{cohort_audits['eICU-CRD']['processed_windows']:,} eICU-CRD windows. "
            "Perturbation stability uses three independent 1% training-SD perturbations "
            "on all MIMIC windows. Explanation consistency compares consecutive eligible "
            "prediction windows within the same ICU stay. Cross-dataset stability compares "
            "full-cohort mean absolute feature rankings without eICU fitting."
        ),
        "",
        markdown_table(result),
        "",
        (
            "Human-readable output form is a structural property. No method in this analysis "
            "has undergone a blinded clinician reader study. The EBM remains a current-state "
            "comparator rather than a 24-hour feature-matched architecture."
        ),
    ]
    (output / "explanation_quality_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    config = {
        "formal_full_data": True,
        "analysis_scope": "all_eligible_prediction_windows",
        "target": TARGET,
        "observation_window_hours": SEQ_LENGTH,
        "mimic_windows": cohort_audits["MIMIC-IV"]["processed_windows"],
        "eicu_windows": cohort_audits["eICU-CRD"]["processed_windows"],
        "perturbation_repeats": args.perturbation_repeats,
        "perturbation_sd_fraction": args.perturbation_sd_fraction,
        "consistency_definition": "consecutive eligible windows within the same ICU stay",
        "top_k": TOP_K,
        "seed": SEED,
        "batch_size": args.batch_size,
        "fnn_batch_size": args.fnn_batch_size,
        "device": str(device),
        "ebm_comparator": "current-state 13-feature EBM; not 24-hour feature matched",
    }
    (output / "analysis_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    # 移除舊版 1,000-case cache，避免正式與探索性資料混在同一輸出目錄。
    for stale_name in (
        "mimic_explanation_sample.csv",
        "eicu_explanation_sample.csv",
        "mimic-iv_selected_hourly_rows.pkl",
        "eicu_selected_hourly_rows.pkl",
    ):
        stale_path = output / stale_name
        if stale_path.exists():
            stale_path.unlink()
    print(result.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
