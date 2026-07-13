"""Compare KG-TFNN intrinsic explanations with TreeSHAP and EBM explanations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
import torch
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

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
from train_fnn import prepare_explicit_temporal_arrays


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/posthoc_explainability_comparison_6h"
TARGET = "label_sofa_increase_ge2_6h"
TIME = "sofa_hour"
SPLIT = "subject_id"
SEQ_LENGTH = 24
N_PER_DATASET = 1_000
PERTURB_REPEATS = 3
TOP_K = 5
SEED = 42

MIMIC_PREDICTIONS = ROOT / "outputs/explicit_kg_tfnn_paired_comparison_6h/inputs/lightgbm_matched/test_predictions.csv.gz"
EICU_PREDICTIONS = ROOT / "outputs/eicu_external_validation/final_frozen_model_evaluation/eicu_external_predictions.csv.gz"
MIMIC_HOURLY = ROOT / "model_hourly_features_v3.csv"
EICU_HOURLY = ROOT / "outputs/eicu_external_validation/eicu_hourly_features.pkl"
BASELINE_ROOT = ROOT / "outputs/feature_matched_baselines_6h_equal_sample/sequence_matched_summary"
EBM_PATH = ROOT / "outputs/fair_comparison_6h_equal_sample/interpretable_6h/protocol/ebm/model.joblib"
EBM_CONFIG = ROOT / "outputs/fair_comparison_6h_equal_sample/interpretable_6h/experiment_config.json"
FNN_CHECKPOINT = ROOT / "outputs/fnn_ablation_6h_equal_sample/seed_42/full/best_model.pt"
SCALING_JSON = ROOT / "outputs/feature_matched_baselines_6h_equal_sample/sequence_matched/gru/sequence_scaling.json"


def sample_one_window_per_stay(path: Path, n: int, seed: int) -> pd.DataFrame:
    columns = [SPLIT, "stay_id", TIME, "y_true"]
    frame = pd.read_csv(path, usecols=columns)
    rng = np.random.default_rng(seed)
    frame = frame.assign(_random=rng.random(len(frame)))
    frame = frame.sort_values("_random").drop_duplicates("stay_id")
    selected = []
    for outcome in (0, 1):
        group = frame[frame.y_true.astype(int) == outcome]
        selected.append(group.sample(n=min(n // 2, len(group)), random_state=seed + outcome))
    result = pd.concat(selected, ignore_index=True).drop(columns="_random")
    if len(result) < n:
        remaining = frame.merge(result[["stay_id"]], on="stay_id", how="left", indicator=True)
        remaining = remaining[remaining._merge == "left_only"].drop(columns="_merge")
        result = pd.concat(
            [result, remaining.sample(n=min(n - len(result), len(remaining)), random_state=seed + 9)],
            ignore_index=True,
        )
    return result.sort_values(["stay_id", TIME]).reset_index(drop=True)


def selected_hourly_rows(keys: pd.DataFrame, database: str) -> pd.DataFrame:
    cache = OUTPUT / f"{database.lower()}_selected_hourly_rows.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    wanted = set(keys.stay_id.astype(np.int64))
    columns = ["stay_id", SPLIT, TIME, TARGET, *explicit_temporal_input_order(FEATURE_ORDER)]
    if database == "MIMIC-IV":
        chunks = []
        for chunk in pd.read_csv(MIMIC_HOURLY, usecols=columns, chunksize=250_000):
            keep = chunk.stay_id.isin(wanted)
            if keep.any():
                chunks.append(chunk.loc[keep].copy())
        frame = pd.concat(chunks, ignore_index=True)
    else:
        source = pd.read_pickle(EICU_HOURLY)
        missing = set(columns) - set(source.columns)
        if missing:
            raise ValueError(f"eICU hourly file is missing columns: {sorted(missing)}")
        frame = source.loc[source.stay_id.isin(wanted), columns].copy()
        del source
    frame.to_pickle(cache)
    return frame


def build_sequences(frame: pd.DataFrame, keys: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    features, labels, stay_ids, subject_ids, hours = prepare_explicit_temporal_arrays(
        frame, target_col=TARGET, time_col=TIME, split_col=SPLIT
    )
    index_by_key = {
        (int(stay), int(hour)): index for index, (stay, hour) in enumerate(zip(stay_ids, hours))
    }
    sequences = []
    metadata = []
    for row in keys.itertuples(index=False):
        index = index_by_key.get((int(row.stay_id), int(row.sofa_hour)))
        if index is None or index < SEQ_LENGTH - 1:
            continue
        start = index - SEQ_LENGTH + 1
        if np.any(stay_ids[start : index + 1] != row.stay_id):
            continue
        if not np.array_equal(hours[start : index + 1], np.arange(row.sofa_hour - 23, row.sofa_hour + 1)):
            continue
        sequences.append(features[start : index + 1])
        metadata.append(
            {
                SPLIT: str(subject_ids[index]), "stay_id": int(row.stay_id), TIME: int(row.sofa_hour),
                "y_true": int(labels[index]),
            }
        )
    if not sequences:
        raise RuntimeError("No complete sampled sequences were reconstructed.")
    return np.stack(sequences).astype(np.float32), pd.DataFrame(metadata)


def summarize_for_trees(sequences: np.ndarray) -> np.ndarray:
    raw = sequences[:, :, : len(FEATURE_ORDER)]
    missing = sequences[:, :, len(FEATURE_ORDER) : 2 * len(FEATURE_ORDER)]
    hours = np.arange(SEQ_LENGTH, dtype=np.float32)
    centered = hours - hours.mean()
    slope = np.einsum("btf,t->bf", raw, centered, optimize=True) / float(np.sum(centered**2))
    return np.concatenate(
        [
            sequences[:, -1, :], raw.mean(1), raw.min(1), raw.max(1), raw.std(1), slope,
            raw[:, -1] - raw[:, -2], raw[:, -1] - raw[:, 0], missing.mean(1),
        ],
        axis=1,
    ).astype(np.float32)


def aggregate_feature_names(names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    mapping = np.empty(len(names), dtype=np.int64)
    temporal = np.zeros(len(names), dtype=bool)
    for index, name in enumerate(names):
        prefix, value = name.split("::", 1)
        matches = [i for i, feature in enumerate(FEATURE_ORDER) if value == feature or value.startswith(feature + "_")]
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


def tree_shap_function(model: object, mapping: np.ndarray, temporal_mask: np.ndarray) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    explainer = shap.TreeExplainer(model)

    def explain(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = explainer.shap_values(x)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        aggregated = aggregate_columns(values, mapping)
        temporal_mass = np.abs(values[:, temporal_mask]).sum(1) / (np.abs(values).sum(1) + 1e-12)
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


def fnn_explain(model: TemporalAttentionFNN, sequences: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    temporal_rows = []
    rule_antecedents = [
        [FEATURE_ORDER.index(feature) for feature, _ in rule["antecedents"]]
        for rule in model.static_fnn.rule_configs
    ]
    weights = model.static_fnn.cross_rule_weights.detach().cpu().numpy()
    for start in range(0, len(sequences), 256):
        x = torch.from_numpy(sequences[start : start + 256]).to(device)
        with torch.no_grad():
            output = model(x)
        attention = output.attention_weights.detach().cpu().numpy()
        feature_risk = output.feature_risks.detach().cpu().numpy()
        rule_activation = output.rule_activations.detach().cpu().numpy()
        base = np.einsum("bt,btf->bf", attention, feature_risk)
        cross = np.zeros_like(base)
        for rule_index, antecedents in enumerate(rule_antecedents):
            contribution = (
                np.einsum("bt,bt->b", attention, rule_activation[:, :, rule_index])
                * weights[rule_index] * model.static_fnn.rule_score_scale / len(antecedents)
            )
            for feature_index in antecedents:
                cross[:, feature_index] += contribution
        explicit = output.explicit_temporal_contributions.detach().cpu().numpy()
        explicit = explicit.sum(2) * model.explicit_temporal_scale / math.sqrt(len(FEATURE_ORDER))
        total = base + cross + explicit
        denominator = np.abs(base).sum(1) + np.abs(cross).sum(1) + np.abs(explicit).sum(1) + 1e-12
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
    return np.asarray([
        len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(left_top, right_top)
    ])


def effective_features(values: np.ndarray, mass: float = 0.80) -> np.ndarray:
    ordered = np.sort(np.abs(values), axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) / (ordered.sum(1, keepdims=True) + 1e-12)
    return (cumulative < mass).sum(1) + 1


def common_neighbor_indices(sequences: np.ndarray) -> np.ndarray:
    raw = sequences[:, :, : len(FEATURE_ORDER)]
    representation = np.concatenate([raw[:, -1], raw.mean(1), raw[:, -1] - raw[:, 0]], axis=1)
    representation = (representation - representation.mean(0)) / (representation.std(0) + 1e-6)
    return NearestNeighbors(n_neighbors=2).fit(representation).kneighbors(return_distance=False)[:, 1]


def global_transport_metrics(mimic: np.ndarray, eicu: np.ndarray) -> tuple[float, float]:
    mimic_rank = np.abs(mimic).mean(0)
    eicu_rank = np.abs(eicu).mean(0)
    correlation = float(spearmanr(mimic_rank, eicu_rank).statistic)
    mimic_top = set(np.argsort(-mimic_rank)[:TOP_K])
    eicu_top = set(np.argsort(-eicu_rank)[:TOP_K])
    return correlation, len(mimic_top & eicu_top) / len(mimic_top | eicu_top)


def perturb_sequences(sequences: np.ndarray, raw_sd: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    perturbed = sequences.copy()
    noise = rng.normal(0.0, raw_sd * 0.01, size=perturbed[:, :, : len(FEATURE_ORDER)].shape)
    perturbed[:, :, : len(FEATURE_ORDER)] += noise.astype(np.float32)
    return perturbed


def performance_lookup() -> dict[str, tuple[float, float]]:
    metrics = pd.read_csv(ROOT / "outputs/explicit_kg_tfnn_paired_comparison_6h/evaluation/advanced_metrics.csv").set_index("model")
    return {
        "KG-TFNN": (metrics.loc["explicit_kg_tfnn", "auroc"], metrics.loc["explicit_kg_tfnn", "auprc"]),
        "LightGBM + TreeSHAP": (metrics.loc["lightgbm_matched", "auroc"], metrics.loc["lightgbm_matched", "auprc"]),
        "XGBoost + TreeSHAP": (metrics.loc["xgboost_matched", "auroc"], metrics.loc["xgboost_matched", "auprc"]),
        "EBM (current state)": (metrics.loc["ebm", "auroc"], metrics.loc["ebm", "auprc"]),
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


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "figures").mkdir(exist_ok=True)
    mimic_keys = sample_one_window_per_stay(MIMIC_PREDICTIONS, N_PER_DATASET, SEED)
    eicu_keys = sample_one_window_per_stay(EICU_PREDICTIONS, N_PER_DATASET, SEED + 1)
    mimic_sequences, mimic_meta = build_sequences(selected_hourly_rows(mimic_keys, "MIMIC-IV"), mimic_keys)
    eicu_sequences, eicu_meta = build_sequences(selected_hourly_rows(eicu_keys, "eICU"), eicu_keys)
    mimic_meta.to_csv(OUTPUT / "mimic_explanation_sample.csv", index=False)
    eicu_meta.to_csv(OUTPUT / "eicu_explanation_sample.csv", index=False)

    scaling_data = json.loads(SCALING_JSON.read_text(encoding="utf-8"))["scaling"]
    mean = np.asarray(scaling_data["mean"], dtype=np.float32)
    sd = np.asarray(scaling_data["std"], dtype=np.float32)
    summary_names = matched_tree_feature_names(explicit_temporal_input_order(FEATURE_ORDER))
    mapping, temporal_mask = aggregate_feature_names(summary_names)

    def tree_inputs(sequence: np.ndarray) -> np.ndarray:
        return summarize_for_trees((sequence - mean) / sd)

    mimic_tree = tree_inputs(mimic_sequences)
    eicu_tree = tree_inputs(eicu_sequences)
    ebm = joblib.load(EBM_PATH)
    ebm_fill = json.loads(EBM_CONFIG.read_text(encoding="utf-8"))["fill_values"]
    mimic_ebm = mimic_sequences[:, -1, : len(FEATURE_ORDER)].copy()
    eicu_ebm = eicu_sequences[:, -1, : len(FEATURE_ORDER)].copy()
    for index, feature in enumerate(FEATURE_ORDER):
        mimic_ebm[:, index] = np.nan_to_num(mimic_ebm[:, index], nan=ebm_fill[feature])
        eicu_ebm[:, index] = np.nan_to_num(eicu_ebm[:, index], nan=ebm_fill[feature])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fnn = load_fnn(device)
    explainers: dict[str, tuple[Callable, np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray], str, str]] = {
        "LightGBM + TreeSHAP": (
            tree_shap_function(joblib.load(BASELINE_ROOT / "lightgbm/model.joblib"), mapping, temporal_mask),
            mimic_tree, eicu_tree, tree_inputs, "TreeSHAP on 24-h matched summaries", "Signed feature attributions",
        ),
        "XGBoost + TreeSHAP": (
            tree_shap_function(joblib.load(BASELINE_ROOT / "xgboost/model.joblib"), mapping, temporal_mask),
            mimic_tree, eicu_tree, tree_inputs, "TreeSHAP on 24-h matched summaries", "Signed feature attributions",
        ),
        "EBM (current state)": (
            ebm_function(ebm), mimic_ebm, eicu_ebm,
            lambda seq: seq[:, -1, : len(FEATURE_ORDER)], "Additive term contributions; current state only", "Additive feature/interaction terms",
        ),
        "KG-TFNN": (
            lambda seq: fnn_explain(fnn, seq, device), mimic_sequences, eicu_sequences,
            lambda seq: seq, "Model-intrinsic fuzzy contributions", "Temporal IF-THEN rules",
        ),
    }

    neighbors = common_neighbor_indices(mimic_sequences)
    rng = np.random.default_rng(SEED)
    performance = performance_lookup()
    result_rows = []
    global_rows = []
    for model_name, (explain, mimic_input, eicu_input, transform, method, output_form) in explainers.items():
        mimic_values, mimic_temporal = explain(mimic_input)
        eicu_values, _ = explain(eicu_input)
        mimic_norm = normalize_explanations(mimic_values)
        consistency_cosine = cosine_rows(mimic_norm, mimic_norm[neighbors])
        consistency_jaccard = top_k_jaccard_rows(mimic_norm, mimic_norm[neighbors], TOP_K)

        stability_cosines = []
        stability_jaccards = []
        for _ in range(PERTURB_REPEATS):
            perturbed_seq = perturb_sequences(mimic_sequences, sd[: len(FEATURE_ORDER)], rng)
            perturbed_values, _ = explain(transform(perturbed_seq))
            perturbed_norm = normalize_explanations(perturbed_values)
            stability_cosines.extend(cosine_rows(mimic_norm, perturbed_norm))
            stability_jaccards.extend(top_k_jaccard_rows(mimic_norm, perturbed_norm, TOP_K))

        transport_spearman, transport_jaccard = global_transport_metrics(mimic_values, eicu_values)
        effective = effective_features(mimic_values)
        auroc, auprc = performance[model_name]
        result_rows.append(
            {
                "model": model_name, "auroc": auroc, "auprc": auprc,
                "explanation_method": method, "output_form": output_form,
                "stability_cosine_mean": np.mean(stability_cosines),
                "stability_top5_jaccard_mean": np.mean(stability_jaccards),
                "neighbor_consistency_cosine_mean": np.mean(consistency_cosine),
                "neighbor_consistency_top5_jaccard_mean": np.mean(consistency_jaccard),
                "effective_features_80_median": np.median(effective),
                "effective_features_80_iqr_low": np.quantile(effective, 0.25),
                "effective_features_80_iqr_high": np.quantile(effective, 0.75),
                "temporal_attribution_mass_mean": np.mean(mimic_temporal),
                "cross_dataset_global_spearman": transport_spearman,
                "cross_dataset_top5_jaccard": transport_jaccard,
                "human_readable_temporal_rule_output": model_name == "KG-TFNN",
                "clinician_validated": False,
            }
        )
        importance = np.abs(mimic_values).mean(0)
        external_importance = np.abs(eicu_values).mean(0)
        for feature, internal, external in zip(FEATURE_ORDER, importance, external_importance):
            global_rows.append(
                {"model": model_name, "feature": feature, "mimic_mean_abs": internal, "eicu_mean_abs": external}
            )

    result = pd.DataFrame(result_rows)
    result.to_csv(OUTPUT / "explanation_quality_comparison.csv", index=False)
    pd.DataFrame(global_rows).to_csv(OUTPUT / "cross_dataset_global_explanations.csv", index=False)

    plot = result.set_index("model")
    metrics = [
        "stability_cosine_mean", "neighbor_consistency_cosine_mean",
        "cross_dataset_global_spearman", "cross_dataset_top5_jaccard",
    ]
    labels = ["Perturbation\nstability", "Neighbor\nconsistency", "Cross-dataset\nrank correlation", "Cross-dataset\nTop-5 Jaccard"]
    figure, axes = plt.subplots(1, 4, figsize=(14, 4), sharey=True)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for axis, metric, label in zip(axes, metrics, labels):
        values = plot[metric]
        axis.barh(np.arange(len(values)), values, color=colors)
        axis.set_title(label)
        axis.set_xlim(-0.05 if "spearman" in metric else 0, 1.02)
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(np.arange(len(plot)), plot.index)
    figure.suptitle("Explanation stability and consistency on frozen models")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(OUTPUT / f"figures/explanation_quality_comparison.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(figure)

    report = [
        "# Post-hoc Explainability Comparison", "",
        "Exploratory analysis only: all explanation-quality metrics use 1,000 prespecified one-window-per-stay samples per database and are not formal full-cohort results. MIMIC perturbation stability uses three 1% training-SD perturbations of raw physiological channels. Neighbor consistency compares each MIMIC case with its nearest trajectory neighbor. Cross-dataset stability compares mean absolute feature-level explanation rankings without eICU fitting.", "",
        markdown_table(result), "",
        "Structural outputs are reported separately from clinician understandability. No explanation method in this analysis has been validated by a clinician reader study.",
    ]
    (OUTPUT / "explanation_quality_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUTPUT / "analysis_config.json").write_text(
        json.dumps(
            {"formal_full_data": False, "analysis_scope": "exploratory_sample", "sample_per_database": N_PER_DATASET, "one_window_per_stay": True, "perturbation_repeats": PERTURB_REPEATS,
             "perturbation_sd_fraction": 0.01, "top_k": TOP_K, "seed": SEED,
             "ebm_comparator": "current-state 13-feature EBM; not 24-h feature matched"},
            indent=2,
        ), encoding="utf-8",
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
