"""Frozen eICU validation for the equal-sample MIMIC baseline models.

The predictive models are loaded from their completed MIMIC experiments. Model
calibration and operating thresholds are estimated from MIMIC validation
predictions only, then transported to every eligible eICU window without model
refitting, checkpoint selection, recalibration, or threshold selection on eICU.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
    module="sklearn",
)

from advanced_model_evaluation import (
    apply_platt_calibration,
    calibration_intercept_slope,
    operating_metrics,
    percentile_ci,
    threshold_at_specificity,
)
from anfis_model import FEATURE_ORDER, explicit_temporal_input_order
from blackbox_baselines import RecurrentRiskModel
from eicu_external_validation import load_frozen_model, optimized_cluster_bootstrap
from full_data_window_utils import (
    FormalWindowData,
    iter_window_batches,
    load_formal_window_data,
    sha256_file,
)
from model_evaluation_report import binary_metrics, calibration_bins


ROOT = Path(__file__).resolve().parent
TARGET = "label_sofa_increase_ge2_6h"
TIME = "sofa_hour"
SPLIT = "subject_id"
SEQ_LENGTH = 24
KEY_COLUMNS = [SPLIT, "stay_id", TIME]

BASELINE_ROOT = ROOT / "outputs/feature_matched_baselines_6h_equal_sample"
TREE_ROOT = BASELINE_ROOT / "sequence_matched_summary"
GRU_ROOT = BASELINE_ROOT / "sequence_matched/gru"
FNN_ROOT = ROOT / "outputs/fnn_ablation_6h_equal_sample/seed_42/full"
EBM_ROOT = ROOT / "outputs/fair_comparison_6h_equal_sample/interpretable_6h/protocol/ebm"
EICU_HOURLY = ROOT / "outputs/eicu_external_validation/eicu_hourly_features.pkl"
MIMIC_HOURLY = ROOT / "model_hourly_features_v3.csv"
EICU_REFERENCE = (
    ROOT
    / "outputs/eicu_external_validation/final_frozen_model_evaluation/eicu_external_predictions.csv.gz"
)
FORMAL_KG_METRICS = (
    ROOT / "outputs/eicu_external_validation/final_frozen_model_evaluation/external_metrics.json"
)


@dataclass
class FrozenModelSpec:
    slug: str
    display_name: str
    kind: str
    artifact_path: Path
    validation_path: Path
    model: Any | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply frozen MIMIC baseline models to the complete harmonized eICU cohort."
    )
    parser.add_argument("--hourly-pickle", default=str(EICU_HOURLY))
    parser.add_argument("--reference-predictions", default=str(EICU_REFERENCE))
    parser.add_argument(
        "--output-dir", default="outputs/eicu_frozen_baseline_validation_6h"
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--bootstrap-reps", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--specificities", default="0.90,0.95")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--progress-every", type=int, default=250_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force-predictions", action="store_true")
    return parser.parse_args()


def choose_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_specs() -> list[FrozenModelSpec]:
    return [
        FrozenModelSpec(
            slug="kg_tfnn_equal_sample",
            display_name="KG-TFNN (equal-sample)",
            kind="fnn",
            artifact_path=FNN_ROOT / "best_model.pt",
            validation_path=FNN_ROOT / "validation_predictions.csv.gz",
        ),
        FrozenModelSpec(
            slug="lightgbm_matched",
            display_name="LightGBM (feature-matched)",
            kind="tree",
            artifact_path=TREE_ROOT / "lightgbm/model.joblib",
            validation_path=TREE_ROOT / "lightgbm/val_predictions.csv.gz",
        ),
        FrozenModelSpec(
            slug="xgboost_matched",
            display_name="XGBoost (feature-matched)",
            kind="tree",
            artifact_path=TREE_ROOT / "xgboost/model.joblib",
            validation_path=TREE_ROOT / "xgboost/val_predictions.csv.gz",
        ),
        FrozenModelSpec(
            slug="gru_matched",
            display_name="GRU (feature-matched)",
            kind="gru",
            artifact_path=GRU_ROOT / "best_model.pt",
            validation_path=GRU_ROOT / "val_predictions.csv.gz",
        ),
        FrozenModelSpec(
            slug="ebm_current_state",
            display_name="EBM (current state)",
            kind="ebm",
            artifact_path=EBM_ROOT / "model.joblib",
            validation_path=EBM_ROOT / "val_predictions.csv.gz",
        ),
    ]


def load_scaling() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = GRU_ROOT / "sequence_scaling.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = explicit_temporal_input_order(FEATURE_ORDER)
    if payload.get("feature_order") != expected:
        raise ValueError("Saved sequence scaling feature order differs from the locked model input.")
    mean = np.asarray(payload["scaling"]["mean"], dtype=np.float32)
    std = np.maximum(
        np.asarray(payload["scaling"]["std"], dtype=np.float32), 1e-6
    )
    if len(mean) != len(expected) or len(std) != len(expected):
        raise ValueError("Sequence scaling width does not match the 39-feature input.")
    return mean[None, None, :], std[None, None, :], payload


def load_models(specs: list[FrozenModelSpec], device: torch.device) -> None:
    for spec in specs:
        if not spec.artifact_path.exists():
            raise FileNotFoundError(spec.artifact_path)
        if not spec.validation_path.exists():
            raise FileNotFoundError(spec.validation_path)
        if spec.kind in {"tree", "ebm"}:
            spec.model = joblib.load(spec.artifact_path)
        elif spec.kind == "gru":
            checkpoint = torch.load(
                spec.artifact_path, map_location=device, weights_only=False
            )
            spec.model = RecurrentRiskModel(
                input_dim=int(checkpoint["input_dim"]),
                hidden_dim=int(checkpoint["hidden_size"]),
                num_layers=int(checkpoint["num_layers"]),
                dropout=float(checkpoint["dropout"]),
                rnn_type="gru",
            ).to(device)
            spec.model.load_state_dict(checkpoint["model_state_dict"])
            spec.model.eval()
        elif spec.kind == "fnn":
            spec.model, _ = load_frozen_model(spec.artifact_path, device, SEQ_LENGTH)
        else:
            raise ValueError(f"Unknown model kind: {spec.kind}")


def summarize_for_trees(sequences: np.ndarray) -> np.ndarray:
    """Reproduce the outcome-agnostic 24-hour summary used during tree training."""

    raw = sequences[:, :, : len(FEATURE_ORDER)]
    missing = sequences[:, :, len(FEATURE_ORDER) : 2 * len(FEATURE_ORDER)]
    hours = np.arange(SEQ_LENGTH, dtype=np.float32)
    centered = hours - hours.mean()
    slope = np.einsum("btf,t->bf", raw, centered, optimize=True) / float(
        np.sum(centered**2)
    )
    return np.concatenate(
        [
            sequences[:, -1, :],
            raw.mean(axis=1),
            raw.min(axis=1),
            raw.max(axis=1),
            raw.std(axis=1),
            slope,
            raw[:, -1, :] - raw[:, -2, :],
            raw[:, -1, :] - raw[:, 0, :],
            missing.mean(axis=1),
        ],
        axis=1,
    ).astype(np.float32)


def positive_probability(model: Any, values: np.ndarray) -> np.ndarray:
    probability = np.asarray(model.predict_proba(values))
    if probability.ndim == 2:
        probability = probability[:, 1]
    return probability.reshape(-1).astype(np.float32)


def validation_transfer(
    specs: list[FrozenModelSpec], specificities: list[float]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    transfers: dict[str, dict[str, Any]] = {}
    audits: dict[str, Any] = {}
    reference_keys: pd.MultiIndex | None = None
    reference_labels: np.ndarray | None = None

    for spec in specs:
        frame = pd.read_csv(spec.validation_path)
        required = {*KEY_COLUMNS, "y_true"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{spec.slug} validation predictions miss {sorted(missing)}")
        probability_column = "y_prob_raw" if "y_prob_raw" in frame else "y_prob"
        if probability_column not in frame:
            raise ValueError(f"{spec.slug} has no raw validation probability column.")
        frame = frame.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)
        keys = pd.MultiIndex.from_frame(frame[KEY_COLUMNS])
        labels = frame["y_true"].to_numpy(dtype=np.int8)
        if keys.has_duplicates:
            raise ValueError(f"{spec.slug} validation predictions contain duplicate windows.")
        if reference_keys is None:
            reference_keys = keys
            reference_labels = labels
        elif not keys.equals(reference_keys) or not np.array_equal(labels, reference_labels):
            raise ValueError("Frozen models do not share the same MIMIC validation windows/outcomes.")

        raw = frame[probability_column].to_numpy(dtype=float)
        intercept, slope = calibration_intercept_slope(labels, raw)
        if slope <= 0:
            raise ValueError(f"{spec.slug} validation calibration slope is not positive.")
        calibrated = apply_platt_calibration(raw, intercept, slope)
        thresholds = {
            str(value): threshold_at_specificity(labels, calibrated, value)
            for value in specificities
        }
        transfers[spec.slug] = {
            "model": spec.display_name,
            "source": "MIMIC-IV validation only",
            "validation_windows": int(len(frame)),
            "validation_patients": int(frame[SPLIT].nunique()),
            "validation_prevalence": float(labels.mean()),
            "raw_probability_column": probability_column,
            "calibration": {
                "method": "Platt logistic calibration on raw-probability logit",
                "intercept": intercept,
                "slope": slope,
            },
            "specificity_thresholds": thresholds,
            "no_eicu_outcome_used": True,
        }
        audits[spec.slug] = {
            "validation_predictions": str(spec.validation_path),
            "validation_predictions_sha256": sha256_file(spec.validation_path),
            "validation_windows": int(len(frame)),
            "validation_positive": int(labels.sum()),
            "model_artifact": str(spec.artifact_path),
            "model_artifact_sha256": sha256_file(spec.artifact_path),
        }
    return transfers, audits


def infer_all_models(
    data: FormalWindowData,
    specs: list[FrozenModelSpec],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
    progress_every: int,
) -> dict[str, np.ndarray]:
    output = {
        spec.slug: np.empty(data.expected_windows, dtype=np.float32) for spec in specs
    }
    cursor = 0
    next_progress = progress_every
    started = perf_counter()

    for sequences, _, _ in iter_window_batches(data, batch_size):
        end = cursor + len(sequences)
        normalized = (sequences - mean) / std
        tree_summary: np.ndarray | None = None
        tensor_normalized: torch.Tensor | None = None
        tensor_raw: torch.Tensor | None = None

        for spec in specs:
            if spec.kind == "tree":
                if tree_summary is None:
                    tree_summary = summarize_for_trees(normalized)
                values = positive_probability(spec.model, tree_summary)
            elif spec.kind == "ebm":
                values = positive_probability(
                    spec.model, sequences[:, -1, : len(FEATURE_ORDER)]
                )
            elif spec.kind == "gru":
                if tensor_normalized is None:
                    tensor_normalized = torch.from_numpy(normalized).to(device)
                with torch.inference_mode():
                    values = (
                        torch.sigmoid(spec.model(tensor_normalized))
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
            elif spec.kind == "fnn":
                if tensor_raw is None:
                    tensor_raw = torch.from_numpy(sequences).to(device)
                with torch.inference_mode():
                    values = (
                        spec.model(tensor_raw)
                        .probabilities.detach()
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
            else:
                raise ValueError(spec.kind)
            if len(values) != len(sequences) or not np.isfinite(values).all():
                raise ValueError(f"Invalid external predictions from {spec.slug}.")
            output[spec.slug][cursor:end] = values

        cursor = end
        if cursor >= next_progress:
            rate = cursor / max(perf_counter() - started, 1e-9)
            print(
                f"Frozen inference: {cursor:,}/{data.expected_windows:,} windows "
                f"({rate:,.0f} windows/s)",
                flush=True,
            )
            next_progress += progress_every

    if cursor != data.expected_windows:
        raise ValueError(f"Predicted {cursor:,} of {data.expected_windows:,} windows.")
    return output


def verify_source_prediction_reproduction(
    *,
    specs: list[FrozenModelSpec],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Require the loaded artifacts to reproduce their saved MIMIC validation scores."""

    print("Reconstructing MIMIC validation windows for source prediction checks...", flush=True)
    data = load_formal_window_data(
        database="MIMIC-IV validation",
        hourly_path=MIMIC_HOURLY,
        prediction_path=specs[0].validation_path,
        target_col=TARGET,
        time_col=TIME,
        split_col=SPLIT,
        seq_length=SEQ_LENGTH,
    )
    reproduced = infer_all_models(
        data,
        specs,
        mean,
        std,
        device,
        batch_size,
        progress_every=max(data.expected_windows, 1),
    )
    target = data.target_indices
    expected_metadata = pd.DataFrame(
        {
            SPLIT: data.subject_ids[target],
            "stay_id": data.stay_ids[target],
            TIME: data.hours[target],
        }
    )
    expected_keys = pd.MultiIndex.from_frame(expected_metadata[KEY_COLUMNS])
    records: dict[str, Any] = {}
    tolerance = 1e-4
    for spec in specs:
        saved = pd.read_csv(spec.validation_path)
        probability_column = "y_prob_raw" if "y_prob_raw" in saved else "y_prob"
        saved = saved.set_index(KEY_COLUMNS).reindex(expected_keys)
        if saved[probability_column].isna().any():
            raise ValueError(f"{spec.slug} source reproduction could not align all windows.")
        expected = saved[probability_column].to_numpy(dtype=float)
        difference = np.abs(reproduced[spec.slug].astype(float) - expected)
        maximum = float(difference.max())
        mean_absolute = float(difference.mean())
        passed = maximum <= tolerance
        records[spec.slug] = {
            "windows": int(len(expected)),
            "maximum_absolute_probability_difference": maximum,
            "mean_absolute_probability_difference": mean_absolute,
            "tolerance": tolerance,
            "passed": passed,
        }
        if not passed:
            raise ValueError(
                f"{spec.slug} does not reproduce frozen MIMIC predictions: max diff={maximum:.6g}"
            )
    del data, reproduced
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def load_cached_predictions(
    specs: list[FrozenModelSpec], output_dir: Path, data: FormalWindowData
) -> dict[str, np.ndarray] | None:
    result: dict[str, np.ndarray] = {}
    target = data.target_indices
    expected_stays = data.stay_ids[target].astype(np.int64, copy=False)
    expected_hours = data.hours[target].astype(np.int64, copy=False)
    expected_labels = data.labels[target].astype(np.int8, copy=False)
    for spec in specs:
        path = output_dir / "predictions" / f"{spec.slug}.csv.gz"
        if not path.exists():
            return None
        frame = pd.read_csv(path, usecols=["stay_id", TIME, "y_true", "y_prob_raw"])
        if len(frame) != data.expected_windows:
            return None
        if not (
            np.array_equal(frame["stay_id"].to_numpy(dtype=np.int64), expected_stays)
            and np.array_equal(frame[TIME].to_numpy(dtype=np.int64), expected_hours)
            and np.array_equal(frame["y_true"].to_numpy(dtype=np.int8), expected_labels)
        ):
            return None
        result[spec.slug] = frame["y_prob_raw"].to_numpy(dtype=np.float32)
    return result


def save_prediction_frame(
    *,
    spec: FrozenModelSpec,
    data: FormalWindowData,
    raw: np.ndarray,
    calibrated: np.ndarray,
    output_dir: Path,
) -> Path:
    target = data.target_indices
    frame = pd.DataFrame(
        {
            SPLIT: data.subject_ids[target],
            "stay_id": data.stay_ids[target],
            TIME: data.hours[target],
            "y_true": data.labels[target].astype(np.int8),
            "y_prob_raw": raw,
            "y_prob": calibrated,
            "model": spec.slug,
            "evaluation_split": "eicu_external_test",
        }
    )
    path = output_dir / "predictions" / f"{spec.slug}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip")
    return path


def paired_p_value(differences: np.ndarray) -> float:
    non_positive = (np.sum(differences <= 0) + 1) / (len(differences) + 1)
    non_negative = (np.sum(differences >= 0) + 1) / (len(differences) + 1)
    return float(min(1.0, 2.0 * min(non_positive, non_negative)))


def load_cached_bootstrap(
    path: Path,
    *,
    model: str,
    reps: int,
) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    expected_replicates = np.arange(reps)
    required = {"model", "replicate", "auroc", "auprc", "brier"}
    if required - set(frame.columns):
        return None
    if len(frame) != reps or not np.array_equal(
        frame["replicate"].to_numpy(dtype=int), expected_replicates
    ):
        return None
    if set(frame["model"].astype(str)) != {model}:
        return None
    return frame


def markdown_table(frame: pd.DataFrame) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=["number"]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.4f}"
        )
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("|", "/") for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def save_figures(
    y: np.ndarray,
    probabilities: dict[str, np.ndarray],
    specs: list[FrozenModelSpec],
    calibration: pd.DataFrame,
    output_dir: Path,
) -> None:
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for spec, color in zip(specs, colors):
        probability = probabilities[spec.slug]
        fpr, tpr, _ = roc_curve(y, probability)
        precision, recall, _ = precision_recall_curve(y, probability)
        axes[0].plot(fpr, tpr, color=color, label=spec.display_name)
        axes[1].plot(recall, precision, color=color, label=spec.display_name)
    axes[0].plot([0, 1], [0, 1], "--", color="#777777")
    axes[1].axhline(y.mean(), linestyle="--", color="#777777")
    axes[0].set(xlabel="1 - Specificity", ylabel="Sensitivity", title="eICU ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="eICU precision-recall")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(figures / f"frozen_baseline_roc_pr.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], "--", color="#777777")
    calibrated = calibration[calibration["probability"] == "mimic_calibrated"]
    for spec, color in zip(specs, colors):
        rows = calibrated[calibrated["model"] == spec.slug].dropna(
            subset=["mean_predicted_probability", "observed_event_rate"]
        )
        axis.plot(
            rows["mean_predicted_probability"],
            rows["observed_event_rate"],
            marker="o",
            color=color,
            label=spec.display_name,
        )
    axis.set(
        xlabel="MIMIC-calibrated predicted probability",
        ylabel="Observed event rate",
        title="Frozen eICU calibration",
    )
    axis.legend(fontsize=8)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            figures / f"frozen_baseline_calibration.{suffix}", dpi=300, bbox_inches="tight"
        )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.bootstrap_reps <= 0:
        raise ValueError("Batch size and bootstrap replicates must be positive.")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    specificities = [float(value) for value in args.specificities.split(",")]
    device = choose_device(args.device)
    specs = model_specs()

    print("Auditing frozen artifacts and MIMIC validation transfer parameters...", flush=True)
    transfers, artifact_audit = validation_transfer(specs, specificities)
    mean, std, _ = load_scaling()
    load_models(specs, device)
    source_audit_path = output_dir / "source_prediction_reproduction.json"
    artifact_fingerprints = {
        slug: record["model_artifact_sha256"] for slug, record in artifact_audit.items()
    }
    source_reproduction: dict[str, Any]
    if source_audit_path.exists():
        cached_source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
        if cached_source_audit.get("artifact_fingerprints") == artifact_fingerprints:
            source_reproduction = cached_source_audit["models"]
            print("Using artifact-matched source prediction reproduction audit.", flush=True)
        else:
            source_audit_path.unlink()
            source_reproduction = verify_source_prediction_reproduction(
                specs=specs,
                mean=mean,
                std=std,
                device=device,
                batch_size=args.batch_size,
            )
    else:
        source_reproduction = verify_source_prediction_reproduction(
            specs=specs,
            mean=mean,
            std=std,
            device=device,
            batch_size=args.batch_size,
        )
    if not source_audit_path.exists():
        source_audit_path.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "artifact_fingerprints": artifact_fingerprints,
                    "models": source_reproduction,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print("Reconstructing every formal eICU window...", flush=True)
    data = load_formal_window_data(
        database="eICU-CRD",
        hourly_path=Path(args.hourly_pickle),
        prediction_path=Path(args.reference_predictions),
        target_col=TARGET,
        time_col=TIME,
        split_col=SPLIT,
        seq_length=SEQ_LENGTH,
    )
    cohort_audit = data.audit_record()
    if not cohort_audit["all_prediction_windows_reconstructed"]:
        raise ValueError("The full external prediction cohort was not reconstructed.")
    print(json.dumps(cohort_audit, indent=2), flush=True)

    raw_predictions = None if args.force_predictions else load_cached_predictions(
        specs, output_dir, data
    )
    predictions_were_cached = raw_predictions is not None
    if raw_predictions is None:
        print(f"Running frozen inference on {device}...", flush=True)
        raw_predictions = infer_all_models(
            data,
            specs,
            mean,
            std,
            device,
            args.batch_size,
            args.progress_every,
        )
    else:
        print("Using audited cached full-cohort external predictions.", flush=True)

    target = data.target_indices
    y = data.labels[target].astype(np.int8, copy=False)
    subjects = data.subject_ids[target]
    summary_rows: list[dict[str, Any]] = []
    operating_rows: list[dict[str, Any]] = []
    calibration_rows: list[pd.DataFrame] = []
    calibrated_predictions: dict[str, np.ndarray] = {}
    bootstrap_results: dict[str, pd.DataFrame] = {}

    for model_index, spec in enumerate(specs):
        print(f"Evaluating {spec.display_name}...", flush=True)
        raw = raw_predictions[spec.slug].astype(float)
        transfer = transfers[spec.slug]
        calibration_parameters = transfer["calibration"]
        probability = apply_platt_calibration(
            raw,
            float(calibration_parameters["intercept"]),
            float(calibration_parameters["slope"]),
        )
        calibrated_predictions[spec.slug] = probability
        prediction_path = output_dir / "predictions" / f"{spec.slug}.csv.gz"
        if not predictions_were_cached or not prediction_path.exists():
            prediction_path = save_prediction_frame(
                spec=spec,
                data=data,
                raw=raw,
                calibrated=probability,
                output_dir=output_dir,
            )

        raw_metrics = binary_metrics(y, raw)
        calibrated_metrics = binary_metrics(y, probability)
        raw_bins, raw_calibration = calibration_bins(
            y, raw, args.n_bins, {"model": spec.slug, "probability": "raw"}
        )
        calibrated_bins, calibrated_calibration = calibration_bins(
            y,
            probability,
            args.n_bins,
            {"model": spec.slug, "probability": "mimic_calibrated"},
        )
        calibration_rows.extend([raw_bins, calibrated_bins])
        raw_metrics.update(raw_calibration)
        calibrated_metrics.update(calibrated_calibration)
        raw_intercept, raw_slope = calibration_intercept_slope(y, raw)
        calibrated_intercept, calibrated_slope = calibration_intercept_slope(y, probability)

        thresholds = {
            float(key): float(value)
            for key, value in transfer["specificity_thresholds"].items()
        }
        model_operating: dict[int, dict[str, float]] = {}
        for specificity, threshold in thresholds.items():
            metrics = operating_metrics(y, probability, threshold)
            tag = int(round(specificity * 100))
            model_operating[tag] = metrics
            operating_rows.append(
                {
                    "model": spec.slug,
                    "display_name": spec.display_name,
                    "target_specificity": specificity,
                    "threshold_source": "MIMIC validation only",
                    **metrics,
                }
            )

        bootstrap_frame = pd.DataFrame(
            {SPLIT: subjects, "y_true": y, "y_prob": probability}
        )
        bootstrap_path = output_dir / f"bootstrap_{spec.slug}.csv.gz"
        bootstrap = load_cached_bootstrap(
            bootstrap_path, model=spec.slug, reps=args.bootstrap_reps
        )
        if bootstrap is None:
            print(
                f"Running {args.bootstrap_reps} patient-clustered bootstrap replicates for "
                f"{spec.display_name}...",
                flush=True,
            )
            bootstrap = optimized_cluster_bootstrap(
                bootstrap_frame, thresholds, args.bootstrap_reps, args.bootstrap_seed
            )
            bootstrap.insert(0, "model", spec.slug)
            bootstrap.to_csv(
                bootstrap_path,
                index=False,
                compression="gzip",
            )
        else:
            print(f"Using audited bootstrap cache for {spec.display_name}.", flush=True)
        bootstrap_results[spec.slug] = bootstrap
        ci = {
            metric: percentile_ci(bootstrap[metric])
            for metric in ("auroc", "auprc", "brier")
        }

        row: dict[str, Any] = {
            "model": spec.slug,
            "display_name": spec.display_name,
            "input_design": (
                "24h x 39 feature-matched"
                if spec.kind != "ebm"
                else "current-state 13 features"
            ),
            "patients": data.expected_patients,
            "stays": data.expected_stays,
            "windows": data.expected_windows,
            "positive_windows": data.expected_positive,
            "prevalence": float(y.mean()),
            "auroc": calibrated_metrics["auroc"],
            "auroc_ci_low": ci["auroc"][0],
            "auroc_ci_high": ci["auroc"][1],
            "auprc": calibrated_metrics["auprc"],
            "auprc_ci_low": ci["auprc"][0],
            "auprc_ci_high": ci["auprc"][1],
            "brier": calibrated_metrics["brier"],
            "brier_ci_low": ci["brier"][0],
            "brier_ci_high": ci["brier"][1],
            "ece": calibrated_metrics["ece"],
            "calibration_intercept": calibrated_intercept,
            "calibration_slope": calibrated_slope,
            "raw_brier": raw_metrics["brier"],
            "raw_ece": raw_metrics["ece"],
            "raw_calibration_intercept": raw_intercept,
            "raw_calibration_slope": raw_slope,
            "mimic_validation_platt_intercept": calibration_parameters["intercept"],
            "mimic_validation_platt_slope": calibration_parameters["slope"],
            "prediction_sha256": sha256_file(prediction_path),
            "no_eicu_fitting": True,
        }
        for tag, metrics in model_operating.items():
            for metric in ("sensitivity", "specificity", "ppv", "npv", "f1"):
                row[f"{metric}_at_spec_{tag}"] = metrics[metric]
                values = bootstrap[f"{metric}_at_spec_{tag}"]
                lower, upper = percentile_ci(values)
                row[f"{metric}_at_spec_{tag}_ci_low"] = lower
                row[f"{metric}_at_spec_{tag}_ci_high"] = upper
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "external_baseline_metrics.csv", index=False)
    operating = pd.DataFrame(operating_rows)
    operating.to_csv(output_dir / "external_fixed_specificity.csv", index=False)
    calibration = pd.concat(calibration_rows, ignore_index=True)
    calibration.to_csv(output_dir / "external_calibration_bins.csv", index=False)

    reference_slug = "kg_tfnn_equal_sample"
    paired_rows: list[dict[str, Any]] = []
    reference_bootstrap = bootstrap_results[reference_slug]
    reference_point = summary.set_index("model").loc[reference_slug]
    for spec in specs:
        if spec.slug == reference_slug:
            continue
        candidate_bootstrap = bootstrap_results[spec.slug]
        if not np.array_equal(
            candidate_bootstrap["replicate"], reference_bootstrap["replicate"]
        ):
            raise ValueError("Bootstrap replicate alignment failed.")
        candidate_point = summary.set_index("model").loc[spec.slug]
        for metric in ("auroc", "auprc", "brier"):
            differences = (
                candidate_bootstrap[metric].to_numpy()
                - reference_bootstrap[metric].to_numpy()
            )
            lower, upper = percentile_ci(pd.Series(differences))
            paired_rows.append(
                {
                    "comparison": f"{spec.slug} minus {reference_slug}",
                    "candidate": spec.slug,
                    "reference": reference_slug,
                    "metric": metric,
                    "difference": float(candidate_point[metric] - reference_point[metric]),
                    "ci_low": lower,
                    "ci_high": upper,
                    "paired_bootstrap_p": paired_p_value(differences),
                    "bootstrap_unit": SPLIT,
                    "replicates": args.bootstrap_reps,
                    "direction": "lower is better" if metric == "brier" else "higher is better",
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output_dir / "paired_vs_equal_sample_kg_tfnn.csv", index=False)

    transfer_payload = {
        "design": "MIMIC validation-only calibration and thresholds transported unchanged",
        "models": transfers,
    }
    (output_dir / "mimic_transfer_parameters.json").write_text(
        json.dumps(transfer_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    audit = {
        "status": "passed",
        "target": TARGET,
        "observation_window_hours": SEQ_LENGTH,
        "comparison_protocol": "equal_sample",
        "external_cohort": cohort_audit,
        "artifact_audit": artifact_audit,
        "source_prediction_reproduction": source_reproduction,
        "scaling_sha256": sha256_file(GRU_ROOT / "sequence_scaling.json"),
        "all_models_share_mimic_validation_windows": True,
        "all_models_share_eicu_windows": True,
        "external_subsampling": False,
        "eicu_model_refitting": False,
        "eicu_calibration_fitting": False,
        "eicu_threshold_selection": False,
        "bootstrap_unit": SPLIT,
        "bootstrap_reps": args.bootstrap_reps,
        "device": str(device),
    }
    (output_dir / "formal_cohort_and_freeze_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "analysis_config.json").write_text(
        json.dumps({**vars(args), "device_used": str(device)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    save_figures(y, calibrated_predictions, specs, calibration, output_dir)

    display_columns = [
        "display_name",
        "input_design",
        "auroc",
        "auroc_ci_low",
        "auroc_ci_high",
        "auprc",
        "auprc_ci_low",
        "auprc_ci_high",
        "brier",
        "ece",
        "calibration_intercept",
        "calibration_slope",
        "sensitivity_at_spec_90",
        "sensitivity_at_spec_95",
    ]
    context_lines: list[str] = []
    if FORMAL_KG_METRICS.exists():
        formal = json.loads(FORMAL_KG_METRICS.read_text(encoding="utf-8"))
        calibrated = formal["mimic_calibrated"]
        context_lines = [
            "## Formal Full-Cohort KG-TFNN Context",
            "",
            (
                "The prespecified formal KG-TFNN used a different, full-cohort training "
                "protocol and is therefore shown as context rather than included in the "
                "equal-sample architecture comparison."
            ),
            "",
            f"- External AUROC: {calibrated['auroc']:.4f}",
            f"- External AUPRC: {calibrated['auprc']:.4f}",
            f"- External Brier score: {calibrated['brier']:.4f}",
            f"- External ECE: {calibrated['ece']:.4f}",
            "",
        ]

    report = [
        "# Frozen Baseline External Validation on eICU",
        "",
        "## Design",
        "",
        "- Outcome: future 6-hour SOFA increase >= 2.",
        "- Observation window: 24 hours.",
        (
            f"- Complete external cohort: {data.expected_patients:,} patients, "
            f"{data.expected_stays:,} ICU stays, and {data.expected_windows:,} windows."
        ),
        "- Predictive parameters were frozen after MIMIC-IV development.",
        "- Calibration and fixed-specificity thresholds used MIMIC validation only.",
        "- No eICU outcome was used for fitting, model selection, calibration, or threshold selection.",
        f"- Uncertainty used {args.bootstrap_reps} patient-clustered bootstrap replicates.",
        "- EBM is a current-state comparator and is not architecture matched.",
        "",
        "## External Performance",
        "",
        markdown_table(summary[display_columns]),
        "",
        "## Paired Equal-Sample Comparisons",
        "",
        "Differences are candidate minus equal-sample KG-TFNN on identical eICU windows.",
        "",
        markdown_table(paired),
        "",
        *context_lines,
        "## Interpretation Boundary",
        "",
        (
            "This analysis evaluates frozen transportability. It does not establish "
            "architecture superiority because the formal full-cohort KG-TFNN and the "
            "equal-sample comparison answer different questions."
        ),
        "",
    ]
    (output_dir / "eicu_frozen_baseline_validation_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(f"Completed frozen external comparison: {output_dir}", flush=True)

    del data, raw_predictions
    gc.collect()


if __name__ == "__main__":
    main()
