"""以凍結的 MIMIC checkpoint 對 eICU 進行真正外部驗證。

Calibration、operating thresholds 與 risk strata 僅由 MIMIC validation 決定；
eICU outcome 不參與模型選擇、調參或 calibration fitting。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, roc_curve
from torch.utils.data import DataLoader, Dataset, Subset

from ablation_fnn_experiments import AblationWindowDataset
from advanced_model_evaluation import (
    apply_platt_calibration,
    calibration_intercept_slope,
    decision_curve,
    operating_metrics,
    percentile_ci,
    threshold_at_specificity,
)
from anfis_model import (
    FEATURE_ORDER,
    TemporalAttentionFNN,
    clinical_rule_priors,
    explicit_temporal_input_order,
    expert_feature_config,
)
from comparison_protocol import window_ids_for_mode
from model_evaluation_report import binary_metrics, calibration_bins
from patient_split import split_ids_for_values
from project_config import (
    EQUAL_SAMPLE_WINDOWS_CSV,
    PATIENT_SPLIT_CSV,
    PRIMARY_HOURLY_FEATURES,
)
from train_fnn import choose_device, load_training_frame, prepare_explicit_temporal_arrays


DEFAULT_CHECKPOINT = (
    "outputs/explicit_temporal_fnn_formal_6h/seed_42/best_model.pt"
)
DEFAULT_MIMIC_VALIDATION_PREDICTIONS = (
    "outputs/final_test_evaluation_6h/predictions/val_predictions.csv.gz"
)
DEFAULT_FINAL_TEST_LOCK = "outputs/final_test_evaluation_6h/FINAL_TEST_LOCK.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen-checkpoint eICU external validation.")
    parser.add_argument(
        "--hourly-pickle",
        default="outputs/eicu_external_validation/eicu_hourly_features.pkl",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--mimic-validation-predictions",
        default=DEFAULT_MIMIC_VALIDATION_PREDICTIONS,
    )
    parser.add_argument("--final-test-lock", default=DEFAULT_FINAL_TEST_LOCK)
    parser.add_argument("--mimic-csv", default=PRIMARY_HOURLY_FEATURES)
    parser.add_argument("--split-manifest", default=PATIENT_SPLIT_CSV)
    parser.add_argument("--equal-sample-windows", default=EQUAL_SAMPLE_WINDOWS_CSV)
    parser.add_argument("--target-col", default="label_sofa_increase_ge2_6h")
    parser.add_argument("--seq-length", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--bootstrap-reps", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--max-external-windows", type=int, default=None)
    parser.add_argument("--specificities", default="0.90,0.95")
    parser.add_argument("--risk-quantiles", default="0.60,0.85")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--dca-max-threshold", type=float, default=0.20)
    parser.add_argument("--output-dir", default="outputs/eicu_external_validation/evaluation")
    parser.add_argument("--force-mimic-predictions", action="store_true")
    parser.add_argument("--force-external-predictions", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_model(
    checkpoint_path: Path,
    device: torch.device,
    seq_length: int,
) -> tuple[TemporalAttentionFNN, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    checkpoint_seq = int(checkpoint_args.get("seq_length", seq_length))
    if checkpoint_seq != seq_length:
        raise ValueError(f"checkpoint seq_length={checkpoint_seq}, requested={seq_length}")
    explicit_temporal = bool(
        checkpoint.get(
            "explicit_temporal_features",
            checkpoint_args.get("explicit_temporal_features", False),
        )
    )
    if not explicit_temporal:
        raise ValueError("External pipeline requires the explicit-temporal checkpoint.")
    model = TemporalAttentionFNN(
        feature_configs=expert_feature_config,
        rule_configs=clinical_rule_priors,
        seq_length=checkpoint_seq,
        attention_hidden=int(checkpoint_args.get("attention_hidden", 64)),
        threshold=float(checkpoint_args.get("threshold", 7.0)),
        rule_score_scale=float(checkpoint_args.get("rule_score_scale", 0.2)),
        use_explicit_temporal_features=True,
        explicit_temporal_scale=float(checkpoint_args.get("explicit_temporal_scale", 1.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def predict_dataset(
    model: TemporalAttentionFNN,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    y_parts = []
    probability_parts = []
    entropy_parts = []
    with torch.inference_mode():
        for batch_x, batch_y in loader:
            output = model(batch_x.to(device, non_blocking=True))
            y_parts.append(batch_y.numpy())
            probability_parts.append(output.probabilities.detach().cpu().numpy())
            attention = output.attention_weights
            entropy = -(attention * torch.log(attention + 1e-8)).sum(dim=1)
            if attention.shape[1] > 1:
                entropy = entropy / math.log(attention.shape[1])
            entropy_parts.append(entropy.detach().cpu().numpy())
    return (
        np.concatenate(y_parts).astype(np.int8),
        np.concatenate(probability_parts).astype(np.float64),
        np.concatenate(entropy_parts).astype(np.float32),
    )


def dataset_target_indices(dataset: Dataset) -> np.ndarray:
    if isinstance(dataset, Subset):
        base = dataset.dataset
        selected = np.asarray(dataset.indices, dtype=np.int64)
        return base.target_indices[selected]
    return np.asarray(dataset.target_indices, dtype=np.int64)


def limit_dataset(dataset: Dataset, maximum: int | None, seed: int) -> Dataset:
    if maximum is None or maximum <= 0 or len(dataset) <= maximum:
        return dataset
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(dataset), size=maximum, replace=False))
    return Subset(dataset, selected.tolist())


def mimic_validation_predictions(
    args: argparse.Namespace,
    model: TemporalAttentionFNN,
    device: torch.device,
    output_dir: Path,
) -> pd.DataFrame:
    cache_path = output_dir / "mimic_validation_predictions.csv.gz"
    if cache_path.exists() and not args.force_mimic_predictions:
        return pd.read_csv(cache_path)

    frozen_predictions = Path(args.mimic_validation_predictions)
    if frozen_predictions.exists() and not args.force_mimic_predictions:
        predictions = pd.read_csv(frozen_predictions)
        if "y_prob_raw" not in predictions.columns:
            if "y_prob" not in predictions.columns:
                raise ValueError("Frozen MIMIC validation predictions lack y_prob.")
            predictions = predictions.rename(columns={"y_prob": "y_prob_raw"})
        required = {"subject_id", "stay_id", "sofa_hour", "y_true", "y_prob_raw"}
        missing = required - set(predictions.columns)
        if missing:
            raise ValueError(f"Frozen MIMIC validation predictions missing: {sorted(missing)}")
        predictions.to_csv(cache_path, index=False, compression="gzip")
        return predictions

    input_order = explicit_temporal_input_order(FEATURE_ORDER)
    frame = load_training_frame(
        Path(args.mimic_csv),
        input_order,
        args.target_col,
        "sofa_hour",
        "subject_id",
        None,
        None,
        args.chunk_size,
        None,
    )
    features, labels, stay_ids, subject_ids, time_values = prepare_explicit_temporal_arrays(
        frame,
        args.target_col,
        "sofa_hour",
        "subject_id",
    )
    del frame
    _, validation_ids, _ = split_ids_for_values(subject_ids, args.split_manifest)
    validation_windows = window_ids_for_mode(
        "equal_sample",
        args.equal_sample_windows,
        args.target_col,
        "validation",
    )
    dataset = AblationWindowDataset(
        features=features,
        labels=labels,
        stay_ids=stay_ids,
        split_values=subject_ids,
        time_values=time_values,
        allowed_split_values=validation_ids,
        input_seq_length=args.seq_length,
        min_history_length=args.seq_length,
        allowed_window_ids=validation_windows,
        require_all_window_ids=True,
    )
    target_indices = dataset_target_indices(dataset)
    y_true, y_prob, entropy = predict_dataset(
        model, dataset, device, args.batch_size, args.num_workers
    )
    predictions = pd.DataFrame(
        {
            "subject_id": subject_ids[target_indices],
            "stay_id": stay_ids[target_indices],
            "sofa_hour": time_values[target_indices],
            "y_true": y_true,
            "y_prob_raw": y_prob,
            "attention_entropy": entropy,
            "evaluation_split": "mimic_validation",
        }
    )
    predictions.to_csv(cache_path, index=False, compression="gzip")
    del features, labels, stay_ids, subject_ids, time_values, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions


def build_mimic_transfer_parameters(
    validation: pd.DataFrame,
    specificities: list[float],
    risk_quantiles: list[float],
    checkpoint_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    y_true = validation["y_true"].to_numpy(dtype=np.int8)
    raw = validation["y_prob_raw"].to_numpy(dtype=float)
    intercept, slope = calibration_intercept_slope(y_true, raw)
    calibrated = apply_platt_calibration(raw, intercept, slope)
    thresholds = {
        str(value): threshold_at_specificity(y_true, calibrated, value)
        for value in specificities
    }
    risk_cuts = [float(np.quantile(calibrated, value)) for value in risk_quantiles]
    payload = {
        "source_database": "MIMIC-IV validation only",
        "checkpoint_sha256": checkpoint_sha256,
        "calibration": {
            "method": "Platt logistic calibration on raw-probability logit",
            "intercept": intercept,
            "slope": slope,
        },
        "specificity_thresholds": thresholds,
        "risk_quantiles": risk_quantiles,
        "risk_cutoffs": risk_cuts,
        "validation_windows": len(validation),
        "validation_prevalence": float(y_true.mean()),
        "no_eicu_outcome_used": True,
    }
    (output_dir / "mimic_transfer_parameters.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def external_predictions(
    args: argparse.Namespace,
    model: TemporalAttentionFNN,
    device: torch.device,
    transfer: dict[str, Any],
    output_dir: Path,
) -> pd.DataFrame:
    cache_path = output_dir / "eicu_external_predictions.csv.gz"
    if cache_path.exists() and not args.force_external_predictions:
        return pd.read_csv(cache_path)

    frame = pd.read_pickle(args.hourly_pickle)
    frame = frame.sort_values(["stay_id", "sofa_hour"], kind="mergesort").reset_index(drop=True)
    input_order = explicit_temporal_input_order(FEATURE_ORDER)
    features, labels, stay_ids, subject_ids, time_values = prepare_explicit_temporal_arrays(
        frame,
        args.target_col,
        "sofa_hour",
        "subject_id",
    )
    all_subjects = set(pd.unique(subject_ids))
    dataset: Dataset = AblationWindowDataset(
        features=features,
        labels=labels,
        stay_ids=stay_ids,
        split_values=subject_ids,
        time_values=time_values,
        allowed_split_values=all_subjects,
        input_seq_length=args.seq_length,
        min_history_length=args.seq_length,
        allowed_window_ids=None,
    )
    dataset = limit_dataset(dataset, args.max_external_windows, args.bootstrap_seed)
    target_indices = dataset_target_indices(dataset)
    started = perf_counter()
    y_true, raw_probability, entropy = predict_dataset(
        model, dataset, device, args.batch_size, args.num_workers
    )
    elapsed = perf_counter() - started
    calibration = transfer["calibration"]
    calibrated = apply_platt_calibration(
        raw_probability,
        float(calibration["intercept"]),
        float(calibration["slope"]),
    )
    metadata_columns = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "hospital_id",
        "gender",
        "age_numeric",
        "ethnicity",
        "unitdischargestatus",
        "sofa_hour",
        "sofa_score",
        "sofa_component_count",
    ]
    predictions = frame.iloc[target_indices][metadata_columns].reset_index(drop=True)
    predictions = predictions.assign(
        y_true=y_true,
        y_prob_raw=raw_probability,
        y_prob=calibrated,
        attention_entropy=entropy,
        evaluation_split="eicu_external_test",
        model="KG-Temporal FNN 24h explicit",
        target_col=args.target_col,
    )
    predictions.to_csv(cache_path, index=False, compression="gzip")
    run_stats = {
        "prediction_windows": len(predictions),
        "seconds": elapsed,
        "windows_per_second": len(predictions) / max(elapsed, 1e-9),
        "device": str(device),
    }
    (output_dir / "external_inference_stats.json").write_text(
        json.dumps(run_stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del frame, features, labels, stay_ids, subject_ids, time_values, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return predictions


def optimized_cluster_bootstrap(
    frame: pd.DataFrame,
    thresholds: dict[float, float],
    reps: int,
    seed: int,
) -> pd.DataFrame:
    y = frame["y_true"].to_numpy(dtype=np.int8)
    score = frame["y_prob"].to_numpy(dtype=np.float64)
    subjects, codes = np.unique(frame["subject_id"].astype(str), return_inverse=True)
    rng = np.random.default_rng(seed)

    ascending = np.argsort(score, kind="mergesort")
    y_asc = y[ascending]
    score_asc = score[ascending]
    codes_asc = codes[ascending]
    tie_starts = np.r_[0, np.flatnonzero(score_asc[1:] != score_asc[:-1]) + 1]

    descending = np.argsort(-score, kind="mergesort")
    y_desc = y[descending]
    codes_desc = codes[descending]
    positive_desc = y_desc == 1
    squared_error = (score - y) ** 2
    subject_window_count = np.bincount(codes, minlength=len(subjects)).astype(np.float64)
    subject_squared_error = np.bincount(
        codes,
        weights=squared_error,
        minlength=len(subjects),
    )
    positive = y == 1
    negative = ~positive
    operating_subject_counts = {}
    for specificity, threshold in thresholds.items():
        predicted = score >= threshold
        operating_subject_counts[specificity] = {
            "tp": np.bincount(codes, weights=(predicted & positive), minlength=len(subjects)),
            "fn": np.bincount(codes, weights=((~predicted) & positive), minlength=len(subjects)),
            "tn": np.bincount(codes, weights=((~predicted) & negative), minlength=len(subjects)),
            "fp": np.bincount(codes, weights=(predicted & negative), minlength=len(subjects)),
        }
    rows = []
    for replicate in range(reps):
        subject_weights = rng.multinomial(
            len(subjects), np.full(len(subjects), 1.0 / len(subjects))
        ).astype(np.float64)
        total_weight = float(np.dot(subject_weights, subject_window_count))

        weights_asc = subject_weights[codes_asc]
        positive_by_tie = np.add.reduceat(weights_asc * (y_asc == 1), tie_starts)
        negative_by_tie = np.add.reduceat(weights_asc * (y_asc == 0), tie_starts)
        total_positive = positive_by_tie.sum()
        total_negative = negative_by_tie.sum()
        negative_before = np.cumsum(negative_by_tie) - negative_by_tie
        auroc = np.sum(positive_by_tie * (negative_before + 0.5 * negative_by_tie))
        auroc = auroc / (total_positive * total_negative)

        weights_desc = subject_weights[codes_desc]
        positive_weight = weights_desc * positive_desc
        cumulative_positive = np.cumsum(positive_weight)
        cumulative_total = np.cumsum(weights_desc)
        precision = np.divide(
            cumulative_positive,
            cumulative_total,
            out=np.zeros_like(cumulative_positive),
            where=cumulative_total > 0,
        )
        auprc = np.sum(precision * positive_weight) / total_positive
        row = {
            "replicate": replicate,
            "auroc": float(auroc),
            "auprc": float(auprc),
            "brier": float(np.dot(subject_weights, subject_squared_error) / total_weight),
        }
        for specificity in thresholds:
            counts = operating_subject_counts[specificity]
            tp = np.dot(subject_weights, counts["tp"])
            fn = np.dot(subject_weights, counts["fn"])
            tn = np.dot(subject_weights, counts["tn"])
            fp = np.dot(subject_weights, counts["fp"])
            tag = int(round(specificity * 100))
            row[f"sensitivity_at_spec_{tag}"] = float(tp / (tp + fn))
            row[f"specificity_at_spec_{tag}"] = float(tn / (tn + fp))
        rows.append(row)
        if (replicate + 1) % 50 == 0 or replicate + 1 == reps:
            print(f"  bootstrap {replicate + 1}/{reps}", flush=True)
    return pd.DataFrame(rows)


def risk_stratification(
    predictions: pd.DataFrame,
    cutoffs: list[float],
) -> pd.DataFrame:
    groups = pd.cut(
        predictions["y_prob"],
        [-np.inf, cutoffs[0], cutoffs[1], np.inf],
        labels=["low", "medium", "high"],
    )
    result = (
        predictions.assign(risk_group=groups)
        .groupby("risk_group", observed=True)["y_true"]
        .agg(windows="size", events="sum", event_rate="mean")
        .reset_index()
    )
    result["mimic_validation_cut_low"] = cutoffs[0]
    result["mimic_validation_cut_high"] = cutoffs[1]
    return result


def save_figures(
    predictions: pd.DataFrame,
    calibration_frame: pd.DataFrame,
    risk: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    y = predictions["y_true"].to_numpy(dtype=np.int8)
    p = predictions["y_prob"].to_numpy(dtype=float)

    fpr, tpr, _ = roc_curve(y, p)
    recall, precision, _ = precision_recall_curve(y, p)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(fpr, tpr, color="#4E79A7", linewidth=2)
    axes[0].plot([0, 1], [0, 1], "--", color="#888888")
    axes[0].set(xlabel="1 - Specificity", ylabel="Sensitivity", title="eICU external ROC")
    axes[1].plot(recall, precision, color="#F28E2B", linewidth=2)
    axes[1].axhline(y.mean(), linestyle="--", color="#888888")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="eICU external precision-recall")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"external_roc_pr.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    valid_bins = calibration_frame.dropna(
        subset=["mean_predicted_probability", "observed_event_rate"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot([0, 1], [0, 1], "--", color="#888888")
    axes[0].plot(
        valid_bins["mean_predicted_probability"],
        valid_bins["observed_event_rate"],
        marker="o",
        color="#59A14F",
    )
    axes[0].set(xlabel="Predicted probability", ylabel="Observed event rate", title="External calibration")
    axes[1].bar(risk["risk_group"].astype(str), risk["event_rate"], color=["#59A14F", "#F28E2B", "#E15759"])
    axes[1].set(xlabel="MIMIC-defined risk stratum", ylabel="Observed event rate", title="External risk stratification")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"external_calibration_risk.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    metrics: dict[str, Any],
    transfer: dict[str, Any],
    checkpoint_path: Path,
    checkpoint_sha256: str,
    output_dir: Path,
) -> None:
    raw = metrics["raw"]
    calibrated = metrics["mimic_calibrated"]
    ci = metrics["clustered_ci95"]
    fixed = metrics["fixed_specificity"]
    fixed_ci = metrics["fixed_specificity_ci95"]
    lines = [
        "# eICU External Validation",
        "",
        "## Design",
        "",
        "- Development database: MIMIC-IV.",
        "- External test database: eICU-CRD.",
        "- Outcome: future 6-hour SOFA increase >= 2.",
        "- Observation window: 24 hours.",
        "- Model, membership functions, rules, attention and weights were frozen.",
        "- Platt calibration, fixed-specificity thresholds and risk strata were fitted on MIMIC validation only.",
        "- No eICU outcome was used for fitting, selection or recalibration.",
        "",
        "## Cohort",
        "",
        f"- Windows: {metrics['windows']:,}",
        f"- Patients: {metrics['patients']:,}",
        f"- ICU stays: {metrics['stays']:,}",
        f"- Hospitals: {metrics['hospitals']:,}",
        f"- Event prevalence: {metrics['prevalence']:.4f}",
        "",
        "## Performance",
        "",
        "| Probability | AUROC | AUPRC | Brier | ECE | Calibration intercept | Calibration slope |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Raw checkpoint | {raw['auroc']:.4f} | {raw['auprc']:.4f} | {raw['brier']:.4f} | {raw['ece']:.4f} | {raw['calibration_intercept']:.3f} | {raw['calibration_slope']:.3f} |",
        f"| MIMIC-calibrated | {calibrated['auroc']:.4f} | {calibrated['auprc']:.4f} | {calibrated['brier']:.4f} | {calibrated['ece']:.4f} | {calibrated['calibration_intercept']:.3f} | {calibrated['calibration_slope']:.3f} |",
        "",
        f"Patient-clustered 95% CI: AUROC {ci['auroc'][0]:.4f}-{ci['auroc'][1]:.4f}; AUPRC {ci['auprc'][0]:.4f}-{ci['auprc'][1]:.4f}; Brier {ci['brier'][0]:.4f}-{ci['brier'][1]:.4f}.",
        "",
        "## MIMIC-Defined Operating Points",
        "",
        "| Target specificity | Threshold | External specificity | External sensitivity | PPV | NPV |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in fixed:
        tag = int(round(float(row["target_specificity"]) * 100))
        sensitivity_ci = fixed_ci[str(tag)]["sensitivity"]
        specificity_ci = fixed_ci[str(tag)]["specificity"]
        lines.append(
            f"| {float(row['target_specificity']):.0%} | {float(row['threshold']):.4f} | "
            f"{float(row['specificity']):.3f} ({specificity_ci[0]:.3f}-{specificity_ci[1]:.3f}) | "
            f"{float(row['sensitivity']):.3f} ({sensitivity_ci[0]:.3f}-{sensitivity_ci[1]:.3f}) | "
            f"{float(row['ppv']):.3f} | {float(row['npv']):.3f} |"
        )
    lines.extend(
        [
        "",
        "## Provenance",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Checkpoint SHA-256: `{checkpoint_sha256}`",
        f"- MIMIC validation windows: {transfer['validation_windows']:,}",
        f"- Cluster bootstrap unit: subject_id ({metrics['bootstrap_reps']} replicates).",
        ]
    )
    (output_dir / "eicu_external_validation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint)
    checkpoint_sha = sha256_file(checkpoint_path)
    lock_path = Path(args.final_test_lock)
    if lock_path.exists():
        final_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if final_lock.get("status") != "complete":
            raise ValueError("Final-test lock is not complete.")
        if final_lock.get("checkpoint_sha256", "").lower() != checkpoint_sha.lower():
            raise ValueError("External checkpoint does not match the locked final checkpoint.")
        frozen_validation = Path(args.mimic_validation_predictions)
        expected_hash = final_lock.get("validation_predictions_sha256")
        if expected_hash and sha256_file(frozen_validation).lower() != expected_hash.lower():
            raise ValueError("MIMIC validation predictions do not match the final-test lock.")
    device = choose_device(args.device)
    model, checkpoint = load_frozen_model(checkpoint_path, device, args.seq_length)
    specificities = [float(value) for value in args.specificities.split(",")]
    risk_quantiles = [float(value) for value in args.risk_quantiles.split(",")]
    if len(risk_quantiles) != 2:
        raise ValueError("risk-quantiles requires exactly two values")

    print("Generating/loading frozen MIMIC validation predictions...")
    mimic_validation = mimic_validation_predictions(args, model, device, output_dir)
    transfer = build_mimic_transfer_parameters(
        mimic_validation,
        specificities,
        risk_quantiles,
        checkpoint_sha,
        output_dir,
    )
    print("Generating/loading eICU external predictions...")
    predictions = external_predictions(args, model, device, transfer, output_dir)
    y = predictions["y_true"].to_numpy(dtype=np.int8)
    raw_probability = predictions["y_prob_raw"].to_numpy(dtype=float)
    probability = predictions["y_prob"].to_numpy(dtype=float)

    raw_metrics = binary_metrics(y, raw_probability)
    calibrated_metrics = binary_metrics(y, probability)
    raw_bins, raw_calibration = calibration_bins(
        y, raw_probability, args.n_bins, {"probability": "raw"}
    )
    calibrated_bins, calibrated_calibration = calibration_bins(
        y, probability, args.n_bins, {"probability": "mimic_calibrated"}
    )
    raw_metrics.update(raw_calibration)
    calibrated_metrics.update(calibrated_calibration)
    raw_intercept, raw_slope = calibration_intercept_slope(y, raw_probability)
    calibrated_intercept, calibrated_slope = calibration_intercept_slope(y, probability)
    raw_metrics.update(calibration_intercept=raw_intercept, calibration_slope=raw_slope)
    calibrated_metrics.update(
        calibration_intercept=calibrated_intercept,
        calibration_slope=calibrated_slope,
    )

    thresholds = {
        float(key): float(value)
        for key, value in transfer["specificity_thresholds"].items()
    }
    operating_rows = []
    for specificity, threshold in thresholds.items():
        row = operating_metrics(y, probability, threshold)
        row["threshold_source"] = f"MIMIC validation specificity {specificity:.2f}"
        row["target_specificity"] = specificity
        operating_rows.append(row)
    pd.DataFrame(operating_rows).to_csv(output_dir / "external_fixed_specificity.csv", index=False)

    risk = risk_stratification(predictions, transfer["risk_cutoffs"])
    risk.to_csv(output_dir / "external_risk_stratification.csv", index=False)
    dca = decision_curve(y, probability, "KG-Temporal FNN external", args.dca_max_threshold)
    dca.to_csv(output_dir / "external_decision_curve.csv", index=False)

    print(f"Running {args.bootstrap_reps} subject-clustered bootstrap replicates...")
    bootstrap = optimized_cluster_bootstrap(
        predictions[["subject_id", "y_true", "y_prob"]],
        thresholds,
        args.bootstrap_reps,
        args.bootstrap_seed,
    )
    bootstrap.to_csv(
        output_dir / "external_patient_cluster_bootstrap.csv.gz",
        index=False,
        compression="gzip",
    )
    ci = {
        metric: percentile_ci(bootstrap[metric])
        for metric in ("auroc", "auprc", "brier")
    }
    fixed_ci = {}
    for specificity in thresholds:
        tag = int(round(specificity * 100))
        fixed_ci[str(tag)] = {
            "sensitivity": percentile_ci(bootstrap[f"sensitivity_at_spec_{tag}"]),
            "specificity": percentile_ci(bootstrap[f"specificity_at_spec_{tag}"]),
        }
    metrics = {
        "windows": len(predictions),
        "patients": int(predictions["subject_id"].nunique()),
        "stays": int(predictions["stay_id"].nunique()),
        "hospitals": int(predictions["hospital_id"].nunique()),
        "prevalence": float(y.mean()),
        "raw": raw_metrics,
        "mimic_calibrated": calibrated_metrics,
        "clustered_ci95": ci,
        "fixed_specificity": operating_rows,
        "fixed_specificity_ci95": fixed_ci,
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_unit": "subject_id",
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_best_epoch": checkpoint.get("best_epoch", checkpoint.get("epoch")),
        "no_eicu_fitting": True,
    }
    (output_dir / "external_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.concat([raw_bins, calibrated_bins], ignore_index=True).to_csv(
        output_dir / "external_calibration_bins.csv", index=False
    )
    save_figures(predictions, calibrated_bins, risk, output_dir)
    write_report(metrics, transfer, checkpoint_path, checkpoint_sha, output_dir)
    config = {**vars(args), "device_used": str(device), "checkpoint_sha256": checkpoint_sha}
    (output_dir / "external_validation_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"External validation complete: AUROC={calibrated_metrics['auroc']:.4f}, "
        f"AUPRC={calibrated_metrics['auprc']:.4f}, Brier={calibrated_metrics['brier']:.4f}"
    )


if __name__ == "__main__":
    main()
