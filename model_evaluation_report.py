"""Unified model evaluation and calibration report.

Outputs:
1. `evaluation_summary.csv`
   Unified AUROC, AUPRC, Brier score, ECE, log loss and classification metrics.
2. `calibration_bins.csv`
   Bin-wise calibration statistics for models evaluated from probabilities.
3. `calibration_curve.png`
   Reliability diagram for available probability predictions.

The script can:
- Collect existing metric CSVs from `outputs/`.
- Recompute clinical score probability calibration for NEWS2/SOFA.
- Recompute FNN test predictions from a saved checkpoint.
- Evaluate `test_predictions.csv` files containing y_true/y_prob columns.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from anfis_model import TemporalAttentionFNN
from paper_figures import (
    build_binary_curve_table,
    generate_evaluation_figures,
    try_generate,
)
from comparison_protocol import (
    filter_frame_to_comparison_windows,
    validate_comparison_args,
)
from patient_split import attach_split, split_ids_for_values
from project_config import (
    COMPARISON_PROTOCOL_JSON,
    DEFAULT_PREDICTION_HORIZONS,
    EQUAL_SAMPLE_WINDOWS_CSV,
    PATIENT_SPLIT_CSV,
    PRIMARY_HOURLY_FEATURES,
)
from sofa_label_utils import horizon_from_target_col
from train_fnn import (
    FEATURE_ORDER,
    ICUWindowDataset,
    choose_device,
    prepare_arrays,
    prepare_explicit_temporal_arrays,
)

import clinical_score_baselines as clinical


def zero_to_none(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return value


def parse_sources(raw: str) -> set[str]:
    sources = {item.strip().lower() for item in raw.split(",") if item.strip()}
    if "all" in sources:
        return {"existing", "clinical", "fnn", "predictions"}
    return sources


def parse_horizons(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def safe_horizon_from_target(target_col: Any, fallback: Any = math.nan) -> Any:
    if target_col is None or (isinstance(target_col, float) and math.isnan(target_col)):
        return fallback
    try:
        return horizon_from_target_col(str(target_col))
    except ValueError:
        return fallback


def roc_auc_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = y_true == 1
    negatives = y_true == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    rank_sum_pos = ranks[positives].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = y_true == 1
    n_pos = int(positives.sum())
    if n_pos == 0:
        return math.nan
    order = np.argsort(-y_score)
    sorted_true = y_true[order]
    tp = np.cumsum(sorted_true == 1)
    precision = tp / (np.arange(sorted_true.size) + 1)
    return float(np.sum(precision[sorted_true == 1]) / n_pos)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = y_true.astype(np.float32)
    y_prob = np.clip(y_prob.astype(np.float64), 1e-7, 1.0 - 1e-7)
    y_pred = (y_prob >= threshold).astype(np.float32)

    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    tn = float(np.sum((y_pred == 0) & (y_true == 0)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    brier = float(np.mean((y_prob - y_true) ** 2))
    log_loss = float(-np.mean(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob)))

    return {
        "auroc": roc_auc_np(y_true, y_prob),
        "auprc": average_precision_np(y_true, y_prob),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "brier": brier,
        "log_loss": log_loss,
        "prevalence": float(np.mean(y_true)),
    }


def calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int,
    model_key: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    y_true = y_true.astype(np.float32)
    y_prob = np.clip(y_prob.astype(np.float64), 1e-7, 1.0 - 1e-7)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Include probability 1.0 in the last bin.
    bin_ids = np.minimum(np.digitize(y_prob, edges[1:-1], right=False), n_bins - 1)

    rows = []
    ece = 0.0
    mce = 0.0
    total = len(y_true)
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        count = int(mask.sum())
        if count == 0:
            mean_pred = math.nan
            frac_pos = math.nan
            abs_gap = math.nan
        else:
            mean_pred = float(np.mean(y_prob[mask]))
            frac_pos = float(np.mean(y_true[mask]))
            abs_gap = abs(mean_pred - frac_pos)
            ece += (count / total) * abs_gap
            mce = max(mce, abs_gap)
        rows.append(
            {
                **model_key,
                "bin": bin_idx,
                "bin_left": float(edges[bin_idx]),
                "bin_right": float(edges[bin_idx + 1]),
                "count": count,
                "mean_predicted_probability": mean_pred,
                "observed_event_rate": frac_pos,
                "absolute_gap": abs_gap,
            }
        )

    return pd.DataFrame(rows), {"ece": float(ece), "mce": float(mce)}


def evaluate_probability_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int,
    model_key: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    finite_mask = np.isfinite(y_true) & np.isfinite(y_prob)
    n_input = int(len(y_true))
    n_dropped_nonfinite = int((~finite_mask).sum())
    y_true = y_true[finite_mask]
    y_prob = y_prob[finite_mask]
    if len(y_true) == 0:
        raise ValueError(f"No finite predictions available for {model_key.get('model', 'unknown model')}.")

    metrics = binary_metrics(y_true, y_prob)
    bins, cal = calibration_bins(y_true, y_prob, n_bins=n_bins, model_key=model_key)
    curves = build_binary_curve_table(y_true, y_prob, model_key=model_key)
    return {
        **model_key,
        "n_input": n_input,
        "n_eval": int(len(y_true)),
        "n_dropped_nonfinite": n_dropped_nonfinite,
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "brier": metrics["brier"],
        "ece": cal["ece"],
        "mce": cal["mce"],
        "log_loss": metrics["log_loss"],
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "f1": metrics["f1"],
        "prevalence": metrics["prevalence"],
        "probability_source": "prediction_level",
    }, bins, curves


def latest_subdir(parent: Path, pattern: str) -> Path | None:
    candidates = [path for path in parent.glob(pattern) if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def is_smoke_artifact(path: Path) -> bool:
    return any("smoke" in part.lower() for part in path.parts)


def collect_existing_metrics(outputs_root: Path, include_smoke: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for path in outputs_root.rglob("clinical_score_metrics.csv"):
        if not include_smoke and is_smoke_artifact(path):
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            prefix = "test" if pd.notna(row.get("test_auroc", math.nan)) else "val"
            rows.append(
                {
                    "source_family": "clinical_score",
                    "comparison_mode": row.get("comparison_mode", "unknown"),
                    "protocol_sha256": row.get("protocol_sha256"),
                    "feature_set": row.get("feature_set", "clinical_score"),
                    "model": row.get("model"),
                    "target_col": row.get("target_col"),
                    "horizon_hours": row.get("horizon_hours"),
                    "threshold_strategy": row.get("threshold_strategy"),
                    "evaluation_split": "test" if prefix == "test" else "validation",
                    "n_eval": row.get(f"n_{prefix}"),
                    "auroc": row.get(f"{prefix}_auroc"),
                    "auprc": row.get(f"{prefix}_auprc"),
                    "brier": row.get(f"{prefix}_brier_calibrated"),
                    "ece": math.nan,
                    "mce": math.nan,
                    "log_loss": row.get(f"{prefix}_log_loss_calibrated"),
                    "accuracy": row.get(f"{prefix}_accuracy"),
                    "precision": row.get(f"{prefix}_precision"),
                    "recall": row.get(f"{prefix}_recall"),
                    "sensitivity": row.get(f"{prefix}_sensitivity", row.get(f"{prefix}_recall")),
                    "specificity": row.get(f"{prefix}_specificity"),
                    "f1": row.get(f"{prefix}_f1"),
                    "prevalence": row.get(f"{prefix}_prevalence"),
                    "probability_source": "existing_metrics",
                    "artifact_path": str(path),
                }
            )

    for path in outputs_root.rglob("baseline_metrics.csv"):
        if not include_smoke and is_smoke_artifact(path):
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            prefix = "test" if pd.notna(row.get("test_auroc", math.nan)) else "val"
            target_col = row.get("target_col", "label_sofa_increase_ge2_6h")
            rows.append(
                {
                    "source_family": "interpretable_ml",
                    "comparison_mode": row.get("comparison_mode", "unknown"),
                    "protocol_sha256": row.get("protocol_sha256"),
                    "feature_set": row.get("feature_set"),
                    "model": row.get("model"),
                    "target_col": target_col,
                    "horizon_hours": row.get("horizon_hours", safe_horizon_from_target(target_col, 6)),
                    "threshold_strategy": "0.5",
                    "evaluation_split": "test" if prefix == "test" else "validation",
                    "n_eval": row.get(f"n_{prefix}"),
                    "auroc": row.get(f"{prefix}_auroc"),
                    "auprc": row.get(f"{prefix}_auprc"),
                    "brier": row.get(f"{prefix}_brier"),
                    "ece": math.nan,
                    "mce": math.nan,
                    "log_loss": row.get(f"{prefix}_log_loss"),
                    "accuracy": row.get(f"{prefix}_accuracy"),
                    "precision": row.get(f"{prefix}_precision"),
                    "recall": row.get(f"{prefix}_recall"),
                    "sensitivity": row.get(f"{prefix}_recall"),
                    "specificity": row.get(f"{prefix}_specificity"),
                    "f1": row.get(f"{prefix}_f1"),
                    "prevalence": math.nan,
                    "probability_source": "existing_metrics",
                    "artifact_path": str(path),
                }
            )

    for path in outputs_root.rglob("blackbox_metrics.csv"):
        if not include_smoke and is_smoke_artifact(path):
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            prefix = "test" if pd.notna(row.get("test_auroc", math.nan)) else "val"
            target_col = row.get("target_col", "label_sofa_increase_ge2_6h")
            rows.append(
                {
                    "source_family": "black_box",
                    "comparison_mode": row.get("comparison_mode", "unknown"),
                    "protocol_sha256": row.get("protocol_sha256"),
                    "feature_set": row.get("feature_set"),
                    "model": row.get("model"),
                    "target_col": target_col,
                    "horizon_hours": row.get("horizon_hours", safe_horizon_from_target(target_col, 6)),
                    "threshold_strategy": "0.5",
                    "evaluation_split": "test" if prefix == "test" else "validation",
                    "n_eval": row.get(f"n_{prefix}"),
                    "auroc": row.get(f"{prefix}_auroc"),
                    "auprc": row.get(f"{prefix}_auprc"),
                    "brier": row.get(f"{prefix}_brier"),
                    "ece": math.nan,
                    "mce": math.nan,
                    "log_loss": row.get(f"{prefix}_log_loss"),
                    "accuracy": row.get(f"{prefix}_accuracy"),
                    "precision": row.get(f"{prefix}_precision"),
                    "recall": row.get(f"{prefix}_recall"),
                    "sensitivity": row.get(f"{prefix}_recall"),
                    "specificity": row.get(f"{prefix}_specificity"),
                    "f1": row.get(f"{prefix}_f1"),
                    "prevalence": math.nan,
                    "probability_source": "existing_metrics",
                    "artifact_path": str(path),
                }
            )

    for path in outputs_root.rglob("metrics.csv"):
        if "fnn_training" not in str(path):
            continue
        if not include_smoke and is_smoke_artifact(path):
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        best_idx = df["val_auroc"].astype(float).idxmax()
        row = df.loc[best_idx]
        test_path = path.parent / "test_metrics.csv"
        test_row = pd.read_csv(test_path).iloc[-1] if test_path.exists() else None
        config_path = path.parent / "train_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        target_col = row.get("target_col", config.get("target_col", "label_sofa_increase_ge2_6h"))
        rows.append(
            {
                "source_family": "fnn",
                "comparison_mode": config.get("comparison_mode", "unknown"),
                "protocol_sha256": config.get("comparison_protocol_sha256"),
                "feature_set": "sequence_raw",
                "model": path.parent.name,
                "target_col": target_col,
                "horizon_hours": row.get("horizon_hours", safe_horizon_from_target(target_col, 6)),
                "threshold_strategy": "0.5",
                "evaluation_split": "test" if test_row is not None else "validation",
                "n_eval": test_row.get("test_windows", math.nan) if test_row is not None else math.nan,
                "auroc": test_row.get("test_auroc") if test_row is not None else row.get("val_auroc"),
                "auprc": test_row.get("test_auprc") if test_row is not None else row.get("val_auprc"),
                "brier": math.nan,
                "ece": math.nan,
                "mce": math.nan,
                "log_loss": math.nan,
                "accuracy": math.nan,
                "precision": test_row.get("test_precision") if test_row is not None else row.get("val_precision"),
                "recall": test_row.get("test_recall") if test_row is not None else row.get("val_recall"),
                "sensitivity": test_row.get("test_recall") if test_row is not None else row.get("val_recall"),
                "specificity": math.nan,
                "f1": math.nan,
                "prevalence": math.nan,
                "probability_source": "existing_metrics",
                "artifact_path": str(path),
            }
        )

    for path in outputs_root.rglob("ablation_summary.csv"):
        if not include_smoke and is_smoke_artifact(path):
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            prefix = "test" if pd.notna(row.get("test_auroc", math.nan)) else "val"
            target_col = row.get("target_col", "label_sofa_increase_ge2_6h")
            rows.append(
                {
                    "source_family": "fnn_ablation",
                    "comparison_mode": row.get("comparison_mode", "unknown"),
                    "protocol_sha256": row.get("protocol_sha256"),
                    "feature_set": "sequence_raw" if row.get("input_seq_length", 24) != 1 else "static",
                    "model": row.get("variant"),
                    "target_col": target_col,
                    "horizon_hours": row.get("horizon_hours", safe_horizon_from_target(target_col, 6)),
                    "threshold_strategy": "0.5",
                    "evaluation_split": "test" if prefix == "test" else "validation",
                    "n_eval": row.get(f"{prefix}_windows"),
                    "auroc": row.get(f"{prefix}_auroc"),
                    "auprc": row.get(f"{prefix}_auprc"),
                    "brier": math.nan,
                    "ece": math.nan,
                    "mce": math.nan,
                    "log_loss": math.nan,
                    "accuracy": math.nan,
                    "precision": row.get(f"{prefix}_precision"),
                    "recall": row.get(f"{prefix}_recall"),
                    "sensitivity": row.get(f"{prefix}_recall"),
                    "specificity": math.nan,
                    "f1": math.nan,
                    "prevalence": row.get(f"{prefix}_positive", math.nan) / row.get(f"{prefix}_windows", math.nan),
                    "probability_source": "existing_metrics",
                    "artifact_path": str(path),
                }
            )
    return rows


def evaluate_prediction_csvs(
    outputs_root: Path,
    n_bins: int,
    include_smoke: bool = False,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame]]:
    rows = []
    bin_tables = []
    curve_tables = []
    prediction_paths = [
        *outputs_root.rglob("test_predictions.csv"),
        *outputs_root.rglob("test_predictions.csv.gz"),
    ]
    for path in prediction_paths:
        if not include_smoke and is_smoke_artifact(path):
            continue
        df = pd.read_csv(path)
        if not {"y_true", "y_prob"}.issubset(df.columns):
            continue
        target_col = "unknown"
        if "target_col" in df.columns and df["target_col"].notna().any():
            target_col = str(df["target_col"].dropna().iloc[0])
        horizon_hours: Any = math.nan
        if "horizon_hours" in df.columns and df["horizon_hours"].notna().any():
            horizon_hours = df["horizon_hours"].dropna().iloc[0]
        else:
            horizon_hours = safe_horizon_from_target(target_col, math.nan)
        model_key = {
            "source_family": "prediction_csv",
            "feature_set": path.parent.parent.name,
            "model": path.parent.name,
            "target_col": target_col,
            "horizon_hours": horizon_hours,
            "threshold_strategy": "0.5",
            "evaluation_split": "test",
            "comparison_mode": (
                str(df["comparison_mode"].dropna().iloc[0])
                if "comparison_mode" in df.columns and df["comparison_mode"].notna().any()
                else "unknown"
            ),
            "protocol_sha256": (
                str(df["protocol_sha256"].dropna().iloc[0])
                if "protocol_sha256" in df.columns and df["protocol_sha256"].notna().any()
                else None
            ),
            "artifact_path": str(path),
        }
        row, bins, curves = evaluate_probability_predictions(
            df["y_true"].to_numpy(dtype=np.float32),
            df["y_prob"].to_numpy(dtype=np.float64),
            n_bins=n_bins,
            model_key=model_key,
        )
        rows.append(row)
        bin_tables.append(bins)
        curve_tables.append(curves)
    return rows, bin_tables, curve_tables


def evaluate_clinical_scores(
    args: argparse.Namespace,
    n_bins: int,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame]]:
    horizons = parse_horizons(args.horizons)
    clinical_args = argparse.Namespace(
        feature_csv=args.feature_csv,
        sofa_csv=args.sofa_csv,
        time_col=args.time_col,
        split_col=args.split_col,
        split_manifest=args.split_manifest,
        horizons=args.horizons,
        val_frac=args.val_frac,
        seed=args.seed,
        max_rows=args.max_rows,
        max_stays=args.max_stays,
        chunk_size=args.chunk_size,
        save_predictions=False,
        output_dir=None,
    )
    score_df, metadata = clinical.build_score_frame(clinical_args, horizons)
    score_df = attach_split(score_df, args.split_manifest, patient_col=args.split_col)
    score_df = score_df.sort_values(["stay_id", args.time_col], kind="mergesort").reset_index(drop=True)
    score_df["_history_index"] = score_df.groupby("stay_id", sort=False).cumcount()

    rows = []
    bin_tables = []
    curve_tables = []
    for horizon in horizons:
        target_col = f"label_sofa_increase_ge2_{horizon}h"
        train_frame = filter_frame_to_comparison_windows(
            score_df.loc[score_df["dataset_split"] == "train"],
            target_col,
            args.time_col,
            "train",
            args.comparison_mode,
            args.equal_sample_windows,
            args.seq_length,
        )
        test_frame = filter_frame_to_comparison_windows(
            score_df.loc[score_df["dataset_split"] == "test"],
            target_col,
            args.time_col,
            "test",
            args.comparison_mode,
            args.equal_sample_windows,
            args.seq_length,
        )
        for score_name in ["news2_score", "sofa_score"]:
            train = train_frame[[score_name, target_col]].dropna()
            test = test_frame[[score_name, target_col]].dropna()
            if train.empty or test.empty:
                continue
            x_train = train[score_name].to_numpy(dtype=np.float64)
            y_train = train[target_col].to_numpy(dtype=np.float32)
            x_test = test[score_name].to_numpy(dtype=np.float64)
            y_test = test[target_col].to_numpy(dtype=np.float32)
            calibrator = clinical.fit_score_calibrator(x_train, y_train)
            y_prob = clinical.calibrated_probabilities(calibrator, x_test)
            model_key = {
                "source_family": "clinical_score",
                "feature_set": "clinical_score",
                "model": f"{score_name}_calibrated",
                "target_col": target_col,
                "horizon_hours": horizon,
                "threshold_strategy": "0.5_calibrated_probability",
                "evaluation_split": "test",
                "comparison_mode": args.comparison_mode,
                "artifact_path": metadata.get("feature_csv", args.feature_csv),
            }
            row, bins, curves = evaluate_probability_predictions(
                y_test,
                y_prob,
                n_bins=n_bins,
                model_key=model_key,
            )
            rows.append(row)
            bin_tables.append(bins)
            curve_tables.append(curves)
    return rows, bin_tables, curve_tables


def find_sofa_target_reference(target_col: str, split_col: str, time_col: str, allowed_stays: set[Any] | None) -> pd.DataFrame:
    match = re.search(r"_(\d+)h$", target_col)
    if match is None:
        raise ValueError(f"Cannot infer SOFA horizon from target column: {target_col}")
    horizon = int(match.group(1))
    sofa_csv = clinical.find_sofa_csv(None)
    if sofa_csv is None:
        raise FileNotFoundError("No sofa_scores_hourly.csv found for target merge.")
    return clinical.load_sofa_reference(
        sofa_csv=sofa_csv,
        horizons=[horizon],
        split_col=split_col,
        time_col=time_col,
        allowed_stays=allowed_stays,
    )[[split_col, time_col, target_col]]


def load_fnn_training_frame(config: dict[str, Any], args: argparse.Namespace) -> pd.DataFrame:
    feature_csv = Path(config.get("csv", args.feature_csv))
    feature_cols = config.get("input_order", config.get("feature_order", FEATURE_ORDER))
    target_col = config.get("target_col", "label_sofa_increase_ge2_6h")
    time_col = config.get("time_col", args.time_col)
    split_col = args.split_col
    header = clinical.read_header(feature_csv)

    usecols = ["stay_id", split_col, time_col, *feature_cols]
    if target_col in header:
        usecols.append(target_col)
    usecols = list(dict.fromkeys(usecols))

    df = clinical.load_feature_frame(
        feature_csv=feature_csv,
        usecols=usecols,
        split_col=split_col,
        max_rows=zero_to_none(args.max_rows),
        max_stays=zero_to_none(args.max_stays),
        chunk_size=args.chunk_size,
    )
    if target_col not in df.columns:
        join_col = "stay_id"
        allowed_stays = set(pd.unique(df[join_col])) if zero_to_none(args.max_stays) is not None else None
        ref = find_sofa_target_reference(target_col, join_col, time_col, allowed_stays)
        df = df.merge(ref, on=[join_col, time_col], how="left", validate="one_to_one")
    return df


def subset_window_dataset(dataset: ICUWindowDataset, max_windows: int | None, seed: int) -> Dataset:
    if max_windows is None or max_windows <= 0 or len(dataset) <= max_windows:
        return dataset
    targets = dataset.labels[dataset.window_starts + dataset.seq_length - 1]
    pos_idx = np.flatnonzero(targets == 1)
    neg_idx = np.flatnonzero(targets == 0)
    rng = np.random.default_rng(seed)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        selected = rng.choice(len(dataset), size=max_windows, replace=False)
    else:
        pos_frac = len(pos_idx) / len(dataset)
        n_pos = min(len(pos_idx), max(1, int(round(max_windows * pos_frac))))
        n_neg = min(len(neg_idx), max_windows - n_pos)
        selected = np.concatenate(
            [
                rng.choice(pos_idx, size=n_pos, replace=False),
                rng.choice(neg_idx, size=n_neg, replace=False),
            ]
        )
        rng.shuffle(selected)
    return Subset(dataset, selected.tolist())


def window_prediction_metadata(dataset: Dataset) -> pd.DataFrame:
    """依 DataLoader 順序取得 patient/stay/hour，供 clustered evaluation 使用。"""
    if isinstance(dataset, Subset):
        base = dataset.dataset
        subset_indices = np.asarray(dataset.indices, dtype=np.int64)
        starts = base.window_starts[subset_indices]
    else:
        base = dataset
        starts = base.window_starts
    target_indices = starts + base.seq_length - 1
    return pd.DataFrame(
        {
            "subject_id": base.split_values[target_indices],
            "stay_id": base.stay_ids[target_indices],
            "sofa_hour": base.time_values[target_indices],
        }
    )


def collect_fnn_predictions(
    model: TemporalAttentionFNN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    probs = []
    targets = []
    activated_counts = []
    top_rule_indices = []
    attention_entropies = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            output = model(batch_x.to(device, non_blocking=True))
            probs.append(output.probabilities.detach().cpu().numpy())
            targets.append(batch_y.detach().cpu().numpy())

            attention = output.attention_weights
            selected_hour = attention.argmax(dim=1)
            batch_index = torch.arange(attention.shape[0], device=attention.device)
            selected_rules = output.rule_activations[batch_index, selected_hour]
            activated_counts.append((selected_rules > 0.1).sum(dim=1).detach().cpu().numpy())
            top_rule_indices.append(selected_rules.argmax(dim=1).detach().cpu().numpy())
            entropy = -(attention * torch.log(attention + 1e-8)).sum(dim=1)
            if attention.shape[1] > 1:
                entropy = entropy / math.log(attention.shape[1])
            attention_entropies.append(entropy.detach().cpu().numpy())
    return (
        np.concatenate(targets).astype(np.float32),
        np.concatenate(probs).astype(np.float64),
        np.concatenate(activated_counts).astype(np.int16),
        np.concatenate(top_rule_indices).astype(np.int16),
        np.concatenate(attention_entropies).astype(np.float32),
    )


def evaluate_fnn_run(
    run_dir: Path,
    args: argparse.Namespace,
    n_bins: int,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame]]:
    config_path = run_dir / "train_config.json"
    checkpoint_path = run_dir / "best_model.pt"
    if not config_path.exists() or not checkpoint_path.exists():
        return [], [], []

    config = json.loads(config_path.read_text(encoding="utf-8"))
    device = choose_device(args.device)
    df = load_fnn_training_frame(config, args)
    target_col = config.get("target_col", "label_sofa_increase_ge2_6h")
    time_col = config.get("time_col", args.time_col)
    split_col = args.split_col
    explicit_temporal = bool(config.get("explicit_temporal_features", False))
    if explicit_temporal:
        features, labels, stay_ids, split_values, time_values = prepare_explicit_temporal_arrays(
            df=df,
            target_col=target_col,
            time_col=time_col,
            split_col=split_col,
        )
    else:
        feature_cols = config.get("feature_order", FEATURE_ORDER)
        features, labels, stay_ids, split_values, time_values = prepare_arrays(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            time_col=time_col,
            split_col=split_col,
        )
    _, val_ids, test_ids = split_ids_for_values(
        split_values,
        config.get("split_manifest", args.split_manifest),
    )
    val_dataset = ICUWindowDataset(
        features=features,
        labels=labels,
        stay_ids=stay_ids,
        split_values=split_values,
        time_values=time_values,
        allowed_split_values=val_ids,
        seq_length=config.get("seq_length", 24),
    )
    test_dataset = ICUWindowDataset(
        features=features,
        labels=labels,
        stay_ids=stay_ids,
        split_values=split_values,
        time_values=time_values,
        allowed_split_values=test_ids,
        seq_length=config.get("seq_length", 24),
    )
    val_dataset = subset_window_dataset(val_dataset, zero_to_none(args.max_val_windows), args.seed + 90)
    test_dataset = subset_window_dataset(test_dataset, zero_to_none(args.max_test_windows), args.seed + 100)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TemporalAttentionFNN(
        seq_length=config.get("seq_length", 24),
        attention_hidden=config.get("attention_hidden", 32),
        threshold=config.get("threshold", 7.0),
        rule_score_scale=config.get("rule_score_scale", 0.2),
        use_explicit_temporal_features=explicit_temporal,
        explicit_temporal_scale=config.get("explicit_temporal_scale", 1.0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if len(test_dataset) == 0:
        return [], [], []

    y_true, y_prob, test_activated, test_top_rule, test_attention_entropy = collect_fnn_predictions(
        model, loader, device
    )
    if args.save_predictions:
        prediction_dir = (
            Path(args.prediction_output_dir)
            if args.prediction_output_dir
            else Path(args.generated_output_dir) / "predictions" / run_dir.name
        )
        prediction_dir.mkdir(parents=True, exist_ok=True)
        protocol_sha = config.get("comparison_protocol_sha256")
        model_name = "explicit_temporal_fnn" if explicit_temporal else run_dir.name
        common = {
            "model": model_name,
            "target_col": target_col,
            "horizon_hours": safe_horizon_from_target(target_col),
            "comparison_mode": config.get("comparison_mode", args.comparison_mode),
            "protocol_sha256": protocol_sha,
        }
        test_frame = window_prediction_metadata(test_dataset)
        test_frame = test_frame.assign(
            y_true=y_true,
            y_prob=y_prob,
            activated_rule_count=test_activated,
            top_rule_index=test_top_rule,
            attention_entropy=test_attention_entropy,
            evaluation_split="test",
            **common,
        )
        test_frame.to_csv(prediction_dir / "test_predictions.csv.gz", index=False, compression="gzip")

        val_true, val_prob, val_activated, val_top_rule, val_attention_entropy = collect_fnn_predictions(
            model, val_loader, device
        )
        val_frame = window_prediction_metadata(val_dataset)
        val_frame = val_frame.assign(
            y_true=val_true,
            y_prob=val_prob,
            activated_rule_count=val_activated,
            top_rule_index=val_top_rule,
            attention_entropy=val_attention_entropy,
            evaluation_split="validation",
            **common,
        )
        val_frame.to_csv(prediction_dir / "val_predictions.csv.gz", index=False, compression="gzip")
    horizon_match = re.search(r"_(\d+)h$", target_col)
    horizon = int(horizon_match.group(1)) if horizon_match else math.nan
    model_key = {
        "source_family": "fnn",
        "feature_set": "sequence_raw",
        "model": "explicit_temporal_fnn" if explicit_temporal else run_dir.name,
        "target_col": target_col,
        "horizon_hours": horizon,
        "threshold_strategy": "0.5",
        "evaluation_split": "test",
        "comparison_mode": config.get("comparison_mode", args.comparison_mode),
        "protocol_sha256": config.get("comparison_protocol_sha256"),
        "artifact_path": str(checkpoint_path),
    }
    row, bins, curves = evaluate_probability_predictions(
        y_true,
        y_prob,
        n_bins=n_bins,
        model_key=model_key,
    )
    return [row], [bins], [curves]


def select_fnn_run_dirs(raw: str | None, outputs_root: Path) -> list[Path]:
    if raw:
        return [Path(item.strip()) for item in raw.split(",") if item.strip()]
    latest = latest_subdir(outputs_root / "fnn_training", "fnn_*")
    return [latest] if latest else []


def plot_calibration_curves(bin_tables: list[pd.DataFrame], output_dir: Path) -> None:
    if not bin_tables:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = pd.concat(bin_tables, ignore_index=True)
    nonempty = bins.dropna(subset=["mean_predicted_probability", "observed_event_rate"])
    if nonempty.empty:
        return

    plt.figure(figsize=(8, 7))
    for (family, model, target), group in nonempty.groupby(["source_family", "model", "target_col"]):
        label = f"{family}:{model}:{target}".replace("label_sofa_increase_ge2_", "")
        plt.plot(
            group["mean_predicted_probability"],
            group["observed_event_rate"],
            marker="o",
            linewidth=1.5,
            label=label,
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.0, label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event rate")
    plt.title("Calibration Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curve.png", dpi=180)
    plt.close()


def artifact_mtime(path_value: Any) -> float:
    if not isinstance(path_value, str) or not path_value:
        return 0.0
    path = Path(path_value)
    return path.stat().st_mtime if path.exists() else 0.0


def filter_and_deduplicate_summary(
    summary_df: pd.DataFrame,
    horizons: list[int],
    comparison_mode: str | None = None,
) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df

    summary_df = summary_df.copy()
    if comparison_mode and "comparison_mode" in summary_df.columns:
        summary_df = summary_df[summary_df["comparison_mode"].astype(str) == comparison_mode].copy()
    if "horizon_hours" in summary_df.columns:
        summary_df["horizon_hours"] = pd.to_numeric(summary_df["horizon_hours"], errors="coerce")
        if horizons:
            summary_df = summary_df[summary_df["horizon_hours"].isin(horizons)].copy()

    dedupe_cols = [
        col
        for col in [
            "source_family",
            "feature_set",
            "model",
            "target_col",
            "horizon_hours",
            "threshold_strategy",
            "evaluation_split",
            "comparison_mode",
            "probability_source",
        ]
        if col in summary_df.columns
    ]
    if "artifact_path" in summary_df.columns and dedupe_cols:
        summary_df["_artifact_mtime"] = summary_df["artifact_path"].apply(artifact_mtime)
        summary_df = summary_df.sort_values("_artifact_mtime").drop_duplicates(dedupe_cols, keep="last")
        summary_df = summary_df.drop(columns=["_artifact_mtime"])
    return summary_df.reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified model evaluation report.")
    parser.add_argument("--sources", default="existing,clinical", help="existing,clinical,fnn,predictions or all")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--feature-csv", default=PRIMARY_HOURLY_FEATURES)
    parser.add_argument("--sofa-csv", default=None)
    parser.add_argument("--time-col", default="sofa_hour")
    parser.add_argument("--split-col", default="subject_id")
    parser.add_argument("--split-manifest", default=PATIENT_SPLIT_CSV)
    parser.add_argument("--comparison-mode", choices=["full", "equal_sample"], default="full")
    parser.add_argument("--comparison-protocol", default=COMPARISON_PROTOCOL_JSON)
    parser.add_argument("--equal-sample-windows", default=EQUAL_SAMPLE_WINDOWS_CSV)
    parser.add_argument("--seq-length", type=int, default=24)
    parser.add_argument(
        "--horizons",
        default=",".join(map(str, DEFAULT_PREDICTION_HORIZONS)),
        help="Primary report defaults to 6; pass 12,24 explicitly for secondary reports.",
    )
    parser.add_argument("--val-frac", type=float, default=0.15, help="舊版相容參數；正式比例由 manifest 決定。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-stays", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--max-val-windows", type=int, default=0, help="0 代表完整 validation set。")
    parser.add_argument("--max-test-windows", type=int, default=0, help="0 代表評估完整 test set。")
    parser.add_argument("--fnn-run-dirs", default=None, help="Comma-separated FNN run dirs containing train_config.json and best_model.pt.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--prediction-output-dir", default=None)
    parser.add_argument("--include-smoke", action="store_true", help="Include smoke-test outputs in collected metrics.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = parse_horizons(args.horizons)
    validate_comparison_args(
        args.comparison_mode,
        args.comparison_protocol,
        f"label_sofa_increase_ge2_{horizons[0]}h",
        args.seq_length,
    )
    sources = parse_sources(args.sources)
    outputs_root = Path(args.outputs_root)
    run_name = datetime.now().strftime("model_evaluation_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else outputs_root / "model_evaluation" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    args.generated_output_dir = str(output_dir)

    summary_rows: list[dict[str, Any]] = []
    calibration_tables: list[pd.DataFrame] = []
    curve_tables: list[pd.DataFrame] = []

    if "existing" in sources:
        summary_rows.extend(collect_existing_metrics(outputs_root, include_smoke=args.include_smoke))

    if "predictions" in sources:
        rows, tables, curves = evaluate_prediction_csvs(
            outputs_root,
            n_bins=args.n_bins,
            include_smoke=args.include_smoke,
        )
        summary_rows.extend(rows)
        calibration_tables.extend(tables)
        curve_tables.extend(curves)

    if "clinical" in sources:
        rows, tables, curves = evaluate_clinical_scores(args, n_bins=args.n_bins)
        summary_rows.extend(rows)
        calibration_tables.extend(tables)
        curve_tables.extend(curves)

    if "fnn" in sources:
        for run_dir in select_fnn_run_dirs(args.fnn_run_dirs, outputs_root):
            rows, tables, curves = evaluate_fnn_run(run_dir, args, n_bins=args.n_bins)
            summary_rows.extend(rows)
            calibration_tables.extend(tables)
            curve_tables.extend(curves)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = filter_and_deduplicate_summary(
        summary_df,
        parse_horizons(args.horizons),
        comparison_mode=args.comparison_mode,
    )
    if not summary_df.empty:
        summary_df.to_csv(output_dir / "evaluation_summary.csv", index=False)
    calibration_df = pd.DataFrame()
    if calibration_tables:
        calibration_df = pd.concat(calibration_tables, ignore_index=True)
        calibration_df.to_csv(output_dir / "calibration_bins.csv", index=False)
    curve_df = pd.DataFrame()
    if curve_tables:
        nonempty_curves = [table for table in curve_tables if not table.empty]
        if nonempty_curves:
            curve_df = pd.concat(nonempty_curves, ignore_index=True)
            curve_df.to_csv(output_dir / "curve_points.csv", index=False)

    try_generate(
        generate_evaluation_figures,
        summary_df=summary_df,
        output_dir=output_dir,
        calibration_df=calibration_df,
        curve_df=curve_df,
    )

    with (output_dir / "evaluation_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    print(f"Evaluation report complete: {output_dir}")
    if not summary_df.empty:
        cols = [
            "source_family",
            "model",
            "target_col",
            "auroc",
            "auprc",
            "brier",
            "ece",
            "log_loss",
            "probability_source",
        ]
        print(summary_df[[col for col in cols if col in summary_df.columns]].tail(20).to_string(index=False))


if __name__ == "__main__":
    main()
