"""Black-box baseline experiments for ICU deterioration prediction.

Models included:
1. Random Forest
2. XGBoost
3. LightGBM
4. LSTM
5. GRU

研究定位：
這些模型主要用來回答「本研究的 Knowledge-Guided Temporal FNN 是否能在預測效能上
接近代表性黑盒模型，同時提供更好的本質可解釋性」。因此這裡統一使用 SOFA
deterioration label、共用 patient-level train/validation/test split，並保存同一套評估指標。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import random
import re
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from anfis_model import FEATURE_ORDER
from comparison_protocol import (
    cohort_record,
    filter_frame_to_comparison_windows,
    validate_cohort_records,
    validate_comparison_args,
    window_id_membership,
    window_ids_for_mode,
    write_cohort_audit,
)
from patient_split import attach_split
from project_config import (
    COMPARISON_PROTOCOL_JSON,
    EQUAL_SAMPLE_WINDOWS_CSV,
    PATIENT_SPLIT_CSV,
    PRIMARY_HOURLY_FEATURES,
)
from sofa_label_utils import horizon_from_target_col, maybe_existing_usecols, merge_sofa_targets
from temporal_feature_utils import is_measurement_process_feature, temporal_feature_window
from train_fnn import CLINICAL_DEFAULTS
from paper_figures import (
    generate_metric_comparison_figures,
    generate_training_figures,
    try_generate,
)


MODEL_ALIASES = {
    "all": "all",
    "rf": "random_forest",
    "random_forest": "random_forest",
    "randomforest": "random_forest",
    "xgb": "xgboost",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
    "lgbm": "lightgbm",
    "lstm": "lstm",
    "gru": "gru",
}


TABULAR_MODELS = {"random_forest", "xgboost", "lightgbm"}
SEQUENCE_MODELS = {"lstm", "gru"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_model_names(model_arg: str) -> list[str]:
    keys = [key.strip().lower() for key in model_arg.split(",") if key.strip()]
    if not keys or "all" in keys:
        return ["random_forest", "xgboost", "lightgbm", "lstm", "gru"]

    models = []
    for key in keys:
        if key not in MODEL_ALIASES:
            raise ValueError(f"Unknown model name: {key}")
        canonical = MODEL_ALIASES[key]
        if canonical != "all" and canonical not in models:
            models.append(canonical)
    return models


def import_optional_dependencies() -> dict[str, Any]:
    deps: dict[str, Any] = {}

    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:
        deps["sklearn_error"] = exc
    else:
        deps["RandomForestClassifier"] = RandomForestClassifier

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        deps["xgboost_error"] = exc
    else:
        deps["XGBClassifier"] = XGBClassifier

    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        deps["lightgbm_error"] = exc
    else:
        deps["LGBMClassifier"] = LGBMClassifier

    try:
        import joblib
    except ImportError:
        deps["joblib"] = None
    else:
        deps["joblib"] = joblib

    return deps


def runnable_models(selected_models: list[str], deps: dict[str, Any]) -> list[str]:
    runnable = []
    for model_name in selected_models:
        if model_name == "random_forest" and "sklearn_error" in deps:
            print("Skip random_forest: scikit-learn is not installed.")
            continue
        if model_name == "xgboost" and "xgboost_error" in deps:
            print("Skip xgboost: xgboost is not installed.")
            continue
        if model_name == "lightgbm" and "lightgbm_error" in deps:
            print("Skip lightgbm: lightgbm is not installed.")
            continue
        runnable.append(model_name)
    return runnable


def dependency_hint(selected_models: list[str], deps: dict[str, Any]) -> str | None:
    packages = []
    if "random_forest" in selected_models and "sklearn_error" in deps:
        packages.append("scikit-learn")
    if "xgboost" in selected_models and "xgboost_error" in deps:
        packages.append("xgboost")
    if "lightgbm" in selected_models and "lightgbm_error" in deps:
        packages.append("lightgbm")
    if not packages:
        return None
    return ".\\env\\Scripts\\python.exe -m pip install " + " ".join(packages)


def read_csv_header(csv_path: Path) -> list[str]:
    return list(pd.read_csv(csv_path, nrows=0).columns)


def is_temporal_feature(column: str, base_features: list[str]) -> bool:
    return is_measurement_process_feature(column, base_features) or (
        temporal_feature_window(column, base_features) is not None
    )


def build_tabular_feature_sets(columns: list[str], feature_set_arg: str) -> dict[str, list[str]]:
    static_features = [feature for feature in FEATURE_ORDER if feature in columns]
    temporal_features = [col for col in columns if is_temporal_feature(col, FEATURE_ORDER)]
    temporal_features = sorted(
        temporal_features,
        key=lambda col: (
            FEATURE_ORDER.index(col.split("_w")[0]) if col.split("_w")[0] in FEATURE_ORDER else 999,
            col,
        ),
    )

    measurement_process = [
        column
        for feature in FEATURE_ORDER
        for column in (
            f"{feature}_is_missing",
            f"{feature}_time_since_last_measurement_h",
        )
        if column in columns
    ]
    window_24h = [column for column in temporal_features if "_w24h_" in column]

    all_sets = {
        "protocol": static_features,
        "static": static_features,
        "temporal": [*static_features, *temporal_features],
        "matched24": list(dict.fromkeys([*static_features, *measurement_process, *window_24h])),
    }
    if feature_set_arg == "compare":
        return all_sets
    return {feature_set_arg: all_sets[feature_set_arg]}


def build_sequence_features(columns: list[str], mode: str) -> list[str]:
    """Build raw or feature-matched recurrent-model inputs."""
    raw = [feature for feature in FEATURE_ORDER if feature in columns]
    if mode == "raw":
        return raw
    if mode != "matched":
        raise ValueError(f"Unknown sequence feature mode: {mode}")

    missing = [f"{feature}_is_missing" for feature in FEATURE_ORDER]
    recency = [f"{feature}_time_since_last_measurement_h" for feature in FEATURE_ORDER]
    expected = [*raw, *missing, *recency]
    absent = [column for column in expected if column not in columns]
    if absent:
        raise ValueError(f"Matched sequence inputs are missing columns: {absent}")
    return expected


def load_baseline_frame(
    csv_path: Path,
    usecols: list[str],
    max_rows: int | None,
    max_stays: int | None,
    chunk_size: int,
) -> pd.DataFrame:
    if max_rows is None and max_stays is None:
        return pd.read_csv(csv_path, usecols=usecols)

    chunks = []
    rows_seen = 0
    selected_stays: set[Any] = set()

    reader = pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size)
    for chunk in reader:
        if max_rows is not None:
            remaining = max_rows - rows_seen
            if remaining <= 0:
                break
            chunk = chunk.iloc[:remaining]
            rows_seen += len(chunk)

        stop_after_chunk = False
        if max_stays is not None:
            keep_stays = set()
            for stay_id in pd.unique(chunk["stay_id"]):
                if stay_id in selected_stays:
                    keep_stays.add(stay_id)
                elif len(selected_stays) < max_stays:
                    selected_stays.add(stay_id)
                    keep_stays.add(stay_id)
                else:
                    stop_after_chunk = True
            chunk = chunk[chunk["stay_id"].isin(keep_stays)]

        if len(chunk) > 0:
            chunks.append(chunk)
        if stop_after_chunk:
            break
        if max_rows is not None and rows_seen >= max_rows:
            break

    if not chunks:
        return pd.DataFrame(columns=usecols)
    return pd.concat(chunks, ignore_index=True)


def default_fill_value(column: str) -> float:
    if column in CLINICAL_DEFAULTS:
        return float(CLINICAL_DEFAULTS[column])

    if column.endswith("_is_missing"):
        return 1.0
    if column.endswith("_time_since_last_measurement_h"):
        return 0.0

    match = re.match(
        r"^(?P<feature>.+)_w\d+h_(?P<stat>mean|min|max|std|slope|change|abnormal_duration|abnormal_frequency)$",
        column,
    )
    if match:
        feature = match.group("feature")
        stat = match.group("stat")
        if stat in {"std", "slope", "change", "abnormal_duration", "abnormal_frequency"}:
            return 0.0
        if feature in CLINICAL_DEFAULTS:
            return float(CLINICAL_DEFAULTS[feature])

    return math.nan


def prepare_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    time_col: str,
    min_history_hours: int,
) -> pd.DataFrame:
    missing = {"stay_id", time_col, target_col, *feature_cols} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df.sort_values(["stay_id", time_col], kind="mergesort").reset_index(drop=True).copy()
    df["_history_index"] = df.groupby("stay_id", sort=False).cumcount()

    for col in [target_col, *feature_cols]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df.groupby("stay_id", sort=False)[feature_cols].ffill()

    deterministic_fill = {
        col: default_fill_value(col)
        for col in feature_cols
        if not math.isnan(default_fill_value(col))
    }
    df[feature_cols] = df[feature_cols].fillna(deterministic_fill)
    return df


def stratified_sample(
    df: pd.DataFrame,
    target_col: str,
    max_samples: int | None,
    seed: int,
) -> pd.DataFrame:
    if max_samples is None or max_samples <= 0 or len(df) <= max_samples:
        return df

    pos_df = df[df[target_col] == 1]
    neg_df = df[df[target_col] == 0]
    if len(pos_df) == 0 or len(neg_df) == 0:
        return df.sample(n=max_samples, random_state=seed)

    pos_frac = len(pos_df) / len(df)
    n_pos = min(len(pos_df), max(1, int(round(max_samples * pos_frac))))
    n_neg = min(len(neg_df), max_samples - n_pos)

    sampled = pd.concat(
        [
            pos_df.sample(n=n_pos, random_state=seed),
            neg_df.sample(n=n_neg, random_state=seed + 1),
        ],
        ignore_index=True,
    )
    return sampled.sample(frac=1.0, random_state=seed + 2).reset_index(drop=True)


def fill_remaining_with_train_medians(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    medians = train_df[feature_cols].median(numeric_only=True)
    fill_values = {}
    for col in feature_cols:
        value = medians.get(col, math.nan)
        if pd.isna(value):
            value = default_fill_value(col)
        if pd.isna(value):
            value = 0.0
        fill_values[col] = float(value)

    train_df[feature_cols] = train_df[feature_cols].fillna(fill_values)
    val_df[feature_cols] = val_df[feature_cols].fillna(fill_values)
    test_df[feature_cols] = test_df[feature_cols].fillna(fill_values)
    return train_df, val_df, test_df, fill_values


def standardize_arrays(
    features: np.ndarray,
    split_values: np.ndarray,
    train_values: set[Any],
) -> tuple[np.ndarray, dict[str, list[float]]]:
    train_mask = np.isin(split_values, list(train_values))
    train_features = features[train_mask]
    mean = np.nanmean(train_features, axis=0).astype(np.float32)
    std = np.nanstd(train_features, axis=0).astype(np.float32)
    mean = np.nan_to_num(mean, nan=0.0)
    std = np.nan_to_num(std, nan=1.0)
    std[std == 0] = 1.0
    return ((features - mean) / std).astype(np.float32), {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


class ICUSequenceDataset(Dataset):
    """保存 hourly array，getitem 時才切出 24h sequence。"""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        stay_ids: np.ndarray,
        split_values: np.ndarray,
        time_values: np.ndarray,
        allowed_split_values: set[Any],
        seq_length: int,
        allowed_window_ids: np.ndarray | None = None,
        require_all_window_ids: bool = True,
    ) -> None:
        self.features = features
        self.labels = labels
        self.seq_length = seq_length
        self.stay_ids = stay_ids
        self.split_values = split_values
        self.time_values = time_values
        self.window_starts = self._build_window_starts(
            stay_ids=stay_ids,
            split_values=split_values,
            time_values=time_values,
            allowed_split_values=allowed_split_values,
            seq_length=seq_length,
            allowed_window_ids=allowed_window_ids,
        )
        if require_all_window_ids and allowed_window_ids is not None and len(self) != len(allowed_window_ids):
            raise ValueError(
                f"Equal-sample windows 缺少 {len(allowed_window_ids) - len(self):,} 筆；"
                "正式比較不可限制 rows/stays/windows。"
            )

    def _build_window_starts(
        self,
        stay_ids: np.ndarray,
        split_values: np.ndarray,
        time_values: np.ndarray,
        allowed_split_values: set[Any],
        seq_length: int,
        allowed_window_ids: np.ndarray | None,
    ) -> np.ndarray:
        starts_by_stay = []
        valid_feature_row = np.isfinite(self.features).all(axis=1)
        valid_label_row = np.isfinite(self.labels)

        boundaries = np.flatnonzero(stay_ids[1:] != stay_ids[:-1]) + 1
        stay_starts = np.concatenate(([0], boundaries))
        stay_ends = np.concatenate((boundaries, [len(stay_ids)]))

        for stay_start, stay_end in zip(stay_starts, stay_ends):
            if split_values[stay_start] not in allowed_split_values:
                continue
            stay_len = stay_end - stay_start
            if stay_len < seq_length:
                continue

            local_starts = np.arange(stay_start, stay_end - seq_length + 1, dtype=np.int64)
            target_indices = local_starts + seq_length - 1
            feature_valid_count = np.concatenate(
                ([0], np.cumsum(valid_feature_row[stay_start:stay_end], dtype=np.int32))
            )
            window_valid = (
                feature_valid_count[seq_length:] - feature_valid_count[:-seq_length]
            ) == seq_length
            label_valid = valid_label_row[target_indices]
            keep = window_valid & label_valid
            if allowed_window_ids is not None:
                target_window_ids = (
                    stay_ids[target_indices].astype(np.int64) * 100_000
                    + time_values[target_indices].astype(np.int64)
                )
                keep &= window_id_membership(target_window_ids, allowed_window_ids)
            if np.any(keep):
                starts_by_stay.append(local_starts[keep])

        if not starts_by_stay:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(starts_by_stay).astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.window_starts.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.window_starts[index])
        end = start + self.seq_length
        target_index = end - 1
        return (
            torch.from_numpy(self.features[start:end]),
            torch.tensor(self.labels[target_index], dtype=torch.float32),
        )

    def label_counts(self) -> tuple[int, int]:
        if len(self) == 0:
            return 0, 0
        targets = self.labels[self.window_starts + self.seq_length - 1]
        positives = int(np.sum(targets == 1))
        negatives = int(np.sum(targets == 0))
        return positives, negatives

    def cohort_record(self, split: str, target_col: str) -> dict[str, Any]:
        target_indices = self.window_starts + self.seq_length - 1
        return cohort_record(
            self.stay_ids[target_indices],
            self.time_values[target_indices],
            self.labels[target_indices],
            split,
            target_col,
        )


def maybe_subset_dataset(dataset: Dataset, max_windows: int | None, seed: int) -> Dataset:
    if max_windows is None or max_windows <= 0 or len(dataset) <= max_windows:
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=max_windows, replace=False)
    return Subset(dataset, indices.tolist())


def dataset_label_counts(dataset: Dataset) -> tuple[int, int]:
    if hasattr(dataset, "label_counts"):
        return dataset.label_counts()

    if isinstance(dataset, Subset) and hasattr(dataset.dataset, "window_starts"):
        base_dataset = dataset.dataset
        subset_indices = np.asarray(dataset.indices, dtype=np.int64)
        starts = base_dataset.window_starts[subset_indices]
        targets = base_dataset.labels[starts + base_dataset.seq_length - 1]
        positives = int(np.sum(targets == 1))
        negatives = int(np.sum(targets == 0))
        return positives, negatives

    positives = 0
    negatives = 0
    for _, target in dataset:
        if float(target) == 1.0:
            positives += 1
        else:
            negatives += 1
    return positives, negatives


def sequence_prediction_metadata(dataset: Dataset) -> pd.DataFrame:
    """依 sequence DataLoader 順序取得 patient/stay/hour identifiers。"""
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


def _sequence_dataset_starts(dataset: Dataset) -> tuple[ICUSequenceDataset, np.ndarray]:
    if isinstance(dataset, Subset):
        base = dataset.dataset
        indices = np.asarray(dataset.indices, dtype=np.int64)
        return base, base.window_starts[indices]
    return dataset, dataset.window_starts


def matched_tree_feature_names(sequence_features: list[str]) -> list[str]:
    raw = sequence_features[: len(FEATURE_ORDER)]
    names = [f"current::{column}" for column in sequence_features]
    for statistic in ("mean", "min", "max", "std", "slope", "short_change", "window_change"):
        names.extend(f"{statistic}::{feature}" for feature in raw)
    names.extend(f"missing_fraction::{feature}" for feature in raw)
    return names


def summarize_sequence_for_trees(
    dataset: Dataset,
    base_feature_count: int,
    batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the same 24x39 hourly input into outcome-agnostic tree summaries."""
    base, starts = _sequence_dataset_starts(dataset)
    n_rows = len(starts)
    output_width = base.features.shape[1] + (8 * base_feature_count)
    output = np.empty((n_rows, output_width), dtype=np.float32)
    targets = base.labels[starts + base.seq_length - 1].astype(np.float32, copy=True)
    offsets = np.arange(base.seq_length, dtype=np.int64)
    hours = np.arange(base.seq_length, dtype=np.float32)
    centered_hours = hours - hours.mean()
    slope_denominator = float(np.sum(centered_hours**2))

    for batch_start in range(0, n_rows, batch_size):
        batch_end = min(batch_start + batch_size, n_rows)
        index = starts[batch_start:batch_end, None] + offsets[None, :]
        sequence = base.features[index]
        raw = sequence[:, :, :base_feature_count]
        missing = sequence[:, :, base_feature_count : 2 * base_feature_count]
        slope = np.einsum("btf,t->bf", raw, centered_hours, optimize=True) / slope_denominator
        output[batch_start:batch_end] = np.concatenate(
            [
                sequence[:, -1, :],
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
        )
    return output, targets


class RecurrentRiskModel(nn.Module):
    """Plain LSTM/GRU black-box baseline."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        rnn_type: str,
    ) -> None:
        super().__init__()
        if rnn_type not in {"lstm", "gru"}:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")
        rnn_cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x_seq)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden).squeeze(-1)


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true = y_true.astype(np.float32)
    y_score = np.clip(y_score.astype(np.float32), 1e-7, 1.0 - 1e-7)
    y_pred = (y_score >= 0.5).astype(np.float32)

    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    tn = float(np.sum((y_pred == 0) & (y_true == 0)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    brier = float(np.mean((y_score - y_true) ** 2))
    log_loss = float(-np.mean(y_true * np.log(y_score) + (1.0 - y_true) * np.log(1.0 - y_score)))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auroc": roc_auc_np(y_true, y_score),
        "auprc": average_precision_np(y_true, y_score),
        "brier": brier,
        "log_loss": log_loss,
    }


def predict_probabilities(model: Any, x: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(x)
    probs = np.asarray(probs)
    if probs.ndim == 2:
        return probs[:, 1].astype(np.float32)
    return probs.reshape(-1).astype(np.float32)


def class_scale_pos_weight(y_train: np.ndarray) -> float:
    positives = max(int(np.sum(y_train == 1)), 1)
    negatives = max(int(np.sum(y_train == 0)), 1)
    return negatives / positives


def fit_random_forest(deps: dict[str, Any], x_train: np.ndarray, y_train: np.ndarray, args: argparse.Namespace) -> Any:
    rf_cls = deps["RandomForestClassifier"]
    return rf_cls(
        n_estimators=args.rf_n_estimators,
        max_depth=args.rf_max_depth,
        min_samples_leaf=args.rf_min_samples_leaf,
        max_features=args.rf_max_features,
        class_weight="balanced_subsample",
        n_jobs=args.n_jobs,
        random_state=args.seed,
    ).fit(x_train, y_train)


def fit_xgboost(deps: dict[str, Any], x_train: np.ndarray, y_train: np.ndarray, args: argparse.Namespace) -> Any:
    xgb_cls = deps["XGBClassifier"]
    return xgb_cls(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method=args.xgb_tree_method,
        scale_pos_weight=class_scale_pos_weight(y_train),
        n_jobs=args.n_jobs,
        random_state=args.seed,
    ).fit(x_train, y_train)


def fit_lightgbm(deps: dict[str, Any], x_train: np.ndarray, y_train: np.ndarray, args: argparse.Namespace) -> Any:
    lgbm_cls = deps["LGBMClassifier"]
    return lgbm_cls(
        n_estimators=args.lgbm_n_estimators,
        max_depth=args.lgbm_max_depth,
        num_leaves=args.lgbm_num_leaves,
        learning_rate=args.lgbm_learning_rate,
        subsample=args.lgbm_subsample,
        colsample_bytree=args.lgbm_colsample_bytree,
        min_child_samples=args.lgbm_min_child_samples,
        objective="binary",
        scale_pos_weight=class_scale_pos_weight(y_train),
        n_jobs=args.n_jobs,
        random_state=args.seed,
        verbosity=-1,
    ).fit(x_train, y_train)


def save_model(model: Any, path: Path, deps: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib = deps.get("joblib")
    if joblib is not None:
        joblib.dump(model, path)
    else:
        with path.open("wb") as f:
            pickle.dump(model, f)


def save_feature_importance(model: Any, feature_names: list[str], path: Path) -> None:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    else:
        values = np.full(len(feature_names), math.nan)

    rows = sorted(
        [
            {
                "feature": feature,
                "importance": float(importance),
            }
            for feature, importance in zip(feature_names, values)
        ],
        key=lambda row: -1 if math.isnan(row["importance"]) else -row["importance"],
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "importance"])
        writer.writeheader()
        writer.writerows(rows)


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train_tabular_model(
    model_name: str,
    deps: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, float], dict[str, float]]:
    start = perf_counter()
    if model_name == "random_forest":
        model = fit_random_forest(deps, x_train, y_train, args)
    elif model_name == "xgboost":
        model = fit_xgboost(deps, x_train, y_train, args)
    elif model_name == "lightgbm":
        model = fit_lightgbm(deps, x_train, y_train, args)
    else:
        raise ValueError(f"Unsupported tabular model: {model_name}")

    train_probs = predict_probabilities(model, x_train)
    val_probs = predict_probabilities(model, x_val)
    train_metrics = binary_metrics(y_train, train_probs)
    val_metrics = binary_metrics(y_val, val_probs)
    val_metrics["fit_seconds"] = perf_counter() - start
    return model, train_metrics, val_metrics


def run_sequence_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    grad_clip: float,
    max_batches: int | None,
) -> tuple[float, dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    n_samples = 0
    all_probs = []
    all_targets = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, (batch_x, batch_y) in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break

            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            if is_train:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            batch_size = int(batch_y.shape[0])
            n_samples += batch_size
            total_loss += float(loss.detach().item()) * batch_size
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_targets.append(batch_y.detach().cpu().numpy())

    avg_loss = total_loss / max(n_samples, 1)
    if not all_probs:
        return avg_loss, {
            "accuracy": math.nan,
            "precision": math.nan,
            "recall": math.nan,
            "specificity": math.nan,
            "f1": math.nan,
            "auroc": math.nan,
            "auprc": math.nan,
            "brier": math.nan,
            "log_loss": math.nan,
        }

    probs = np.concatenate(all_probs).astype(np.float32)
    targets = np.concatenate(all_targets).astype(np.float32)
    return avg_loss, binary_metrics(targets, probs)


def collect_sequence_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            logits = model(batch_x.to(device, non_blocking=True))
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_targets.append(batch_y.detach().cpu().numpy())
    if not all_probs:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    return (
        np.concatenate(all_targets).astype(np.float32),
        np.concatenate(all_probs).astype(np.float32),
    )


def train_sequence_model(
    model_name: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    input_dim: int,
    device: torch.device,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[nn.Module, dict[str, float], dict[str, float], dict[str, float]]:
    start = perf_counter()
    model = RecurrentRiskModel(
        input_dim=input_dim,
        hidden_dim=args.rnn_hidden_size,
        num_layers=args.rnn_num_layers,
        dropout=args.rnn_dropout,
        rnn_type=model_name,
    ).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.sequence_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.sequence_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.sequence_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    pos, neg = dataset_label_counts(train_dataset)
    pos_weight = neg / max(pos, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.rnn_learning_rate,
        weight_decay=args.rnn_weight_decay,
    )

    metrics_path = output_dir / "epoch_metrics.csv"
    best_auc = -math.inf
    best_loss = math.inf
    best_state = None
    best_train_metrics: dict[str, float] = {}
    best_val_metrics: dict[str, float] = {}

    for epoch in range(1, args.sequence_epochs + 1):
        train_loss, train_metrics = run_sequence_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            grad_clip=args.rnn_grad_clip,
            max_batches=args.limit_train_batches,
        )
        val_loss, val_metrics = run_sequence_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            grad_clip=0.0,
            max_batches=args.limit_val_batches,
        )

        if math.isnan(val_metrics["auroc"]):
            improved = val_loss < best_loss
        else:
            improved = val_metrics["auroc"] > best_auc

        if improved:
            best_auc = val_metrics["auroc"] if not math.isnan(val_metrics["auroc"]) else best_auc
            best_loss = val_loss
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            best_train_metrics = train_metrics
            best_val_metrics = val_metrics

        append_metrics(
            metrics_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_auroc": train_metrics["auroc"],
                "train_auprc": train_metrics["auprc"],
                "val_loss": val_loss,
                "val_auroc": val_metrics["auroc"],
                "val_auprc": val_metrics["auprc"],
                "val_f1": val_metrics["f1"],
            },
        )
        print(
            f"{model_name} epoch {epoch:03d}/{args.sequence_epochs} | "
            f"train loss {train_loss:.4f} auroc {train_metrics['auroc']:.4f} | "
            f"val loss {val_loss:.4f} auroc {val_metrics['auroc']:.4f} auprc {val_metrics['auprc']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_metrics = run_sequence_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        grad_clip=0.0,
        max_batches=args.limit_test_batches,
    )
    test_metrics["loss"] = test_loss

    if args.save_predictions:
        protocol_sha = json.loads(Path(args.comparison_protocol).read_text(encoding="utf-8"))[
            "protocol_sha256"
        ]
        y_true, y_prob = collect_sequence_predictions(
            model=model,
            loader=val_loader,
            device=device,
            max_batches=args.limit_val_batches,
        )
        val_predictions = sequence_prediction_metadata(val_dataset).iloc[: len(y_true)].copy()
        val_predictions = val_predictions.assign(
            y_true=y_true,
            y_prob=y_prob,
            model=model_name,
            feature_set="sequence_raw",
            target_col=args.target_col,
            horizon_hours=horizon_from_target_col(args.target_col),
            evaluation_split="validation",
            comparison_mode=args.comparison_mode,
            protocol_sha256=protocol_sha,
        )
        val_predictions.to_csv(
            output_dir / "val_predictions.csv.gz", index=False, compression="gzip"
        )
        y_true, y_prob = collect_sequence_predictions(
            model=model,
            loader=test_loader,
            device=device,
            max_batches=args.limit_test_batches,
        )
        test_predictions = sequence_prediction_metadata(test_dataset).iloc[: len(y_true)].copy()
        test_predictions = test_predictions.assign(
            y_true=y_true,
            y_prob=y_prob,
            model=model_name,
            feature_set="sequence_raw",
            target_col=args.target_col,
            horizon_hours=horizon_from_target_col(args.target_col),
            evaluation_split="test",
            comparison_mode=args.comparison_mode,
            protocol_sha256=protocol_sha,
        )
        test_predictions.to_csv(
            output_dir / "test_predictions.csv.gz", index=False, compression="gzip"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "input_dim": input_dim,
            "hidden_size": args.rnn_hidden_size,
            "num_layers": args.rnn_num_layers,
            "dropout": args.rnn_dropout,
            "feature_order": FEATURE_ORDER,
            "args": vars(args),
        },
        output_dir / "best_model.pt",
    )

    best_val_metrics["fit_seconds"] = perf_counter() - start
    test_metrics["fit_seconds"] = best_val_metrics["fit_seconds"]
    return model, best_train_metrics, best_val_metrics, test_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run black-box ML baselines.")
    parser.add_argument("--csv", default=PRIMARY_HOURLY_FEATURES)
    parser.add_argument("--sofa-csv", default=None, help="若 CSV 缺 SOFA label，可指定 sofa_scores_hourly.csv。")
    parser.add_argument("--target-col", default="label_sofa_increase_ge2_6h")
    parser.add_argument("--time-col", default="sofa_hour")
    parser.add_argument("--split-col", default="subject_id")
    parser.add_argument("--split-manifest", default=PATIENT_SPLIT_CSV)
    parser.add_argument("--comparison-mode", choices=["full", "equal_sample"], default="full")
    parser.add_argument("--comparison-protocol", default=COMPARISON_PROTOCOL_JSON)
    parser.add_argument("--equal-sample-windows", default=EQUAL_SAMPLE_WINDOWS_CSV)
    parser.add_argument("--allow-incomplete-cohort", action="store_true", help="僅限 smoke test。")
    parser.add_argument(
        "--feature-set",
        choices=["protocol", "static", "temporal", "matched24", "compare"],
        default="protocol",
        help="Tabular models only: temporal includes rolling mean/min/max/std/slope.",
    )
    parser.add_argument("--models", default="all", help="all or comma list: rf,xgboost,lightgbm,lstm,gru")
    parser.add_argument(
        "--sequence-feature-set",
        choices=["raw", "matched"],
        default="raw",
        help="matched uses raw + missingness + time-since channels (39 hourly inputs).",
    )
    parser.add_argument("--seq-length", type=int, default=24)
    parser.add_argument("--min-history-hours", type=int, default=24)
    parser.add_argument("--val-frac", type=float, default=0.15, help="舊版相容參數；正式比例由 manifest 決定。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-stays", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--max-train-samples", type=int, default=None, help="僅供 smoke test。")
    parser.add_argument("--max-val-samples", type=int, default=None, help="僅供 smoke test。")
    parser.add_argument("--max-test-samples", type=int, default=None, help="預設評估完整 test set。")
    parser.add_argument("--max-train-windows", type=int, default=None, help="僅供 smoke test。")
    parser.add_argument("--max-val-windows", type=int, default=None, help="僅供 smoke test。")
    parser.add_argument("--max-test-windows", type=int, default=None, help="預設評估完整 test set。")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--output-dir", default=None)

    parser.add_argument("--rf-n-estimators", type=int, default=300)
    parser.add_argument("--rf-max-depth", type=int, default=12)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=50)
    parser.add_argument("--rf-max-features", default="sqrt")

    parser.add_argument("--xgb-n-estimators", type=int, default=500)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.03)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--xgb-tree-method", default="hist")

    parser.add_argument("--lgbm-n-estimators", type=int, default=500)
    parser.add_argument("--lgbm-max-depth", type=int, default=-1)
    parser.add_argument("--lgbm-num-leaves", type=int, default=31)
    parser.add_argument("--lgbm-learning-rate", type=float, default=0.03)
    parser.add_argument("--lgbm-subsample", type=float, default=0.8)
    parser.add_argument("--lgbm-colsample-bytree", type=float, default=0.8)
    parser.add_argument("--lgbm-min-child-samples", type=int, default=50)

    parser.add_argument("--sequence-epochs", type=int, default=10)
    parser.add_argument("--sequence-batch-size", type=int, default=128)
    parser.add_argument("--rnn-hidden-size", type=int, default=64)
    parser.add_argument("--rnn-num-layers", type=int, default=1)
    parser.add_argument("--rnn-dropout", type=float, default=0.2)
    parser.add_argument("--rnn-learning-rate", type=float, default=1e-3)
    parser.add_argument("--rnn-weight-decay", type=float, default=1e-5)
    parser.add_argument("--rnn-grad-clip", type=float, default=5.0)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    protocol = validate_comparison_args(
        args.comparison_mode,
        args.comparison_protocol,
        args.target_col,
        args.seq_length,
    )
    if args.feature_set not in {"protocol", "matched24"}:
        raise ValueError("正式公平比較固定使用 protocol 的 13 個 predictors。")
    if args.feature_set == "matched24" and args.sequence_feature_set != "matched":
        raise ValueError("matched24 tree baselines require --sequence-feature-set matched.")
    if args.min_history_hours != args.seq_length:
        raise ValueError("min-history-hours 必須與 24h comparison window 相同。")
    debug_limits = [
        args.max_rows, args.max_stays, args.max_train_samples, args.max_val_samples,
        args.max_test_samples, args.max_train_windows, args.max_val_windows, args.max_test_windows,
    ]
    if not args.allow_incomplete_cohort and any(value is not None for value in debug_limits):
        raise ValueError("正式比較不可額外限制 rows/stays/samples/windows。")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find CSV: {csv_path}")

    selected_models = parse_model_names(args.models)
    deps = import_optional_dependencies()
    models_to_run = runnable_models(selected_models, deps)
    if not models_to_run:
        hint = dependency_hint(selected_models, deps)
        if hint:
            print("No selected black-box baseline can run because dependencies are missing.")
            print(f"Install command: {hint}")
        return

    run_name = datetime.now().strftime("blackbox_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "blackbox_baselines" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = read_csv_header(csv_path)
    tabular_feature_sets = build_tabular_feature_sets(columns, args.feature_set)
    sequence_features = build_sequence_features(columns, args.sequence_feature_set)

    all_feature_cols = list(
        dict.fromkeys(
            [
                *sequence_features,
                *[feature for feature_cols in tabular_feature_sets.values() for feature in feature_cols],
            ]
        )
    )
    usecols = list(dict.fromkeys(["stay_id", args.time_col, args.split_col, args.target_col, *all_feature_cols]))
    existing_usecols, missing_usecols = maybe_existing_usecols(csv_path, usecols)
    missing_required = [col for col in missing_usecols if col != args.target_col]
    if missing_required:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_required)}")
    horizon_hours = horizon_from_target_col(args.target_col)

    print(f"Reading data: {csv_path}")
    print(f"Models: {', '.join(models_to_run)}")
    print(f"Tabular feature sets: {', '.join(tabular_feature_sets)}")
    print(f"Selected feature columns: {len(all_feature_cols)}")

    load_start = perf_counter()
    df = load_baseline_frame(
        csv_path=csv_path,
        usecols=existing_usecols,
        max_rows=args.max_rows,
        max_stays=args.max_stays,
        chunk_size=args.chunk_size,
    )
    df = merge_sofa_targets(
        df=df,
        target_cols=[args.target_col],
        split_col=args.split_col,
        time_col=args.time_col,
        sofa_csv=args.sofa_csv,
    )
    print(f"Loaded rows: {len(df):,}; stays: {df['stay_id'].nunique():,}; seconds: {perf_counter() - load_start:.1f}")

    df = prepare_frame(
        df=df,
        feature_cols=all_feature_cols,
        target_col=args.target_col,
        time_col=args.time_col,
        min_history_hours=args.min_history_hours,
    )
    if args.split_col != "subject_id":
        raise ValueError("正式 baseline 實驗必須以 subject_id 做 patient-level split。")
    df = attach_split(df, args.split_manifest, patient_col=args.split_col)
    train_values = set(df.loc[df["dataset_split"] == "train", args.split_col].unique().tolist())
    val_values = set(df.loc[df["dataset_split"] == "validation", args.split_col].unique().tolist())
    test_values = set(df.loc[df["dataset_split"] == "test", args.split_col].unique().tolist())
    train_df = filter_frame_to_comparison_windows(
        df.loc[df["dataset_split"] == "train"], args.target_col, args.time_col, "train",
        args.comparison_mode, args.equal_sample_windows, args.seq_length,
        require_all_window_ids=not args.allow_incomplete_cohort,
    )
    val_df = filter_frame_to_comparison_windows(
        df.loc[df["dataset_split"] == "validation"], args.target_col, args.time_col, "validation",
        args.comparison_mode, args.equal_sample_windows, args.seq_length,
        require_all_window_ids=not args.allow_incomplete_cohort,
    )
    test_df = filter_frame_to_comparison_windows(
        df.loc[df["dataset_split"] == "test"], args.target_col, args.time_col, "test",
        args.comparison_mode, args.equal_sample_windows, args.seq_length,
        require_all_window_ids=not args.allow_incomplete_cohort,
    )
    hourly_df = df

    cohort_records = [
        cohort_record(train_df["stay_id"], train_df[args.time_col], train_df[args.target_col], "train", args.target_col),
        cohort_record(val_df["stay_id"], val_df[args.time_col], val_df[args.target_col], "validation", args.target_col),
        cohort_record(test_df["stay_id"], test_df[args.time_col], test_df[args.target_col], "test", args.target_col),
    ]
    validate_cohort_records(
        cohort_records, protocol, args.comparison_mode, allow_incomplete=args.allow_incomplete_cohort
    )
    write_cohort_audit(output_dir / "cohort_audit.json", cohort_records)

    train_df, val_df, test_df, fill_values = fill_remaining_with_train_medians(
        train_df, val_df, test_df, all_feature_cols
    )
    print(
        f"Train rows: {len(train_df):,}; positive rate: {train_df[args.target_col].mean():.4f} | "
        f"Val rows: {len(val_df):,}; positive rate: {val_df[args.target_col].mean():.4f} | "
        f"Test rows: {len(test_df):,}; positive rate: {test_df[args.target_col].mean():.4f}"
    )

    with (output_dir / "experiment_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "models_to_run": models_to_run,
                "tabular_feature_sets": tabular_feature_sets,
                "sequence_features": sequence_features,
                "fill_values": fill_values,
                "comparison_protocol_sha256": protocol["protocol_sha256"],
                "cohort_audit": cohort_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    metrics_path = output_dir / "blackbox_metrics.csv"

    tabular_models = [model_name for model_name in models_to_run if model_name in TABULAR_MODELS]
    matched_tree_models = tabular_models if args.feature_set == "matched24" else []
    if matched_tree_models:
        tabular_models = []
    if tabular_models:
        sampled_train_df = train_df
        sampled_val_df = val_df
        sampled_test_df = test_df
        if args.allow_incomplete_cohort:
            sampled_train_df = stratified_sample(train_df, args.target_col, args.max_train_samples, args.seed)
            sampled_val_df = stratified_sample(val_df, args.target_col, args.max_val_samples, args.seed + 10)
            sampled_test_df = stratified_sample(test_df, args.target_col, args.max_test_samples, args.seed + 20)
        y_train = sampled_train_df[args.target_col].to_numpy(dtype=np.float32)
        y_val = sampled_val_df[args.target_col].to_numpy(dtype=np.float32)
        y_test = sampled_test_df[args.target_col].to_numpy(dtype=np.float32)
        print(
            f"Tabular train rows: {len(sampled_train_df):,}; val rows: {len(sampled_val_df):,}; "
            f"test rows: {len(sampled_test_df):,}"
        )

        for feature_set_name, feature_cols in tabular_feature_sets.items():
            print(f"\n=== Tabular feature set: {feature_set_name} ({len(feature_cols)} features) ===")
            x_train = sampled_train_df[feature_cols].to_numpy(dtype=np.float32, copy=True)
            x_val = sampled_val_df[feature_cols].to_numpy(dtype=np.float32, copy=True)
            x_test = sampled_test_df[feature_cols].to_numpy(dtype=np.float32, copy=True)

            for model_name in tabular_models:
                print(f"Training {model_name}...")
                model_dir = output_dir / feature_set_name / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                model, train_metrics, val_metrics = train_tabular_model(
                    model_name=model_name,
                    deps=deps,
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    args=args,
                )
                save_model(model, model_dir / "model.joblib", deps)
                save_feature_importance(model, feature_cols, model_dir / "feature_importance.csv")
                test_probs = predict_probabilities(model, x_test)
                test_metrics = binary_metrics(y_test, test_probs)

                if args.save_predictions:
                    id_cols = [args.split_col, "stay_id", args.time_col]
                    val_predictions = sampled_val_df[id_cols].reset_index(drop=True).copy()
                    val_predictions = val_predictions.assign(
                        y_true=y_val,
                        y_prob=predict_probabilities(model, x_val),
                        model=model_name,
                        feature_set=feature_set_name,
                        target_col=args.target_col,
                        horizon_hours=horizon_hours,
                        evaluation_split="validation",
                        comparison_mode=args.comparison_mode,
                        protocol_sha256=protocol["protocol_sha256"],
                    )
                    val_predictions.to_csv(
                        model_dir / "val_predictions.csv.gz", index=False, compression="gzip"
                    )
                    test_predictions = sampled_test_df[id_cols].reset_index(drop=True).copy()
                    test_predictions = test_predictions.assign(
                        y_true=y_test,
                        y_prob=test_probs,
                        model=model_name,
                        feature_set=feature_set_name,
                        target_col=args.target_col,
                        horizon_hours=horizon_hours,
                        evaluation_split="test",
                        comparison_mode=args.comparison_mode,
                        protocol_sha256=protocol["protocol_sha256"],
                    )
                    test_predictions.to_csv(
                        model_dir / "test_predictions.csv.gz", index=False, compression="gzip"
                    )

                append_metrics(
                    metrics_path,
                    {
                        "comparison_mode": args.comparison_mode,
                        "protocol_sha256": protocol["protocol_sha256"],
                        "target_col": args.target_col,
                        "horizon_hours": horizon_hours,
                        "feature_set": feature_set_name,
                        "model": model_name,
                        "n_train": len(sampled_train_df),
                        "n_val": len(sampled_val_df),
                        "n_test": len(sampled_test_df),
                        "n_features": len(feature_cols),
                        "train_auroc": train_metrics["auroc"],
                        "train_auprc": train_metrics["auprc"],
                        "train_log_loss": train_metrics["log_loss"],
                        "val_auroc": val_metrics["auroc"],
                        "val_auprc": val_metrics["auprc"],
                        "val_accuracy": val_metrics["accuracy"],
                        "val_precision": val_metrics["precision"],
                        "val_recall": val_metrics["recall"],
                        "val_specificity": val_metrics["specificity"],
                        "val_f1": val_metrics["f1"],
                        "val_brier": val_metrics["brier"],
                        "val_log_loss": val_metrics["log_loss"],
                        "test_auroc": test_metrics["auroc"],
                        "test_auprc": test_metrics["auprc"],
                        "test_accuracy": test_metrics["accuracy"],
                        "test_precision": test_metrics["precision"],
                        "test_recall": test_metrics["recall"],
                        "test_specificity": test_metrics["specificity"],
                        "test_f1": test_metrics["f1"],
                        "test_brier": test_metrics["brier"],
                        "test_log_loss": test_metrics["log_loss"],
                        "fit_seconds": val_metrics["fit_seconds"],
                    },
                )
                print(
                    f"{model_name} | test AUROC {test_metrics['auroc']:.4f} | "
                    f"AUPRC {test_metrics['auprc']:.4f} | F1 {test_metrics['f1']:.4f} | "
                    f"seconds {val_metrics['fit_seconds']:.1f}"
                )

    sequence_models = [model_name for model_name in models_to_run if model_name in SEQUENCE_MODELS]
    if sequence_models or matched_tree_models:
        sequence_feature_name = f"sequence_{args.sequence_feature_set}"
        print(
            f"\n=== Sequence feature set: {sequence_feature_name} "
            f"({len(sequence_features)} features, {args.seq_length}h) ==="
        )
        device = choose_device(args.device)
        print(f"Sequence device: {device}")

        seq_df = hourly_df
        seq_df = seq_df.sort_values(["stay_id", args.time_col], kind="mergesort").reset_index(drop=True)
        seq_features = seq_df[sequence_features].to_numpy(dtype=np.float32, copy=True)
        seq_labels = seq_df[args.target_col].to_numpy(dtype=np.float32, copy=True)
        seq_stay_ids = seq_df["stay_id"].to_numpy(copy=True)
        seq_split_values = seq_df[args.split_col].to_numpy(copy=True)
        seq_time_values = seq_df[args.time_col].to_numpy(dtype=np.int64, copy=True)
        seq_features, scaling = standardize_arrays(seq_features, seq_split_values, train_values)
        train_window_ids = window_ids_for_mode(
            args.comparison_mode, args.equal_sample_windows, args.target_col, "train"
        )
        val_window_ids = window_ids_for_mode(
            args.comparison_mode, args.equal_sample_windows, args.target_col, "validation"
        )

        base_train_dataset = ICUSequenceDataset(
            features=seq_features,
            labels=seq_labels,
            stay_ids=seq_stay_ids,
            split_values=seq_split_values,
            time_values=seq_time_values,
            allowed_split_values=train_values,
            seq_length=args.seq_length,
            allowed_window_ids=train_window_ids,
            require_all_window_ids=not args.allow_incomplete_cohort,
        )
        base_val_dataset = ICUSequenceDataset(
            features=seq_features,
            labels=seq_labels,
            stay_ids=seq_stay_ids,
            split_values=seq_split_values,
            time_values=seq_time_values,
            allowed_split_values=val_values,
            seq_length=args.seq_length,
            allowed_window_ids=val_window_ids,
            require_all_window_ids=not args.allow_incomplete_cohort,
        )
        base_test_dataset = ICUSequenceDataset(
            features=seq_features,
            labels=seq_labels,
            stay_ids=seq_stay_ids,
            split_values=seq_split_values,
            time_values=seq_time_values,
            allowed_split_values=test_values,
            seq_length=args.seq_length,
        )
        sequence_cohort_records = [
            base_train_dataset.cohort_record("train", args.target_col),
            base_val_dataset.cohort_record("validation", args.target_col),
            base_test_dataset.cohort_record("test", args.target_col),
        ]
        validate_cohort_records(
            sequence_cohort_records,
            protocol,
            args.comparison_mode,
            allow_incomplete=args.allow_incomplete_cohort,
        )
        write_cohort_audit(output_dir / "sequence_cohort_audit.json", sequence_cohort_records)
        train_sequence_dataset = maybe_subset_dataset(base_train_dataset, args.max_train_windows, args.seed)
        val_sequence_dataset = maybe_subset_dataset(base_val_dataset, args.max_val_windows, args.seed + 20)
        test_sequence_dataset = maybe_subset_dataset(base_test_dataset, args.max_test_windows, args.seed + 30)

        print(
            f"Sequence train windows: {len(train_sequence_dataset):,}; "
            f"val windows: {len(val_sequence_dataset):,}; test windows: {len(test_sequence_dataset):,}"
        )

        if len(train_sequence_dataset) == 0 or len(val_sequence_dataset) == 0 or len(test_sequence_dataset) == 0:
            raise ValueError("Sequence train, validation, or test set has no usable windows.")

        if matched_tree_models:
            feature_set_name = "sequence_matched_summary"
            feature_names = matched_tree_feature_names(sequence_features)
            print(f"Building {len(feature_names)} feature-matched tree summaries...")
            x_train, y_train = summarize_sequence_for_trees(
                train_sequence_dataset, len(FEATURE_ORDER)
            )
            x_val, y_val = summarize_sequence_for_trees(
                val_sequence_dataset, len(FEATURE_ORDER)
            )
            x_test, y_test = summarize_sequence_for_trees(
                test_sequence_dataset, len(FEATURE_ORDER)
            )
            val_metadata = sequence_prediction_metadata(val_sequence_dataset)
            test_metadata = sequence_prediction_metadata(test_sequence_dataset)

            for model_name in matched_tree_models:
                print(f"Training feature-matched {model_name}...")
                model_dir = output_dir / feature_set_name / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                model, train_metrics, val_metrics = train_tabular_model(
                    model_name=model_name,
                    deps=deps,
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                    args=args,
                )
                save_model(model, model_dir / "model.joblib", deps)
                save_feature_importance(model, feature_names, model_dir / "feature_importance.csv")
                val_probs = predict_probabilities(model, x_val)
                test_probs = predict_probabilities(model, x_test)
                test_metrics = binary_metrics(y_test, test_probs)

                if args.save_predictions:
                    for metadata, targets, probabilities, split_name, path in (
                        (val_metadata, y_val, val_probs, "validation", model_dir / "val_predictions.csv.gz"),
                        (test_metadata, y_test, test_probs, "test", model_dir / "test_predictions.csv.gz"),
                    ):
                        predictions = metadata.copy().assign(
                            y_true=targets,
                            y_prob=probabilities,
                            model=model_name,
                            feature_set=feature_set_name,
                            target_col=args.target_col,
                            horizon_hours=horizon_hours,
                            evaluation_split=split_name,
                            comparison_mode=args.comparison_mode,
                            protocol_sha256=protocol["protocol_sha256"],
                        )
                        predictions.to_csv(path, index=False, compression="gzip")

                append_metrics(
                    metrics_path,
                    {
                        "comparison_mode": args.comparison_mode,
                        "protocol_sha256": protocol["protocol_sha256"],
                        "target_col": args.target_col,
                        "horizon_hours": horizon_hours,
                        "feature_set": feature_set_name,
                        "model": model_name,
                        "n_train": len(y_train),
                        "n_val": len(y_val),
                        "n_test": len(y_test),
                        "n_features": len(feature_names),
                        "train_auroc": train_metrics["auroc"],
                        "train_auprc": train_metrics["auprc"],
                        "train_log_loss": train_metrics["log_loss"],
                        "val_auroc": val_metrics["auroc"],
                        "val_auprc": val_metrics["auprc"],
                        "val_accuracy": val_metrics["accuracy"],
                        "val_precision": val_metrics["precision"],
                        "val_recall": val_metrics["recall"],
                        "val_specificity": val_metrics["specificity"],
                        "val_f1": val_metrics["f1"],
                        "val_brier": val_metrics["brier"],
                        "val_log_loss": val_metrics["log_loss"],
                        "test_auroc": test_metrics["auroc"],
                        "test_auprc": test_metrics["auprc"],
                        "test_accuracy": test_metrics["accuracy"],
                        "test_precision": test_metrics["precision"],
                        "test_recall": test_metrics["recall"],
                        "test_specificity": test_metrics["specificity"],
                        "test_f1": test_metrics["f1"],
                        "test_brier": test_metrics["brier"],
                        "test_log_loss": test_metrics["log_loss"],
                        "fit_seconds": val_metrics["fit_seconds"],
                    },
                )
                print(
                    f"{model_name} matched | test AUROC {test_metrics['auroc']:.4f} | "
                    f"AUPRC {test_metrics['auprc']:.4f}"
                )
            del x_train, x_val, x_test, y_train, y_val, y_test, val_probs, test_probs, model

        for model_name in sequence_models:
            print(f"Training {model_name}...")
            model_dir = output_dir / sequence_feature_name / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            with (model_dir / "sequence_scaling.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "feature_order": sequence_features,
                        "scaling": scaling,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            _, train_metrics, val_metrics, test_metrics = train_sequence_model(
                model_name=model_name,
                train_dataset=train_sequence_dataset,
                val_dataset=val_sequence_dataset,
                test_dataset=test_sequence_dataset,
                input_dim=len(sequence_features),
                device=device,
                output_dir=model_dir,
                args=args,
            )
            append_metrics(
                metrics_path,
                {
                    "comparison_mode": args.comparison_mode,
                    "protocol_sha256": protocol["protocol_sha256"],
                    "target_col": args.target_col,
                    "horizon_hours": horizon_hours,
                    "feature_set": sequence_feature_name,
                    "model": model_name,
                    "n_train": len(train_sequence_dataset),
                    "n_val": len(val_sequence_dataset),
                    "n_test": len(test_sequence_dataset),
                    "n_features": len(sequence_features),
                    "train_auroc": train_metrics["auroc"],
                    "train_auprc": train_metrics["auprc"],
                    "train_log_loss": train_metrics["log_loss"],
                    "val_auroc": val_metrics["auroc"],
                    "val_auprc": val_metrics["auprc"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_precision": val_metrics["precision"],
                    "val_recall": val_metrics["recall"],
                    "val_specificity": val_metrics["specificity"],
                    "val_f1": val_metrics["f1"],
                    "val_brier": val_metrics["brier"],
                    "val_log_loss": val_metrics["log_loss"],
                    "test_auroc": test_metrics["auroc"],
                    "test_auprc": test_metrics["auprc"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_precision": test_metrics["precision"],
                    "test_recall": test_metrics["recall"],
                    "test_specificity": test_metrics["specificity"],
                    "test_f1": test_metrics["f1"],
                    "test_brier": test_metrics["brier"],
                    "test_log_loss": test_metrics["log_loss"],
                    "fit_seconds": val_metrics["fit_seconds"],
                },
            )
            print(
                f"{model_name} | test AUROC {test_metrics['auroc']:.4f} | "
                f"AUPRC {test_metrics['auprc']:.4f} | F1 {test_metrics['f1']:.4f} | "
                f"seconds {val_metrics['fit_seconds']:.1f}"
            )

    print(f"\nFinished. Metrics: {metrics_path}")
    print(f"Artifacts: {output_dir}")
    try_generate(
        generate_metric_comparison_figures,
        metrics_path,
        output_dir,
        "Black-box Machine Learning Baselines",
    )
    for epoch_metrics_path in output_dir.rglob("epoch_metrics.csv"):
        try_generate(generate_training_figures, epoch_metrics_path, epoch_metrics_path.parent)


if __name__ == "__main__":
    main()
