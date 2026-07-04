"""Interpretable baseline experiments for ICU deterioration prediction.

Models included:
1. Logistic Regression
2. Decision Tree
3. Generalized Additive Model (pyGAM)
4. Explainable Boosting Machine (InterpretML EBM)

The script is designed to compare traditional interpretable models against the
Knowledge-Guided Temporal FNN. It uses the same SOFA deterioration label and
the shared patient-level train/validation/test manifest.
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

from anfis_model import FEATURE_ORDER
from comparison_protocol import (
    cohort_record,
    filter_frame_to_comparison_windows,
    validate_cohort_records,
    validate_comparison_args,
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
from paper_figures import generate_metric_comparison_figures, try_generate


class StandardizedModel:
    """Small wrapper for models that need standardized numeric inputs."""

    def __init__(self, model: Any, mean: np.ndarray, scale: np.ndarray) -> None:
        self.model = model
        self.mean = mean.astype(np.float32)
        self.scale = scale.astype(np.float32)

    @classmethod
    def fit_transform(cls, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.mean(x, axis=0).astype(np.float32)
        scale = np.std(x, axis=0).astype(np.float32)
        scale[scale == 0] = 1.0
        return ((x - mean) / scale).astype(np.float32), mean, scale

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale).astype(np.float32)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.transform(x))


MODEL_ALIASES = {
    "lr": "logistic_regression",
    "logistic": "logistic_regression",
    "logistic_regression": "logistic_regression",
    "decision_tree": "decision_tree",
    "tree": "decision_tree",
    "dt": "decision_tree",
    "gam": "gam",
    "generalized_additive_model": "gam",
    "ebm": "ebm",
    "explainable_boosting_machine": "ebm",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_model_names(model_arg: str) -> list[str]:
    if model_arg.strip().lower() == "all":
        return ["logistic_regression", "decision_tree", "gam", "ebm"]

    models = []
    for raw_name in model_arg.split(","):
        key = raw_name.strip().lower()
        if not key:
            continue
        if key not in MODEL_ALIASES:
            raise ValueError(f"Unknown model name: {raw_name}")
        canonical_name = MODEL_ALIASES[key]
        if canonical_name not in models:
            models.append(canonical_name)
    if not models:
        raise ValueError("No baseline model was selected.")
    return models


def import_optional_dependencies() -> dict[str, Any]:
    deps: dict[str, Any] = {}

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier
    except ImportError as exc:
        deps["sklearn_error"] = exc
    else:
        deps["LogisticRegression"] = LogisticRegression
        deps["Pipeline"] = Pipeline
        deps["StandardScaler"] = StandardScaler
        deps["DecisionTreeClassifier"] = DecisionTreeClassifier

    try:
        from pygam import LogisticGAM, s
    except ImportError as exc:
        deps["pygam_error"] = exc
    else:
        deps["LogisticGAM"] = LogisticGAM
        deps["spline_term"] = s

    try:
        from interpret.glassbox import ExplainableBoostingClassifier
    except ImportError as exc:
        deps["interpret_error"] = exc
    else:
        deps["ExplainableBoostingClassifier"] = ExplainableBoostingClassifier

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
        if model_name in {"logistic_regression", "decision_tree"} and "sklearn_error" in deps:
            print(f"Skip {model_name}: scikit-learn is not installed.")
            continue
        if model_name == "gam" and "pygam_error" in deps:
            print("Skip gam: pyGAM is not installed.")
            continue
        if model_name == "ebm" and "interpret_error" in deps:
            print("Skip ebm: interpret is not installed.")
            continue
        runnable.append(model_name)
    return runnable


def dependency_hint(selected_models: list[str], deps: dict[str, Any]) -> str | None:
    packages = []
    if any(name in selected_models for name in ["logistic_regression", "decision_tree"]):
        if "sklearn_error" in deps:
            packages.append("scikit-learn")
    if "gam" in selected_models and "pygam_error" in deps:
        packages.append("pygam")
    if "ebm" in selected_models and "interpret_error" in deps:
        packages.append("interpret")

    if not packages:
        return None
    unique_packages = list(dict.fromkeys(packages))
    return ".\\env\\Scripts\\python.exe -m pip install " + " ".join(unique_packages)


def read_csv_header(csv_path: Path) -> list[str]:
    return list(pd.read_csv(csv_path, nrows=0).columns)


def is_temporal_feature(column: str, base_features: list[str]) -> bool:
    return is_measurement_process_feature(column, base_features) or (
        temporal_feature_window(column, base_features) is not None
    )


def build_feature_sets(
    columns: list[str],
    feature_set_arg: str,
    include_current_sofa: bool,
) -> dict[str, list[str]]:
    static_features = [feature for feature in FEATURE_ORDER if feature in columns]
    temporal_features = [col for col in columns if is_temporal_feature(col, FEATURE_ORDER)]
    temporal_features = sorted(
        temporal_features,
        key=lambda col: (
            FEATURE_ORDER.index(col.split("_w")[0]) if col.split("_w")[0] in FEATURE_ORDER else 999,
            col,
        ),
    )

    if include_current_sofa and "sofa_score" in columns:
        static_features = [*static_features, "sofa_score"]

    available_sets = {
        "protocol": static_features,
        "static": static_features,
        "temporal": [*static_features, *temporal_features],
    }

    if feature_set_arg == "compare":
        return available_sets
    return {feature_set_arg: available_sets[feature_set_arg]}


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

    if column == "sofa_score":
        return 0.0
    return math.nan


def prepare_frame(
    df: pd.DataFrame,
    all_feature_cols: list[str],
    target_col: str,
    time_col: str,
    min_history_hours: int,
) -> pd.DataFrame:
    missing = {"stay_id", time_col, target_col, *all_feature_cols} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df.sort_values(["stay_id", time_col], kind="mergesort").reset_index(drop=True).copy()
    df["_history_index"] = df.groupby("stay_id", sort=False).cumcount()

    for col in [target_col, *all_feature_cols]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[all_feature_cols] = df[all_feature_cols].replace([np.inf, -np.inf], np.nan)

    # Forward fill only within the same ICU stay; this avoids using future values.
    df[all_feature_cols] = df.groupby("stay_id", sort=False)[all_feature_cols].ffill()

    deterministic_fill = {
        col: default_fill_value(col)
        for col in all_feature_cols
        if not math.isnan(default_fill_value(col))
    }
    df[all_feature_cols] = df[all_feature_cols].fillna(deterministic_fill)
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


def balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    y = y.astype(int)
    n_pos = max(int(np.sum(y == 1)), 1)
    n_neg = max(int(np.sum(y == 0)), 1)
    n_total = y.size
    weights = np.ones_like(y, dtype=np.float32)
    weights[y == 1] = n_total / (2.0 * n_pos)
    weights[y == 0] = n_total / (2.0 * n_neg)
    return weights


def select_features_by_signal(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    feature_cols: list[str],
    max_features: int | None,
) -> list[str]:
    if max_features is None or max_features <= 0 or len(feature_cols) <= max_features:
        return feature_cols

    protected = [col for col in feature_cols if col in FEATURE_ORDER or col == "sofa_score"]
    candidates = [col for col in feature_cols if col not in protected]
    scores = []
    y_std = np.std(y_train)

    for col in candidates:
        values = x_train[col].to_numpy(dtype=np.float64)
        if np.std(values) == 0 or y_std == 0:
            score = 0.0
        else:
            score = abs(float(np.corrcoef(values, y_train)[0, 1]))
            if math.isnan(score):
                score = 0.0
        scores.append((score, col))

    scores.sort(reverse=True)
    remaining_slots = max(max_features - len(protected), 0)
    selected = [*protected, *[col for _, col in scores[:remaining_slots]]]
    return selected[:max_features]


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


def predict_probabilities(model: Any, model_name: str, x: np.ndarray) -> np.ndarray:
    if model_name == "gam":
        probs = model.predict_proba(x)
        return np.asarray(probs, dtype=np.float32).reshape(-1)

    probs = model.predict_proba(x)
    probs = np.asarray(probs)
    if probs.ndim == 2:
        return probs[:, 1].astype(np.float32)
    return probs.reshape(-1).astype(np.float32)


def fit_logistic_regression(
    deps: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
) -> Any:
    pipeline_cls = deps["Pipeline"]
    scaler_cls = deps["StandardScaler"]
    logistic_cls = deps["LogisticRegression"]
    return pipeline_cls(
        [
            ("scaler", scaler_cls()),
            (
                "model",
                logistic_cls(
                    max_iter=args.lr_max_iter,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=args.seed,
                ),
            ),
        ]
    ).fit(x_train, y_train)


def fit_decision_tree(
    deps: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
) -> Any:
    tree_cls = deps["DecisionTreeClassifier"]
    return tree_cls(
        max_depth=args.tree_max_depth,
        min_samples_leaf=args.tree_min_samples_leaf,
        class_weight="balanced",
        random_state=args.seed,
    ).fit(x_train, y_train)


def fit_gam(
    deps: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
) -> Any:
    gam_cls = deps["LogisticGAM"]
    spline_term = deps["spline_term"]
    x_scaled, mean, scale = StandardizedModel.fit_transform(x_train)

    terms = spline_term(0, n_splines=args.gam_splines)
    for feature_index in range(1, x_scaled.shape[1]):
        terms += spline_term(feature_index, n_splines=args.gam_splines)

    gam = gam_cls(terms, lam=args.gam_lam, max_iter=args.gam_max_iter, verbose=False)
    if args.gam_use_sample_weights:
        gam.fit(x_scaled, y_train, weights=balanced_sample_weights(y_train))
    else:
        gam.fit(x_scaled, y_train)
    return StandardizedModel(gam, mean, scale)


def fit_ebm(
    deps: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    args: argparse.Namespace,
) -> Any:
    ebm_cls = deps["ExplainableBoostingClassifier"]
    kwargs = {
        "feature_names": feature_names,
        "interactions": args.ebm_interactions,
        "max_bins": args.ebm_max_bins,
        "learning_rate": args.ebm_learning_rate,
        "outer_bags": args.ebm_outer_bags,
        "validation_size": args.ebm_validation_size,
        "random_state": args.seed,
        "n_jobs": args.n_jobs,
    }

    try:
        ebm = ebm_cls(**kwargs, class_weight="balanced")
    except TypeError:
        ebm = ebm_cls(**kwargs)
    return ebm.fit(x_train, y_train)


def save_model(model: Any, path: Path, deps: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib = deps.get("joblib")
    if joblib is not None:
        joblib.dump(model, path)
    else:
        with path.open("wb") as f:
            pickle.dump(model, f)


def save_feature_report(
    model: Any,
    model_name: str,
    feature_names: list[str],
    path: Path,
) -> None:
    rows = []

    if model_name == "logistic_regression":
        estimator = model.named_steps["model"]
        coefficients = estimator.coef_.reshape(-1)
        rows = [
            {
                "feature": feature,
                "importance": abs(float(coef)),
                "coefficient": float(coef),
            }
            for feature, coef in zip(feature_names, coefficients)
        ]
    elif model_name == "decision_tree":
        rows = [
            {
                "feature": feature,
                "importance": float(importance),
                "coefficient": math.nan,
            }
            for feature, importance in zip(feature_names, model.feature_importances_)
        ]
    elif model_name == "ebm" and hasattr(model, "term_importances"):
        importances = np.asarray(model.term_importances(), dtype=float)
        term_names = getattr(model, "term_names_", None) or getattr(model, "term_names", None)
        if term_names is None:
            term_names = [f"term_{idx}" for idx in range(len(importances))]
        rows = [
            {
                "feature": str(term),
                "importance": float(importance),
                "coefficient": math.nan,
            }
            for term, importance in zip(term_names, importances)
        ]
    else:
        rows = [
            {
                "feature": feature,
                "importance": math.nan,
                "coefficient": math.nan,
            }
            for feature in feature_names
        ]

    rows = sorted(rows, key=lambda row: -1 if math.isnan(row["importance"]) else -row["importance"])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "importance", "coefficient"])
        writer.writeheader()
        writer.writerows(rows)


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def train_one_model(
    model_name: str,
    deps: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    args: argparse.Namespace,
) -> tuple[Any, dict[str, float], dict[str, float]]:
    start_time = perf_counter()

    if model_name == "logistic_regression":
        model = fit_logistic_regression(deps, x_train, y_train, args)
    elif model_name == "decision_tree":
        model = fit_decision_tree(deps, x_train, y_train, args)
    elif model_name == "gam":
        model = fit_gam(deps, x_train, y_train, args)
    elif model_name == "ebm":
        model = fit_ebm(deps, x_train, y_train, feature_names, args)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    train_probs = predict_probabilities(model, model_name, x_train)
    val_probs = predict_probabilities(model, model_name, x_val)
    train_metrics = binary_metrics(y_train, train_probs)
    val_metrics = binary_metrics(y_val, val_probs)
    val_metrics["fit_seconds"] = perf_counter() - start_time
    return model, train_metrics, val_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run interpretable ML baselines.")
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
        choices=["protocol", "static", "temporal", "compare"],
        default="protocol",
        help="static uses current-hour variables; temporal also includes rolling mean/min/max/std/slope.",
    )
    parser.add_argument("--include-current-sofa", action="store_true")
    parser.add_argument("--models", default="all", help="all or comma list: lr,tree,gam,ebm")
    parser.add_argument("--min-history-hours", type=int, default=24)
    parser.add_argument("--val-frac", type=float, default=0.15, help="舊版相容參數；正式比例由 manifest 決定。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-stays", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--max-train-samples", type=int, default=None, help="僅供 smoke test。")
    parser.add_argument("--max-val-samples", type=int, default=None, help="僅供 smoke test。")
    parser.add_argument("--max-test-samples", type=int, default=None, help="預設評估完整 test set。")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--lr-max-iter", type=int, default=1000)
    parser.add_argument("--tree-max-depth", type=int, default=5)
    parser.add_argument("--tree-min-samples-leaf", type=int, default=100)
    parser.add_argument("--gam-max-features", type=int, default=40)
    parser.add_argument("--gam-splines", type=int, default=8)
    parser.add_argument("--gam-max-iter", type=int, default=100)
    parser.add_argument("--gam-lam", type=float, default=1.0)
    parser.add_argument("--gam-use-sample-weights", action="store_true")
    parser.add_argument("--ebm-max-features", type=int, default=80)
    parser.add_argument("--ebm-interactions", type=int, default=10)
    parser.add_argument("--ebm-max-bins", type=int, default=256)
    parser.add_argument("--ebm-learning-rate", type=float, default=0.01)
    parser.add_argument("--ebm-outer-bags", type=int, default=4)
    parser.add_argument("--ebm-validation-size", type=float, default=0.15)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    protocol = validate_comparison_args(
        args.comparison_mode,
        args.comparison_protocol,
        args.target_col,
        args.min_history_hours,
    )
    if args.feature_set != "protocol" or args.include_current_sofa:
        raise ValueError("正式公平比較固定使用 protocol 的 13 個 predictors，不可加入 SOFA 或其他特徵。")
    if not args.allow_incomplete_cohort and any(
        value is not None
        for value in [args.max_rows, args.max_stays, args.max_train_samples, args.max_val_samples, args.max_test_samples]
    ):
        raise ValueError("正式比較不可額外限制 rows/stays/samples。")

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot find CSV: {csv_path}")

    selected_models = parse_model_names(args.models)
    deps = import_optional_dependencies()
    models_to_run = runnable_models(selected_models, deps)
    if not models_to_run:
        hint = dependency_hint(selected_models, deps)
        if hint:
            print("No selected baseline can run because dependencies are missing.")
            print(f"Install command: {hint}")
        return

    run_name = datetime.now().strftime("baseline_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "interpretable_baselines" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    columns = read_csv_header(csv_path)
    feature_sets = build_feature_sets(
        columns=columns,
        feature_set_arg=args.feature_set,
        include_current_sofa=args.include_current_sofa,
    )
    all_feature_cols = list(dict.fromkeys(col for cols in feature_sets.values() for col in cols))
    usecols = list(dict.fromkeys(["stay_id", args.time_col, args.split_col, args.target_col, *all_feature_cols]))
    existing_usecols, missing_usecols = maybe_existing_usecols(csv_path, usecols)
    missing_required = [col for col in missing_usecols if col != args.target_col]
    if missing_required:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_required)}")
    horizon_hours = horizon_from_target_col(args.target_col)

    print(f"Reading data: {csv_path}")
    print(f"Models: {', '.join(models_to_run)}")
    print(f"Feature sets: {', '.join(feature_sets)}")
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
        all_feature_cols=all_feature_cols,
        target_col=args.target_col,
        time_col=args.time_col,
        min_history_hours=args.min_history_hours,
    )
    if args.split_col != "subject_id":
        raise ValueError("正式 baseline 實驗必須以 subject_id 做 patient-level split。")
    df = attach_split(df, args.split_manifest, patient_col=args.split_col)
    train_df = filter_frame_to_comparison_windows(
        df.loc[df["dataset_split"] == "train"], args.target_col, args.time_col, "train",
        args.comparison_mode, args.equal_sample_windows, args.min_history_hours,
        require_all_window_ids=not args.allow_incomplete_cohort,
    )
    val_df = filter_frame_to_comparison_windows(
        df.loc[df["dataset_split"] == "validation"], args.target_col, args.time_col, "validation",
        args.comparison_mode, args.equal_sample_windows, args.min_history_hours,
        require_all_window_ids=not args.allow_incomplete_cohort,
    )
    test_df = filter_frame_to_comparison_windows(
        df.loc[df["dataset_split"] == "test"], args.target_col, args.time_col, "test",
        args.comparison_mode, args.equal_sample_windows, args.min_history_hours,
        require_all_window_ids=not args.allow_incomplete_cohort,
    )
    del df

    if args.allow_incomplete_cohort:
        train_df = stratified_sample(train_df, args.target_col, args.max_train_samples, args.seed)
        val_df = stratified_sample(val_df, args.target_col, args.max_val_samples, args.seed + 10)
        test_df = stratified_sample(test_df, args.target_col, args.max_test_samples, args.seed + 20)

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
                "feature_sets": feature_sets,
                "fill_values": fill_values,
                "comparison_protocol_sha256": protocol["protocol_sha256"],
                "cohort_audit": cohort_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    metrics_path = output_dir / "baseline_metrics.csv"
    y_train = train_df[args.target_col].to_numpy(dtype=np.float32)
    y_val = val_df[args.target_col].to_numpy(dtype=np.float32)
    y_test = test_df[args.target_col].to_numpy(dtype=np.float32)

    for feature_set_name, feature_cols in feature_sets.items():
        print(f"\n=== Feature set: {feature_set_name} ({len(feature_cols)} features) ===")

        for model_name in models_to_run:
            model_feature_cols = feature_cols
            if model_name == "gam":
                model_feature_cols = select_features_by_signal(
                    train_df[feature_cols],
                    y_train,
                    feature_cols,
                    args.gam_max_features,
                )
            elif model_name == "ebm":
                model_feature_cols = select_features_by_signal(
                    train_df[feature_cols],
                    y_train,
                    feature_cols,
                    args.ebm_max_features,
                )

            x_train = train_df[model_feature_cols].to_numpy(dtype=np.float32, copy=True)
            x_val = val_df[model_feature_cols].to_numpy(dtype=np.float32, copy=True)
            x_test = test_df[model_feature_cols].to_numpy(dtype=np.float32, copy=True)

            print(f"Training {model_name} with {len(model_feature_cols)} features...")
            model, train_metrics, val_metrics = train_one_model(
                model_name=model_name,
                deps=deps,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                feature_names=model_feature_cols,
                args=args,
            )

            model_dir = output_dir / feature_set_name / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            save_model(model, model_dir / "model.joblib", deps)
            save_feature_report(model, model_name, model_feature_cols, model_dir / "feature_report.csv")
            test_probs = predict_probabilities(model, model_name, x_test)
            test_metrics = binary_metrics(y_test, test_probs)

            if args.save_predictions:
                id_cols = [args.split_col, "stay_id", args.time_col]
                val_predictions = val_df[id_cols].reset_index(drop=True).copy()
                val_predictions = val_predictions.assign(
                    y_true=y_val,
                    y_prob=predict_probabilities(model, model_name, x_val),
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
                test_predictions = test_df[id_cols].reset_index(drop=True).copy()
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

            row = {
                "comparison_mode": args.comparison_mode,
                "protocol_sha256": protocol["protocol_sha256"],
                "target_col": args.target_col,
                "horizon_hours": horizon_hours,
                "feature_set": feature_set_name,
                "model": model_name,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "n_features": len(model_feature_cols),
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
            }
            append_metrics(metrics_path, row)

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
        "Interpretable Machine Learning Baselines",
    )


if __name__ == "__main__":
    main()
