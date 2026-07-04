"""Clinical score baselines for ICU deterioration prediction.

Models:
1. NEWS2 Score
2. Current SOFA Score

This script evaluates traditional clinical scores against the same SOFA
deterioration labels used by the FNN experiments:

    label_sofa_increase_ge2_6h
    label_sofa_increase_ge2_12h
    label_sofa_increase_ge2_24h

The script is robust to two project states:
1. `model_hourly_features_v3.csv` already contains `sofa_score` and label columns.
2. The hourly feature table only contains predictors; SOFA labels are
   merged from `sofa_scores_hourly.csv`, including the archived copy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from paper_figures import generate_metric_comparison_figures, try_generate
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
    SOFA_HOURLY_CSV,
)


NEWS2_CANDIDATES = {
    "heart_rate": ["heart_rate", "HeartRate", "Heart Rate"],
    "respiratory_rate": ["respiratory_rate", "RespRate", "Respiratory Rate"],
    "spo2": ["spo2", "SpO2", "O2 saturation pulseoxymetry"],
    "fio2": ["fio2", "FiO2", "Inspired O2 Fraction"],
    "temperature_c": ["temperature_c", "Temperature_C", "Temperature Celsius"],
    "sbp": ["sbp", "SBP", "sbp_arterial", "sbp_noninvasive"],
    "gcs_total": ["gcs_total", "GCS_Total"],
    "gcs_eye": ["gcs_eye", "GCS - Eye Opening"],
    "gcs_verbal": ["gcs_verbal", "GCS - Verbal Response"],
    "gcs_motor": ["gcs_motor", "GCS - Motor Response"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NEWS2/SOFA clinical score baselines.")
    parser.add_argument("--feature-csv", default=PRIMARY_HOURLY_FEATURES)
    parser.add_argument("--sofa-csv", default=None, help="Optional sofa_scores_hourly.csv path.")
    parser.add_argument("--time-col", default="sofa_hour")
    parser.add_argument("--split-col", default="subject_id")
    parser.add_argument("--split-manifest", default=PATIENT_SPLIT_CSV)
    parser.add_argument("--comparison-mode", choices=["full", "equal_sample"], default="full")
    parser.add_argument("--comparison-protocol", default=COMPARISON_PROTOCOL_JSON)
    parser.add_argument("--equal-sample-windows", default=EQUAL_SAMPLE_WINDOWS_CSV)
    parser.add_argument("--seq-length", type=int, default=24)
    parser.add_argument("--allow-incomplete-cohort", action="store_true", help="僅限 smoke test。")
    parser.add_argument("--horizons", default="6,12,24", help="Comma list, e.g. 6,12,24.")
    parser.add_argument("--val-frac", type=float, default=0.15, help="舊版相容參數；正式比例由 manifest 決定。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0, help="0 means unlimited.")
    parser.add_argument("--max-stays", type=int, default=0, help="0 means unlimited.")
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def zero_to_none(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return value


def parse_horizons(raw: str) -> list[int]:
    horizons = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            horizons.append(int(item))
    if not horizons:
        raise ValueError("At least one horizon is required.")
    return horizons


def horizon_from_target_col(target_col: str) -> int:
    suffix = target_col.rsplit("_", 1)[-1]
    if not suffix.endswith("h"):
        raise ValueError(f"無法由 target 欄位判斷 horizon: {target_col}")
    return int(suffix[:-1])


def read_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def find_sofa_csv(user_path: str | None) -> Path | None:
    if user_path:
        path = Path(user_path)
        if not path.exists():
            raise FileNotFoundError(f"Cannot find SOFA CSV: {path}")
        return path

    candidates = [Path(SOFA_HOURLY_CSV)]
    candidates.extend(Path(".").glob("_archive_unused_*/old_data/sofa_scores_hourly.csv"))
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def load_feature_frame(
    feature_csv: Path,
    usecols: list[str],
    split_col: str,
    max_rows: int | None,
    max_stays: int | None,
    chunk_size: int,
) -> pd.DataFrame:
    if max_rows is None and max_stays is None:
        return pd.read_csv(feature_csv, usecols=usecols)

    chunks = []
    rows_seen = 0
    selected_stays: set[Any] = set()
    reader = pd.read_csv(feature_csv, usecols=usecols, chunksize=chunk_size)

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
            for stay_id in pd.unique(chunk[split_col]):
                if stay_id in selected_stays:
                    keep_stays.add(stay_id)
                elif len(selected_stays) < max_stays:
                    selected_stays.add(stay_id)
                    keep_stays.add(stay_id)
                else:
                    stop_after_chunk = True
            chunk = chunk[chunk[split_col].isin(keep_stays)]

        if len(chunk) > 0:
            chunks.append(chunk)
        if stop_after_chunk:
            break
        if max_rows is not None and rows_seen >= max_rows:
            break

    if not chunks:
        return pd.DataFrame(columns=usecols)
    return pd.concat(chunks, ignore_index=True)


def load_sofa_reference(
    sofa_csv: Path,
    horizons: list[int],
    split_col: str,
    time_col: str,
    allowed_stays: set[Any] | None,
) -> pd.DataFrame:
    header = read_header(sofa_csv)
    target_cols = [f"label_sofa_increase_ge2_{h}h" for h in horizons]
    required = [split_col, time_col, "sofa_score", *target_cols]
    missing = [col for col in required if col not in header]
    if missing:
        raise ValueError(f"SOFA reference lacks required columns: {missing}")

    if allowed_stays is None:
        return pd.read_csv(sofa_csv, usecols=required)

    chunks = []
    for chunk in pd.read_csv(sofa_csv, usecols=required, chunksize=500_000):
        selected = chunk[chunk[split_col].isin(allowed_stays)]
        if not selected.empty:
            chunks.append(selected)
    if not chunks:
        return pd.DataFrame(columns=required)
    return pd.concat(chunks, ignore_index=True)


def numeric_series(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def normalize_fio2(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        np.where((values > 0) & (values <= 1.0), values * 100.0, values),
        index=values.index,
        dtype="float64",
    )


def calculate_news2(df: pd.DataFrame, column_map: dict[str, str | None]) -> pd.Series:
    rr = numeric_series(df, column_map.get("respiratory_rate"))
    spo2 = numeric_series(df, column_map.get("spo2"))
    fio2 = normalize_fio2(numeric_series(df, column_map.get("fio2")))
    temp = numeric_series(df, column_map.get("temperature_c"))
    sbp = numeric_series(df, column_map.get("sbp"))
    hr = numeric_series(df, column_map.get("heart_rate"))
    gcs = numeric_series(df, column_map.get("gcs_total"))

    if gcs.isna().all():
        eye = numeric_series(df, column_map.get("gcs_eye"))
        verbal = numeric_series(df, column_map.get("gcs_verbal"))
        motor = numeric_series(df, column_map.get("gcs_motor"))
        gcs = pd.concat([eye, verbal, motor], axis=1).sum(axis=1, min_count=3)

    score = pd.Series(0, index=df.index, dtype="int16")

    score += np.select(
        [rr <= 8, rr >= 25, (rr >= 21) & (rr <= 24), (rr >= 9) & (rr <= 11)],
        [3, 3, 2, 1],
        default=0,
    ).astype("int16")
    score += np.select(
        [spo2 <= 91, (spo2 >= 92) & (spo2 <= 93), (spo2 >= 94) & (spo2 <= 95)],
        [3, 2, 1],
        default=0,
    ).astype("int16")
    score += ((fio2 > 21) & fio2.notna()).astype("int16") * 2
    score += np.select(
        [temp <= 35.0, temp >= 39.1, ((temp >= 35.1) & (temp <= 36.0)) | ((temp >= 38.1) & (temp <= 39.0))],
        [3, 2, 1],
        default=0,
    ).astype("int16")
    score += np.select(
        [sbp <= 90, sbp >= 220, (sbp >= 91) & (sbp <= 100), (sbp >= 101) & (sbp <= 110)],
        [3, 3, 2, 1],
        default=0,
    ).astype("int16")
    score += np.select(
        [hr <= 40, hr >= 131, (hr >= 111) & (hr <= 130), ((hr >= 41) & (hr <= 50)) | ((hr >= 91) & (hr <= 110))],
        [3, 3, 2, 1],
        default=0,
    ).astype("int16")
    score += ((gcs < 15) & gcs.notna()).astype("int16") * 3
    return score


def build_score_frame(args: argparse.Namespace, horizons: list[int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_csv = Path(args.feature_csv)
    if not feature_csv.exists():
        raise FileNotFoundError(f"Cannot find feature CSV: {feature_csv}")

    header = read_header(feature_csv)
    target_cols = [f"label_sofa_increase_ge2_{h}h" for h in horizons]
    column_map = {key: first_existing(header, candidates) for key, candidates in NEWS2_CANDIDATES.items()}

    feature_usecols = ["stay_id", args.split_col, args.time_col]
    feature_usecols.extend(col for col in column_map.values() if col is not None)
    feature_usecols.extend(col for col in ["sofa_score", *target_cols] if col in header)
    feature_usecols = list(dict.fromkeys(feature_usecols))

    missing_news2 = [
        key
        for key in ["heart_rate", "respiratory_rate", "spo2", "fio2", "temperature_c", "sbp"]
        if column_map.get(key) is None
    ]
    if missing_news2:
        print(f"Warning: missing NEWS2 source columns: {missing_news2}")

    start = perf_counter()
    feature_df = load_feature_frame(
        feature_csv=feature_csv,
        usecols=feature_usecols,
        split_col="stay_id",
        max_rows=zero_to_none(args.max_rows),
        max_stays=zero_to_none(args.max_stays),
        chunk_size=args.chunk_size,
    )
    feature_df["news2_score"] = calculate_news2(feature_df, column_map)
    print(f"Loaded feature rows: {len(feature_df):,}; seconds: {perf_counter() - start:.1f}")

    needed_from_sofa = [col for col in ["sofa_score", *target_cols] if col not in feature_df.columns]
    sofa_csv = None
    if needed_from_sofa:
        sofa_csv = find_sofa_csv(args.sofa_csv)
        if sofa_csv is None:
            raise FileNotFoundError(
                "Feature CSV lacks SOFA score/labels and no sofa_scores_hourly.csv was found."
            )
        join_col = "stay_id"
        allowed_stays = set(pd.unique(feature_df[join_col])) if zero_to_none(args.max_stays) is not None else None
        print(f"Merging SOFA reference: {sofa_csv}")
        sofa_ref = load_sofa_reference(
            sofa_csv=sofa_csv,
            horizons=horizons,
            split_col=join_col,
            time_col=args.time_col,
            allowed_stays=allowed_stays,
        )
        feature_df = feature_df.merge(
            sofa_ref,
            on=[join_col, args.time_col],
            how="left",
            validate="one_to_one",
        )

    keep_cols = ["stay_id", args.split_col, args.time_col, "news2_score", "sofa_score", *target_cols]
    score_df = feature_df[keep_cols].copy()
    for col in ["news2_score", "sofa_score", *target_cols]:
        score_df[col] = pd.to_numeric(score_df[col], errors="coerce")

    metadata = {
        "feature_csv": str(feature_csv),
        "sofa_csv": str(sofa_csv) if sofa_csv else None,
        "column_map": column_map,
        "rows": len(score_df),
        "stays": int(score_df["stay_id"].nunique()),
        "patients": int(score_df[args.split_col].nunique()),
    }
    return score_df, metadata


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


def fit_score_calibrator(x_train: np.ndarray, y_train: np.ndarray):
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    if len(np.unique(y_train)) < 2:
        return None
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(x_train.reshape(-1, 1), y_train.astype(int))
    return model


def calibrated_probabilities(calibrator: Any, x: np.ndarray) -> np.ndarray:
    if calibrator is None:
        x = x.astype("float64")
        min_value = np.nanmin(x)
        max_value = np.nanmax(x)
        if not np.isfinite(min_value) or not np.isfinite(max_value) or max_value <= min_value:
            return np.full_like(x, fill_value=0.5, dtype="float64")
        return np.clip((x - min_value) / (max_value - min_value), 1e-7, 1 - 1e-7)
    return calibrator.predict_proba(x.reshape(-1, 1))[:, 1]


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(np.float32)
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    tn = float(np.sum((y_pred == 0) & (y_true == 0)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def brier_and_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_prob = np.clip(y_prob.astype("float64"), 1e-7, 1.0 - 1e-7)
    y_true = y_true.astype("float64")
    return {
        "brier": float(np.mean((y_prob - y_true) ** 2)),
        "log_loss": float(-np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))),
    }


def youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    thresholds = np.unique(y_score[np.isfinite(y_score)])
    if thresholds.size == 0 or len(np.unique(y_true)) < 2:
        return math.nan
    best_threshold = float(thresholds[0])
    best_youden = -math.inf
    for threshold in thresholds:
        metrics = classification_metrics(y_true, y_score, float(threshold))
        youden = metrics["sensitivity"] + metrics["specificity"] - 1.0
        if youden > best_youden:
            best_youden = youden
            best_threshold = float(threshold)
    return best_threshold


def score_thresholds(score_name: str, train_youden: float) -> list[tuple[str, float]]:
    if score_name == "news2_score":
        return [
            ("fixed_news2_ge5", 5.0),
            ("fixed_news2_ge7", 7.0),
            ("train_youden", train_youden),
        ]
    if score_name == "sofa_score":
        return [
            ("fixed_sofa_ge2", 2.0),
            ("train_youden", train_youden),
        ]
    return [("train_youden", train_youden)]


def evaluate_score(
    df: pd.DataFrame,
    score_name: str,
    target_col: str,
    train_mask: pd.Series,
    val_mask: pd.Series,
    test_mask: pd.Series,
) -> list[dict[str, Any]]:
    train = df.loc[train_mask, [score_name, target_col]].dropna()
    val = df.loc[val_mask, [score_name, target_col]].dropna()
    test = df.loc[test_mask, [score_name, target_col]].dropna()
    if train.empty or val.empty or test.empty:
        return []

    x_train = train[score_name].to_numpy(dtype="float64")
    y_train = train[target_col].to_numpy(dtype="float32")
    x_val = val[score_name].to_numpy(dtype="float64")
    y_val = val[target_col].to_numpy(dtype="float32")
    x_test = test[score_name].to_numpy(dtype="float64")
    y_test = test[target_col].to_numpy(dtype="float32")

    train_auroc = roc_auc_np(y_train, x_train)
    train_auprc = average_precision_np(y_train, x_train)
    val_auroc = roc_auc_np(y_val, x_val)
    val_auprc = average_precision_np(y_val, x_val)
    test_auroc = roc_auc_np(y_test, x_test)
    test_auprc = average_precision_np(y_test, x_test)

    calibrator = fit_score_calibrator(x_train, y_train)
    val_prob = calibrated_probabilities(calibrator, x_val)
    calibration = brier_and_log_loss(y_val, val_prob)
    test_prob = calibrated_probabilities(calibrator, x_test)
    test_calibration = brier_and_log_loss(y_test, test_prob)
    train_threshold = youden_threshold(y_train, x_train)

    rows = []
    for strategy, threshold in score_thresholds(score_name, train_threshold):
        if math.isnan(threshold):
            continue
        cls = classification_metrics(y_val, x_val, threshold)
        test_cls = classification_metrics(y_test, x_test, threshold)
        rows.append(
            {
                "feature_set": "clinical_score",
                "model": score_name,
                "target_col": target_col,
                "horizon_hours": target_col.split("_")[-1].replace("h", ""),
                "threshold_strategy": strategy,
                "threshold": threshold,
                "n_train": len(train),
                "n_val": len(val),
                "n_test": len(test),
                "n_features": 1,
                "train_prevalence": float(np.mean(y_train)),
                "val_prevalence": float(np.mean(y_val)),
                "test_prevalence": float(np.mean(y_test)),
                "train_auroc": train_auroc,
                "train_auprc": train_auprc,
                "val_auroc": val_auroc,
                "val_auprc": val_auprc,
                "test_auroc": test_auroc,
                "test_auprc": test_auprc,
                "val_accuracy": cls["accuracy"],
                "val_precision": cls["precision"],
                "val_recall": cls["recall"],
                "val_sensitivity": cls["sensitivity"],
                "val_specificity": cls["specificity"],
                "val_f1": cls["f1"],
                "val_brier_calibrated": calibration["brier"],
                "val_log_loss_calibrated": calibration["log_loss"],
                "test_accuracy": test_cls["accuracy"],
                "test_precision": test_cls["precision"],
                "test_recall": test_cls["recall"],
                "test_sensitivity": test_cls["sensitivity"],
                "test_specificity": test_cls["specificity"],
                "test_f1": test_cls["f1"],
                "test_brier_calibrated": test_calibration["brier"],
                "test_log_loss_calibrated": test_calibration["log_loss"],
                "val_score_mean_positive": float(np.mean(x_val[y_val == 1])) if np.any(y_val == 1) else math.nan,
                "val_score_mean_negative": float(np.mean(x_val[y_val == 0])) if np.any(y_val == 0) else math.nan,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    horizons = parse_horizons(args.horizons)
    protocol = validate_comparison_args(
        args.comparison_mode,
        args.comparison_protocol,
        f"label_sofa_increase_ge2_{horizons[0]}h",
        args.seq_length,
    )
    if not args.allow_incomplete_cohort and (
        zero_to_none(args.max_rows) is not None or zero_to_none(args.max_stays) is not None
    ):
        raise ValueError("正式比較不可額外限制 rows/stays。")
    run_name = datetime.now().strftime("clinical_scores_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "clinical_score_baselines" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    score_df, metadata = build_score_frame(args, horizons)
    if args.split_col != "subject_id":
        raise ValueError("正式 clinical baseline 必須以 subject_id 做 patient-level split。")
    score_df = attach_split(score_df, args.split_manifest, patient_col=args.split_col)
    score_df = score_df.sort_values(["stay_id", args.time_col], kind="mergesort").reset_index(drop=True)
    score_df["_history_index"] = score_df.groupby("stay_id", sort=False).cumcount()
    patient_counts = {
        split: int(score_df.loc[score_df["dataset_split"] == split, args.split_col].nunique())
        for split in ["train", "validation", "test"]
    }

    target_cols = [f"label_sofa_increase_ge2_{h}h" for h in horizons]
    rows: list[dict[str, Any]] = []
    cohort_records: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for target_col in target_cols:
        if target_col not in score_df.columns:
            print(f"Skip missing target: {target_col}")
            continue
        comparison_parts = []
        for split in ["train", "validation", "test"]:
            part = filter_frame_to_comparison_windows(
                score_df.loc[score_df["dataset_split"] == split],
                target_col,
                args.time_col,
                split,
                args.comparison_mode,
                args.equal_sample_windows,
                args.seq_length,
                require_all_window_ids=not args.allow_incomplete_cohort,
            )
            comparison_parts.append(part)
            cohort_records.append(
                cohort_record(part["stay_id"], part[args.time_col], part[target_col], split, target_col)
            )
        comparison_df = pd.concat(comparison_parts, ignore_index=True)
        train_mask = comparison_df["dataset_split"] == "train"
        val_mask = comparison_df["dataset_split"] == "validation"
        test_mask = comparison_df["dataset_split"] == "test"

        export = comparison_df[
            ["stay_id", args.split_col, args.time_col, "dataset_split", "news2_score", "sofa_score", target_col]
        ].copy()
        export = export.rename(columns={target_col: "y_true"})
        export["target_col"] = target_col
        export["comparison_mode"] = args.comparison_mode
        export["protocol_sha256"] = protocol["protocol_sha256"]
        prediction_frames.append(export)

        for score_name in ["news2_score", "sofa_score"]:
            if score_name not in score_df.columns:
                print(f"Skip missing score: {score_name}")
                continue
            rows.extend(evaluate_score(comparison_df, score_name, target_col, train_mask, val_mask, test_mask))
            if args.save_predictions:
                id_cols = [args.split_col, "stay_id", args.time_col]
                train_score = comparison_df.loc[
                    train_mask, [*id_cols, score_name, target_col]
                ].dropna(subset=[score_name, target_col])
                calibrator = fit_score_calibrator(
                    train_score[score_name].to_numpy(dtype=float),
                    train_score[target_col].to_numpy(dtype=np.float32),
                )
                model_name = f"{score_name}_calibrated"
                model_dir = output_dir / "clinical_score" / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                for split_name, split_mask, file_stem in [
                    ("validation", val_mask, "val_predictions.csv.gz"),
                    ("test", test_mask, "test_predictions.csv.gz"),
                ]:
                    part = comparison_df.loc[
                        split_mask, [*id_cols, score_name, target_col]
                    ].dropna(subset=[score_name, target_col])
                    prediction = part[id_cols].reset_index(drop=True).copy()
                    prediction = prediction.assign(
                        y_true=part[target_col].to_numpy(dtype=np.float32),
                        y_prob=calibrated_probabilities(
                            calibrator, part[score_name].to_numpy(dtype=float)
                        ),
                        model=model_name,
                        feature_set="clinical_score",
                        target_col=target_col,
                        horizon_hours=horizon_from_target_col(target_col),
                        evaluation_split=split_name,
                        comparison_mode=args.comparison_mode,
                        protocol_sha256=protocol["protocol_sha256"],
                    )
                    prediction.to_csv(model_dir / file_stem, index=False, compression="gzip")

    validate_cohort_records(
        cohort_records, protocol, args.comparison_mode, allow_incomplete=args.allow_incomplete_cohort
    )
    write_cohort_audit(output_dir / "cohort_audit.json", cohort_records)
    for row in rows:
        row["comparison_mode"] = args.comparison_mode
        row["protocol_sha256"] = protocol["protocol_sha256"]

    metrics_path = output_dir / "clinical_score_metrics.csv"
    write_csv(metrics_path, rows)

    if args.save_predictions:
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            output_dir / "clinical_score_predictions.csv", index=False
        )

    with (output_dir / "clinical_score_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "horizons": horizons,
                "metadata": metadata,
                "train_patients": patient_counts["train"],
                "val_patients": patient_counts["validation"],
                "test_patients": patient_counts["test"],
                "comparison_protocol_sha256": protocol["protocol_sha256"],
                "cohort_audit": cohort_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Clinical score baseline complete: {metrics_path}")
    try_generate(
        generate_metric_comparison_figures,
        metrics_path,
        output_dir,
        "Clinical Score Baselines",
    )
    if rows:
        preview = pd.DataFrame(rows)
        print(
            preview[
                [
                    "model",
                    "horizon_hours",
                    "threshold_strategy",
                    "test_auroc",
                    "test_auprc",
                    "test_sensitivity",
                    "test_specificity",
                    "test_f1",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
