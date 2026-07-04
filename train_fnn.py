"""Train the Knowledge-Guided Temporal FNN on hourly ICU features.

這支訓練腳本對應研究計畫二版的主要設定：
1. 輸入為每個 ICU stay 的 24 小時 sliding window。
2. 主要標籤預設為未來 6 小時 SOFA increase >= 2。
3. 以 subject_id 固定切成 train/validation/test，避免同一病人的多次 stay 跨組。
4. 缺值只做同 stay 內 forward fill；開頭仍缺的欄位補臨床正常值，避免 bfill 造成未來資訊外洩。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from anfis_model import (
    FEATURE_ORDER,
    NeuroSymbolicLoss,
    TemporalAttentionFNN,
    explicit_temporal_input_order,
)
from comparison_protocol import (
    cohort_record,
    validate_cohort_records,
    validate_comparison_args,
    window_id_membership,
    window_ids_for_mode,
    write_cohort_audit,
)
from paper_figures import generate_training_figures, try_generate
from patient_split import split_ids_for_values
from project_config import (
    COMPARISON_PROTOCOL_JSON,
    EQUAL_SAMPLE_WINDOWS_CSV,
    PATIENT_SPLIT_CSV,
    PRIMARY_HOURLY_FEATURES,
)
from sofa_label_utils import horizon_from_target_col, maybe_existing_usecols, merge_sofa_targets


# 這些值使用各 membership function 的臨床正常中心點。
# 只在同 stay 內 forward fill 之後仍然缺值時使用，避免使用未來測量值補過去。
CLINICAL_DEFAULTS = {
    "heart_rate": 70.0,
    "respiratory_rate": 16.0,
    "spo2": 97.0,
    "fio2": 0.21,
    "temperature_c": 37.0,
    "sbp": 130.0,
    "gcs_total": 15.0,
    "map": 85.0,
    "pao2_fio2": 450.0,
    "platelets": 220.0,
    "bilirubin": 0.8,
    "creatinine": 0.8,
    "lactate": 1.2,
}


class ICUWindowDataset(Dataset):
    """以 numpy array 保存原始 hourly table，getitem 時才切出 24h window。"""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        stay_ids: np.ndarray,
        split_values: np.ndarray,
        time_values: np.ndarray,
        allowed_split_values: set,
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
        self.allowed_window_ids = allowed_window_ids
        self.window_starts = self._build_window_starts(
            stay_ids=stay_ids,
            split_values=split_values,
            allowed_split_values=allowed_split_values,
            time_values=time_values,
            allowed_window_ids=allowed_window_ids,
            seq_length=seq_length,
        )
        if require_all_window_ids and allowed_window_ids is not None and len(self) != len(allowed_window_ids):
            raise ValueError(
                f"Equal-sample windows 缺少 {len(allowed_window_ids) - len(self):,} 筆；"
                "正式比較不可限制讀取 rows/stays。"
            )

    def _build_window_starts(
        self,
        stay_ids: np.ndarray,
        split_values: np.ndarray,
        allowed_split_values: set,
        time_values: np.ndarray,
        allowed_window_ids: np.ndarray | None,
        seq_length: int,
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
        x = torch.from_numpy(self.features[start:end])
        y = torch.tensor(self.labels[target_index], dtype=torch.float32)
        return x, y

    def label_counts(self) -> tuple[int, int]:
        if len(self) == 0:
            return 0, 0
        targets = self.labels[self.window_starts + self.seq_length - 1]
        positives = int(np.sum(targets == 1))
        negatives = int(np.sum(targets == 0))
        return positives, negatives

    def cohort_record(self, split: str, target_col: str) -> dict[str, object]:
        target_indices = self.window_starts + self.seq_length - 1
        return cohort_record(
            self.stay_ids[target_indices],
            self.time_values[target_indices],
            self.labels[target_indices],
            split,
            target_col,
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_training_frame(
    csv_path: Path,
    feature_cols: list[str],
    target_col: str,
    time_col: str,
    split_col: str,
    max_rows: int | None,
    max_stays: int | None,
    chunk_size: int,
    sofa_csv: str | Path | None = None,
) -> pd.DataFrame:
    usecols = list(dict.fromkeys(["stay_id", time_col, split_col, target_col, *feature_cols]))
    existing_usecols, missing_usecols = maybe_existing_usecols(csv_path, usecols)
    missing_required = [col for col in missing_usecols if col != target_col]
    if missing_required:
        raise ValueError(f"CSV 缺少必要欄位: {sorted(missing_required)}")

    if max_rows is None and max_stays is None:
        df = pd.read_csv(csv_path, usecols=existing_usecols)
        return merge_sofa_targets(
            df=df,
            target_cols=[target_col],
            split_col=split_col,
            time_col=time_col,
            sofa_csv=sofa_csv,
        )

    chunks = []
    rows_seen = 0
    selected_stays: set = set()

    reader = pd.read_csv(csv_path, usecols=existing_usecols, chunksize=chunk_size)
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
        return pd.DataFrame(columns=existing_usecols)
    df = pd.concat(chunks, ignore_index=True)
    return merge_sofa_targets(
        df=df,
        target_cols=[target_col],
        split_col=split_col,
        time_col=time_col,
        sofa_csv=sofa_csv,
    )


def prepare_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    time_col: str,
    split_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    missing = set(["stay_id", time_col, split_col, target_col, *feature_cols]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少必要欄位: {sorted(missing)}")

    df = df.sort_values(["stay_id", time_col], kind="mergesort").reset_index(drop=True)

    numeric_cols = [target_col, *feature_cols]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[feature_cols] = df.groupby("stay_id", sort=False)[feature_cols].ffill()
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna({col: CLINICAL_DEFAULTS[col] for col in feature_cols})

    features = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    labels = df[target_col].to_numpy(dtype=np.float32, copy=True)
    stay_ids = df["stay_id"].to_numpy(copy=True)
    split_values = df[split_col].to_numpy(copy=True)
    time_values = pd.to_numeric(df[time_col], errors="raise").to_numpy(dtype=np.int64, copy=True)
    return features, labels, stay_ids, split_values, time_values


def prepare_explicit_temporal_arrays(
    df: pd.DataFrame,
    target_col: str,
    time_col: str,
    split_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """建立 raw、missingness 與 time-since 三段式 leakage-free 輸入。"""
    input_cols = explicit_temporal_input_order(FEATURE_ORDER)
    required = {"stay_id", time_col, split_col, target_col, *input_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV 缺少 explicit temporal 必要欄位: {sorted(missing)}")

    df = df.sort_values(["stay_id", time_col], kind="mergesort").reset_index(drop=True)
    raw_cols = list(FEATURE_ORDER)
    missing_cols = [f"{feature}_is_missing" for feature in FEATURE_ORDER]
    time_since_cols = [
        f"{feature}_time_since_last_measurement_h" for feature in FEATURE_ORDER
    ]

    for column in [target_col, *input_cols]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # 僅在同一 stay 內向前補值；開頭缺值使用固定臨床正常值，禁止 bfill。
    df[raw_cols] = df.groupby("stay_id", sort=False)[raw_cols].ffill()
    df[raw_cols] = df[raw_cols].replace([np.inf, -np.inf], np.nan)
    df[raw_cols] = df[raw_cols].fillna(CLINICAL_DEFAULTS)
    df[missing_cols] = df[missing_cols].fillna(1.0).clip(0.0, 1.0)

    fallback_time = pd.to_numeric(df[time_col], errors="coerce").fillna(0.0) + 1.0
    for column in time_since_cols:
        df[column] = df[column].replace([np.inf, -np.inf], np.nan).fillna(fallback_time)
        df[column] = df[column].clip(lower=0.0)

    features = df[input_cols].to_numpy(dtype=np.float32, copy=True)
    labels = df[target_col].to_numpy(dtype=np.float32, copy=True)
    stay_ids = df["stay_id"].to_numpy(copy=True)
    split_values = df[split_col].to_numpy(copy=True)
    time_values = pd.to_numeric(df[time_col], errors="raise").to_numpy(
        dtype=np.int64,
        copy=True,
    )
    return features, labels, stay_ids, split_values, time_values


def choose_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def average_losses(loss_sums: dict[str, float], n_samples: int) -> dict[str, float]:
    if n_samples == 0:
        return {key: math.nan for key in loss_sums}
    return {key: value / n_samples for key, value in loss_sums.items()}


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


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_score = np.clip(y_score.astype(np.float64), 1e-7, 1.0 - 1e-7)
    y_pred = (y_score >= 0.5).astype(np.float32)
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    tn = float(np.sum((y_pred == 0) & (y_true == 0)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) else math.nan
    recall = tp / (tp + fn) if (tp + fn) else math.nan
    specificity = tn / (tn + fp) if (tn + fp) else math.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else math.nan
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    brier = float(np.mean((y_score - y_true) ** 2))
    log_loss = float(
        -np.mean(y_true * np.log(y_score) + (1.0 - y_true) * np.log(1.0 - y_score))
    )

    # 固定使用 10 個等寬機率區間，讓所有模型的 ECE/MCE 可直接比較。
    edges = np.linspace(0.0, 1.0, 11)
    bin_ids = np.minimum(np.digitize(y_score, edges[1:-1], right=False), 9)
    ece = 0.0
    mce = 0.0
    for bin_idx in range(10):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            continue
        gap = abs(float(np.mean(y_score[mask])) - float(np.mean(y_true[mask])))
        ece += float(np.mean(mask)) * gap
        mce = max(mce, gap)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "auroc": roc_auc_np(y_true, y_score),
        "auprc": average_precision_np(y_true, y_score),
        "brier": brier,
        "ece": ece,
        "mce": mce,
        "log_loss": log_loss,
        "prevalence": float(np.mean(y_true)),
    }


def run_epoch(
    model: TemporalAttentionFNN,
    loader: DataLoader,
    criterion: NeuroSymbolicLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
    max_batches: int | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    criterion.train(is_train)

    loss_sums = {
        "total": 0.0,
        "prediction": 0.0,
        "clinical_consistency": 0.0,
        "rule_sparsity": 0.0,
        "rule_drift": 0.0,
        "nonnegative_weights": 0.0,
    }
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

            output = model(batch_x)
            loss_dict = criterion(output, batch_y, model)
            loss = loss_dict["total"]

            if is_train:
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            batch_size = int(batch_y.shape[0])
            n_samples += batch_size
            for key, value in loss_dict.items():
                loss_sums[key] += float(value.detach().item()) * batch_size

            all_probs.append(output.probabilities.detach().cpu().numpy())
            all_targets.append(batch_y.detach().cpu().numpy())

    losses = average_losses(loss_sums, n_samples)
    if all_probs:
        probs = np.concatenate(all_probs).astype(np.float32)
        targets = np.concatenate(all_targets).astype(np.float32)
        metrics = classification_metrics(targets, probs)
    else:
        metrics = {
            "accuracy": math.nan,
            "precision": math.nan,
            "recall": math.nan,
            "sensitivity": math.nan,
            "specificity": math.nan,
            "f1": math.nan,
            "auroc": math.nan,
            "auprc": math.nan,
            "brier": math.nan,
            "ece": math.nan,
            "mce": math.nan,
            "log_loss": math.nan,
            "prevalence": math.nan,
        }
    return losses, metrics


def save_metrics_row(path: Path, row: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_best_params(path_value: str | None) -> dict[str, object]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"找不到 Optuna best params: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("best_params", payload)
    if not isinstance(params, dict):
        raise ValueError(f"無法解析 best params: {path}")
    return params


def apply_best_params(args: argparse.Namespace, params: dict[str, object]) -> argparse.Namespace:
    allowed = {
        "learning_rate",
        "weight_decay",
        "batch_size",
        "grad_clip",
        "attention_hidden",
        "threshold",
        "rule_score_scale",
        "lambda_cons",
        "lambda_sparse",
        "lambda_drift",
        "lambda_nonnegative",
        "explicit_temporal_scale",
    }
    applied = {}
    for key, value in params.items():
        if key in allowed:
            setattr(args, key, value)
            applied[key] = value
    args.applied_best_params = applied
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Knowledge-Guided Temporal FNN.")
    parser.add_argument("--csv", default=PRIMARY_HOURLY_FEATURES, help="訓練資料 CSV。")
    parser.add_argument("--sofa-csv", default=None, help="若訓練資料缺 SOFA label，可指定 sofa_scores_hourly.csv。")
    parser.add_argument(
        "--target-col",
        default="label_sofa_increase_ge2_6h",
        help="預測標籤欄位，可改成 label_sofa_increase_ge2_12h 或 24h。",
    )
    parser.add_argument("--time-col", default="sofa_hour", help="同一 stay 內排序用欄位。")
    parser.add_argument("--split-col", default="subject_id", help="固定使用 subject_id 進行 patient-level split。")
    parser.add_argument("--split-manifest", default=PATIENT_SPLIT_CSV, help="固定 train/validation/test 名單。")
    parser.add_argument("--comparison-mode", choices=["full", "equal_sample"], default="full")
    parser.add_argument("--comparison-protocol", default=COMPARISON_PROTOCOL_JSON)
    parser.add_argument("--equal-sample-windows", default=EQUAL_SAMPLE_WINDOWS_CSV)
    parser.add_argument("--allow-incomplete-cohort", action="store_true", help="僅限 smoke test。")
    parser.add_argument("--seq-length", type=int, default=24, help="回看幾小時。")
    parser.add_argument("--val-frac", type=float, default=0.15, help="舊版相容參數；正式比例由 split manifest 決定。")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--best-params-json", default=None, help="Optuna best_params.json。")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--lambda-cons", type=float, default=0.1)
    parser.add_argument("--lambda-sparse", type=float, default=0.001)
    parser.add_argument("--lambda-drift", type=float, default=0.001)
    parser.add_argument("--lambda-nonnegative", type=float, default=0.05)
    parser.add_argument("--rule-score-scale", type=float, default=0.2)
    parser.add_argument("--attention-hidden", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=7.0)
    parser.add_argument(
        "--explicit-temporal-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="啟用 raw + missingness + time-since 輸入與明確時序特徵層。",
    )
    parser.add_argument("--explicit-temporal-scale", type=float, default=1.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0 ...")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None, help="測試用：最多讀取幾列。")
    parser.add_argument("--max-stays", type=int, default=None, help="測試用：最多讀取幾個 ICU stay。")
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--limit-train-batches", type=int, default=None)
    parser.add_argument("--limit-val-batches", type=int, default=None)
    parser.add_argument("--limit-test-batches", type=int, default=None)
    parser.add_argument("--no-pos-weight", action="store_true", help="停用 class imbalance pos_weight。")
    parser.add_argument("--dry-run", action="store_true", help="只建立資料集並顯示統計，不訓練。")
    parser.add_argument("--output-dir", default=None, help="checkpoint 與 metrics 儲存位置。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args = apply_best_params(args, load_best_params(args.best_params_json))
    set_seed(args.seed)
    protocol = validate_comparison_args(
        args.comparison_mode,
        args.comparison_protocol,
        args.target_col,
        args.seq_length,
    )

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到訓練資料: {csv_path}")

    run_name = datetime.now().strftime("fnn_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "fnn_training" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = perf_counter()
    print(f"讀取資料: {csv_path}")
    input_order = (
        explicit_temporal_input_order(FEATURE_ORDER)
        if args.explicit_temporal_features
        else list(FEATURE_ORDER)
    )
    df = load_training_frame(
        csv_path=csv_path,
        feature_cols=input_order,
        target_col=args.target_col,
        time_col=args.time_col,
        split_col=args.split_col,
        max_rows=args.max_rows,
        max_stays=args.max_stays,
        chunk_size=args.chunk_size,
        sofa_csv=args.sofa_csv,
    )
    print(f"讀入列數: {len(df):,}，stay 數: {df['stay_id'].nunique():,}")

    if args.explicit_temporal_features:
        features, labels, stay_ids, split_values, time_values = prepare_explicit_temporal_arrays(
            df=df,
            target_col=args.target_col,
            time_col=args.time_col,
            split_col=args.split_col,
        )
    else:
        features, labels, stay_ids, split_values, time_values = prepare_arrays(
            df=df,
            feature_cols=FEATURE_ORDER,
            target_col=args.target_col,
            time_col=args.time_col,
            split_col=args.split_col,
        )
    del df

    if args.split_col != "subject_id":
        raise ValueError("正式實驗必須以 subject_id 做 patient-level split。")
    train_ids, val_ids, test_ids = split_ids_for_values(split_values, args.split_manifest)
    train_window_ids = window_ids_for_mode(
        args.comparison_mode, args.equal_sample_windows, args.target_col, "train"
    )
    val_window_ids = window_ids_for_mode(
        args.comparison_mode, args.equal_sample_windows, args.target_col, "validation"
    )
    train_dataset = ICUWindowDataset(
        features=features,
        labels=labels,
        stay_ids=stay_ids,
        split_values=split_values,
        time_values=time_values,
        allowed_split_values=train_ids,
        seq_length=args.seq_length,
        allowed_window_ids=train_window_ids,
        require_all_window_ids=not args.allow_incomplete_cohort,
    )
    val_dataset = ICUWindowDataset(
        features=features,
        labels=labels,
        stay_ids=stay_ids,
        split_values=split_values,
        time_values=time_values,
        allowed_split_values=val_ids,
        seq_length=args.seq_length,
        allowed_window_ids=val_window_ids,
        require_all_window_ids=not args.allow_incomplete_cohort,
    )
    test_dataset = ICUWindowDataset(
        features=features,
        labels=labels,
        stay_ids=stay_ids,
        split_values=split_values,
        time_values=time_values,
        allowed_split_values=test_ids,
        seq_length=args.seq_length,
    )

    cohort_records = [
        train_dataset.cohort_record("train", args.target_col),
        val_dataset.cohort_record("validation", args.target_col),
        test_dataset.cohort_record("test", args.target_col),
    ]
    validate_cohort_records(
        cohort_records,
        protocol,
        args.comparison_mode,
        allow_incomplete=args.allow_incomplete_cohort,
    )
    write_cohort_audit(output_dir / "cohort_audit.json", cohort_records)

    train_pos, train_neg = train_dataset.label_counts()
    val_pos, val_neg = val_dataset.label_counts()
    test_pos, test_neg = test_dataset.label_counts()
    print(f"Train windows: {len(train_dataset):,} | positive: {train_pos:,} | negative: {train_neg:,}")
    print(f"Val windows:   {len(val_dataset):,} | positive: {val_pos:,} | negative: {val_neg:,}")
    print(f"Test windows:  {len(test_dataset):,} | positive: {test_pos:,} | negative: {test_neg:,}")

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError("train、validation 或 test 沒有可用 window；請增加 max_rows/max_stays 或檢查 label。")

    config_path = output_dir / "train_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "feature_order": FEATURE_ORDER,
                "input_order": input_order,
                "clinical_defaults": CLINICAL_DEFAULTS,
                "train_windows": len(train_dataset),
                "val_windows": len(val_dataset),
                "train_positive": train_pos,
                "train_negative": train_neg,
                "val_positive": val_pos,
                "val_negative": val_neg,
                "test_windows": len(test_dataset),
                "test_positive": test_pos,
                "test_negative": test_neg,
                "split_method": "fixed patient-level train/validation/test manifest",
                "comparison_protocol_sha256": protocol["protocol_sha256"],
                "cohort_audit": cohort_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if args.dry_run:
        print(f"Dry run 完成，設定已存到: {config_path}")
        return

    device = choose_device(args.device)
    print(f"使用裝置: {device}")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = TemporalAttentionFNN(
        seq_length=args.seq_length,
        attention_hidden=args.attention_hidden,
        threshold=args.threshold,
        rule_score_scale=args.rule_score_scale,
        use_explicit_temporal_features=args.explicit_temporal_features,
        explicit_temporal_scale=args.explicit_temporal_scale,
    ).to(device)

    criterion = NeuroSymbolicLoss(
        lambda_cons=args.lambda_cons,
        lambda_sparse=args.lambda_sparse,
        lambda_drift=args.lambda_drift,
        lambda_nonnegative=args.lambda_nonnegative,
    )
    if not args.no_pos_weight and train_pos > 0 and train_neg > 0:
        pos_weight = train_neg / train_pos
        criterion.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
        print(f"啟用 pos_weight: {pos_weight:.3f}")
    elif not args.no_pos_weight:
        print("訓練集缺少正類或負類，pos_weight 自動停用。")
    criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    metrics_path = output_dir / "metrics.csv"
    best_score = -math.inf
    best_loss = math.inf
    best_path = output_dir / "best_model.pt"
    best_epoch = 0
    epochs_without_improvement = 0
    actual_epochs = 0
    stopped_early = False

    for epoch in range(1, args.epochs + 1):
        actual_epochs = epoch
        epoch_start = perf_counter()
        train_losses, train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
            max_batches=args.limit_train_batches,
        )
        val_losses, val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            grad_clip=None,
            max_batches=args.limit_val_batches,
        )

        val_score = val_metrics["auroc"]
        improved = False
        if math.isnan(val_score):
            improved = val_losses["total"] < best_loss - args.early_stopping_min_delta
        else:
            improved = val_score > best_score + args.early_stopping_min_delta

        if improved:
            best_epoch = epoch
            epochs_without_improvement = 0
            best_score = val_score if not math.isnan(val_score) else best_score
            best_loss = val_losses["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                    "feature_order": FEATURE_ORDER,
                    "input_order": input_order,
                    "clinical_defaults": CLINICAL_DEFAULTS,
                    "val_losses": val_losses,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "comparison_mode": args.comparison_mode,
            "protocol_sha256": protocol["protocol_sha256"],
            "target_col": args.target_col,
            "horizon_hours": horizon_from_target_col(args.target_col),
            "train_total_loss": train_losses["total"],
            "train_bce": train_losses["prediction"],
            "train_auroc": train_metrics["auroc"],
            "train_auprc": train_metrics["auprc"],
            "val_total_loss": val_losses["total"],
            "val_bce": val_losses["prediction"],
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "val_recall": val_metrics["recall"],
            "val_precision": val_metrics["precision"],
            "seconds": perf_counter() - epoch_start,
        }
        save_metrics_row(metrics_path, row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_losses['total']:.4f} bce {train_losses['prediction']:.4f} "
            f"auroc {train_metrics['auroc']:.4f} | "
            f"val loss {val_losses['total']:.4f} bce {val_losses['prediction']:.4f} "
            f"auroc {val_metrics['auroc']:.4f} auprc {val_metrics['auprc']:.4f}"
        )

        if (
            args.early_stopping_patience > 0
            and epoch >= args.early_stopping_min_epochs
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            stopped_early = True
            print(
                f"Early stopping at epoch {epoch}: validation AUROC 已連續 "
                f"{epochs_without_improvement} epochs 未改善。"
            )
            break

    last_path = output_dir / "last_model.pt"
    torch.save(
        {
            "epoch": actual_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "feature_order": FEATURE_ORDER,
            "input_order": input_order,
            "clinical_defaults": CLINICAL_DEFAULTS,
        },
        last_path,
    )

    training_summary = {
        "requested_epochs": args.epochs,
        "actual_epochs": actual_epochs,
        "best_epoch": best_epoch,
        "best_val_auroc": best_score if math.isfinite(best_score) else None,
        "best_val_loss": best_loss if math.isfinite(best_loss) else None,
        "stopped_early": stopped_early,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_epochs": args.early_stopping_min_epochs,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "applied_best_params": args.applied_best_params,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(training_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # test set 僅在 validation 已選出最佳 checkpoint 後評估一次。
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_losses, test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        grad_clip=None,
        max_batches=args.limit_test_batches,
    )
    test_row = {
        "comparison_mode": args.comparison_mode,
        "protocol_sha256": protocol["protocol_sha256"],
        "target_col": args.target_col,
        "horizon_hours": horizon_from_target_col(args.target_col),
        "evaluation_split": "test",
        "checkpoint_epoch": best_checkpoint["epoch"],
        "test_windows": len(test_dataset),
        "test_positive": test_pos,
        "test_negative": test_neg,
        "test_total_loss": test_losses["total"],
        "test_bce": test_losses["prediction"],
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }
    save_metrics_row(output_dir / "test_metrics.csv", test_row)

    print(f"訓練完成，總耗時 {perf_counter() - start_time:.1f} 秒")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Test AUROC: {test_metrics['auroc']:.4f} | Test AUPRC: {test_metrics['auprc']:.4f}")
    try_generate(generate_training_figures, metrics_path, output_dir)


if __name__ == "__main__":
    main()
