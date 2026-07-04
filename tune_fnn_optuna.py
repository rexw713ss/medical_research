"""Optuna hyperparameter tuning for the Knowledge-Guided Temporal FNN.

這支腳本用 Optuna 搜尋 `anfis_model.py` 裡 TemporalAttentionFNN 的關鍵超參數。
設計重點：
1. 資料只載入一次，每個 trial 只重建模型與 optimizer。
2. 使用與正式訓練相同的 SOFA label、patient-level split、24h sliding window。
3. 支援新版 explicit temporal inputs，並可搜尋其 contribution scale。
4. 目標函數預設最大化 validation AUROC，並記錄 AUPRC、loss 與各項正則化設定。
5. Optuna 不建立 test dataset；正式 test 僅在完成 tuning 與 validation 選模後使用。
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

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
    window_ids_for_mode,
    write_cohort_audit,
)
from train_fnn import (
    ICUWindowDataset,
    choose_device,
    load_training_frame,
    prepare_arrays,
    prepare_explicit_temporal_arrays,
    run_epoch,
)
from paper_figures import generate_tuning_figures, try_generate
from patient_split import split_ids_for_values
from project_config import (
    COMPARISON_PROTOCOL_JSON,
    EQUAL_SAMPLE_WINDOWS_CSV,
    PATIENT_SPLIT_CSV,
    PRIMARY_HOURLY_FEATURES,
)


@dataclass
class TuningData:
    train_dataset: Dataset
    val_dataset: Dataset
    train_pos: int
    train_neg: int
    cohort_records: list[dict[str, Any]]
    input_order: list[str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def zero_to_none(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return value


def dataset_label_counts(dataset: Dataset) -> tuple[int, int]:
    """計算 Dataset 或 Subset 的正負類數量，用於 BCE pos_weight。"""
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


def dataset_cohort_record(dataset: Dataset, split: str, target_col: str) -> dict[str, Any]:
    """建立 Dataset/Subset 的 target-window fingerprint。"""
    if hasattr(dataset, "cohort_record"):
        return dataset.cohort_record(split, target_col)
    if isinstance(dataset, Subset) and hasattr(dataset.dataset, "window_starts"):
        base = dataset.dataset
        subset_indices = np.asarray(dataset.indices, dtype=np.int64)
        starts = base.window_starts[subset_indices]
        targets = starts + base.seq_length - 1
        return cohort_record(
            base.stay_ids[targets],
            base.time_values[targets],
            base.labels[targets],
            split,
            target_col,
        )
    raise TypeError("Unsupported tuning dataset type for cohort audit.")


def stratified_window_subset(dataset: Dataset, max_windows: int | None, seed: int) -> Dataset:
    """保留原始正負類比例抽樣 window，避免 tuning 子集剛好抽不到惡化樣本。"""
    if max_windows is None or max_windows <= 0 or len(dataset) <= max_windows:
        return dataset

    if not hasattr(dataset, "window_starts"):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(dataset), size=max_windows, replace=False)
        return Subset(dataset, indices.tolist())

    targets = dataset.labels[dataset.window_starts + dataset.seq_length - 1]
    pos_indices = np.flatnonzero(targets == 1)
    neg_indices = np.flatnonzero(targets == 0)
    rng = np.random.default_rng(seed)

    if len(pos_indices) == 0 or len(neg_indices) == 0:
        indices = rng.choice(len(dataset), size=max_windows, replace=False)
        return Subset(dataset, indices.tolist())

    pos_frac = len(pos_indices) / len(dataset)
    n_pos = min(len(pos_indices), max(1, int(round(max_windows * pos_frac))))
    n_neg = min(len(neg_indices), max_windows - n_pos)

    selected = np.concatenate(
        [
            rng.choice(pos_indices, size=n_pos, replace=False),
            rng.choice(neg_indices, size=n_neg, replace=False),
        ]
    )
    rng.shuffle(selected)
    return Subset(dataset, selected.tolist())


def build_tuning_data(
    args: argparse.Namespace,
    protocol: dict[str, Any],
) -> TuningData:
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到訓練資料: {csv_path}")

    print(f"讀取 tuning 資料: {csv_path}")
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
        max_rows=zero_to_none(args.max_rows),
        max_stays=zero_to_none(args.max_stays),
        chunk_size=args.chunk_size,
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
    gc.collect()

    if args.split_col != "subject_id":
        raise ValueError("正式 tuning 必須以 subject_id 做 patient-level split。")
    train_ids, val_ids, _ = split_ids_for_values(split_values, args.split_manifest)
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

    train_dataset = stratified_window_subset(train_dataset, zero_to_none(args.max_train_windows), args.seed)
    val_dataset = stratified_window_subset(val_dataset, zero_to_none(args.max_val_windows), args.seed + 11)

    cohort_records = [
        dataset_cohort_record(train_dataset, "train", args.target_col),
        dataset_cohort_record(val_dataset, "validation", args.target_col),
    ]
    validate_cohort_records(
        cohort_records,
        protocol,
        args.comparison_mode,
        allow_incomplete=args.allow_incomplete_cohort,
    )

    train_pos, train_neg = dataset_label_counts(train_dataset)
    val_pos, val_neg = dataset_label_counts(val_dataset)
    print(f"Train windows: {len(train_dataset):,} | positive: {train_pos:,} | negative: {train_neg:,}")
    print(f"Val windows:   {len(val_dataset):,} | positive: {val_pos:,} | negative: {val_neg:,}")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("train 或 validation 沒有可用 window；請增加 max_stays/max_rows。")
    if train_pos == 0 or train_neg == 0:
        raise ValueError("tuning train set 缺少正類或負類；請增加 max_stays/max_train_windows。")

    return TuningData(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_pos=train_pos,
        train_neg=train_neg,
        cohort_records=cohort_records,
        input_order=input_order,
    )


def make_trial_loaders(
    tuning_data: TuningData,
    batch_size: int,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader]:
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        tuning_data.train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        tuning_data.val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def suggest_hparams(trial: optuna.Trial, args: argparse.Namespace) -> dict[str, Any]:
    """定義搜尋空間；範圍保守，避免破壞臨床先驗太多。"""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", args.lr_min, args.lr_max, log=True),
        "weight_decay": trial.suggest_float("weight_decay", args.weight_decay_min, args.weight_decay_max, log=True),
        "batch_size": trial.suggest_categorical("batch_size", args.batch_size_choices),
        "rule_score_scale": trial.suggest_float("rule_score_scale", args.rule_score_scale_min, args.rule_score_scale_max),
        "threshold": trial.suggest_float("threshold", args.threshold_min, args.threshold_max),
        "attention_hidden": trial.suggest_categorical("attention_hidden", args.attention_hidden_choices),
        "lambda_cons": trial.suggest_float("lambda_cons", args.lambda_cons_min, args.lambda_cons_max),
        "lambda_sparse": trial.suggest_float("lambda_sparse", args.lambda_sparse_min, args.lambda_sparse_max, log=True),
        "lambda_drift": trial.suggest_float("lambda_drift", args.lambda_drift_min, args.lambda_drift_max, log=True),
        "lambda_nonnegative": trial.suggest_float(
            "lambda_nonnegative",
            args.lambda_nonnegative_min,
            args.lambda_nonnegative_max,
        ),
        "grad_clip": trial.suggest_categorical("grad_clip", args.grad_clip_choices),
    }
    if args.explicit_temporal_features:
        params["explicit_temporal_scale"] = trial.suggest_float(
            "explicit_temporal_scale",
            args.explicit_temporal_scale_min,
            args.explicit_temporal_scale_max,
            log=True,
        )
    return params


def objective_factory(
    tuning_data: TuningData,
    device: torch.device,
    output_dir: Path,
    args: argparse.Namespace,
):
    trial_metrics_path = output_dir / "trial_metrics.csv"

    def objective(trial: optuna.Trial) -> float:
        set_seed(args.seed + trial.number)
        hparams = suggest_hparams(trial, args)

        train_loader, val_loader = make_trial_loaders(
            tuning_data=tuning_data,
            batch_size=int(hparams["batch_size"]),
            device=device,
            args=args,
        )

        model = TemporalAttentionFNN(
            seq_length=args.seq_length,
            attention_hidden=int(hparams["attention_hidden"]),
            threshold=float(hparams["threshold"]),
            rule_score_scale=float(hparams["rule_score_scale"]),
            use_explicit_temporal_features=args.explicit_temporal_features,
            explicit_temporal_scale=float(
                hparams.get("explicit_temporal_scale", args.explicit_temporal_scale)
            ),
        ).to(device)

        pos_weight = torch.tensor(
            [tuning_data.train_neg / max(tuning_data.train_pos, 1)],
            dtype=torch.float32,
            device=device,
        )
        criterion = NeuroSymbolicLoss(
            lambda_cons=float(hparams["lambda_cons"]),
            lambda_sparse=float(hparams["lambda_sparse"]),
            lambda_drift=float(hparams["lambda_drift"]),
            lambda_nonnegative=float(hparams["lambda_nonnegative"]),
        )
        criterion.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        criterion = criterion.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(hparams["learning_rate"]),
            weight_decay=float(hparams["weight_decay"]),
        )

        best_val_auroc = 0.0
        best_val_auprc = 0.0
        best_val_loss = math.inf
        trial_start = perf_counter()

        try:
            for epoch in range(1, args.trial_epochs + 1):
                train_losses, train_metrics = run_epoch(
                    model=model,
                    loader=train_loader,
                    criterion=criterion,
                    device=device,
                    optimizer=optimizer,
                    grad_clip=float(hparams["grad_clip"]),
                    max_batches=zero_to_none(args.limit_train_batches),
                )
                val_losses, val_metrics = run_epoch(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                    optimizer=None,
                    grad_clip=None,
                    max_batches=zero_to_none(args.limit_val_batches),
                )

                current_auroc = float(val_metrics.get("auroc", 0.0))
                current_auprc = float(val_metrics.get("auprc", 0.0))
                current_loss = float(val_losses.get("total", math.inf))
                if math.isnan(current_auroc):
                    current_auroc = 0.0
                if math.isnan(current_auprc):
                    current_auprc = 0.0

                best_val_auroc = max(best_val_auroc, current_auroc)
                best_val_auprc = max(best_val_auprc, current_auprc)
                best_val_loss = min(best_val_loss, current_loss)

                trial.report(current_auroc, epoch)
                trial.set_user_attr("best_val_auprc", best_val_auprc)
                trial.set_user_attr("best_val_loss", best_val_loss)
                trial.set_user_attr("last_train_loss", float(train_losses.get("total", math.nan)))
                trial.set_user_attr("last_val_loss", current_loss)

                append_trial_metrics(
                    trial_metrics_path,
                    {
                        "trial": trial.number,
                        "epoch": epoch,
                        "train_total_loss": train_losses["total"],
                        "train_bce": train_losses["prediction"],
                        "train_auroc": train_metrics["auroc"],
                        "train_auprc": train_metrics["auprc"],
                        "val_total_loss": val_losses["total"],
                        "val_bce": val_losses["prediction"],
                        "val_auroc": current_auroc,
                        "val_auprc": current_auprc,
                        "seconds": perf_counter() - trial_start,
                        **hparams,
                    },
                )

                if trial.should_prune():
                    raise optuna.TrialPruned()

            return best_val_auroc
        finally:
            del model, criterion, optimizer, train_loader, val_loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return objective


def append_trial_metrics(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_best_outputs(study: optuna.Study, output_dir: Path, args: argparse.Namespace) -> None:
    best = {
        "best_value_val_auroc": study.best_value,
        "best_params": study.best_params,
        "best_trial_number": study.best_trial.number,
        "best_trial_user_attrs": study.best_trial.user_attrs,
        "model_design": "explicit_temporal" if args.explicit_temporal_features else "sequence_only",
        "target_col": args.target_col,
        "seq_length": args.seq_length,
        "comparison_mode": args.comparison_mode,
    }
    best_path = output_dir / "best_params.json"
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    command_parts = [
        ".\\env\\Scripts\\python.exe",
        ".\\train_fnn.py",
        f"--csv {args.csv}",
        f"--target-col {args.target_col}",
        f"--time-col {args.time_col}",
        f"--split-col {args.split_col}",
        f"--split-manifest {args.split_manifest}",
        "--comparison-mode full",
        f"--comparison-protocol {args.comparison_protocol}",
        f"--equal-sample-windows {args.equal_sample_windows}",
        f"--seq-length {args.seq_length}",
        f"--val-frac {args.val_frac}",
        f"--learning-rate {study.best_params['learning_rate']}",
        f"--weight-decay {study.best_params['weight_decay']}",
        f"--batch-size {study.best_params['batch_size']}",
        f"--grad-clip {study.best_params['grad_clip']}",
        f"--attention-hidden {study.best_params['attention_hidden']}",
        f"--threshold {study.best_params['threshold']}",
        f"--rule-score-scale {study.best_params['rule_score_scale']}",
        f"--lambda-cons {study.best_params['lambda_cons']}",
        f"--lambda-sparse {study.best_params['lambda_sparse']}",
        f"--lambda-drift {study.best_params['lambda_drift']}",
        f"--lambda-nonnegative {study.best_params['lambda_nonnegative']}",
        "--epochs 20",
        "--early-stopping-patience 5",
        "--early-stopping-min-epochs 10",
        "--output-dir outputs\\explicit_temporal_fnn_formal_6h\\seed_42",
    ]
    if args.explicit_temporal_features:
        command_parts.extend(
            [
                "--explicit-temporal-features",
                f"--explicit-temporal-scale {study.best_params['explicit_temporal_scale']}",
            ]
        )
    command_path = output_dir / "train_with_best_params.ps1"
    command_path.write_text(" ".join(command_parts) + "\n", encoding="utf-8")

    trials_path = output_dir / "optuna_trials.csv"
    study.trials_dataframe().to_csv(trials_path, index=False)

    print("\n" + "=" * 60)
    print("Optuna tuning 完成")
    print("=" * 60)
    print(f"最佳 Validation AUROC: {study.best_value:.4f}")
    print(f"最佳參數已儲存: {best_path}")
    print(f"正式訓練指令已儲存: {command_path}")
    print("最佳超參數:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")


def parse_csv_list(raw: str, cast_type: type) -> list[Any]:
    return [cast_type(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna tuning for Knowledge-Guided Temporal FNN.")
    parser.add_argument("--csv", default=PRIMARY_HOURLY_FEATURES)
    parser.add_argument("--target-col", default="label_sofa_increase_ge2_6h")
    parser.add_argument("--time-col", default="sofa_hour")
    parser.add_argument("--split-col", default="subject_id")
    parser.add_argument("--split-manifest", default=PATIENT_SPLIT_CSV)
    parser.add_argument("--comparison-mode", choices=["full", "equal_sample"], default="full")
    parser.add_argument("--comparison-protocol", default=COMPARISON_PROTOCOL_JSON)
    parser.add_argument("--equal-sample-windows", default=EQUAL_SAMPLE_WINDOWS_CSV)
    parser.add_argument("--seq-length", type=int, default=24)
    parser.add_argument(
        "--explicit-temporal-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Tune the explicit raw + missingness + time-since temporal FNN.",
    )
    parser.add_argument("--explicit-temporal-scale", type=float, default=1.0)
    parser.add_argument("--allow-incomplete-cohort", action="store_true", help="僅供 smoke test。")
    parser.add_argument("--val-frac", type=float, default=0.15, help="舊版相容參數；正式比例由 manifest 決定。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--max-rows", type=int, default=0, help="0 代表不限制；tuning 通常用 max-stays 即可。")
    parser.add_argument("--max-stays", type=int, default=0, help="0 代表不限制。")
    parser.add_argument("--max-train-windows", type=int, default=0, help="0 代表不限制。")
    parser.add_argument("--max-val-windows", type=int, default=0, help="0 代表不限制。")
    parser.add_argument("--chunk-size", type=int, default=500_000)

    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--trial-epochs", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=None, help="Optuna 搜尋秒數上限。")
    parser.add_argument("--limit-train-batches", type=int, default=0, help="0 代表每個 epoch 跑完整 tuning train set。")
    parser.add_argument("--limit-val-batches", type=int, default=0, help="0 代表每個 epoch 跑完整 tuning val set。")

    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--lr-max", type=float, default=5e-3)
    parser.add_argument("--weight-decay-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay-max", type=float, default=1e-3)
    parser.add_argument("--rule-score-scale-min", type=float, default=0.01)
    parser.add_argument("--rule-score-scale-max", type=float, default=0.5)
    parser.add_argument("--threshold-min", type=float, default=5.0)
    parser.add_argument("--threshold-max", type=float, default=9.0)
    parser.add_argument("--lambda-cons-min", type=float, default=0.0)
    parser.add_argument("--lambda-cons-max", type=float, default=0.5)
    parser.add_argument("--lambda-sparse-min", type=float, default=1e-5)
    parser.add_argument("--lambda-sparse-max", type=float, default=1e-2)
    parser.add_argument("--lambda-drift-min", type=float, default=1e-5)
    parser.add_argument("--lambda-drift-max", type=float, default=1e-2)
    parser.add_argument("--lambda-nonnegative-min", type=float, default=0.01)
    parser.add_argument("--lambda-nonnegative-max", type=float, default=0.1)
    parser.add_argument("--explicit-temporal-scale-min", type=float, default=0.1)
    parser.add_argument("--explicit-temporal-scale-max", type=float, default=2.0)
    parser.add_argument("--batch-size-choices", default="256,512,1024")
    parser.add_argument("--attention-hidden-choices", default="16,32,64,128")
    parser.add_argument("--grad-clip-choices", default="1.0,3.0,5.0,10.0")

    parser.add_argument("--study-name", default=None)
    parser.add_argument("--storage", default=None, help="例如 sqlite:///outputs/fnn_tuning/study.db；空值會自動建立。")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.batch_size_choices = parse_csv_list(args.batch_size_choices, int)
    args.attention_hidden_choices = parse_csv_list(args.attention_hidden_choices, int)
    args.grad_clip_choices = parse_csv_list(args.grad_clip_choices, float)
    return args


def main() -> None:
    args = normalize_args(parse_args())
    set_seed(args.seed)
    protocol = validate_comparison_args(
        args.comparison_mode,
        args.comparison_protocol,
        args.target_col,
        args.seq_length,
    )

    run_name = datetime.now().strftime("optuna_fnn_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "fnn_tuning" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"使用裝置: {device}")
    print(f"輸出資料夾: {output_dir}")

    tuning_data = build_tuning_data(args, protocol)
    write_cohort_audit(output_dir / "cohort_audit.json", tuning_data.cohort_records)

    with (output_dir / "tuning_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "feature_order": FEATURE_ORDER,
                "input_order": tuning_data.input_order,
                "model_design": (
                    "explicit_temporal" if args.explicit_temporal_features else "sequence_only"
                ),
                "train_windows": len(tuning_data.train_dataset),
                "val_windows": len(tuning_data.val_dataset),
                "train_positive": tuning_data.train_pos,
                "train_negative": tuning_data.train_neg,
                "comparison_protocol_sha256": protocol["protocol_sha256"],
                "cohort_audit": tuning_data.cohort_records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    storage = args.storage
    if storage is None:
        storage = f"sqlite:///{(output_dir / 'optuna_study.db').resolve().as_posix()}"
    study_name = args.study_name or output_dir.name

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )

    remaining_trials = max(args.n_trials - len(study.trials), 0)
    print(
        f"啟動 Optuna 搜尋：目標共 {args.n_trials} trials，已存在 {len(study.trials)}，"
        f"本次執行 {remaining_trials}；每個 trial {args.trial_epochs} epochs"
    )
    if remaining_trials > 0:
        study.optimize(
            objective_factory(tuning_data, device, output_dir, args),
            n_trials=remaining_trials,
            timeout=args.timeout,
            gc_after_trial=True,
        )

    write_best_outputs(study, output_dir, args)
    try_generate(generate_tuning_figures, output_dir / "optuna_trials.csv", output_dir)


if __name__ == "__main__":
    main()
