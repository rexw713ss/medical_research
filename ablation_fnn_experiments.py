"""Ablation experiments for Knowledge-Guided Temporal FNN.

消融目的：
1. Randomly initialized FNN
   檢驗 expert knowledge initialization 是否有貢獻。
2. Guideline-guided FNN without temporal features
   檢驗 temporal design 是否有貢獻。
3. Temporal FNN without clinical consistency regularization
   檢驗 clinical consistency regularization 是否有貢獻。
4. Full Knowledge-Guided Temporal FNN
   作為完整模型。

所有 variant 預設使用相同 SOFA outcome、相同 patient-level train/validation/test split、
相同訓練超參數，並輸出效能與規則品質指標。
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
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from anfis_model import (
    FEATURE_ORDER,
    NeuroSymbolicLoss,
    TemporalAttentionFNN,
    clinical_rule_priors,
    expert_feature_config,
)
from comparison_protocol import (
    cohort_record,
    load_protocol,
    validate_cohort_records,
    validate_comparison_args,
    window_id_membership,
    window_ids_for_mode,
    write_cohort_audit,
)
from paper_figures import generate_ablation_figures, generate_training_figures, try_generate
from patient_split import split_ids_for_values
from project_config import (
    COMPARISON_PROTOCOL_JSON,
    EQUAL_SAMPLE_WINDOWS_CSV,
    PATIENT_SPLIT_CSV,
    PRIMARY_HOURLY_FEATURES,
)
from sofa_label_utils import horizon_from_target_col
from train_fnn import (
    choose_device,
    load_training_frame,
    prepare_arrays,
    run_epoch,
)


VARIANT_ORDER = [
    "random_init",
    "static_guideline",
    "no_consistency",
    "full",
]

SUMMARY_METRICS = [
    "val_auroc",
    "val_auprc",
    "test_auroc",
    "test_auprc",
    "test_brier",
    "test_ece",
    "test_mce",
    "test_log_loss",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_specificity",
    "test_f1",
    "rule_drift_loss",
    "active_rule_fraction_gt_0_1",
    "attention_entropy",
    "attention_last_6h_mass",
]

T_CRITICAL_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


@dataclass
class VariantSpec:
    name: str
    display_name: str
    description: str
    input_seq_length: int
    min_history_length: int
    feature_configs: dict[str, list[dict]]
    rule_configs: list[dict] | None
    expert_init: bool
    temporal_design: bool
    clinical_consistency: bool
    lambda_cons: float
    lambda_drift: float
    explicit_temporal_features: bool = False


class AblationWindowDataset(Dataset):
    """建立固定 target time 的 sliding-window dataset。

    `input_seq_length` 控制模型實際看幾小時。
    `min_history_length` 控制 target row 至少需要累積幾小時歷史。
    因此 static variant 可設定 input_seq_length=1, min_history_length=24，
    讓它只看當前小時，但評估時間點仍與 24h temporal model 對齊。
    """

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        stay_ids: np.ndarray,
        split_values: np.ndarray,
        time_values: np.ndarray,
        allowed_split_values: set,
        input_seq_length: int,
        min_history_length: int,
        allowed_window_ids: np.ndarray | None = None,
        require_all_window_ids: bool = True,
    ) -> None:
        self.features = features
        self.labels = labels
        self.input_seq_length = input_seq_length
        self.min_history_length = min_history_length
        self.stay_ids = stay_ids
        self.time_values = time_values
        self.window_starts, self.target_indices = self._build_windows(
            stay_ids=stay_ids,
            split_values=split_values,
            allowed_split_values=allowed_split_values,
            time_values=time_values,
            allowed_window_ids=allowed_window_ids,
        )
        if require_all_window_ids and allowed_window_ids is not None and len(self) != len(allowed_window_ids):
            raise ValueError(
                f"Equal-sample windows 缺少 {len(allowed_window_ids) - len(self):,} 筆；"
                "正式比較不可限制 rows/stays/windows。"
            )

    def _build_windows(
        self,
        stay_ids: np.ndarray,
        split_values: np.ndarray,
        allowed_split_values: set,
        time_values: np.ndarray,
        allowed_window_ids: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        starts_by_stay = []
        targets_by_stay = []
        valid_feature_row = np.isfinite(self.features).all(axis=1)
        valid_label_row = np.isfinite(self.labels)

        boundaries = np.flatnonzero(stay_ids[1:] != stay_ids[:-1]) + 1
        stay_starts = np.concatenate(([0], boundaries))
        stay_ends = np.concatenate((boundaries, [len(stay_ids)]))

        for stay_start, stay_end in zip(stay_starts, stay_ends):
            if split_values[stay_start] not in allowed_split_values:
                continue

            stay_len = stay_end - stay_start
            if stay_len < self.min_history_length:
                continue

            local_targets = np.arange(
                stay_start + self.min_history_length - 1,
                stay_end,
                dtype=np.int64,
            )
            local_starts = local_targets - self.input_seq_length + 1
            valid_range = local_starts >= stay_start
            local_targets = local_targets[valid_range]
            local_starts = local_starts[valid_range]

            feature_valid_count = np.concatenate(
                ([0], np.cumsum(valid_feature_row[stay_start:stay_end], dtype=np.int32))
            )
            relative_starts = local_starts - stay_start
            relative_ends = relative_starts + self.input_seq_length
            window_valid = (
                feature_valid_count[relative_ends] - feature_valid_count[relative_starts]
            ) == self.input_seq_length
            label_valid = valid_label_row[local_targets]

            keep = window_valid & label_valid
            if allowed_window_ids is not None:
                target_window_ids = (
                    stay_ids[local_targets].astype(np.int64) * 100_000
                    + time_values[local_targets].astype(np.int64)
                )
                keep &= window_id_membership(target_window_ids, allowed_window_ids)
            if np.any(keep):
                starts_by_stay.append(local_starts[keep])
                targets_by_stay.append(local_targets[keep])

        if not starts_by_stay:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        return (
            np.concatenate(starts_by_stay).astype(np.int64, copy=False),
            np.concatenate(targets_by_stay).astype(np.int64, copy=False),
        )

    def __len__(self) -> int:
        return int(self.window_starts.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.window_starts[index])
        end = start + self.input_seq_length
        target_index = int(self.target_indices[index])
        return (
            torch.from_numpy(self.features[start:end]),
            torch.tensor(self.labels[target_index], dtype=torch.float32),
        )

    def label_counts(self) -> tuple[int, int]:
        if len(self) == 0:
            return 0, 0
        targets = self.labels[self.target_indices]
        positives = int(np.sum(targets == 1))
        negatives = int(np.sum(targets == 0))
        return positives, negatives

    def cohort_record(self, split: str, target_col: str) -> dict[str, Any]:
        return cohort_record(
            self.stay_ids[self.target_indices],
            self.time_values[self.target_indices],
            self.labels[self.target_indices],
            split,
            target_col,
        )


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
    if hasattr(dataset, "label_counts"):
        return dataset.label_counts()

    if isinstance(dataset, Subset) and hasattr(dataset.dataset, "target_indices"):
        base_dataset = dataset.dataset
        subset_indices = np.asarray(dataset.indices, dtype=np.int64)
        targets = base_dataset.labels[base_dataset.target_indices[subset_indices]]
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
    if hasattr(dataset, "cohort_record"):
        return dataset.cohort_record(split, target_col)
    if isinstance(dataset, Subset) and hasattr(dataset.dataset, "target_indices"):
        base = dataset.dataset
        subset_indices = np.asarray(dataset.indices, dtype=np.int64)
        targets = base.target_indices[subset_indices]
        return cohort_record(
            base.stay_ids[targets],
            base.time_values[targets],
            base.labels[targets],
            split,
            target_col,
        )
    raise TypeError("Unsupported dataset type for cohort audit.")


def stratified_subset(dataset: Dataset, max_windows: int | None, seed: int) -> Dataset:
    if max_windows is None or max_windows <= 0 or len(dataset) <= max_windows:
        return dataset

    if not hasattr(dataset, "target_indices"):
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(dataset), size=max_windows, replace=False)
        return Subset(dataset, indices.tolist())

    targets = dataset.labels[dataset.target_indices]
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


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def parse_seeds(raw: str | None, fallback_seed: int) -> list[int]:
    if not raw:
        return [fallback_seed]
    seeds = []
    for value in raw.split(","):
        value = value.strip()
        if value:
            seeds.append(int(value))
    if not seeds:
        raise ValueError("--seeds 至少要包含一個整數。")
    if len(seeds) != len(set(seeds)):
        raise ValueError("--seeds 不可包含重複值。")
    return seeds


def mean_std_ci(values: pd.Series) -> tuple[int, float, float, float, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    n = int(len(numeric))
    if n == 0:
        return 0, math.nan, math.nan, math.nan, math.nan
    mean = float(numeric.mean())
    std = float(numeric.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        t_critical = T_CRITICAL_95.get(n, 1.96)
        half_width = t_critical * std / math.sqrt(n)
    else:
        half_width = math.nan
    return n, mean, std, mean - half_width, mean + half_width


def aggregate_ablation_results(summary: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    aggregate_rows = []
    group_cols = [
        "target_col",
        "horizon_hours",
        "variant",
        "display_name",
        "expert_init",
        "temporal_design",
        "clinical_consistency",
    ]
    for keys, group in summary.groupby(group_cols, dropna=False, sort=False):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(group["seed"].nunique())
        row["seeds"] = ",".join(str(value) for value in sorted(group["seed"].astype(int).unique()))
        for metric in SUMMARY_METRICS:
            if metric not in group.columns:
                continue
            _, mean, std, ci_low, ci_high = mean_std_ci(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = ci_low
            row[f"{metric}_ci95_high"] = ci_high
        aggregate_rows.append(row)

    aggregate_path = output_dir / "ablation_aggregate.csv"
    pd.DataFrame(aggregate_rows).to_csv(aggregate_path, index=False)

    # 配對差值能分別對應三個研究問題，避免 full-vs-static 同時混入兩種設計差異。
    contrasts = [
        ("expert_knowledge_initialization", "full", "random_init"),
        ("temporal_feature_design", "no_consistency", "static_guideline"),
        ("clinical_consistency_regularization", "full", "no_consistency"),
    ]
    effect_rows = []
    for (target_col, horizon_hours), horizon_frame in summary.groupby(
        ["target_col", "horizon_hours"], sort=False
    ):
        for component, enabled_variant, disabled_variant in contrasts:
            enabled = horizon_frame[horizon_frame["variant"] == enabled_variant]
            disabled = horizon_frame[horizon_frame["variant"] == disabled_variant]
            paired = enabled.merge(
                disabled,
                on=["target_col", "horizon_hours", "seed"],
                suffixes=("_enabled", "_disabled"),
                validate="one_to_one",
            )
            if paired.empty:
                continue
            row: dict[str, Any] = {
                "component": component,
                "enabled_variant": enabled_variant,
                "disabled_variant": disabled_variant,
                "target_col": target_col,
                "horizon_hours": int(horizon_hours),
                "n_paired_seeds": int(len(paired)),
                "seeds": ",".join(
                    str(value) for value in sorted(paired["seed"].astype(int).unique())
                ),
            }
            for metric in ["test_auroc", "test_auprc", "test_brier", "test_ece", "rule_drift_loss"]:
                enabled_col = f"{metric}_enabled"
                disabled_col = f"{metric}_disabled"
                if enabled_col not in paired or disabled_col not in paired:
                    continue
                enabled_values = pd.to_numeric(paired[enabled_col], errors="coerce")
                disabled_values = pd.to_numeric(paired[disabled_col], errors="coerce")
                delta = enabled_values - disabled_values
                n, mean, std, ci_low, ci_high = mean_std_ci(delta)
                row[f"delta_{metric}_n"] = n
                row[f"delta_{metric}_mean"] = mean
                row[f"delta_{metric}_std"] = std
                row[f"delta_{metric}_ci95_low"] = ci_low
                row[f"delta_{metric}_ci95_high"] = ci_high
                try:
                    from scipy.stats import ttest_rel

                    finite = np.isfinite(enabled_values) & np.isfinite(disabled_values)
                    row[f"delta_{metric}_paired_t_p"] = float(
                        ttest_rel(enabled_values[finite], disabled_values[finite]).pvalue
                    ) if int(finite.sum()) >= 2 else math.nan
                except (ImportError, ValueError):
                    row[f"delta_{metric}_paired_t_p"] = math.nan
            effect_rows.append(row)

    effects_path = output_dir / "paired_component_effects.csv"
    pd.DataFrame(effect_rows).to_csv(effects_path, index=False)
    return aggregate_path, effects_path


def rebuild_summaries(output_dir: Path) -> pd.DataFrame:
    rows = []
    for result_path in output_dir.glob("seed_*/*/result.json"):
        try:
            rows.append(json.loads(result_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    if not rows:
        return pd.DataFrame()
    summary = pd.DataFrame(rows).sort_values(["horizon_hours", "seed", "variant"])
    summary.to_csv(output_dir / "ablation_summary.csv", index=False)
    aggregate_ablation_results(summary, output_dir)
    return summary


def load_best_params(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("best_params", data)


def apply_best_params(args: argparse.Namespace, params: dict[str, Any]) -> argparse.Namespace:
    mapping = {
        "learning_rate": "learning_rate",
        "weight_decay": "weight_decay",
        "batch_size": "batch_size",
        "rule_score_scale": "rule_score_scale",
        "threshold": "threshold",
        "attention_hidden": "attention_hidden",
        "lambda_cons": "lambda_cons",
        "lambda_sparse": "lambda_sparse",
        "lambda_drift": "lambda_drift",
        "lambda_nonnegative": "lambda_nonnegative",
        "explicit_temporal_scale": "explicit_temporal_scale",
        "grad_clip": "grad_clip",
    }
    for source_key, arg_key in mapping.items():
        if source_key in params:
            setattr(args, arg_key, params[source_key])
    return args


def make_random_feature_config(
    features: np.ndarray,
    seed: int,
    row_mask: np.ndarray | None = None,
) -> dict[str, list[dict]]:
    rng = np.random.default_rng(seed)
    configs: dict[str, list[dict]] = {}
    for feature_idx, feature in enumerate(FEATURE_ORDER):
        expert_terms = expert_feature_config[feature]
        values = features[:, feature_idx]
        if row_mask is not None:
            values = values[row_mask]
        values = values[np.isfinite(values)]
        if values.size == 0:
            low, high = 0.0, 1.0
        else:
            low, high = np.percentile(values, [1, 99])
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                low, high = float(np.min(values)), float(np.max(values) + 1.0)
            if high <= low:
                high = low + 1.0

        n_terms = len(expert_terms)
        span = max(float(high - low), 1e-3)
        centers = np.sort(rng.uniform(low, high, size=n_terms))
        sigmas = rng.uniform(span / (n_terms * 3.0), span / max(n_terms, 1), size=n_terms)
        sigmas = np.clip(sigmas, 1e-3, 80.0)
        weights = rng.uniform(0.0, 4.0, size=n_terms)

        configs[feature] = [
            {
                "name": expert_term["name"],
                "center": float(center),
                "sigma": float(sigma),
                "weight": float(weight),
            }
            for expert_term, center, sigma, weight in zip(expert_terms, centers, sigmas, weights)
        ]
    return configs


def make_random_rule_configs(
    feature_configs: dict[str, list[dict]],
    seed: int,
    n_rules: int | None = None,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    features = list(feature_configs.keys())
    n_rules = n_rules if n_rules is not None else len(clinical_rule_priors)
    rules = []
    for rule_idx in range(n_rules):
        antecedent_count = len(clinical_rule_priors[rule_idx % len(clinical_rule_priors)]["antecedents"])
        selected_features = rng.choice(features, size=min(antecedent_count, len(features)), replace=False)
        antecedents = []
        for feature in selected_features:
            terms = [config["name"] for config in feature_configs[feature]]
            antecedents.append((feature, str(rng.choice(terms))))
        rules.append(
            {
                "name": f"random_rule_{rule_idx + 1}",
                "antecedents": antecedents,
                "weight": float(rng.uniform(0.0, 5.0)),
            }
        )
    return rules


def build_variants(
    args: argparse.Namespace,
    random_feature_configs: dict[str, list[dict]],
    random_rule_configs: list[dict],
) -> dict[str, VariantSpec]:
    static_min_history = args.static_min_history_hours
    return {
        "random_init": VariantSpec(
            name="random_init",
            display_name="Randomly initialized FNN",
            description="Same temporal architecture, but fuzzy sets and cross-feature rules are randomly initialized.",
            input_seq_length=args.seq_length,
            min_history_length=args.seq_length,
            feature_configs=random_feature_configs,
            rule_configs=random_rule_configs,
            expert_init=False,
            temporal_design=True,
            clinical_consistency=True,
            lambda_cons=args.lambda_cons,
            lambda_drift=args.random_lambda_drift,
        ),
        "static_guideline": VariantSpec(
            name="static_guideline",
            display_name="Guideline-guided FNN without temporal features",
            description="Expert-guided FNN using only the current hour; target times remain aligned to temporal variants.",
            input_seq_length=1,
            min_history_length=static_min_history,
            feature_configs=expert_feature_config,
            rule_configs=clinical_rule_priors,
            expert_init=True,
            temporal_design=False,
            clinical_consistency=False,
            lambda_cons=0.0,
            lambda_drift=args.lambda_drift,
        ),
        "no_consistency": VariantSpec(
            name="no_consistency",
            display_name="Temporal FNN without clinical consistency regularization",
            description="Expert-guided temporal FNN with lambda_cons set to zero.",
            input_seq_length=args.seq_length,
            min_history_length=args.seq_length,
            feature_configs=expert_feature_config,
            rule_configs=clinical_rule_priors,
            expert_init=True,
            temporal_design=True,
            clinical_consistency=False,
            lambda_cons=0.0,
            lambda_drift=args.lambda_drift,
        ),
        "full": VariantSpec(
            name="full",
            display_name="Full Knowledge-Guided Temporal FNN",
            description="Expert-guided temporal FNN with clinical consistency regularization.",
            input_seq_length=args.seq_length,
            min_history_length=args.seq_length,
            feature_configs=expert_feature_config,
            rule_configs=clinical_rule_priors,
            expert_init=True,
            temporal_design=True,
            clinical_consistency=True,
            lambda_cons=args.lambda_cons,
            lambda_drift=args.lambda_drift,
        ),
    }


def parse_variants(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return VARIANT_ORDER
    selected = []
    for item in raw.split(","):
        name = item.strip().lower()
        if not name:
            continue
        if name not in VARIANT_ORDER:
            raise ValueError(f"Unknown variant: {name}")
        selected.append(name)
    return selected


def build_datasets_for_variant(
    spec: VariantSpec,
    features: np.ndarray,
    labels: np.ndarray,
    stay_ids: np.ndarray,
    split_values: np.ndarray,
    time_values: np.ndarray,
    train_ids: set,
    val_ids: set,
    test_ids: set,
    train_window_ids: np.ndarray | None,
    val_window_ids: np.ndarray | None,
    args: argparse.Namespace,
) -> tuple[Dataset, Dataset, Dataset]:
    train_dataset = AblationWindowDataset(
        features=features,
        labels=labels,
        stay_ids=stay_ids,
        split_values=split_values,
        time_values=time_values,
        allowed_split_values=train_ids,
        input_seq_length=spec.input_seq_length,
        min_history_length=spec.min_history_length,
        allowed_window_ids=train_window_ids,
        require_all_window_ids=not args.allow_incomplete_cohort,
    )
    val_dataset = AblationWindowDataset(
        features=features,
        labels=labels,
        stay_ids=stay_ids,
        split_values=split_values,
        time_values=time_values,
        allowed_split_values=val_ids,
        input_seq_length=spec.input_seq_length,
        min_history_length=spec.min_history_length,
        allowed_window_ids=val_window_ids,
        require_all_window_ids=not args.allow_incomplete_cohort,
    )
    test_dataset = AblationWindowDataset(
        features=features,
        labels=labels,
        stay_ids=stay_ids,
        split_values=split_values,
        time_values=time_values,
        allowed_split_values=test_ids,
        input_seq_length=spec.input_seq_length,
        min_history_length=spec.min_history_length,
    )
    train_dataset = stratified_subset(train_dataset, zero_to_none(args.max_train_windows), args.seed)
    val_dataset = stratified_subset(val_dataset, zero_to_none(args.max_val_windows), args.seed + 17)
    test_dataset = stratified_subset(test_dataset, zero_to_none(args.max_test_windows), args.seed + 29)
    return train_dataset, val_dataset, test_dataset


def make_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    device: torch.device,
    args: argparse.Namespace,
    run_seed: int,
) -> tuple[DataLoader, DataLoader]:
    pin_memory = device.type == "cuda"
    generator = torch.Generator()
    generator.manual_seed(run_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def finite_or_zero(value: float) -> float:
    return 0.0 if value is None or math.isnan(float(value)) else float(value)


def evaluate_rule_quality(
    model: TemporalAttentionFNN,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    max_batches: int | None,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    activation_sum = 0.0
    activation_count = 0
    active_count = 0
    top_activation_sum = 0.0
    top_activation_count = 0
    attention_entropy_sum = 0.0
    attention_last_6h_sum = 0.0
    attention_count = 0
    with torch.no_grad():
        for batch_idx, (batch_x, _) in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            output = model(batch_x.to(device))
            if output.rule_activations.numel() > 0:
                activations = output.rule_activations.detach()
                activation_sum += float(activations.sum().item())
                activation_count += int(activations.numel())
                active_count += int((activations > 0.1).sum().item())
                top_values = activations.max(dim=-1).values
                top_activation_sum += float(top_values.sum().item())
                top_activation_count += int(top_values.numel())

            attention = output.attention_weights.detach()
            seq_len = attention.shape[1]
            if seq_len > 1:
                entropy = -(attention * torch.log(attention + 1e-8)).sum(dim=1) / math.log(seq_len)
                attention_entropy_sum += float(entropy.sum().item())
            attention_last_6h_sum += float(
                attention[:, -min(6, seq_len) :].sum(dim=1).sum().item()
            )
            attention_count += int(attention.shape[0])

    rule_count = len(model.static_fnn.rule_configs)
    antecedent_counts = [len(rule["antecedents"]) for rule in model.static_fnn.rule_configs]
    avg_antecedents = float(np.mean(antecedent_counts)) if antecedent_counts else 0.0
    cross_weights = model.static_fnn.cross_rule_weights.detach().cpu() if rule_count else torch.empty(0)

    mean_rule_activation = activation_sum / max(activation_count, 1)
    active_rule_fraction = active_count / max(activation_count, 1)
    top_rule_activation = top_activation_sum / max(top_activation_count, 1)
    attention_entropy = attention_entropy_sum / max(attention_count, 1)
    attention_last_6h = attention_last_6h_sum / max(attention_count, 1)

    return {
        "num_cross_rules": float(rule_count),
        "avg_rule_antecedents": avg_antecedents,
        "mean_rule_activation": mean_rule_activation,
        "active_rule_fraction_gt_0_1": active_rule_fraction,
        "mean_top_rule_activation": top_rule_activation,
        "rule_sparsity_loss": float(model.static_fnn.sparsity_loss().detach().cpu().item()),
        "rule_drift_loss": float(model.static_fnn.drift_loss().detach().cpu().item()),
        "nonnegative_weight_loss": float(model.static_fnn.nonnegative_weight_loss().detach().cpu().item()),
        "negative_cross_rule_count": float((cross_weights < 0).sum().item()) if rule_count else 0.0,
        "attention_entropy": attention_entropy,
        "attention_last_6h_mass": attention_last_6h,
    }


def train_variant(
    spec: VariantSpec,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    device: torch.device,
    output_dir: Path,
    args: argparse.Namespace,
    run_seed: int,
) -> dict[str, Any]:
    variant_dir = output_dir / f"seed_{run_seed}" / spec.name
    variant_dir.mkdir(parents=True, exist_ok=True)

    train_pos, train_neg = dataset_label_counts(train_dataset)
    val_pos, val_neg = dataset_label_counts(val_dataset)
    test_pos, test_neg = dataset_label_counts(test_dataset)
    if train_pos == 0 or train_neg == 0:
        raise ValueError(f"{spec.name} train set 缺少正類或負類，請增加資料量。")
    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        raise ValueError(f"{spec.name} 沒有可用 window。")

    set_seed(run_seed)
    train_loader, val_loader = make_loaders(train_dataset, val_dataset, device, args, run_seed)

    model = TemporalAttentionFNN(
        feature_configs=spec.feature_configs,
        rule_configs=spec.rule_configs,
        seq_length=spec.input_seq_length,
        attention_hidden=args.attention_hidden,
        threshold=args.threshold,
        rule_score_scale=args.rule_score_scale,
        use_explicit_temporal_features=spec.explicit_temporal_features,
        explicit_temporal_scale=args.explicit_temporal_scale,
    ).to(device)

    criterion = NeuroSymbolicLoss(
        lambda_cons=spec.lambda_cons,
        lambda_sparse=args.lambda_sparse,
        lambda_drift=spec.lambda_drift,
        lambda_nonnegative=args.lambda_nonnegative,
    )
    pos_weight = torch.tensor([train_neg / max(train_pos, 1)], dtype=torch.float32, device=device)
    criterion.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    epoch_metrics_path = variant_dir / "epoch_metrics.csv"
    if epoch_metrics_path.exists():
        epoch_metrics_path.unlink()
    best_state = None
    best_epoch = 0
    best_val_auroc = -math.inf
    best_val_loss = math.inf
    best_train_losses: dict[str, float] = {}
    best_train_metrics: dict[str, float] = {}
    best_val_losses: dict[str, float] = {}
    best_val_metrics: dict[str, float] = {}
    start_time = perf_counter()
    actual_epochs = 0
    epochs_without_improvement = 0
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

        val_auroc = finite_or_zero(val_metrics["auroc"])
        val_loss = float(val_losses["total"])
        improved = (
            val_auroc > best_val_auroc + args.early_stopping_min_delta
            if val_auroc > 0
            else val_loss < best_val_loss - args.early_stopping_min_delta
        )
        if improved:
            best_epoch = epoch
            best_val_auroc = val_auroc
            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            best_train_losses = train_losses
            best_train_metrics = train_metrics
            best_val_losses = val_losses
            best_val_metrics = val_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        append_csv(
            epoch_metrics_path,
            {
                "epoch": epoch,
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
            },
        )
        print(
            f"{spec.name} epoch {epoch:03d}/{args.epochs} | "
            f"train AUROC {finite_or_zero(train_metrics['auroc']):.4f} | "
            f"val AUROC {finite_or_zero(val_metrics['auroc']):.4f} "
            f"AUPRC {finite_or_zero(val_metrics['auprc']):.4f}"
        )

        if (
            args.early_stopping_patience > 0
            and epoch >= args.early_stopping_min_epochs
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            stopped_early = True
            print(
                f"{spec.name} seed {run_seed} early stopping at epoch {epoch}; "
                f"validation AUROC 已連續 {epochs_without_improvement} epochs 未改善。"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # validation 只用於選 checkpoint；test 在選定後評估一次。
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_losses, test_metrics = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        grad_clip=None,
        max_batches=zero_to_none(args.limit_test_batches),
    )

    rule_quality = evaluate_rule_quality(
        model=model,
        dataset=val_dataset,
        device=device,
        batch_size=args.batch_size,
        max_batches=zero_to_none(args.rule_quality_batches),
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": run_seed,
            "best_epoch": best_epoch,
            "variant": spec.name,
            "display_name": spec.display_name,
            "args": vars(args),
            "feature_order": FEATURE_ORDER,
            "explicit_temporal_features": spec.explicit_temporal_features,
            "best_val_losses": best_val_losses,
            "best_val_metrics": best_val_metrics,
            "test_losses": test_losses,
            "test_metrics": test_metrics,
            "rule_quality": rule_quality,
        },
        variant_dir / "best_model.pt",
    )
    (variant_dir / "variant_config.json").write_text(
        json.dumps(
            {
                "name": spec.name,
                "seed": run_seed,
                "display_name": spec.display_name,
                "description": spec.description,
                "expert_init": spec.expert_init,
                "temporal_design": spec.temporal_design,
                "clinical_consistency": spec.clinical_consistency,
                "input_seq_length": spec.input_seq_length,
                "min_history_length": spec.min_history_length,
                "lambda_cons": spec.lambda_cons,
                "lambda_drift": spec.lambda_drift,
                "explicit_temporal_features": spec.explicit_temporal_features,
                "train_windows": len(train_dataset),
                "val_windows": len(val_dataset),
                "train_positive": train_pos,
                "train_negative": train_neg,
                "val_positive": val_pos,
                "val_negative": val_neg,
                "test_windows": len(test_dataset),
                "test_positive": test_pos,
                "test_negative": test_neg,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    elapsed = perf_counter() - start_time
    result = {
        "comparison_mode": args.comparison_mode,
        "protocol_sha256": load_protocol(args.comparison_protocol)["protocol_sha256"],
        "target_col": args.target_col,
        "horizon_hours": horizon_from_target_col(args.target_col),
        "seed": run_seed,
        "variant": spec.name,
        "display_name": spec.display_name,
        "expert_init": int(spec.expert_init),
        "temporal_design": int(spec.temporal_design),
        "clinical_consistency": int(spec.clinical_consistency),
        "explicit_temporal_features": int(spec.explicit_temporal_features),
        "input_seq_length": spec.input_seq_length,
        "min_history_length": spec.min_history_length,
        "train_windows": len(train_dataset),
        "val_windows": len(val_dataset),
        "train_positive": train_pos,
        "train_negative": train_neg,
        "val_positive": val_pos,
        "val_negative": val_neg,
        "test_windows": len(test_dataset),
        "test_positive": test_pos,
        "test_negative": test_neg,
        "train_total_loss": best_train_losses.get("total", math.nan),
        "train_auroc": best_train_metrics.get("auroc", math.nan),
        "train_auprc": best_train_metrics.get("auprc", math.nan),
        "val_total_loss": best_val_losses.get("total", math.nan),
        "val_bce": best_val_losses.get("prediction", math.nan),
        "val_auroc": best_val_metrics.get("auroc", math.nan),
        "val_auprc": best_val_metrics.get("auprc", math.nan),
        "val_recall": best_val_metrics.get("recall", math.nan),
        "val_precision": best_val_metrics.get("precision", math.nan),
        "test_total_loss": test_losses.get("total", math.nan),
        "test_auroc": test_metrics.get("auroc", math.nan),
        "test_auprc": test_metrics.get("auprc", math.nan),
        "test_recall": test_metrics.get("recall", math.nan),
        "test_precision": test_metrics.get("precision", math.nan),
        "test_sensitivity": test_metrics.get("sensitivity", math.nan),
        "test_specificity": test_metrics.get("specificity", math.nan),
        "test_f1": test_metrics.get("f1", math.nan),
        "test_accuracy": test_metrics.get("accuracy", math.nan),
        "test_brier": test_metrics.get("brier", math.nan),
        "test_ece": test_metrics.get("ece", math.nan),
        "test_mce": test_metrics.get("mce", math.nan),
        "test_log_loss": test_metrics.get("log_loss", math.nan),
        "best_epoch": best_epoch,
        "actual_epochs": actual_epochs,
        "stopped_early": int(stopped_early),
        "seconds": elapsed,
        **rule_quality,
    }
    (variant_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FNN ablation experiments.")
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
    parser.add_argument("--variants", default="all", help="all or comma list: random_init,static_guideline,no_consistency,full")
    parser.add_argument("--seq-length", type=int, default=24)
    parser.add_argument("--static-min-history-hours", type=int, default=24)
    parser.add_argument("--val-frac", type=float, default=0.15, help="舊版相容參數；正式比例由 manifest 決定。")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        default=None,
        help="逗號分隔的 random seeds；未指定時沿用 --seed。",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-stays", type=int, default=0)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--max-test-windows", type=int, default=0, help="0 代表評估完整 test set。")
    parser.add_argument("--chunk-size", type=int, default=500_000)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--attention-hidden", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=7.0)
    parser.add_argument("--rule-score-scale", type=float, default=0.2)
    parser.add_argument("--lambda-cons", type=float, default=0.1)
    parser.add_argument("--lambda-sparse", type=float, default=0.001)
    parser.add_argument("--lambda-drift", type=float, default=0.001)
    parser.add_argument("--random-lambda-drift", type=float, default=0.0)
    parser.add_argument("--lambda-nonnegative", type=float, default=0.05)
    parser.add_argument("--explicit-temporal-scale", type=float, default=1.0)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-val-batches", type=int, default=0)
    parser.add_argument("--limit-test-batches", type=int, default=0)
    parser.add_argument("--rule-quality-batches", type=int, default=0)
    parser.add_argument("--best-params-json", default=None, help="可讀取 Optuna best_params.json 作為共同超參數。")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-resume", action="store_true", help="忽略已完成的 result.json 並重新訓練。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args = apply_best_params(args, load_best_params(args.best_params_json))
    selected_variants = parse_variants(args.variants)
    parsed_seeds = parse_seeds(args.seeds, args.seed)
    if len(parsed_seeds) != 1:
        raise ValueError("此 CLI 一次只接受一個 seed；多 seed 請分次執行並使用不同 output-dir。")
    args.seed = parsed_seeds[0]
    set_seed(args.seed)
    protocol = validate_comparison_args(
        args.comparison_mode,
        args.comparison_protocol,
        args.target_col,
        args.seq_length,
    )
    if not args.allow_incomplete_cohort and any(
        value > 0 for value in [args.max_train_windows, args.max_val_windows, args.max_test_windows]
    ):
        raise ValueError("正式比較不可再抽樣 windows；equal-sample 已由共用 manifest 固定。")

    device = choose_device(args.device)
    run_name = datetime.now().strftime("fnn_ablation_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "fnn_ablation" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"使用裝置: {device}")
    print(f"輸出資料夾: {output_dir}")
    print(f"Variants: {', '.join(selected_variants)}")

    df = load_training_frame(
        csv_path=Path(args.csv),
        feature_cols=FEATURE_ORDER,
        target_col=args.target_col,
        time_col=args.time_col,
        split_col=args.split_col,
        max_rows=zero_to_none(args.max_rows),
        max_stays=zero_to_none(args.max_stays),
        chunk_size=args.chunk_size,
        sofa_csv=args.sofa_csv,
    )
    print(f"讀入列數: {len(df):,}，stay 數: {df['stay_id'].nunique():,}")

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
        raise ValueError("正式消融實驗必須以 subject_id 做 patient-level split。")
    train_ids, val_ids, test_ids = split_ids_for_values(split_values, args.split_manifest)
    train_window_ids = window_ids_for_mode(
        args.comparison_mode, args.equal_sample_windows, args.target_col, "train"
    )
    val_window_ids = window_ids_for_mode(
        args.comparison_mode, args.equal_sample_windows, args.target_col, "validation"
    )
    train_id_array = np.fromiter(train_ids, dtype=split_values.dtype)
    train_row_mask = np.isin(split_values, train_id_array)
    random_feature_configs = make_random_feature_config(
        features, args.seed + 1000, row_mask=train_row_mask
    )
    random_rule_configs = make_random_rule_configs(random_feature_configs, args.seed + 2000)
    variants = build_variants(args, random_feature_configs, random_rule_configs)

    (output_dir / "ablation_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "selected_variants": selected_variants,
                "feature_order": FEATURE_ORDER,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_path = output_dir / "ablation_summary.csv"
    for variant_name in selected_variants:
        spec = variants[variant_name]
        print("\n" + "=" * 80)
        print(f"Running: {spec.display_name}")
        print("=" * 80)

        train_dataset, val_dataset, test_dataset = build_datasets_for_variant(
            spec=spec,
            features=features,
            labels=labels,
            stay_ids=stay_ids,
            split_values=split_values,
            time_values=time_values,
            train_ids=train_ids,
            val_ids=val_ids,
            test_ids=test_ids,
            train_window_ids=train_window_ids,
            val_window_ids=val_window_ids,
            args=args,
        )
        train_pos, train_neg = dataset_label_counts(train_dataset)
        val_pos, val_neg = dataset_label_counts(val_dataset)
        test_pos, test_neg = dataset_label_counts(test_dataset)
        cohort_records = [
            dataset_cohort_record(train_dataset, "train", args.target_col),
            dataset_cohort_record(val_dataset, "validation", args.target_col),
            dataset_cohort_record(test_dataset, "test", args.target_col),
        ]
        validate_cohort_records(
            cohort_records,
            protocol,
            args.comparison_mode,
            allow_incomplete=args.allow_incomplete_cohort,
        )
        write_cohort_audit(
            output_dir / f"seed_{args.seed}" / variant_name / "cohort_audit.json",
            cohort_records,
        )
        print(
            f"Train windows: {len(train_dataset):,} | pos {train_pos:,} | neg {train_neg:,} | "
            f"Val windows: {len(val_dataset):,} | pos {val_pos:,} | neg {val_neg:,} | "
            f"Test windows: {len(test_dataset):,} | pos {test_pos:,} | neg {test_neg:,}"
        )

        row = train_variant(
            spec=spec,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            device=device,
            output_dir=output_dir,
            args=args,
            run_seed=args.seed,
        )
        append_csv(summary_path, row)
        print(
            f"{spec.name} done | test AUROC {finite_or_zero(row['test_auroc']):.4f} | "
            f"AUPRC {finite_or_zero(row['test_auprc']):.4f} | "
            f"rule drift {finite_or_zero(row['rule_drift_loss']):.4f}"
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n消融實驗完成")
    print(f"Summary: {summary_path}")
    print(f"Artifacts: {output_dir}")
    rebuild_summaries(output_dir)
    try_generate(generate_ablation_figures, summary_path, output_dir)
    for epoch_metrics_path in output_dir.rglob("epoch_metrics.csv"):
        try_generate(generate_training_figures, epoch_metrics_path, epoch_metrics_path.parent)


if __name__ == "__main__":
    main()
