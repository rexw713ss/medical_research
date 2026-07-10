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
from itertools import combinations
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
    ExpertGuidedStaticFNN,
    NeuroSymbolicLoss,
    TemporalAttentionFNN,
    clinical_rule_priors,
    expert_feature_config,
    explicit_temporal_input_order,
)
from advanced_model_evaluation import apply_platt_calibration, calibration_intercept_slope
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
    CLINICAL_DEFAULTS,
    classification_metrics,
    choose_device,
    load_training_frame,
    prepare_arrays,
    prepare_explicit_temporal_arrays,
    run_epoch,
)


VARIANT_ORDER = [
    "random_init",
    "static_guideline",
    "no_consistency",
    "full",
    "no_missingness",
    "missingness_only",
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
    "rule_drift_normalized",
    "rule_concordance",
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
    input_mask_mode: str = "full"


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
        input_mask_mode: str = "full",
    ) -> None:
        self.features = features
        self.labels = labels
        self.input_seq_length = input_seq_length
        self.min_history_length = min_history_length
        self.stay_ids = stay_ids
        self.split_values = split_values
        self.time_values = time_values
        self.input_mask_mode = input_mask_mode
        if input_mask_mode not in {"full", "no_missingness", "missingness_only"}:
            raise ValueError(f"Unknown input mask mode: {input_mask_mode}")
        self.clinical_defaults = torch.tensor(
            [CLINICAL_DEFAULTS[feature] for feature in FEATURE_ORDER],
            dtype=torch.float32,
        )
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
        window = torch.from_numpy(self.features[start:end])
        if self.input_mask_mode != "full":
            window = window.clone()
            base_count = len(FEATURE_ORDER)
            if self.input_mask_mode == "no_missingness":
                window[:, base_count:] = 0.0
            elif self.input_mask_mode == "missingness_only":
                window[:, :base_count] = self.clinical_defaults
        return window, torch.tensor(self.labels[target_index], dtype=torch.float32)

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


def calculate_rule_stability(output_dir: Path, top_k: int = 10) -> dict[str, dict[str, float]]:
    rows = []
    summaries: dict[str, dict[str, float]] = {}
    for variant in VARIANT_ORDER:
        inventories = {}
        for path in sorted(output_dir.glob(f"seed_*/{variant}/rule_inventory.csv")):
            seed = int(path.parents[1].name.removeprefix("seed_"))
            frame = pd.read_csv(path).sort_values("importance", ascending=False)
            inventories[seed] = set(frame.head(top_k)["rule_id"].astype(str))
        similarities = []
        for (seed_a, rules_a), (seed_b, rules_b) in combinations(inventories.items(), 2):
            union = rules_a | rules_b
            jaccard = len(rules_a & rules_b) / len(union) if union else math.nan
            rows.append(
                {
                    "variant": variant,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "top_k": top_k,
                    "intersection": len(rules_a & rules_b),
                    "union": len(union),
                    "jaccard": jaccard,
                }
            )
            similarities.append(jaccard)
        finite = np.asarray(similarities, dtype=float)
        finite = finite[np.isfinite(finite)]
        summaries[variant] = {
            "mean": float(finite.mean()) if finite.size else math.nan,
            "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0 if finite.size else math.nan,
            "pairs": int(finite.size),
        }
    pd.DataFrame(rows).to_csv(output_dir / "rule_stability_pairwise.csv", index=False)
    return summaries


def write_publication_table(aggregate: pd.DataFrame, output_dir: Path) -> None:
    columns = {
        "AUROC": "test_auroc_mean",
        "AUPRC": "test_auprc_mean",
        "Brier": "test_brier_mean",
        "ECE": "test_ece_mean",
        "Rule Concordance": "rule_concordance_mean",
        "Rule Stability": "rule_stability",
        "Rule Drift": "rule_drift_normalized_mean",
    }
    rows = []
    for _, source in aggregate.iterrows():
        row = {"Model": source["display_name"]}
        for label, source_col in columns.items():
            row[label] = source.get(source_col, math.nan)
        row["Seeds"] = source.get("seeds", "")
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "ablation_publication_table.csv", index=False)

    def formatted(source: pd.Series, metric: str, stability: bool = False) -> str:
        value = float(source.get(metric, math.nan))
        if not np.isfinite(value):
            return "NA"
        if stability:
            return f"{value:.3f}"
        std = float(source.get(metric.replace("_mean", "_std"), math.nan))
        return f"{value:.3f} +/- {std:.3f}" if np.isfinite(std) else f"{value:.3f}"

    lines = [
        "# Formal 6-hour FNN Ablation Study",
        "",
        "Values are mean +/- SD across random seeds. Brier and ECE use validation-only Platt calibration.",
        "Rule Stability is mean pairwise Top-10 Jaccard similarity.",
        "",
        "| Model | AUROC | AUPRC | Brier | ECE | Rule Concordance | Rule Stability | Rule Drift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in aggregate.iterrows():
        lines.append(
            "| " + " | ".join(
                [
                    str(row["display_name"]),
                    formatted(row, "test_auroc_mean"),
                    formatted(row, "test_auprc_mean"),
                    formatted(row, "test_brier_mean"),
                    formatted(row, "test_ece_mean"),
                    formatted(row, "rule_concordance_mean"),
                    formatted(row, "rule_stability", stability=True),
                    formatted(row, "rule_drift_normalized_mean"),
                ]
            ) + " |"
        )
    (output_dir / "ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_ablation_results(summary: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    stability = calculate_rule_stability(output_dir)
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
        row["rule_stability"] = stability.get(str(row["variant"]), {}).get("mean", math.nan)
        row["rule_stability_std"] = stability.get(str(row["variant"]), {}).get("std", math.nan)
        row["rule_stability_pairs"] = stability.get(str(row["variant"]), {}).get("pairs", 0)
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
    aggregate = pd.DataFrame(aggregate_rows)
    variant_rank = {name: index for index, name in enumerate(VARIANT_ORDER)}
    aggregate["_variant_rank"] = aggregate["variant"].map(variant_rank)
    aggregate = aggregate.sort_values(["horizon_hours", "_variant_rank"]).drop(
        columns="_variant_rank"
    )
    aggregate.to_csv(aggregate_path, index=False)
    write_publication_table(aggregate, output_dir)

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
            for metric in [
                "test_auroc",
                "test_auprc",
                "test_brier",
                "test_ece",
                "rule_concordance",
                "rule_drift_normalized",
            ]:
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
        quantiles = np.linspace(0.05, 0.95, n_terms)
        if values.size:
            coverage_centers = np.quantile(values, quantiles)
        else:
            coverage_centers = np.linspace(low, high, n_terms)
        spacing = span / max(n_terms - 1, 1)
        coverage_centers = np.clip(
            coverage_centers + rng.normal(0.0, spacing * 0.15, size=n_terms),
            low,
            high,
        )
        # 打散 term 與 center 的對應，但保留數值空間覆蓋，避免無梯度的壞初始化。
        centers = coverage_centers[rng.permutation(n_terms)]
        sigmas = rng.uniform(spacing * 0.75, spacing * 1.50, size=n_terms)
        sigmas = np.clip(sigmas, 1e-3, 80.0)
        # 13 個 feature score 會相加；近零初始化可避免 sigmoid 一開始即飽和。
        weights = rng.uniform(0.0, 0.5, size=n_terms)

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
                "weight": float(rng.uniform(0.0, 1.0)),
            }
        )
    return rules


def build_variants(
    args: argparse.Namespace,
    random_feature_configs: dict[str, list[dict]],
    random_rule_configs: list[dict],
) -> dict[str, VariantSpec]:
    static_min_history = args.static_min_history_hours
    random_lambda_drift = (
        args.lambda_drift
        if args.random_lambda_drift is None
        else args.random_lambda_drift
    )
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
            lambda_drift=random_lambda_drift,
            explicit_temporal_features=True,
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
            explicit_temporal_features=True,
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
            explicit_temporal_features=True,
        ),
        "no_missingness": VariantSpec(
            name="no_missingness",
            display_name="Explicit KG-TFNN without missingness channels",
            description="Full architecture with missingness and time-since channels fixed to zero.",
            input_seq_length=args.seq_length,
            min_history_length=args.seq_length,
            feature_configs=expert_feature_config,
            rule_configs=clinical_rule_priors,
            expert_init=True,
            temporal_design=True,
            clinical_consistency=True,
            lambda_cons=args.lambda_cons,
            lambda_drift=args.lambda_drift,
            explicit_temporal_features=True,
            input_mask_mode="no_missingness",
        ),
        "missingness_only": VariantSpec(
            name="missingness_only",
            display_name="Missingness-only Temporal FNN",
            description="Clinical values fixed to defaults; only missingness and time-since channels vary.",
            input_seq_length=args.seq_length,
            min_history_length=args.seq_length,
            feature_configs=expert_feature_config,
            rule_configs=clinical_rule_priors,
            expert_init=True,
            temporal_design=True,
            clinical_consistency=True,
            lambda_cons=args.lambda_cons,
            lambda_drift=args.lambda_drift,
            explicit_temporal_features=True,
            input_mask_mode="missingness_only",
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
    variant_features = (
        features
        if spec.explicit_temporal_features
        else features[:, : len(FEATURE_ORDER)]
    )
    train_dataset = AblationWindowDataset(
        features=variant_features,
        labels=labels,
        stay_ids=stay_ids,
        split_values=split_values,
        time_values=time_values,
        allowed_split_values=train_ids,
        input_seq_length=spec.input_seq_length,
        min_history_length=spec.min_history_length,
        allowed_window_ids=train_window_ids,
        require_all_window_ids=not args.allow_incomplete_cohort,
        input_mask_mode=spec.input_mask_mode,
    )
    val_dataset = AblationWindowDataset(
        features=variant_features,
        labels=labels,
        stay_ids=stay_ids,
        split_values=split_values,
        time_values=time_values,
        allowed_split_values=val_ids,
        input_seq_length=spec.input_seq_length,
        min_history_length=spec.min_history_length,
        allowed_window_ids=val_window_ids,
        require_all_window_ids=not args.allow_incomplete_cohort,
        input_mask_mode=spec.input_mask_mode,
    )
    test_dataset = AblationWindowDataset(
        features=variant_features,
        labels=labels,
        stay_ids=stay_ids,
        split_values=split_values,
        time_values=time_values,
        allowed_split_values=test_ids,
        input_seq_length=spec.input_seq_length,
        min_history_length=spec.min_history_length,
        input_mask_mode=spec.input_mask_mode,
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


def evaluate_dataset_with_predictions(
    model: TemporalAttentionFNN,
    dataset: Dataset,
    criterion: NeuroSymbolicLoss,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_batches: int | None = None,
) -> tuple[dict[str, float], dict[str, float], np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    loss_sums = {
        "total": 0.0,
        "prediction": 0.0,
        "clinical_consistency": 0.0,
        "rule_sparsity": 0.0,
        "rule_drift": 0.0,
        "nonnegative_weights": 0.0,
    }
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    n_samples = 0
    model.eval()
    criterion.eval()
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            output = model(batch_x)
            losses = criterion(output, batch_y, model)
            n_batch = int(batch_y.shape[0])
            n_samples += n_batch
            for key, value in losses.items():
                loss_sums[key] += float(value.detach().item()) * n_batch
            probabilities.append(output.probabilities.detach().cpu().numpy())
            targets.append(batch_y.detach().cpu().numpy())

    if not probabilities:
        raise ValueError("Evaluation dataset 沒有可用樣本。")
    y_true = np.concatenate(targets).astype(np.float32, copy=False)
    y_prob = np.concatenate(probabilities).astype(np.float32, copy=False)
    mean_losses = {key: value / n_samples for key, value in loss_sums.items()}
    return mean_losses, classification_metrics(y_true, y_prob), y_true, y_prob


def dataset_prediction_metadata(dataset: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(dataset, Subset):
        base = dataset.dataset
        selected = np.asarray(dataset.indices, dtype=np.int64)
        target_indices = base.target_indices[selected]
    else:
        base = dataset
        target_indices = base.target_indices
    return (
        np.asarray(base.split_values[target_indices]),
        np.asarray(base.stay_ids[target_indices]),
        np.asarray(base.time_values[target_indices]),
    )


def save_predictions(
    path: Path,
    dataset: Dataset,
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    y_prob_calibrated: np.ndarray,
) -> None:
    subject_id, stay_id, time_value = dataset_prediction_metadata(dataset)
    if len(y_true) > len(subject_id):
        raise ValueError("Prediction metadata 少於預測列數。")
    subject_id = subject_id[: len(y_true)]
    stay_id = stay_id[: len(y_true)]
    time_value = time_value[: len(y_true)]
    pd.DataFrame(
        {
            "subject_id": subject_id,
            "stay_id": stay_id,
            "sofa_hour": time_value,
            "y_true": y_true.astype(np.int8),
            "y_prob_raw": y_prob_raw,
            "y_prob": y_prob_calibrated,
        }
    ).to_csv(path, index=False, compression="gzip")


def normalized_rule_drift(model: TemporalAttentionFNN) -> float:
    """以每組參數的初始 RMS 正規化，避免不同生理尺度主導 drift。"""
    ratios = []
    static = model.static_fnn
    for feature in static.feature_names:
        pairs = [
            (static.centers[feature], getattr(static, f"initial_centers__{feature}")),
            (static.sigma(feature), getattr(static, f"initial_sigmas__{feature}")),
            (static.rule_weights[feature], getattr(static, f"initial_rule_weights__{feature}")),
        ]
        for current, initial in pairs:
            delta_rms = torch.sqrt(torch.mean((current - initial).square()))
            initial_rms = torch.sqrt(torch.mean(initial.square()))
            ratios.append(float((delta_rms / (initial_rms + 1.0)).detach().cpu().item()))
    if static.cross_rule_weights.numel() > 0:
        delta_rms = torch.sqrt(
            torch.mean((static.cross_rule_weights - static.initial_cross_rule_weights).square())
        )
        initial_rms = torch.sqrt(torch.mean(static.initial_cross_rule_weights.square()))
        ratios.append(float((delta_rms / (initial_rms + 1.0)).detach().cpu().item()))
    return float(np.mean(ratios)) if ratios else 0.0


def rule_inventory(model: TemporalAttentionFNN) -> pd.DataFrame:
    rows = []
    static = model.static_fnn
    for feature in static.feature_names:
        centers = static.centers[feature].detach().cpu().numpy()
        sigmas = static.sigma(feature).detach().cpu().numpy()
        weights = static.rule_weights[feature].detach().cpu().numpy()
        initial_centers = getattr(static, f"initial_centers__{feature}").cpu().numpy()
        initial_sigmas = getattr(static, f"initial_sigmas__{feature}").cpu().numpy()
        initial_weights = getattr(static, f"initial_rule_weights__{feature}").cpu().numpy()
        for index, term in enumerate(static.term_names[feature]):
            rows.append(
                {
                    "rule_id": f"feature::{feature}::{term}",
                    "rule_type": "feature",
                    "rule": f"IF {feature} IS {term}",
                    "initial_weight": float(initial_weights[index]),
                    "trained_weight": float(weights[index]),
                    "effective_weight": float(weights[index]),
                    "importance": float(abs(weights[index])),
                    "initial_center": float(initial_centers[index]),
                    "trained_center": float(centers[index]),
                    "initial_sigma": float(initial_sigmas[index]),
                    "trained_sigma": float(sigmas[index]),
                }
            )
    for index, rule in enumerate(static.rule_configs):
        antecedents = " AND ".join(
            f"{feature} IS {term}" for feature, term in rule["antecedents"]
        )
        signature = "&".join(
            sorted(f"{feature}={term}" for feature, term in rule["antecedents"])
        )
        initial_weight = float(static.initial_cross_rule_weights[index].detach().cpu().item())
        trained_weight = float(static.cross_rule_weights[index].detach().cpu().item())
        effective_weight = trained_weight * float(static.rule_score_scale)
        rows.append(
            {
                "rule_id": f"cross::{signature}",
                "rule_type": "cross_feature",
                "rule": f"IF {antecedents}",
                "initial_weight": initial_weight,
                "trained_weight": trained_weight,
                "effective_weight": effective_weight,
                "importance": abs(effective_weight),
                "initial_center": math.nan,
                "trained_center": math.nan,
                "initial_sigma": math.nan,
                "trained_sigma": math.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)


def evaluate_rule_quality(
    model: TemporalAttentionFNN,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    max_batches: int | None,
    rule_score_scale: float,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    guideline_reference = ExpertGuidedStaticFNN(
        expert_feature_config,
        clinical_rule_priors,
        rule_score_scale=rule_score_scale,
    ).to(device)
    guideline_reference.eval()
    model.eval()
    activation_sum = 0.0
    activation_count = 0
    active_count = 0
    top_activation_sum = 0.0
    top_activation_count = 0
    attention_entropy_sum = 0.0
    attention_last_6h_sum = 0.0
    attention_count = 0
    learned_static_scores = []
    guideline_static_scores = []
    with torch.no_grad():
        for batch_idx, (batch_x, _) in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break
            output = model(batch_x.to(device))
            raw_current = batch_x[:, -1, : len(FEATURE_ORDER)].to(device)
            guideline_output = guideline_reference(raw_current)
            learned_static_scores.append(output.hourly_risk_scores[:, -1].detach().cpu().numpy())
            guideline_static_scores.append(
                guideline_output["static_risk_score"].detach().cpu().numpy()
            )
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
    if learned_static_scores:
        learned = np.concatenate(learned_static_scores)
        guideline = np.concatenate(guideline_static_scores)
        rule_concordance = float(pd.Series(learned).corr(pd.Series(guideline), method="spearman"))
    else:
        rule_concordance = math.nan

    return {
        "num_cross_rules": float(rule_count),
        "avg_rule_antecedents": avg_antecedents,
        "mean_rule_activation": mean_rule_activation,
        "active_rule_fraction_gt_0_1": active_rule_fraction,
        "mean_top_rule_activation": top_rule_activation,
        "rule_sparsity_loss": float(model.static_fnn.sparsity_loss().detach().cpu().item()),
        "rule_drift_loss": float(model.static_fnn.drift_loss().detach().cpu().item()),
        "rule_drift_normalized": normalized_rule_drift(model),
        "rule_concordance": rule_concordance,
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
    if not spec.expert_init and model.explicit_temporal_weights is not None:
        with torch.no_grad():
            model.explicit_temporal_weights.uniform_(0.0, 0.20)
            model.initial_explicit_temporal_weights.copy_(model.explicit_temporal_weights)

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

    # validation 只用於 checkpoint selection、calibration；test 在定案後評估一次。
    _, _, val_targets, val_prob_raw = evaluate_dataset_with_predictions(
        model=model,
        dataset=val_dataset,
        criterion=criterion,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=zero_to_none(args.limit_val_batches),
    )
    calibration_intercept, calibration_slope = calibration_intercept_slope(
        val_targets.astype(np.int8),
        val_prob_raw,
    )
    val_prob = apply_platt_calibration(
        val_prob_raw,
        calibration_intercept,
        calibration_slope,
    )
    test_losses, test_metrics_raw, test_targets, test_prob_raw = evaluate_dataset_with_predictions(
        model=model,
        dataset=test_dataset,
        criterion=criterion,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=zero_to_none(args.limit_test_batches),
    )
    test_prob = apply_platt_calibration(
        test_prob_raw,
        calibration_intercept,
        calibration_slope,
    )
    test_metrics = classification_metrics(test_targets, test_prob)
    save_predictions(
        variant_dir / "validation_predictions.csv.gz",
        val_dataset,
        val_targets,
        val_prob_raw,
        val_prob,
    )
    save_predictions(
        variant_dir / "test_predictions.csv.gz",
        test_dataset,
        test_targets,
        test_prob_raw,
        test_prob,
    )

    rule_quality = evaluate_rule_quality(
        model=model,
        dataset=val_dataset,
        device=device,
        batch_size=args.batch_size,
        max_batches=zero_to_none(args.rule_quality_batches),
        rule_score_scale=args.rule_score_scale,
    )
    inventory = rule_inventory(model)
    inventory.to_csv(variant_dir / "rule_inventory.csv", index=False)

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
            "input_mask_mode": spec.input_mask_mode,
            "best_val_losses": best_val_losses,
            "best_val_metrics": best_val_metrics,
            "test_losses": test_losses,
            "test_metrics": test_metrics,
            "test_metrics_raw": test_metrics_raw,
            "calibration_intercept": calibration_intercept,
            "calibration_slope": calibration_slope,
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
                "input_mask_mode": spec.input_mask_mode,
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
        "input_mask_mode": spec.input_mask_mode,
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
        "test_raw_brier": test_metrics_raw.get("brier", math.nan),
        "test_raw_ece": test_metrics_raw.get("ece", math.nan),
        "test_raw_log_loss": test_metrics_raw.get("log_loss", math.nan),
        "validation_platt_intercept": calibration_intercept,
        "validation_platt_slope": calibration_slope,
        "calibration_method": "validation_only_platt",
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
    parser.add_argument(
        "--variants",
        default="all",
        help="all or comma list: random_init,static_guideline,no_consistency,full,no_missingness,missingness_only",
    )
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
    parser.add_argument(
        "--random-lambda-drift",
        type=float,
        default=None,
        help="預設與 full model 相同；僅供額外 sensitivity analysis 覆寫。",
    )
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


def run_seed_ablation(
    args: argparse.Namespace,
    run_seed: int,
    selected_variants: list[str],
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
    protocol: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> None:
    args.seed = run_seed
    set_seed(run_seed)
    train_id_array = np.fromiter(train_ids, dtype=split_values.dtype)
    train_row_mask = np.isin(split_values, train_id_array)
    random_feature_configs = make_random_feature_config(
        features[:, : len(FEATURE_ORDER)],
        run_seed + 1000,
        row_mask=train_row_mask,
    )
    random_rule_configs = make_random_rule_configs(
        random_feature_configs,
        run_seed + 2000,
    )
    variants = build_variants(args, random_feature_configs, random_rule_configs)
    seed_dir = output_dir / f"seed_{run_seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "selected_variants": selected_variants,
                "feature_order": FEATURE_ORDER,
                "input_order": explicit_temporal_input_order(FEATURE_ORDER),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_path = output_dir / "ablation_summary.csv"
    for variant_name in selected_variants:
        spec = variants[variant_name]
        result_path = seed_dir / variant_name / "result.json"
        if result_path.exists() and not args.no_resume:
            print(f"Skip completed: seed {run_seed} / {variant_name}")
            continue

        print("\n" + "=" * 80)
        print(f"Seed {run_seed} | Running: {spec.display_name}")
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
        write_cohort_audit(seed_dir / variant_name / "cohort_audit.json", cohort_records)
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
            run_seed=run_seed,
        )
        append_csv(summary_path, row)
        rebuild_summaries(output_dir)
        print(
            f"{spec.name} done | test AUROC {finite_or_zero(row['test_auroc']):.4f} | "
            f"AUPRC {finite_or_zero(row['test_auprc']):.4f} | "
            f"rule concordance {finite_or_zero(row['rule_concordance']):.4f}"
        )
        del train_dataset, val_dataset, test_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    args = apply_best_params(args, load_best_params(args.best_params_json))
    selected_variants = parse_variants(args.variants)
    parsed_seeds = parse_seeds(args.seeds, args.seed)
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

    use_explicit_inputs = any(name != "static_guideline" for name in selected_variants)
    input_order = (
        explicit_temporal_input_order(FEATURE_ORDER)
        if use_explicit_inputs
        else list(FEATURE_ORDER)
    )
    df = load_training_frame(
        csv_path=Path(args.csv),
        feature_cols=input_order,
        target_col=args.target_col,
        time_col=args.time_col,
        split_col=args.split_col,
        max_rows=zero_to_none(args.max_rows),
        max_stays=zero_to_none(args.max_stays),
        chunk_size=args.chunk_size,
        sofa_csv=args.sofa_csv,
    )
    print(f"讀入列數: {len(df):,}，stay 數: {df['stay_id'].nunique():,}")

    if use_explicit_inputs:
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
        raise ValueError("正式消融實驗必須以 subject_id 做 patient-level split。")
    train_ids, val_ids, test_ids = split_ids_for_values(split_values, args.split_manifest)
    train_window_ids = window_ids_for_mode(
        args.comparison_mode, args.equal_sample_windows, args.target_col, "train"
    )
    val_window_ids = window_ids_for_mode(
        args.comparison_mode, args.equal_sample_windows, args.target_col, "validation"
    )
    (output_dir / "ablation_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "seeds": parsed_seeds,
                "selected_variants": selected_variants,
                "feature_order": FEATURE_ORDER,
                "input_order": input_order,
                "calibration_method": "validation_only_platt",
                "rule_concordance_definition": "Spearman correlation with frozen guideline static risk on validation windows",
                "rule_stability_definition": "Mean pairwise Top-10 rule-importance Jaccard across seeds",
                "rule_drift_definition": "Mean parameter-group RMSE normalized by initial RMS plus one",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    for run_seed in parsed_seeds:
        run_seed_ablation(
            args=args,
            run_seed=run_seed,
            selected_variants=selected_variants,
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
            protocol=protocol,
            device=device,
            output_dir=output_dir,
        )

    print("\n消融實驗完成")
    summary_path = output_dir / "ablation_summary.csv"
    print(f"Summary: {summary_path}")
    print(f"Artifacts: {output_dir}")
    rebuild_summaries(output_dir)
    try_generate(generate_ablation_figures, summary_path, output_dir)
    for epoch_metrics_path in output_dir.rglob("epoch_metrics.csv"):
        try_generate(generate_training_figures, epoch_metrics_path, epoch_metrics_path.parent)


if __name__ == "__main__":
    main()
