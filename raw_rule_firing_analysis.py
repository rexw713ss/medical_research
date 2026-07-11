"""Analyze unnormalized cross-rule firing and activation-threshold sensitivity."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from anfis_model import TemporalAttentionFNN
from extract_temporal_fuzzy_rules import sha256_file
from patient_split import split_ids_for_values
from train_fnn import (
    ICUWindowDataset,
    choose_device,
    load_training_frame,
    prepare_explicit_temporal_arrays,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="outputs/explicit_temporal_fnn_formal_6h/seed_42")
    parser.add_argument("--output-dir", default="outputs/raw_rule_firing_6h")
    parser.add_argument("--thresholds", default="0.01,0.025,0.05,0.1,0.2,0.35,0.5")
    parser.add_argument("--reference-threshold", type=float, default=0.1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-stays", type=int, default=0)
    parser.add_argument("--max-test-windows", type=int, default=0)
    return parser.parse_args()


def load_frozen_model(run_dir: Path, config: dict, device: torch.device) -> TemporalAttentionFNN:
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model = TemporalAttentionFNN(
        seq_length=config["seq_length"],
        attention_hidden=config["attention_hidden"],
        threshold=config["threshold"],
        rule_score_scale=config["rule_score_scale"],
        use_explicit_temporal_features=True,
        explicit_temporal_scale=config["explicit_temporal_scale"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def median_from_histogram(histogram: np.ndarray) -> float:
    total = int(histogram.sum())
    if total == 0:
        return math.nan
    cumulative = np.cumsum(histogram)
    left = int(np.searchsorted(cumulative, (total - 1) / 2, side="right"))
    right = int(np.searchsorted(cumulative, total / 2, side="right"))
    return (left + right) / 2


def analyze(
    model: TemporalAttentionFNN,
    dataset: ICUWindowDataset | Subset,
    thresholds: list[float],
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rule_count = len(model.static_fnn.rule_configs)
    bases = ("current_hour", "attention_selected_hour")
    support = {(basis, threshold): np.zeros(rule_count, dtype=np.int64) for basis in bases for threshold in thresholds}
    positives = {(basis, threshold): np.zeros(rule_count, dtype=np.int64) for basis in bases for threshold in thresholds}
    count_hist = {
        (basis, threshold, outcome): np.zeros(rule_count + 1, dtype=np.int64)
        for basis in bases
        for threshold in thresholds
        for outcome in (0, 1)
    }
    firing_sum = {basis: np.zeros(rule_count, dtype=np.float64) for basis in bases}
    firing_positive_sum = {basis: np.zeros(rule_count, dtype=np.float64) for basis in bases}
    firing_negative_sum = {basis: np.zeros(rule_count, dtype=np.float64) for basis in bases}
    outcome_count = np.zeros(2, dtype=np.int64)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            labels = batch_y.to(device, non_blocking=True).long()
            output = model(batch_x)
            current = output.raw_rule_firing[:, -1, :]
            selected_index = torch.argmax(output.attention_weights, dim=1)
            selected = output.raw_rule_firing[
                torch.arange(len(batch_x), device=device), selected_index
            ]
            arrays = {
                "current_hour": current.detach().cpu().numpy(),
                "attention_selected_hour": selected.detach().cpu().numpy(),
            }
            y = labels.detach().cpu().numpy().astype(np.int8)
            outcome_count += np.bincount(y, minlength=2)

            for basis, values in arrays.items():
                firing_sum[basis] += values.sum(axis=0)
                firing_positive_sum[basis] += values[y == 1].sum(axis=0)
                firing_negative_sum[basis] += values[y == 0].sum(axis=0)
                for threshold in thresholds:
                    active = values >= threshold
                    support[(basis, threshold)] += active.sum(axis=0)
                    positives[(basis, threshold)] += active[y == 1].sum(axis=0)
                    activated_count = active.sum(axis=1)
                    for outcome in (0, 1):
                        count_hist[(basis, threshold, outcome)] += np.bincount(
                            activated_count[y == outcome], minlength=rule_count + 1
                        )

    total = int(outcome_count.sum())
    prevalence = outcome_count[1] / total
    weights = model.static_fnn.cross_rule_weights.detach().cpu().numpy()
    rule_rows = []
    summary_rows = []
    for basis in bases:
        mean_firing = firing_sum[basis] / total
        for threshold in thresholds:
            for rule_index, config in enumerate(model.static_fnn.rule_configs):
                n = int(support[(basis, threshold)][rule_index])
                positive = int(positives[(basis, threshold)][rule_index])
                positive_rate = positive / n if n else math.nan
                rule_rows.append(
                    {
                        "basis": basis,
                        "threshold": threshold,
                        "rule_index": rule_index,
                        "rule_id": config["name"],
                        "antecedents": " AND ".join(f"{feature} IS {term}" for feature, term in config["antecedents"]),
                        "trained_weight": float(weights[rule_index]),
                        "support": n,
                        "support_fraction": n / total,
                        "positive": positive,
                        "positive_rate": positive_rate,
                        "positive_rate_lift": positive_rate / prevalence if n else math.nan,
                        "mean_raw_firing": float(mean_firing[rule_index]),
                        "mean_raw_firing_positive": float(firing_positive_sum[basis][rule_index] / outcome_count[1]),
                        "mean_raw_firing_negative": float(firing_negative_sum[basis][rule_index] / outcome_count[0]),
                        "weighted_mean_firing": float(abs(weights[rule_index]) * mean_firing[rule_index]),
                        "threshold_ranking_score": float(
                            abs(weights[rule_index]) * math.sqrt(n / total)
                        ),
                    }
                )
            for outcome in (0, 1):
                histogram = count_hist[(basis, threshold, outcome)]
                counts = np.arange(rule_count + 1)
                summary_rows.append(
                    {
                        "basis": basis,
                        "threshold": threshold,
                        "outcome": outcome,
                        "windows": int(histogram.sum()),
                        "mean_activated_rules": float(np.sum(histogram * counts) / histogram.sum()),
                        "median_activated_rules": median_from_histogram(histogram),
                        "windows_with_any_rule": int(histogram[1:].sum()),
                        "fraction_with_any_rule": float(histogram[1:].sum() / histogram.sum()),
                    }
                )
    return pd.DataFrame(rule_rows), pd.DataFrame(summary_rows)


def threshold_stability(frame: pd.DataFrame, reference: float, top_k: int) -> pd.DataFrame:
    rows = []
    for basis, group in frame.groupby("basis", sort=False):
        reference_frame = group[np.isclose(group["threshold"], reference)]
        reference_rules = set(
            reference_frame.nlargest(top_k, "threshold_ranking_score")["rule_id"]
        )
        for threshold, threshold_frame in group.groupby("threshold", sort=True):
            rules = set(threshold_frame.nlargest(top_k, "threshold_ranking_score")["rule_id"])
            union = rules | reference_rules
            rows.append(
                {
                    "basis": basis,
                    "threshold": threshold,
                    "reference_threshold": reference,
                    "top_k": top_k,
                    "top_k_jaccard": len(rules & reference_rules) / len(union) if union else math.nan,
                    "rules_with_support_ge_100": int((threshold_frame["support"] >= 100).sum()),
                }
            )
    return pd.DataFrame(rows)


def save_figures(rule_frame: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    current = rule_frame[rule_frame["basis"] == "current_hour"]
    for rule_id, group in current.groupby("rule_id"):
        axes[0].plot(group["threshold"], group["support_fraction"], marker="o", label=rule_id)
    axes[0].set(xlabel="Raw firing threshold", ylabel="Support fraction", title="Cross-rule support sensitivity")
    axes[0].legend(fontsize=7, frameon=False)
    for outcome, label in ((0, "Negative windows"), (1, "Positive windows")):
        group = summary[(summary["basis"] == "current_hour") & (summary["outcome"] == outcome)]
        axes[1].plot(group["threshold"], group["mean_activated_rules"], marker="o", label=label)
    axes[1].set(xlabel="Raw firing threshold", ylabel="Mean activated rules", title="Activated-rule count sensitivity")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"raw_rule_firing_threshold_sensitivity.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output_dir: Path,
    rules: pd.DataFrame,
    summary: pd.DataFrame,
    stability: pd.DataFrame,
    checkpoint_hash: str,
) -> None:
    current = rules[(rules["basis"] == "current_hour") & np.isclose(rules["threshold"], 0.1)]
    lines = [
        "# Raw Rule Firing and Activation-Threshold Sensitivity",
        "",
        f"Frozen checkpoint SHA-256: `{checkpoint_hash}`.",
        "Raw firing is the product t-norm before normalization across rules.",
        "",
        "## Rule-Level Results at Raw Firing >= 0.10",
        "",
        "| Rule | Support | Positive rate | Lift | Mean firing positive | Mean firing negative |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in current.sort_values("weighted_mean_firing", ascending=False).iterrows():
        lines.append(
            f"| {row['rule_id']} | {int(row['support']):,} | {row['positive_rate']:.3f} | "
            f"{row['positive_rate_lift']:.2f} | {row['mean_raw_firing_positive']:.4f} | "
            f"{row['mean_raw_firing_negative']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Activated Rules",
            "",
            "| Basis | Threshold | Outcome | Mean activated | Median activated | Any rule |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['basis']} | {row['threshold']:.3f} | {int(row['outcome'])} | "
            f"{row['mean_activated_rules']:.3f} | {row['median_activated_rules']:.1f} | "
            f"{row['fraction_with_any_rule']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Top-K Threshold Stability",
            "",
            "| Basis | Threshold | Reference | Top-K | Jaccard | Rules with support >=100 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in stability.iterrows():
        lines.append(
            f"| {row['basis']} | {row['threshold']:.3f} | {row['reference_threshold']:.3f} | "
            f"{int(row['top_k'])} | {row['top_k_jaccard']:.3f} | "
            f"{int(row['rules_with_support_ge_100'])} |"
        )
    lines.extend(
        [
            "",
            "These are model-internal firing diagnostics, not clinician validation or causal evidence.",
        ]
    )
    (output_dir / "raw_rule_firing_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
    thresholds = sorted({float(value) for value in args.thresholds.split(",")})
    if not any(np.isclose(thresholds, args.reference_threshold)):
        raise ValueError("Reference threshold must be included in --thresholds.")

    frame = load_training_frame(
        csv_path=Path(config["csv"]),
        feature_cols=config["input_order"],
        target_col=config["target_col"],
        time_col=config["time_col"],
        split_col=config["split_col"],
        max_rows=args.max_rows or None,
        max_stays=args.max_stays or None,
        chunk_size=config.get("chunk_size", 500_000),
        sofa_csv=config.get("sofa_csv"),
    )
    features, labels, stay_ids, split_values, time_values = prepare_explicit_temporal_arrays(
        frame, config["target_col"], config["time_col"], config["split_col"]
    )
    _, _, test_ids = split_ids_for_values(split_values, config["split_manifest"])
    dataset: ICUWindowDataset | Subset = ICUWindowDataset(
        features=features,
        labels=labels,
        stay_ids=stay_ids,
        split_values=split_values,
        time_values=time_values,
        allowed_split_values=test_ids,
        seq_length=config["seq_length"],
    )
    if args.max_test_windows and len(dataset) > args.max_test_windows:
        rng = np.random.default_rng(42)
        indices = np.sort(rng.choice(len(dataset), args.max_test_windows, replace=False))
        dataset = Subset(dataset, indices.tolist())

    device = choose_device(args.device)
    model = load_frozen_model(run_dir, config, device)
    rules, summary = analyze(model, dataset, thresholds, device, args.batch_size)
    stability = threshold_stability(rules, args.reference_threshold, args.top_k)
    rules.to_csv(output_dir / "raw_rule_firing_by_threshold.csv", index=False)
    summary.to_csv(output_dir / "activation_threshold_summary.csv", index=False)
    stability.to_csv(output_dir / "top_k_threshold_stability.csv", index=False)
    save_figures(rules, summary, output_dir)
    checkpoint_hash = sha256_file(run_dir / "best_model.pt")
    write_report(output_dir, rules, summary, stability, checkpoint_hash)
    (output_dir / "analysis_config.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "checkpoint_sha256": checkpoint_hash,
                "thresholds": thresholds,
                "reference_threshold": args.reference_threshold,
                "top_k": args.top_k,
                "windows": len(dataset),
                "raw_firing_definition": "product t-norm before normalization across cross-rules",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote raw firing analysis to {output_dir}")


if __name__ == "__main__":
    main()
