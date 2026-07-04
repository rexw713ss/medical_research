"""Explicit temporal-feature and 4/6/12/24-hour observation-window sensitivity.

All variants predict the same outcome on the same target windows. A 24-hour history
is required for eligibility, while each model only receives its assigned observation
window. Explicit variants also receive leakage-free missingness and time-since-last-
measurement channels from the v3 preprocessing table.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ablation_fnn_experiments import (
    VariantSpec,
    apply_best_params,
    build_datasets_for_variant,
    dataset_cohort_record,
    mean_std_ci,
    parse_seeds,
    train_variant,
)
from anfis_model import (
    EXPLICIT_TEMPORAL_SIGNAL_NAMES,
    FEATURE_ORDER,
    clinical_rule_priors,
    explicit_temporal_input_order,
    expert_feature_config,
)
from comparison_protocol import (
    validate_cohort_records,
    validate_comparison_args,
    window_ids_for_mode,
    write_cohort_audit,
)
from patient_split import split_ids_for_values
from project_config import (
    COMPARISON_PROTOCOL_JSON,
    EQUAL_SAMPLE_WINDOWS_CSV,
    PATIENT_SPLIT_CSV,
    PRIMARY_HOURLY_FEATURES,
)
from train_fnn import (
    choose_device,
    load_training_frame,
    prepare_explicit_temporal_arrays,
)


SUMMARY_METRICS = ("test_auroc", "test_auprc", "test_brier", "test_ece")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run explicit temporal-feature observation-window sensitivity."
    )
    parser.add_argument("--csv", default=PRIMARY_HOURLY_FEATURES)
    parser.add_argument("--target-col", default="label_sofa_increase_ge2_6h")
    parser.add_argument("--time-col", default="sofa_hour")
    parser.add_argument("--split-col", default="subject_id")
    parser.add_argument("--split-manifest", default=PATIENT_SPLIT_CSV)
    parser.add_argument("--comparison-mode", choices=["full", "equal_sample"], default="equal_sample")
    parser.add_argument("--comparison-protocol", default=COMPARISON_PROTOCOL_JSON)
    parser.add_argument("--equal-sample-windows", default=EQUAL_SAMPLE_WINDOWS_CSV)
    parser.add_argument("--observation-windows", default="4,6,12,24")
    parser.add_argument("--eligibility-hours", type=int, default=24)
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument(
        "--include-sequence-only-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also train a 24-hour sequence-only FNN without explicit temporal signals.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-epochs", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-stays", type=int, default=None)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--max-test-windows", type=int, default=0)
    parser.add_argument("--allow-incomplete-cohort", action="store_true")
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-val-batches", type=int, default=0)
    parser.add_argument("--limit-test-batches", type=int, default=0)
    parser.add_argument("--rule-quality-batches", type=int, default=0)
    parser.add_argument(
        "--best-params-json",
        default="outputs/explicit_temporal_fnn_tuning_6h/best_params.json",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/explicit_temporal_observation_sensitivity_6h",
    )
    parser.add_argument("--force", action="store_true")
    parser.set_defaults(
        seed=42,
        seq_length=24,
        static_min_history_hours=24,
        sofa_csv=None,
        val_frac=0.15,
        batch_size=128,
        learning_rate=1e-3,
        weight_decay=1e-5,
        grad_clip=5.0,
        attention_hidden=32,
        threshold=7.0,
        rule_score_scale=0.2,
        lambda_cons=0.1,
        lambda_sparse=0.001,
        lambda_drift=0.001,
        random_lambda_drift=0.001,
        lambda_nonnegative=0.05,
        explicit_temporal_scale=1.0,
    )
    return parser.parse_args()


def run_name(window: int, explicit: bool) -> str:
    suffix = "explicit" if explicit else "sequence_only"
    return f"observation_{window}h_{suffix}"


def spec_for_run(args: argparse.Namespace, window: int, explicit: bool) -> VariantSpec:
    design = "explicit temporal features" if explicit else "sequence-only temporal attention"
    return VariantSpec(
        name=run_name(window, explicit),
        display_name=f"FNN ({window}h, {design})",
        description=(
            f"Knowledge-guided temporal FNN using {window} hours and {design}; "
            f"target eligibility remains fixed at {args.eligibility_hours} hours."
        ),
        input_seq_length=window,
        min_history_length=args.eligibility_hours,
        feature_configs=expert_feature_config,
        rule_configs=clinical_rule_priors,
        expert_init=True,
        temporal_design=True,
        clinical_consistency=True,
        lambda_cons=args.lambda_cons,
        lambda_drift=args.lambda_drift,
        explicit_temporal_features=explicit,
    )


def completed(output_dir: Path, seed: int, window: int, explicit: bool) -> bool:
    run_dir = output_dir / f"seed_{seed}" / run_name(window, explicit)
    return (run_dir / "result.json").exists() and (run_dir / "best_model.pt").exists()


def collect_results(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for path in output_dir.glob("seed_*/observation_*h_*/result.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["artifact_path"] = str(path.parent)
        rows.append(row)
    if not rows:
        return pd.DataFrame(), pd.DataFrame()

    runs = pd.DataFrame(rows)
    runs = runs.sort_values(["explicit_temporal_features", "observation_window_hours", "seed"])
    runs.to_csv(output_dir / "observation_window_runs.csv", index=False)

    aggregate_rows = []
    for (explicit, window), group in runs.groupby(
        ["explicit_temporal_features", "observation_window_hours"]
    ):
        row = {
            "feature_design": "explicit" if int(explicit) else "sequence_only",
            "explicit_temporal_features": int(explicit),
            "observation_window_hours": int(window),
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in SUMMARY_METRICS:
            _, mean, std, low, high = mean_std_ci(group[metric])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows).sort_values(
        ["explicit_temporal_features", "observation_window_hours"],
        ascending=[False, True],
    )
    aggregate.to_csv(output_dir / "observation_window_aggregate.csv", index=False)
    write_paired_differences(runs, output_dir)
    write_temporal_weights(output_dir)
    return runs, aggregate


def write_paired_differences(runs: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    explicit = runs[runs["explicit_temporal_features"] == 1]
    reference = explicit[explicit["observation_window_hours"] == 24].set_index("seed")
    for window, group in explicit.groupby("observation_window_hours"):
        current = group.set_index("seed")
        common = current.index.intersection(reference.index)
        for metric in ("test_auroc", "test_auprc"):
            delta = current.loc[common, metric] - reference.loc[common, metric]
            _, mean, std, low, high = mean_std_ci(delta)
            rows.append(
                {
                    "comparison": f"explicit_{int(window)}h_minus_explicit_24h",
                    "metric": metric,
                    "n_paired_seeds": len(common),
                    "mean_difference": mean,
                    "std_difference": std,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )

    control = runs[
        (runs["explicit_temporal_features"] == 0)
        & (runs["observation_window_hours"] == 24)
    ].set_index("seed")
    common = reference.index.intersection(control.index)
    for metric in ("test_auroc", "test_auprc"):
        delta = reference.loc[common, metric] - control.loc[common, metric]
        _, mean, std, low, high = mean_std_ci(delta)
        rows.append(
            {
                "comparison": "explicit_24h_minus_sequence_only_24h",
                "metric": metric,
                "n_paired_seeds": len(common),
                "mean_difference": mean,
                "std_difference": std,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "paired_sensitivity_differences.csv", index=False)


def write_temporal_weights(output_dir: Path) -> None:
    rows = []
    for checkpoint_path in output_dir.glob("seed_*/observation_*h_explicit/best_model.pt"):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        weights = checkpoint["model_state_dict"].get("explicit_temporal_weights")
        if weights is None:
            continue
        result_path = checkpoint_path.with_name("result.json")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for feature_idx, feature in enumerate(FEATURE_ORDER):
            for signal_idx, signal in enumerate(EXPLICIT_TEMPORAL_SIGNAL_NAMES):
                rows.append(
                    {
                        "seed": result["seed"],
                        "observation_window_hours": result["observation_window_hours"],
                        "feature": feature,
                        "temporal_signal": signal,
                        "learned_weight": float(weights[feature_idx, signal_idx]),
                    }
                )
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(output_dir / "explicit_temporal_weights.csv", index=False)
        summary = (
            frame.assign(abs_weight=frame["learned_weight"].abs())
            .groupby(["observation_window_hours", "temporal_signal"], as_index=False)
            .agg(
                mean_weight=("learned_weight", "mean"),
                std_weight=("learned_weight", "std"),
                mean_abs_weight=("abs_weight", "mean"),
                std_abs_weight=("abs_weight", "std"),
                n_coefficients=("learned_weight", "size"),
            )
            .sort_values(["observation_window_hours", "mean_abs_weight"], ascending=[True, False])
        )
        summary.to_csv(output_dir / "explicit_temporal_weight_summary.csv", index=False)


def plot_results(output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.read_csv(output_dir / "observation_window_aggregate.csv")
    explicit = frame[frame["feature_design"] == "explicit"].sort_values(
        "observation_window_hours"
    )
    control = frame[frame["feature_design"] == "sequence_only"]
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, metric, color in [
        (axes[0], "test_auroc", "#4E79A7"),
        (axes[1], "test_auprc", "#F28E2B"),
    ]:
        lower = explicit[f"{metric}_mean"] - explicit[f"{metric}_ci95_low"]
        upper = explicit[f"{metric}_ci95_high"] - explicit[f"{metric}_mean"]
        axis.errorbar(
            explicit["observation_window_hours"],
            explicit[f"{metric}_mean"],
            yerr=np.vstack([lower, upper]),
            marker="o",
            color=color,
            capsize=4,
            label="Explicit temporal FNN",
        )
        if not control.empty:
            axis.scatter(
                control["observation_window_hours"],
                control[f"{metric}_mean"],
                marker="s",
                s=60,
                color="#7F7F7F",
                label="Sequence-only control",
                zorder=3,
            )
        axis.set(
            xlabel="Observation window (hours)",
            ylabel=metric.removeprefix("test_").upper(),
        )
        axis.set_xticks(explicit["observation_window_hours"])
        axis.legend(fontsize=8)
    fig.suptitle("Explicit temporal-feature observation-window sensitivity")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"observation_window_sensitivity.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    weight_path = output_dir / "explicit_temporal_weights.csv"
    if not weight_path.exists():
        return

    weights = pd.read_csv(weight_path)
    weights_24h = weights[weights["observation_window_hours"] == 24]
    if weights_24h.empty:
        return

    coefficient_matrix = (
        weights_24h.groupby(["feature", "temporal_signal"])["learned_weight"]
        .mean()
        .unstack("temporal_signal")
        .reindex(index=FEATURE_ORDER, columns=EXPLICIT_TEMPORAL_SIGNAL_NAMES)
    )
    signal_profile = (
        weights.assign(abs_weight=weights["learned_weight"].abs())
        .groupby(["observation_window_hours", "temporal_signal"])["abs_weight"]
        .mean()
        .unstack("observation_window_hours")
        .reindex(index=EXPLICIT_TEMPORAL_SIGNAL_NAMES)
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.7, 1]})
    limit = float(np.nanmax(np.abs(coefficient_matrix.to_numpy())))
    image = axes[0].imshow(
        coefficient_matrix.to_numpy(),
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    axes[0].set_xticks(range(len(coefficient_matrix.columns)))
    axes[0].set_xticklabels(coefficient_matrix.columns, rotation=55, ha="right", fontsize=8)
    axes[0].set_yticks(range(len(coefficient_matrix.index)))
    axes[0].set_yticklabels(coefficient_matrix.index, fontsize=8)
    axes[0].set_title("24-hour mean learned coefficients")
    colorbar = fig.colorbar(image, ax=axes[0], fraction=0.03, pad=0.02)
    colorbar.set_label("Coefficient")

    colors = {4: "#4E79A7", 6: "#59A14F", 12: "#F28E2B", 24: "#E15759"}
    y = np.arange(len(signal_profile.index))
    offsets = np.linspace(-0.27, 0.27, len(signal_profile.columns))
    for offset, window in zip(offsets, signal_profile.columns):
        axes[1].barh(
            y + offset,
            signal_profile[window],
            height=0.16,
            label=f"{int(window)} h",
            color=colors.get(int(window)),
        )
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(signal_profile.index, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean absolute coefficient")
    axes[1].set_title("Temporal coefficient profile")
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("Learned explicit temporal-feature coefficients")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(figures / f"explicit_temporal_coefficients.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    windows = sorted({int(value) for value in args.observation_windows.split(",")})
    if any(window < 2 for window in windows):
        raise ValueError("Explicit temporal observation windows must be at least 2 hours.")
    seeds = parse_seeds(args.seeds, 42)
    best_payload = json.loads(Path(args.best_params_json).read_text(encoding="utf-8"))
    args = apply_best_params(args, best_payload.get("best_params", best_payload))
    protocol = validate_comparison_args(
        args.comparison_mode,
        args.comparison_protocol,
        args.target_col,
        args.eligibility_hours,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    designs = [(window, True) for window in windows]
    if args.include_sequence_only_control:
        designs.append((24, False))
    pending = [
        (seed, window, explicit)
        for seed in seeds
        for window, explicit in designs
        if args.force or not completed(output_dir, seed, window, explicit)
    ]
    if not pending:
        collect_results(output_dir)
        plot_results(output_dir)
        print("All explicit temporal sensitivity runs are already complete.")
        return

    input_order = explicit_temporal_input_order(FEATURE_ORDER)
    frame = load_training_frame(
        Path(args.csv),
        input_order,
        args.target_col,
        args.time_col,
        args.split_col,
        args.max_rows,
        args.max_stays,
        args.chunk_size,
        args.sofa_csv,
    )
    explicit_features, labels, stay_ids, split_values, time_values = prepare_explicit_temporal_arrays(
        frame,
        args.target_col,
        args.time_col,
        args.split_col,
    )
    del frame
    raw_features = explicit_features[:, : len(FEATURE_ORDER)]

    if args.split_col != "subject_id":
        raise ValueError("Formal sensitivity analysis requires patient-level subject_id split.")
    train_ids, val_ids, test_ids = split_ids_for_values(split_values, args.split_manifest)
    train_window_ids = window_ids_for_mode(
        args.comparison_mode,
        args.equal_sample_windows,
        args.target_col,
        "train",
    )
    val_window_ids = window_ids_for_mode(
        args.comparison_mode,
        args.equal_sample_windows,
        args.target_col,
        "validation",
    )

    dataset_cache = {}
    for window, explicit in sorted(set((window, explicit) for _, window, explicit in pending)):
        spec = spec_for_run(args, window, explicit)
        model_features = explicit_features if explicit else raw_features
        datasets = build_datasets_for_variant(
            spec,
            model_features,
            labels,
            stay_ids,
            split_values,
            time_values,
            train_ids,
            val_ids,
            test_ids,
            train_window_ids,
            val_window_ids,
            args,
        )
        records = [
            dataset_cohort_record(datasets[0], "train", args.target_col),
            dataset_cohort_record(datasets[1], "validation", args.target_col),
            dataset_cohort_record(datasets[2], "test", args.target_col),
        ]
        validate_cohort_records(
            records,
            protocol,
            args.comparison_mode,
            allow_incomplete=args.allow_incomplete_cohort,
        )
        dataset_cache[(window, explicit)] = (*datasets, records)

    experiment_config = {
        **vars(args),
        "observation_windows": windows,
        "eligibility_hours": args.eligibility_hours,
        "seeds": seeds,
        "input_order": input_order,
        "explicit_temporal_signals": EXPLICIT_TEMPORAL_SIGNAL_NAMES,
        "protocol_sha256": protocol["protocol_sha256"],
        "designs": [
            {"window": window, "explicit_temporal_features": explicit}
            for window, explicit in designs
        ],
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(experiment_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    device = choose_device(args.device)
    for seed, window, explicit in pending:
        spec = spec_for_run(args, window, explicit)
        train_dataset, val_dataset, test_dataset, records = dataset_cache[(window, explicit)]
        run_dir = output_dir / f"seed_{seed}" / spec.name
        run_dir.mkdir(parents=True, exist_ok=True)
        write_cohort_audit(run_dir / "cohort_audit.json", records)
        result = train_variant(
            spec,
            train_dataset,
            val_dataset,
            test_dataset,
            device,
            output_dir,
            args,
            seed,
        )
        result["observation_window_hours"] = window
        result["explicit_temporal_features"] = int(explicit)
        result["feature_design"] = "explicit" if explicit else "sequence_only"
        (run_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        collect_results(output_dir)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    collect_results(output_dir)
    plot_results(output_dir)
    print(f"Explicit temporal sensitivity complete: {output_dir}")


if __name__ == "__main__":
    main()
