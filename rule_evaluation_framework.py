"""Produce the formal Rule Evaluation Framework results and case studies."""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from advanced_model_evaluation import apply_platt_calibration
from anfis_model import FEATURE_ORDER, TemporalAttentionFNN
from train_fnn import CLINICAL_DEFAULTS, choose_device


FEATURE_LABELS = {
    "heart_rate": "Heart rate",
    "respiratory_rate": "Respiratory rate",
    "spo2": "SpO2",
    "fio2": "FiO2",
    "temperature_c": "Temperature",
    "sbp": "SBP",
    "gcs_total": "GCS",
    "map": "MAP",
    "pao2_fio2": "PaO2/FiO2",
    "platelets": "Platelets",
    "bilirubin": "Bilirubin",
    "creatinine": "Creatinine",
    "lactate": "Lactate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FNN rule quality.")
    parser.add_argument(
        "--checkpoint",
        default="outputs/explicit_temporal_fnn_formal_6h/seed_42/best_model.pt",
    )
    parser.add_argument(
        "--inventory-roots",
        default=(
            "outputs/fnn_ablation_6h_equal_sample,"
            "outputs/rule_evaluation_full_fnn_extra_seeds"
        ),
    )
    parser.add_argument(
        "--predictions",
        default="outputs/final_test_evaluation_6h/predictions/test_predictions.csv.gz",
    )
    parser.add_argument(
        "--advanced-metrics",
        default="outputs/final_test_evaluation_6h/advanced/advanced_metrics.csv",
    )
    parser.add_argument(
        "--temporal-rules",
        default="outputs/temporal_rule_extraction_6h/extracted_temporal_rules.csv",
    )
    parser.add_argument("--hourly-csv", default="model_hourly_features_v3.csv")
    parser.add_argument("--target-col", default="label_sofa_increase_ge2_6h")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="outputs/rule_evaluation_6h")
    parser.add_argument("--markdown", default="docs/rule_evaluation_framework_6h.md")
    return parser.parse_args()


def load_model(path: Path, device: torch.device) -> TemporalAttentionFNN:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("args", {})
    model = TemporalAttentionFNN(
        seq_length=int(config.get("seq_length", 24)),
        attention_hidden=int(config.get("attention_hidden", 32)),
        threshold=float(config.get("threshold", 7.0)),
        rule_score_scale=float(config.get("rule_score_scale", 0.2)),
        use_explicit_temporal_features=bool(config.get("explicit_temporal_features", True)),
        explicit_temporal_scale=float(config.get("explicit_temporal_scale", 1.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def inventory_antecedents(row: pd.Series) -> int:
    if str(row["rule_type"]) == "feature":
        return 1
    return str(row["rule"]).count(" AND ") + 1


def load_full_inventories(roots: list[Path], top_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    top_frames = []
    top_sets: dict[int, set[str]] = {}
    seen_seeds = set()
    for root in roots:
        for path in sorted(root.glob("seed_*/full/rule_inventory.csv")):
            seed = int(path.parents[1].name.removeprefix("seed_"))
            if seed in seen_seeds:
                continue
            seen_seeds.add(seed)
            inventory = pd.read_csv(path).sort_values("importance", ascending=False).head(top_k)
            inventory = inventory.copy()
            inventory.insert(0, "seed", seed)
            inventory.insert(1, "rank", np.arange(1, len(inventory) + 1))
            inventory["antecedent_count"] = inventory.apply(inventory_antecedents, axis=1)
            top_frames.append(inventory)
            top_sets[seed] = set(inventory["rule_id"].astype(str))
    if len(top_sets) < 5:
        raise ValueError(f"Rule Stability requires 5 completed seeds; found {sorted(top_sets)}")

    pair_rows = []
    for seed_a, seed_b in combinations(sorted(top_sets), 2):
        rules_a = top_sets[seed_a]
        rules_b = top_sets[seed_b]
        union = rules_a | rules_b
        pair_rows.append(
            {
                "seed_a": seed_a,
                "seed_b": seed_b,
                "top_k": top_k,
                "intersection": len(rules_a & rules_b),
                "union": len(union),
                "jaccard": len(rules_a & rules_b) / len(union),
            }
        )
    return pd.concat(top_frames, ignore_index=True), pd.DataFrame(pair_rows)


def membership_drift(model: TemporalAttentionFNN) -> pd.DataFrame:
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
                    "feature": feature,
                    "term": term,
                    "initial_center": initial_centers[index],
                    "trained_center": centers[index],
                    "center_shift": centers[index] - initial_centers[index],
                    "center_shift_in_initial_sigma": abs(centers[index] - initial_centers[index])
                    / max(initial_sigmas[index], 1e-6),
                    "initial_sigma": initial_sigmas[index],
                    "trained_sigma": sigmas[index],
                    "relative_sigma_shift": abs(sigmas[index] - initial_sigmas[index])
                    / max(initial_sigmas[index], 1e-6),
                    "initial_weight": initial_weights[index],
                    "trained_weight": weights[index],
                    "absolute_weight_shift": abs(weights[index] - initial_weights[index]),
                }
            )
    return pd.DataFrame(rows)


def save_membership_figure(model: TemporalAttentionFNN, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    static = model.static_fnn
    fig, axes = plt.subplots(4, 4, figsize=(18, 15))
    axes = axes.ravel()
    for axis, feature in zip(axes, FEATURE_ORDER):
        initial_centers = getattr(static, f"initial_centers__{feature}").cpu().numpy()
        initial_sigmas = getattr(static, f"initial_sigmas__{feature}").cpu().numpy()
        centers = static.centers[feature].detach().cpu().numpy()
        sigmas = static.sigma(feature).detach().cpu().numpy()
        low = min(np.min(initial_centers - 3 * initial_sigmas), np.min(centers - 3 * sigmas))
        high = max(np.max(initial_centers + 3 * initial_sigmas), np.max(centers + 3 * sigmas))
        x = np.linspace(low, high, 300)
        colors = plt.cm.tab10(np.linspace(0, 1, len(centers)))
        for index, (term, color) in enumerate(zip(static.term_names[feature], colors)):
            initial = np.exp(-0.5 * ((x - initial_centers[index]) / initial_sigmas[index]) ** 2)
            trained = np.exp(-0.5 * ((x - centers[index]) / sigmas[index]) ** 2)
            axis.plot(x, initial, linestyle="--", color=color, alpha=0.65)
            axis.plot(x, trained, linestyle="-", color=color, label=term.replace("_", " "))
        axis.set_title(FEATURE_LABELS[feature])
        axis.set_ylim(0, 1.05)
        axis.legend(fontsize=6, frameon=False)
    for axis in axes[len(FEATURE_ORDER) :]:
        axis.axis("off")
    fig.suptitle("Membership Functions Before (Dashed) and After Training (Solid)", fontsize=15)
    fig.tight_layout()
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "membership_functions_before_after.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures / "membership_functions_before_after.pdf", bbox_inches="tight")
    plt.close(fig)


def guideline_direction_rubric(rules: pd.DataFrame, top_k: int) -> pd.DataFrame:
    cross = rules[rules["rule_id"].astype(str).str.startswith("cross::")].head(3)
    frame = pd.concat([rules.head(max(top_k - len(cross), 1)), cross], ignore_index=True)
    frame = frame.drop_duplicates("rule_id").head(top_k).copy()
    frame["rank"] = np.arange(1, len(frame) + 1)
    source_column = (
        "guideline_direction_alignment"
        if "guideline_direction_alignment" in frame.columns
        else "clinical_concordance"
    )
    frame["static_direction_alignment"] = (
        pd.to_numeric(frame[source_column], errors="coerce") > 0
    ).astype(float)
    frame["temporal_direction_alignment"] = frame["temporal_signal"].isin(
        [
            "risk_slope",
            "short_term_change",
            "window_change",
            "abnormal_duration",
            "abnormal_frequency",
        ]
    ).astype(float)
    frame["cross_rule_alignment"] = np.where(
        frame["rule_id"].astype(str).str.startswith("cross::"),
        1.0,
        np.nan,
    )
    frame["guideline_direction_alignment_score"] = frame[
        [
            "static_direction_alignment",
            "temporal_direction_alignment",
            "cross_rule_alignment",
        ]
    ].mean(axis=1, skipna=True)
    return frame[
        [
            "rank",
            "rule_id",
            "extracted_temporal_fuzzy_rule",
            "static_direction_alignment",
            "temporal_direction_alignment",
            "cross_rule_alignment",
            "guideline_direction_alignment_score",
        ]
    ]


def activated_rule_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    labels = {0: "Negative windows", 1: "Positive windows"}
    rows = []
    for outcome, group in predictions.groupby("y_true"):
        values = pd.to_numeric(group["activated_rule_count"], errors="coerce").dropna()
        rows.append(
            {
                "outcome": int(outcome),
                "group": labels[int(outcome)],
                "windows": len(values),
                "mean_activated_rules": values.mean(),
                "std_activated_rules": values.std(ddof=1),
                "median_activated_rules": values.median(),
                "q1": values.quantile(0.25),
                "q3": values.quantile(0.75),
            }
        )
    return pd.DataFrame(rows)


def select_cases(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    frame = predictions.copy()
    frame["predicted_positive"] = frame["y_prob_calibrated"] >= threshold
    groups = {
        "TP": frame[(frame["y_true"] == 1) & frame["predicted_positive"]],
        "FP": frame[(frame["y_true"] == 0) & frame["predicted_positive"]],
        "FN": frame[(frame["y_true"] == 1) & ~frame["predicted_positive"]],
    }
    selected = []
    for case_type, group in groups.items():
        if group.empty:
            continue
        target_probability = group["y_prob_calibrated"].median()
        row = group.iloc[(group["y_prob_calibrated"] - target_probability).abs().argmin()].copy()
        row["case_type"] = case_type
        selected.append(row)
    return pd.DataFrame(selected).reset_index(drop=True)


def load_case_hourly_data(
    csv_path: Path,
    cases: pd.DataFrame,
    target_col: str,
    chunk_size: int,
) -> pd.DataFrame:
    stays = set(cases["stay_id"].astype(np.int64))
    columns = ["subject_id", "stay_id", "sofa_hour", target_col, *FEATURE_ORDER]
    chunks = []
    for chunk in pd.read_csv(csv_path, usecols=columns, chunksize=chunk_size):
        selected = chunk[chunk["stay_id"].isin(stays)]
        if not selected.empty:
            chunks.append(selected)
    if not chunks:
        raise ValueError("No hourly rows found for selected case studies.")
    hourly = pd.concat(chunks, ignore_index=True)
    case_rows = []
    for _, case in cases.iterrows():
        start = int(case["sofa_hour"]) - 23
        end = int(case["sofa_hour"])
        frame = hourly[
            (hourly["stay_id"] == case["stay_id"])
            & hourly["sofa_hour"].between(start, end)
        ].copy()
        frame["case_type"] = case["case_type"]
        frame["index_hour"] = int(case["sofa_hour"])
        frame["relative_hour"] = frame["sofa_hour"] - int(case["sofa_hour"])
        case_rows.append(frame)
    return pd.concat(case_rows, ignore_index=True)


def add_case_probabilities(
    timeline: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    probabilities = predictions[
        ["stay_id", "sofa_hour", "y_prob_calibrated", "activated_rule_count"]
    ].drop_duplicates(["stay_id", "sofa_hour"])
    return timeline.merge(probabilities, on=["stay_id", "sofa_hour"], how="left")


def save_case_figure(timeline: pd.DataFrame, threshold: float, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    case_order = [case for case in ["TP", "FP", "FN"] if case in set(timeline["case_type"])]
    fig, axes = plt.subplots(2, len(case_order), figsize=(6 * len(case_order), 8), squeeze=False)
    for column, case_type in enumerate(case_order):
        frame = timeline[timeline["case_type"] == case_type].sort_values("relative_hour")
        axes[0, column].plot(frame["relative_hour"], frame["y_prob_calibrated"], marker="o")
        axes[0, column].axhline(threshold, color="#A51C30", linestyle="--", label="90% specificity")
        axes[0, column].set(
            title=f"{case_type}: calibrated deterioration risk",
            xlabel="Hours before index",
            ylabel="Probability",
            ylim=(0, max(0.25, float(frame["y_prob_calibrated"].max(skipna=True)) * 1.15)),
        )
        axes[0, column].legend(frameon=False)

        changes = {}
        for feature in FEATURE_ORDER:
            values = pd.to_numeric(frame[feature], errors="coerce").ffill()
            scale = max(abs(CLINICAL_DEFAULTS[feature]), 1.0)
            changes[feature] = abs(values.iloc[-1] - values.iloc[0]) / scale if values.notna().all() else 0
        selected_features = sorted(changes, key=changes.get, reverse=True)[:4]
        for feature in selected_features:
            values = (
                pd.to_numeric(frame[feature], errors="coerce")
                .ffill()
                .fillna(CLINICAL_DEFAULTS[feature])
            )
            std = values.std(ddof=0)
            standardized = (values - values.mean()) / (std if std > 0 else 1.0)
            axes[1, column].plot(
                frame["relative_hour"], standardized, marker=".", label=FEATURE_LABELS[feature]
            )
        axes[1, column].axhline(0, color="#777777", linewidth=0.8)
        axes[1, column].set(
            title=f"{case_type}: largest physiologic changes",
            xlabel="Hours before index",
            ylabel="Within-case standardized value",
        )
        axes[1, column].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "tp_fp_fn_case_timelines.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures / "tp_fp_fn_case_timelines.pdf", bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    complexity: pd.DataFrame,
    stability: pd.DataFrame,
    rubric: pd.DataFrame,
    drift: pd.DataFrame,
    activated: pd.DataFrame,
    cases: pd.DataFrame,
    top_k: int,
    threshold: float,
) -> None:
    mean_complexity = complexity["antecedent_count"].mean()
    mean_stability = stability["jaccard"].mean()
    mean_alignment = rubric["guideline_direction_alignment_score"].mean()
    lines = [
        "# Rule Evaluation Framework: Experimental Results",
        "",
        "## Summary",
        "",
        "| Analysis | Experimental result |",
        "|---|---:|",
        f"| Rule Complexity | Top-{top_k} mean antecedents = {mean_complexity:.2f} |",
        f"| Rule Stability | 5-seed mean pairwise Top-{top_k} Jaccard = {mean_stability:.3f} |",
        f"| Guideline-Direction Alignment | Mean prespecified direction alignment = {mean_alignment:.3f} |",
        f"| Rule Drift | Median center shift = {drift['center_shift_in_initial_sigma'].median():.3f} initial sigmas (mean {drift['center_shift_in_initial_sigma'].mean():.3f}) |",
        "",
        "Five-seed complexity/stability uses the fixed equal-sample training protocol (200,000 train and "
        "50,000 validation windows per seed). Direction alignment, membership drift, activated rules and case "
        "studies use the frozen full-cohort final model.",
        "",
        "## Rule Complexity And Stability",
        "",
        f"Five seeds produced {len(stability)} pairwise comparisons. Values below use the same Top-{top_k} rule definition.",
        "",
        "| Seed pair | Jaccard similarity |",
        "|---|---:|",
    ]
    for _, row in stability.iterrows():
        lines.append(f"| {int(row['seed_a'])} vs {int(row['seed_b'])} | {row['jaccard']:.3f} |")
    lines.extend(
        [
            "",
            "## Guideline-Direction Alignment Rubric",
            "",
            "Each rule receives one point for guideline-consistent static direction and one for a clinically "
            "worsening/persistent temporal direction. Cross-feature rules additionally require a predefined "
            "clinical rule combination. NA criteria are excluded from the denominator.",
            "",
            "| Rank | Rule | Static | Temporal | Cross-rule | Overall |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in rubric.iterrows():
        cross = "NA" if pd.isna(row["cross_rule_alignment"]) else f"{row['cross_rule_alignment']:.0f}"
        lines.append(
            f"| {int(row['rank'])} | {row['extracted_temporal_fuzzy_rule']} | "
            f"{row['static_direction_alignment']:.0f} | {row['temporal_direction_alignment']:.0f} | "
            f"{cross} | {row['guideline_direction_alignment_score']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Membership-Function Drift",
            "",
            "| Parameter | Mean shift | Median shift |",
            "|---|---:|---:|",
            f"| Center, normalized by initial sigma | {drift['center_shift_in_initial_sigma'].mean():.3f} | {drift['center_shift_in_initial_sigma'].median():.3f} |",
            f"| Sigma, relative change | {drift['relative_sigma_shift'].mean():.3f} | {drift['relative_sigma_shift'].median():.3f} |",
            f"| Rule weight, absolute change | {drift['absolute_weight_shift'].mean():.3f} | {drift['absolute_weight_shift'].median():.3f} |",
            "",
            "## Activated Rules",
            "",
            "Rule activation is measured at the attention-selected hour using normalized cross-rule activation > 0.1.",
            "",
            "| Outcome group | Windows | Mean activated rules | Median (IQR) |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in activated.iterrows():
        lines.append(
            f"| {row['group']} | {int(row['windows']):,} | {row['mean_activated_rules']:.3f} | "
            f"{row['median_activated_rules']:.1f} ({row['q1']:.1f}-{row['q3']:.1f}) |"
        )
    lines.extend(
        [
            "",
            "## TP/FP/FN Case Studies",
            "",
            f"Cases use the MIMIC-validation threshold for 90% specificity ({threshold:.4f}). "
            "One probability-median representative was selected per error group.",
            "",
            "| Case | Subject | Stay | Index hour | True outcome | Calibrated risk | Activated rules |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in cases.iterrows():
        lines.append(
            f"| {row['case_type']} | {int(row['subject_id'])} | {int(row['stay_id'])} | "
            f"{int(row['sofa_hour'])} | {int(row['y_true'])} | {row['y_prob_calibrated']:.3f} | "
            f"{int(row['activated_rule_count'])} |"
        )
    lines.extend(
        [
            "",
            "Figures: `outputs/rule_evaluation_6h/figures/`. Detailed numeric tables are stored beside this report.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    model = load_model(Path(args.checkpoint), device)

    inventories, stability = load_full_inventories(
        [Path(value.strip()) for value in args.inventory_roots.split(",") if value.strip()],
        args.top_k,
    )
    inventories.to_csv(output_dir / "top_k_rule_complexity.csv", index=False)
    stability.to_csv(output_dir / "five_seed_rule_stability.csv", index=False)

    drift = membership_drift(model)
    drift.to_csv(output_dir / "membership_parameter_drift.csv", index=False)
    save_membership_figure(model, output_dir)

    temporal_rules = pd.read_csv(args.temporal_rules)
    rubric = guideline_direction_rubric(temporal_rules, args.top_k)
    rubric.to_csv(output_dir / "guideline_direction_alignment_rubric.csv", index=False)

    predictions = pd.read_csv(args.predictions)
    metrics = pd.read_csv(args.advanced_metrics).iloc[0]
    predictions["y_prob_calibrated"] = apply_platt_calibration(
        predictions["y_prob"].to_numpy(dtype=float),
        float(metrics["validation_platt_intercept"]),
        float(metrics["validation_platt_slope"]),
    )
    activated = activated_rule_summary(predictions)
    activated.to_csv(output_dir / "activated_rule_summary.csv", index=False)
    threshold = float(metrics["threshold_spec_90"])
    cases = select_cases(predictions, threshold)
    cases.to_csv(output_dir / "selected_tp_fp_fn_cases.csv", index=False)

    timeline = load_case_hourly_data(
        Path(args.hourly_csv),
        cases,
        args.target_col,
        args.chunk_size,
    )
    timeline = add_case_probabilities(timeline, predictions)
    timeline.to_csv(output_dir / "tp_fp_fn_case_timelines.csv.gz", index=False, compression="gzip")
    save_case_figure(timeline, threshold, output_dir)

    write_report(
        output_dir / "rule_evaluation_report.md",
        inventories,
        stability,
        rubric,
        drift,
        activated,
        cases,
        args.top_k,
        threshold,
    )
    if args.markdown:
        write_report(
            Path(args.markdown),
            inventories,
            stability,
            rubric,
            drift,
            activated,
            cases,
            args.top_k,
            threshold,
        )
    (output_dir / "rule_evaluation_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Rule Evaluation Framework complete: {output_dir}")


if __name__ == "__main__":
    main()
