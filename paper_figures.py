"""Publication-ready figures for ICU deterioration experiments.

All plotting functions save both 300-dpi PNG and vector PDF files under a
`figures/` folder. Matplotlib is imported lazily so model training can still
finish with a clear warning when the plotting dependency is unavailable.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
    "#882255",
    "#44AA99",
    "#999999",
]

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "gam": "GAM",
    "ebm": "EBM",
    "lstm": "LSTM",
    "gru": "GRU",
    "news2_score_calibrated": "NEWS2",
    "sofa_score_calibrated": "SOFA",
    "random_init": "Random-init FNN",
    "static_guideline": "Guideline FNN",
    "no_consistency": "Temporal FNN without consistency",
    "full": "Full Knowledge-Guided Temporal FNN",
}


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Matplotlib is required for figures. Run scripts with env\\Scripts\\python.exe."
        ) from exc

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    return plt


def _figure_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(fig: Any, output_dir: str | Path, stem: str) -> list[Path]:
    figure_dir = _figure_dir(output_dir)
    png_path = figure_dir / f"{stem}.png"
    pdf_path = figure_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    return [png_path, pdf_path]


def _model_label(model: Any, feature_set: Any = None) -> str:
    raw = str(model)
    if raw.startswith("fnn_"):
        horizon = next((part for part in raw.split("_") if part.endswith("h")), "")
        label = f"Knowledge-Guided Temporal FNN ({horizon})" if horizon else "Knowledge-Guided Temporal FNN"
    else:
        label = MODEL_LABELS.get(raw, raw.replace("_", " ").title())
    if feature_set in {"static", "temporal"}:
        return f"{label} ({feature_set})"
    return label


def _finite_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=columns)


def _downsample_curve(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        return x, y
    indices = np.unique(np.linspace(0, len(x) - 1, max_points).round().astype(int))
    return x[indices], y[indices]


def build_binary_curve_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_key: dict[str, Any],
    max_points: int = 500,
) -> pd.DataFrame:
    """Create downsampled ROC and precision-recall curve coordinates."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    finite = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[finite].astype(np.int8)
    y_prob = y_prob[finite]
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    if positives == 0 or negatives == 0:
        return pd.DataFrame()

    order = np.argsort(-y_prob, kind="mergesort")
    sorted_true = y_true[order]
    sorted_prob = y_prob[order]
    distinct = np.r_[np.where(np.diff(sorted_prob))[0], len(sorted_prob) - 1]
    tp = np.cumsum(sorted_true == 1)[distinct].astype(np.float64)
    fp = np.cumsum(sorted_true == 0)[distinct].astype(np.float64)

    fpr = np.r_[0.0, fp / negatives, 1.0]
    tpr = np.r_[0.0, tp / positives, 1.0]
    recall = np.r_[0.0, tp / positives]
    precision = np.r_[1.0, tp / np.maximum(tp + fp, 1.0)]
    fpr, tpr = _downsample_curve(fpr, tpr, max_points)
    recall, precision = _downsample_curve(recall, precision, max_points)

    rows = []
    for curve_type, x, y in [
        ("roc", fpr, tpr),
        ("precision_recall", recall, precision),
    ]:
        for point_index, (x_value, y_value) in enumerate(zip(x, y)):
            rows.append(
                {
                    **model_key,
                    "curve_type": curve_type,
                    "point_index": point_index,
                    "x": float(x_value),
                    "y": float(y_value),
                }
            )
    return pd.DataFrame(rows)


def generate_training_figures(metrics_csv: str | Path, output_dir: str | Path) -> list[Path]:
    path = Path(metrics_csv)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty or "epoch" not in df.columns:
        return []
    df = _finite_frame(df, ["epoch"])
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    for prefix, color in [("train", PALETTE[0]), ("val", PALETTE[1])]:
        loss_col = f"{prefix}_total_loss"
        if loss_col in df:
            axes[0].plot(df["epoch"], df[loss_col], marker="o", color=color, label=prefix.title())
        auroc_col = f"{prefix}_auroc"
        if auroc_col in df:
            axes[1].plot(df["epoch"], df[auroc_col], marker="o", color=color, label=prefix.title())
        auprc_col = f"{prefix}_auprc"
        if auprc_col in df:
            axes[2].plot(df["epoch"], df[auprc_col], marker="o", color=color, label=prefix.title())

    for ax, title, ylabel in zip(
        axes,
        ["Loss", "AUROC", "AUPRC"],
        ["Loss", "Score", "Score"],
    ):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
    axes[1].set_ylim(0, 1)
    axes[2].set_ylim(0, 1)
    fig.suptitle("Training and Validation History", y=1.03, fontsize=13)
    fig.tight_layout()
    outputs = _save(fig, output_dir, "training_history")
    plt.close(fig)
    return outputs


def generate_metric_comparison_figures(
    metrics_csv: str | Path,
    output_dir: str | Path,
    title: str,
) -> list[Path]:
    path = Path(metrics_csv)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    metric_prefix = "test" if {"test_auroc", "test_auprc"}.issubset(df.columns) else "val"
    auroc_col = f"{metric_prefix}_auroc"
    auprc_col = f"{metric_prefix}_auprc"
    required = ["model", auroc_col, auprc_col]
    if df.empty or not set(required).issubset(df.columns):
        return []
    df = _finite_frame(df, [auroc_col, auprc_col])
    if df.empty:
        return []
    df["label"] = [
        _model_label(model, feature_set)
        for model, feature_set in zip(df["model"], df.get("feature_set", pd.Series([None] * len(df))))
    ]
    if "threshold_strategy" in df.columns:
        df["label"] = df["label"] + " (" + df["threshold_strategy"].astype(str) + ")"
    df = df.sort_values(auroc_col)

    plt = _plt()
    height = max(4.0, 0.42 * len(df) + 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(12, height), sharey=True)
    for ax, metric, label, color in [
        (axes[0], auroc_col, "AUROC", PALETTE[0]),
        (axes[1], auprc_col, "AUPRC", PALETTE[1]),
    ]:
        ax.barh(df["label"], df[metric], color=color, alpha=0.9)
        ax.set_xlim(0, 1)
        ax.set_xlabel(label)
        ax.set_title(label)
        for index, value in enumerate(df[metric]):
            ax.text(min(value + 0.01, 0.97), index, f"{value:.3f}", va="center", fontsize=8)
    fig.suptitle(title, y=1.01, fontsize=13)
    fig.tight_layout()
    outputs = _save(fig, output_dir, "model_metric_comparison")
    plt.close(fig)
    return outputs


def generate_ablation_figures(summary_csv: str | Path, output_dir: str | Path) -> list[Path]:
    path = Path(summary_csv)
    if not path.exists():
        return []
    raw = pd.read_csv(path)
    if raw.empty or "variant" not in raw.columns:
        return []
    plt = _plt()
    outputs: list[Path] = []

    if "horizon_hours" in raw.columns:
        horizons = sorted(pd.to_numeric(raw["horizon_hours"], errors="coerce").dropna().astype(int).unique())
    else:
        horizons = [None]

    for horizon in horizons:
        frame = raw if horizon is None else raw[pd.to_numeric(raw["horizon_hours"], errors="coerce") == horizon]
        frame = frame.copy()
        metric_prefix = "test" if {"test_auroc", "test_auprc"}.issubset(frame.columns) else "val"
        auroc_col = f"{metric_prefix}_auroc"
        auprc_col = f"{metric_prefix}_auprc"
        quality_cols = [
            ("rule_drift_loss", "Rule drift loss"),
            ("active_rule_fraction_gt_0_1", "Activated rule fraction"),
            ("attention_entropy", "Attention entropy"),
        ]
        numeric_cols = [auroc_col, auprc_col, *[col for col, _ in quality_cols]]
        available_numeric = [col for col in numeric_cols if col in frame.columns]
        for col in available_numeric:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

        group_cols = ["variant"]
        if "display_name" in frame.columns:
            group_cols.append("display_name")
        means = frame.groupby(group_cols, sort=False)[available_numeric].mean().reset_index()
        stds = frame.groupby(group_cols, sort=False)[available_numeric].std().fillna(0.0).reset_index()
        stds = stds.rename(columns={col: f"{col}_std" for col in available_numeric})
        df = means.merge(stds, on=group_cols, validate="one_to_one")
        variant_order = {
            "random_init": 0,
            "static_guideline": 1,
            "no_consistency": 2,
            "full": 3,
        }
        df["variant_order"] = df["variant"].map(variant_order).fillna(len(variant_order))
        df = df.sort_values("variant_order")
        df["label"] = df.get("display_name", df["variant"]).fillna(df["variant"])
        short_labels = {
            "random_init": "Random\ninit",
            "static_guideline": "Static\nguideline",
            "no_consistency": "No\nconsistency",
            "full": "Full\ntemporal",
        }
        df["short_label"] = df["variant"].map(short_labels).fillna(df["variant"])
        suffix = f"_{horizon}h" if horizon is not None else ""
        title_suffix = f" ({horizon}-hour outcome)" if horizon is not None else ""

        predictive = _finite_frame(df, [auroc_col, auprc_col])
        if not predictive.empty:
            y = np.arange(len(predictive))
            width = 0.36
            fig, ax = plt.subplots(figsize=(10, max(4.8, 0.7 * len(predictive) + 1.5)))
            ax.barh(
                y - width / 2,
                predictive[auroc_col],
                width,
                xerr=predictive[f"{auroc_col}_std"],
                label="AUROC",
                color=PALETTE[0],
                capsize=3,
            )
            ax.barh(
                y + width / 2,
                predictive[auprc_col],
                width,
                xerr=predictive[f"{auprc_col}_std"],
                label="AUPRC",
                color=PALETTE[1],
                capsize=3,
            )
            ax.set_yticks(y)
            ax.set_yticklabels(predictive["label"])
            ax.invert_yaxis()
            ax.set_xlim(0, 1)
            ax.set_xlabel("Mean score across seeds (error bar: SD)")
            ax.set_title(f"Ablation Study: Predictive Performance{title_suffix}")
            ax.legend()
            fig.tight_layout()
            outputs.extend(_save(fig, output_dir, f"ablation_predictive_performance{suffix}"))
            plt.close(fig)

        available = [(col, label) for col, label in quality_cols if col in df.columns]
        if available:
            fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4.8))
            axes = np.atleast_1d(axes)
            for index, (col, label) in enumerate(available):
                axes[index].bar(
                    df["short_label"],
                    df[col],
                    yerr=df[f"{col}_std"],
                    color=PALETTE[index % len(PALETTE)],
                    capsize=3,
                )
                axes[index].set_title(label)
                axes[index].tick_params(axis="x", labelsize=9)
            fig.suptitle(f"Ablation Study: Rule and Temporal Quality{title_suffix}", y=1.03, fontsize=13)
            fig.tight_layout()
            outputs.extend(_save(fig, output_dir, f"ablation_rule_quality{suffix}"))
            plt.close(fig)

        # 每個差值皆為成對 seed 比較，正值代表啟用該元件後 AUROC 較高。
        if "seed" in frame.columns and "test_auroc" in frame.columns:
            contrasts = [
                ("Expert initialization", "full", "random_init"),
                ("Temporal design", "no_consistency", "static_guideline"),
                ("Consistency regularization", "full", "no_consistency"),
            ]
            effects = []
            for label, enabled, disabled in contrasts:
                left = frame[frame["variant"] == enabled][["seed", "test_auroc"]]
                right = frame[frame["variant"] == disabled][["seed", "test_auroc"]]
                paired = left.merge(right, on="seed", suffixes=("_enabled", "_disabled"))
                if paired.empty:
                    continue
                delta = paired["test_auroc_enabled"] - paired["test_auroc_disabled"]
                effects.append((label, float(delta.mean()), float(delta.std(ddof=1)) if len(delta) > 1 else 0.0))
            if effects:
                labels, values, errors = zip(*effects)
                fig, ax = plt.subplots(figsize=(9, 4.8))
                colors = [PALETTE[0] if value >= 0 else PALETTE[3] for value in values]
                ax.bar(labels, values, yerr=errors, color=colors, capsize=4)
                ax.axhline(0.0, color="#444444", linewidth=1)
                ax.set_ylabel("Paired test AUROC difference (mean, error bar: SD)")
                ax.set_title(f"Estimated Component Contributions{title_suffix}")
                ax.tick_params(axis="x", rotation=15)
                fig.tight_layout()
                outputs.extend(_save(fig, output_dir, f"ablation_component_contributions{suffix}"))
                plt.close(fig)
    return outputs


def generate_tuning_figures(trials_csv: str | Path, output_dir: str | Path) -> list[Path]:
    path = Path(trials_csv)
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if df.empty or "value" not in df.columns:
        return []
    values = pd.to_numeric(df["value"], errors="coerce")
    complete = df[values.notna()].copy()
    complete["value"] = values[values.notna()]
    if complete.empty:
        return []

    plt = _plt()
    outputs: list[Path] = []
    trials = np.arange(1, len(complete) + 1)
    best = np.maximum.accumulate(complete["value"].to_numpy())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(trials, complete["value"], color=PALETTE[0], alpha=0.7, label="Trial AUROC")
    ax.plot(trials, best, color=PALETTE[1], linewidth=2, label="Best so far")
    ax.set_xlabel("Completed trial")
    ax.set_ylabel("Validation AUROC")
    ax.set_title("Optuna Optimization History")
    ax.legend()
    fig.tight_layout()
    outputs.extend(_save(fig, output_dir, "optuna_optimization_history"))
    plt.close(fig)

    param_cols = [col for col in complete.columns if col.startswith("params_")]
    correlations = []
    for col in param_cols:
        numeric = pd.to_numeric(complete[col], errors="coerce")
        if numeric.notna().sum() >= 3 and numeric.nunique(dropna=True) > 1:
            corr = numeric.corr(complete["value"], method="spearman")
            if pd.notna(corr):
                correlations.append((col.removeprefix("params_"), abs(float(corr)), float(corr)))
    if correlations:
        corr_df = pd.DataFrame(correlations, columns=["parameter", "absolute", "signed"])
        corr_df = corr_df.nlargest(10, "absolute").sort_values("absolute")
        colors = [PALETTE[0] if value >= 0 else PALETTE[1] for value in corr_df["signed"]]
        fig, ax = plt.subplots(figsize=(8, max(4, 0.42 * len(corr_df) + 1.5)))
        ax.barh(corr_df["parameter"], corr_df["absolute"], color=colors)
        ax.set_xlabel("Absolute Spearman correlation with validation AUROC")
        ax.set_title("Hyperparameter Association")
        fig.tight_layout()
        outputs.extend(_save(fig, output_dir, "optuna_parameter_association"))
        plt.close(fig)
    return outputs


def _preferred_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    df = summary_df.copy()
    if "probability_source" in df.columns and (df["probability_source"] == "prediction_level").any():
        prediction_rows = df[df["probability_source"] == "prediction_level"].copy()
        key_cols = ["model", "target_col", "feature_set"]
        keys = set(map(tuple, prediction_rows[key_cols].astype(str).to_numpy()))
        keep_existing = ~df[key_cols].astype(str).apply(tuple, axis=1).isin(keys)
        df = pd.concat([prediction_rows, df[keep_existing]], ignore_index=True)
    return df


def _top_model_keys(summary: pd.DataFrame, horizon: int, top_n: int) -> set[tuple[str, str]]:
    frame = summary[pd.to_numeric(summary["horizon_hours"], errors="coerce") == horizon].copy()
    frame["auroc"] = pd.to_numeric(frame["auroc"], errors="coerce")
    frame = frame.dropna(subset=["auroc"]).sort_values("auroc", ascending=False)
    selected = frame.head(top_n)
    fnn = frame[frame["source_family"].astype(str).str.contains("fnn", case=False, na=False)]
    clinical = frame[
        frame["source_family"].eq("clinical_score")
        & frame.get("probability_source", pd.Series("", index=frame.index)).eq("prediction_level")
    ]
    selected = pd.concat([selected, fnn.head(1), clinical], ignore_index=True).drop_duplicates(
        ["model", "feature_set"]
    )
    return set(zip(selected["model"].astype(str), selected["feature_set"].astype(str)))


def generate_evaluation_figures(
    summary_df: pd.DataFrame,
    output_dir: str | Path,
    calibration_df: pd.DataFrame | None = None,
    curve_df: pd.DataFrame | None = None,
    top_n: int = 8,
) -> list[Path]:
    if summary_df.empty:
        return []
    summary = _preferred_summary(summary_df)
    summary["horizon_hours"] = pd.to_numeric(summary["horizon_hours"], errors="coerce")
    horizons = sorted(summary["horizon_hours"].dropna().astype(int).unique())
    plt = _plt()
    outputs: list[Path] = []

    for horizon in horizons:
        frame = summary[summary["horizon_hours"] == horizon].copy()
        frame = _finite_frame(frame, ["auroc", "auprc"])
        if frame.empty:
            continue
        frame["label"] = [
            _model_label(model, feature_set)
            for model, feature_set in zip(frame["model"], frame.get("feature_set", pd.Series([None] * len(frame))))
        ]
        ranked = frame.sort_values("auroc", ascending=False).head(top_n)
        required = frame[
            frame["source_family"].astype(str).str.contains("fnn", case=False, na=False)
            | (
                frame["source_family"].eq("clinical_score")
                & frame.get("probability_source", pd.Series("", index=frame.index)).eq("prediction_level")
            )
        ]
        frame = (
            pd.concat([ranked, required], ignore_index=True)
            .drop_duplicates(["model", "feature_set"])
            .sort_values("auroc")
        )
        fig, axes = plt.subplots(1, 2, figsize=(12, max(4.5, 0.45 * len(frame) + 1.5)), sharey=True)
        for ax, metric, color in [
            (axes[0], "auroc", PALETTE[0]),
            (axes[1], "auprc", PALETTE[1]),
        ]:
            ax.barh(frame["label"], frame[metric], color=color)
            ax.set_xlim(0, 1)
            ax.set_xlabel(metric.upper())
            ax.set_title(metric.upper())
            for index, value in enumerate(frame[metric]):
                ax.text(min(value + 0.01, 0.97), index, f"{value:.3f}", va="center", fontsize=8)
        fig.suptitle(f"Model Performance for {horizon}-hour Prediction", y=1.01, fontsize=13)
        fig.tight_layout()
        outputs.extend(_save(fig, output_dir, f"model_performance_{horizon}h"))
        plt.close(fig)

        calibration_frame = _finite_frame(frame, ["brier", "ece"])
        if not calibration_frame.empty:
            x = np.arange(len(calibration_frame))
            width = 0.36
            fig, ax = plt.subplots(figsize=(11, 4.8))
            ax.bar(x - width / 2, calibration_frame["brier"], width, label="Brier", color=PALETTE[2])
            ax.bar(x + width / 2, calibration_frame["ece"], width, label="ECE", color=PALETTE[3])
            ax.set_xticks(x)
            ax.set_xticklabels(calibration_frame["label"], rotation=22, ha="right")
            ax.set_ylabel("Error (lower is better)")
            ax.set_title(f"Calibration Error for {horizon}-hour Prediction")
            ax.legend()
            fig.tight_layout()
            outputs.extend(_save(fig, output_dir, f"calibration_error_{horizon}h"))
            plt.close(fig)

        selected_keys = _top_model_keys(summary, horizon, top_n)
        if curve_df is not None and not curve_df.empty:
            curves = curve_df[pd.to_numeric(curve_df["horizon_hours"], errors="coerce") == horizon].copy()
            curves = curves[
                curves.apply(
                    lambda row: (str(row["model"]), str(row.get("feature_set", ""))) in selected_keys,
                    axis=1,
                )
            ]
            for curve_type, xlabel, ylabel, stem in [
                ("roc", "False positive rate", "True positive rate", "roc_curves"),
                ("precision_recall", "Recall", "Precision", "precision_recall_curves"),
            ]:
                subset = curves[curves["curve_type"] == curve_type]
                if subset.empty:
                    continue
                fig, ax = plt.subplots(figsize=(7, 6))
                for index, ((model, feature_set), group) in enumerate(
                    subset.groupby(["model", "feature_set"], sort=False)
                ):
                    group = group.sort_values("point_index")
                    metric_name = "auroc" if curve_type == "roc" else "auprc"
                    score_rows = summary[
                        (summary["horizon_hours"] == horizon)
                        & summary["model"].astype(str).eq(str(model))
                        & summary["feature_set"].astype(str).eq(str(feature_set))
                    ]
                    score = pd.to_numeric(score_rows.get(metric_name), errors="coerce").dropna()
                    legend_label = _model_label(model, feature_set)
                    if not score.empty:
                        legend_label += f" ({metric_name.upper()}={score.iloc[0]:.3f})"
                    is_fnn = str(model).startswith("fnn_")
                    is_clinical = str(model) in {"news2_score_calibrated", "sofa_score_calibrated"}
                    ax.plot(
                        group["x"],
                        group["y"],
                        linewidth=3.0 if is_fnn else 1.8,
                        linestyle="--" if is_fnn else (":" if is_clinical else "-"),
                        color="#A51C30" if is_fnn else PALETTE[index % len(PALETTE)],
                        label=legend_label,
                        zorder=5 if is_fnn else 2,
                    )
                if curve_type == "roc":
                    ax.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1)
                else:
                    prevalence = pd.to_numeric(
                        summary.loc[summary["horizon_hours"] == horizon, "prevalence"],
                        errors="coerce",
                    ).dropna()
                    if not prevalence.empty:
                        ax.axhline(prevalence.iloc[0], linestyle="--", color="#777777", linewidth=1)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(f"{stem.replace('_', ' ').title()} ({horizon}h)")
                ax.legend(loc="best")
                fig.tight_layout()
                outputs.extend(_save(fig, output_dir, f"{stem}_{horizon}h"))
                plt.close(fig)

        if calibration_df is not None and not calibration_df.empty:
            bins = calibration_df[
                pd.to_numeric(calibration_df["horizon_hours"], errors="coerce") == horizon
            ].copy()
            bins = bins[
                bins.apply(
                    lambda row: (str(row["model"]), str(row.get("feature_set", ""))) in selected_keys,
                    axis=1,
                )
            ]
            bins = bins.dropna(subset=["mean_predicted_probability", "observed_event_rate"])
            if not bins.empty:
                fig, ax = plt.subplots(figsize=(7, 6))
                for index, ((model, feature_set), group) in enumerate(
                    bins.groupby(["model", "feature_set"], sort=False)
                ):
                    group = group.sort_values("bin")
                    is_fnn = str(model).startswith("fnn_")
                    is_clinical = str(model) in {"news2_score_calibrated", "sofa_score_calibrated"}
                    ax.plot(
                        group["mean_predicted_probability"],
                        group["observed_event_rate"],
                        marker="o",
                        linewidth=2.8 if is_fnn else 1.6,
                        linestyle="--" if is_fnn else (":" if is_clinical else "-"),
                        color="#A51C30" if is_fnn else PALETTE[index % len(PALETTE)],
                        label=_model_label(model, feature_set),
                        zorder=5 if is_fnn else 2,
                    )
                ax.plot([0, 1], [0, 1], "--", color="#777777", linewidth=1)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Mean predicted probability")
                ax.set_ylabel("Observed event rate")
                ax.set_title(f"Calibration Curves ({horizon}h)")
                ax.legend(loc="best")
                fig.tight_layout()
                outputs.extend(_save(fig, output_dir, f"calibration_curves_{horizon}h"))
                plt.close(fig)

    trend = _finite_frame(summary, ["horizon_hours", "auroc", "auprc"])
    counts = trend.groupby(["model", "feature_set"])["horizon_hours"].nunique()
    repeated = counts[counts >= 2].index
    if len(repeated):
        trend = trend.set_index(["model", "feature_set"]).loc[repeated].reset_index()
        top_models = (
            trend.groupby(["model", "feature_set"])["auroc"].mean().nlargest(top_n).index
        )
        trend = trend.set_index(["model", "feature_set"]).loc[top_models].reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for index, ((model, feature_set), group) in enumerate(
            trend.groupby(["model", "feature_set"], sort=False)
        ):
            group = group.sort_values("horizon_hours")
            label = _model_label(model, feature_set)
            color = PALETTE[index % len(PALETTE)]
            axes[0].plot(group["horizon_hours"], group["auroc"], marker="o", color=color, label=label)
            axes[1].plot(group["horizon_hours"], group["auprc"], marker="o", color=color, label=label)
        for ax, metric in zip(axes, ["AUROC", "AUPRC"]):
            ax.set_xlabel("Prediction horizon (hours)")
            ax.set_ylabel(metric)
            ax.set_ylim(0, 1)
            ax.set_title(metric)
        axes[1].legend(loc="best")
        fig.suptitle("Performance Across Prediction Horizons", y=1.02, fontsize=13)
        fig.tight_layout()
        outputs.extend(_save(fig, output_dir, "horizon_performance_trends"))
        plt.close(fig)
    return outputs


def try_generate(generator: Any, *args: Any, **kwargs: Any) -> list[Path]:
    """Run a figure generator without interrupting a completed experiment."""
    try:
        outputs = generator(*args, **kwargs)
        if outputs:
            print(f"Figures saved to: {Path(outputs[0]).parent}")
        return outputs
    except Exception as exc:
        print(f"Figure generation skipped: {exc}")
        return []
