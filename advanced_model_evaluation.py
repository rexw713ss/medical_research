"""Patient-clustered advanced evaluation for ICU deterioration models.

Required prediction columns:
subject_id, stay_id, sofa_hour, y_true, y_prob, evaluation_split.
Validation predictions determine operating thresholds and risk strata. All reported
performance is evaluated on the independent patient-level test split.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import precision_recall_curve, roc_curve

from model_evaluation_report import binary_metrics, calibration_bins
from project_config import PATIENT_SPLIT_CSV, SOFA_HOURLY_CSV


KEY_COLS = ["subject_id", "stay_id", "sofa_hour"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Advanced clustered model evaluation.")
    parser.add_argument("--predictions-root", default="outputs/fair_comparison_6h_equal_sample")
    parser.add_argument("--patient-split", default=PATIENT_SPLIT_CSV)
    parser.add_argument("--sofa-csv", default=SOFA_HOURLY_CSV)
    parser.add_argument("--target-col", default="label_sofa_increase_ge2_6h")
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--specificities", default="0.90,0.95")
    parser.add_argument("--risk-quantiles", default="0.60,0.85")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="沿用既有 DCA/risk CSV，只重新建立 quantile calibration 與論文圖。",
    )
    parser.add_argument(
        "--dca-max-threshold",
        type=float,
        default=0.20,
        help="Decision curve 的最高風險閾值；低盛行率 outcome 預設顯示 1%% 到 20%%。",
    )
    parser.add_argument("--output-dir", default="outputs/advanced_evaluation_6h_equal_sample")
    return parser.parse_args()


def display_model_name(model: str) -> str:
    """將輸出路徑與內部名稱轉成適合論文圖例的短名稱。"""
    lowered = model.lower()
    aliases = [
        ("news2", "NEWS2"),
        ("sofa_score", "SOFA"),
        ("logistic_regression", "Logistic Regression"),
        ("decision_tree", "Decision Tree"),
        ("random_forest", "Random Forest"),
        ("xgboost", "XGBoost"),
        ("lightgbm", "LightGBM"),
        ("lstm", "LSTM"),
        ("gru", "GRU"),
        ("ebm", "EBM"),
        ("gam", "GAM"),
        ("fnn", "KG-Temporal FNN"),
        ("training_", "KG-Temporal FNN"),
    ]
    for token, label in aliases:
        if token in lowered:
            return label
    return model.replace("_", " ")


def read_prediction(path: Path, target_col: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set([*KEY_COLS, "y_true", "y_prob"]) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} 缺少欄位: {sorted(missing)}")
    if "target_col" in frame and frame["target_col"].notna().any():
        frame = frame[frame["target_col"].astype(str) == target_col].copy()
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="coerce")
    frame["y_prob"] = pd.to_numeric(frame["y_prob"], errors="coerce")
    frame = frame.dropna(subset=[*KEY_COLS, "y_true", "y_prob"])
    if frame.duplicated(KEY_COLS).any():
        raise ValueError(f"{path} 有重複 prediction window。")
    return frame.sort_values(KEY_COLS).reset_index(drop=True)


def prediction_model_name(frame: pd.DataFrame, path: Path, root: Path) -> str:
    model = str(frame["model"].dropna().iloc[0]) if "model" in frame and frame["model"].notna().any() else path.parent.name
    feature = (
        str(frame["feature_set"].dropna().iloc[0])
        if "feature_set" in frame and frame["feature_set"].notna().any()
        else path.parent.parent.name
    )
    relative = str(path.parent.relative_to(root)).replace("\\", "/")
    return f"{feature}:{model}" if model not in relative else relative


def find_prediction_pairs(root: Path, target_col: str) -> dict[str, dict[str, pd.DataFrame]]:
    pairs: dict[str, dict[str, pd.DataFrame]] = {}
    test_paths = [*root.rglob("test_predictions.csv"), *root.rglob("test_predictions.csv.gz")]
    for test_path in sorted(test_paths):
        val_candidates = [
            test_path.with_name("val_predictions.csv"),
            test_path.with_name("val_predictions.csv.gz"),
        ]
        val_path = next((path for path in val_candidates if path.exists()), None)
        if val_path is None:
            continue
        test = read_prediction(test_path, target_col)
        val = read_prediction(val_path, target_col)
        if test.empty or val.empty:
            continue
        name = prediction_model_name(test, test_path, root)
        if name in pairs:
            name = f"{name}@{test_path.parent.name}"
        pairs[name] = {"validation": val, "test": test, "path": test_path}
    if not pairs:
        raise FileNotFoundError(f"{root} 中找不到成對的 validation/test predictions。")
    return pairs


def validate_independent_splits(
    models: dict[str, dict[str, pd.DataFrame]],
    split_path: Path,
) -> dict[str, Any]:
    manifest = pd.read_csv(split_path, usecols=["subject_id", "split"])
    split_map = dict(zip(manifest["subject_id"], manifest["split"]))
    errors = []
    reference_keys = None
    for model, parts in models.items():
        val = parts["validation"]
        test = parts["test"]
        val_subjects = set(val["subject_id"].unique())
        test_subjects = set(test["subject_id"].unique())
        if val_subjects & test_subjects:
            errors.append(f"{model}: validation/test subject overlap")
        if any(split_map.get(subject) != "validation" for subject in val_subjects):
            errors.append(f"{model}: validation subject 不符 manifest")
        if any(split_map.get(subject) != "test" for subject in test_subjects):
            errors.append(f"{model}: test subject 不符 manifest")
        keys = pd.MultiIndex.from_frame(test[KEY_COLS])
        if reference_keys is None:
            reference_keys = keys
        elif not keys.equals(reference_keys):
            errors.append(f"{model}: test windows 未與其他模型完全對齊")
    return {
        "status": "passed" if not errors else "failed",
        "patient_split": str(split_path),
        "models": len(models),
        "errors": errors,
    }


def threshold_at_specificity(y_true: np.ndarray, y_prob: np.ndarray, specificity: float) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, drop_intermediate=False)
    valid = np.flatnonzero((1.0 - fpr) >= specificity)
    if len(valid) == 0:
        return math.nan
    # 在達到指定 specificity 的 thresholds 中選 validation sensitivity 最高者。
    best_tpr = np.max(tpr[valid])
    best = valid[tpr[valid] == best_tpr]
    return float(np.min(thresholds[best]))


def operating_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = y_prob >= threshold
    positive = y_true == 1
    negative = ~positive
    tp = float(np.sum(pred & positive))
    fn = float(np.sum(~pred & positive))
    tn = float(np.sum(~pred & negative))
    fp = float(np.sum(pred & negative))
    sensitivity = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    ppv = tp / (tp + fp) if tp + fp else math.nan
    npv = tn / (tn + fn) if tn + fn else math.nan
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if ppv + sensitivity else math.nan
    return {
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
    }


def calibration_intercept_slope(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    p = np.clip(y_prob.astype(float), 1e-6, 1 - 1e-6)
    x = np.log(p / (1 - p))

    def objective(params: np.ndarray) -> float:
        logits = params[0] + params[1] * x
        return float(np.sum(np.logaddexp(0.0, logits) - y_true * logits))

    result = minimize(objective, np.array([0.0, 1.0]), method="BFGS")
    return float(result.x[0]), float(result.x[1])


def apply_platt_calibration(y_prob: np.ndarray, intercept: float, slope: float) -> np.ndarray:
    probability = np.clip(y_prob.astype(float), 1e-6, 1 - 1e-6)
    logit = np.log(probability / (1 - probability))
    return expit(intercept + slope * logit)


def weighted_auc(y: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float:
    order = np.argsort(score, kind="mergesort")
    y_sorted = y[order]
    score_sorted = score[order]
    w_sorted = weight[order]
    starts = np.r_[0, np.flatnonzero(score_sorted[1:] != score_sorted[:-1]) + 1]
    pos = np.add.reduceat(w_sorted * (y_sorted == 1), starts)
    neg = np.add.reduceat(w_sorted * (y_sorted == 0), starts)
    total_pos = pos.sum()
    total_neg = neg.sum()
    if total_pos <= 0 or total_neg <= 0:
        return math.nan
    neg_before = np.cumsum(neg) - neg
    return float(np.sum(pos * (neg_before + 0.5 * neg)) / (total_pos * total_neg))


def weighted_auprc(y: np.ndarray, score: np.ndarray, weight: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    y_sorted = y[order]
    w_sorted = weight[order]
    pos_weight = w_sorted * (y_sorted == 1)
    total_pos = pos_weight.sum()
    if total_pos <= 0:
        return math.nan
    tp = np.cumsum(pos_weight)
    total = np.cumsum(w_sorted)
    precision = np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)
    return float(np.sum(precision * pos_weight) / total_pos)


def weighted_operating_metrics(
    y: np.ndarray,
    score: np.ndarray,
    weight: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    pred = score >= threshold
    tp = np.sum(weight * pred * (y == 1))
    fn = np.sum(weight * (~pred) * (y == 1))
    tn = np.sum(weight * (~pred) * (y == 0))
    fp = np.sum(weight * pred * (y == 0))
    sensitivity = tp / (tp + fn) if tp + fn else math.nan
    specificity = tn / (tn + fp) if tn + fp else math.nan
    ppv = tp / (tp + fp) if tp + fp else math.nan
    npv = tn / (tn + fn) if tn + fn else math.nan
    f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if ppv + sensitivity else math.nan
    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1),
    }


def weighted_ece(
    y: np.ndarray,
    score: np.ndarray,
    weight: np.ndarray,
    n_bins: int = 10,
) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum(np.digitize(score, edges[1:-1], right=False), n_bins - 1)
    total_weight = weight.sum()
    if total_weight <= 0:
        return math.nan
    ece = 0.0
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        bin_weight = weight[mask].sum()
        if bin_weight <= 0:
            continue
        mean_probability = np.sum(weight[mask] * score[mask]) / bin_weight
        event_rate = np.sum(weight[mask] * y[mask]) / bin_weight
        ece += (bin_weight / total_weight) * abs(mean_probability - event_rate)
    return float(ece)


def patient_bootstrap(
    frame: pd.DataFrame,
    thresholds: dict[float, float],
    reps: int,
    seed: int,
) -> pd.DataFrame:
    y = frame["y_true"].to_numpy(dtype=np.int8)
    score = frame["y_prob"].to_numpy(dtype=float)
    subjects, subject_codes = np.unique(frame["subject_id"], return_inverse=True)
    rng = np.random.default_rng(seed)

    ascending = np.argsort(score, kind="mergesort")
    y_asc = y[ascending]
    score_asc = score[ascending]
    codes_asc = subject_codes[ascending]
    tie_starts = np.r_[0, np.flatnonzero(score_asc[1:] != score_asc[:-1]) + 1]

    descending = np.argsort(-score, kind="mergesort")
    y_desc = y[descending]
    codes_desc = subject_codes[descending]
    positive_desc = y_desc == 1

    n_subjects = len(subjects)
    subject_window_count = np.bincount(subject_codes, minlength=n_subjects).astype(float)
    subject_squared_error = np.bincount(
        subject_codes,
        weights=(score - y) ** 2,
        minlength=n_subjects,
    )
    bin_ids = np.minimum(np.digitize(score, np.linspace(0.1, 0.9, 9)), 9)
    bin_subject_count = np.vstack(
        [
            np.bincount(subject_codes, weights=(bin_ids == bin_idx), minlength=n_subjects)
            for bin_idx in range(10)
        ]
    )
    bin_subject_score = np.vstack(
        [
            np.bincount(
                subject_codes,
                weights=score * (bin_ids == bin_idx),
                minlength=n_subjects,
            )
            for bin_idx in range(10)
        ]
    )
    bin_subject_event = np.vstack(
        [
            np.bincount(
                subject_codes,
                weights=(y == 1) * (bin_ids == bin_idx),
                minlength=n_subjects,
            )
            for bin_idx in range(10)
        ]
    )
    operating_subject_counts = {}
    for specificity, threshold in thresholds.items():
        predicted = score >= threshold
        operating_subject_counts[specificity] = {
            "tp": np.bincount(subject_codes, weights=predicted & (y == 1), minlength=n_subjects),
            "fn": np.bincount(subject_codes, weights=(~predicted) & (y == 1), minlength=n_subjects),
            "tn": np.bincount(subject_codes, weights=(~predicted) & (y == 0), minlength=n_subjects),
            "fp": np.bincount(subject_codes, weights=predicted & (y == 0), minlength=n_subjects),
        }

    rows = []
    for replicate in range(reps):
        subject_weight = rng.multinomial(len(subjects), np.full(len(subjects), 1 / len(subjects)))
        subject_weight = subject_weight.astype(float)
        total_weight = float(np.dot(subject_weight, subject_window_count))

        weights_asc = subject_weight[codes_asc]
        positive_by_tie = np.add.reduceat(weights_asc * (y_asc == 1), tie_starts)
        negative_by_tie = np.add.reduceat(weights_asc * (y_asc == 0), tie_starts)
        total_positive = positive_by_tie.sum()
        total_negative = negative_by_tie.sum()
        negative_before = np.cumsum(negative_by_tie) - negative_by_tie
        auroc = np.sum(positive_by_tie * (negative_before + 0.5 * negative_by_tie))
        auroc = auroc / (total_positive * total_negative)

        weights_desc = subject_weight[codes_desc]
        positive_weight = weights_desc * positive_desc
        cumulative_positive = np.cumsum(positive_weight)
        cumulative_total = np.cumsum(weights_desc)
        precision = np.divide(
            cumulative_positive,
            cumulative_total,
            out=np.zeros_like(cumulative_positive),
            where=cumulative_total > 0,
        )
        auprc = np.sum(precision * positive_weight) / total_positive

        bin_count = bin_subject_count @ subject_weight
        bin_score = bin_subject_score @ subject_weight
        bin_event = bin_subject_event @ subject_weight
        valid_bins = bin_count > 0
        ece = np.sum(
            (bin_count[valid_bins] / total_weight)
            * np.abs(
                bin_score[valid_bins] / bin_count[valid_bins]
                - bin_event[valid_bins] / bin_count[valid_bins]
            )
        )
        row = {
            "replicate": replicate,
            "auroc": float(auroc),
            "auprc": float(auprc),
            "brier": float(np.dot(subject_weight, subject_squared_error) / total_weight),
            "ece": float(ece),
        }
        for specificity, threshold in thresholds.items():
            counts = operating_subject_counts[specificity]
            tp = np.dot(subject_weight, counts["tp"])
            fn = np.dot(subject_weight, counts["fn"])
            tn = np.dot(subject_weight, counts["tn"])
            fp = np.dot(subject_weight, counts["fp"])
            sensitivity = tp / (tp + fn) if tp + fn else math.nan
            specificity_value = tn / (tn + fp) if tn + fp else math.nan
            ppv = tp / (tp + fp) if tp + fp else math.nan
            npv = tn / (tn + fn) if tn + fn else math.nan
            f1 = 2 * ppv * sensitivity / (ppv + sensitivity) if ppv + sensitivity else math.nan
            operating = {
                "sensitivity": sensitivity,
                "specificity": specificity_value,
                "ppv": ppv,
                "npv": npv,
                "f1": f1,
            }
            tag = int(round(specificity * 100))
            for metric, value in operating.items():
                row[f"{metric}_at_spec_{tag}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def percentile_ci(values: pd.Series) -> tuple[float, float]:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return math.nan, math.nan
    return tuple(np.quantile(finite, [0.025, 0.975]).astype(float))


def decision_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model: str,
    max_threshold: float,
) -> pd.DataFrame:
    prevalence = float(np.mean(y_true))
    rows = []
    for threshold in np.linspace(0.01, max_threshold, 50):
        pred = y_prob >= threshold
        tp = np.mean(pred & (y_true == 1))
        fp = np.mean(pred & (y_true == 0))
        odds = threshold / (1 - threshold)
        rows.append(
            {
                "model": model,
                "threshold": threshold,
                "net_benefit": tp - fp * odds,
                "treat_all": prevalence - (1 - prevalence) * odds,
                "treat_none": 0.0,
            }
        )
    return pd.DataFrame(rows)


def quantile_calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int,
    model: str,
) -> pd.DataFrame:
    """以等人數分箱呈現低盛行率 outcome 的 calibration curve。"""
    frame = pd.DataFrame(
        {
            "y_true": np.asarray(y_true, dtype=np.int8),
            "y_prob": np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1.0 - 1e-7),
        }
    )
    frame["bin"] = pd.qcut(frame["y_prob"], q=n_bins, labels=False, duplicates="drop")
    rows = []
    for bin_id, group in frame.dropna(subset=["bin"]).groupby("bin", sort=True):
        rows.append(
            {
                "model": model,
                "bin": int(bin_id),
                "bin_left": float(group["y_prob"].min()),
                "bin_right": float(group["y_prob"].max()),
                "count": int(len(group)),
                "mean_predicted_probability": float(group["y_prob"].mean()),
                "observed_event_rate": float(group["y_true"].mean()),
                "absolute_gap": float(
                    abs(group["y_prob"].mean() - group["y_true"].mean())
                ),
            }
        )
    return pd.DataFrame(rows)


def risk_strata(
    val: pd.DataFrame,
    test: pd.DataFrame,
    model: str,
    quantiles: tuple[float, float],
) -> pd.DataFrame:
    cut_low, cut_high = np.quantile(val["y_prob"], quantiles)
    groups = pd.cut(
        test["y_prob"],
        bins=[-np.inf, cut_low, cut_high, np.inf],
        labels=["low", "medium", "high"],
    )
    frame = test.assign(risk_group=groups)
    result = (
        frame.groupby("risk_group", observed=True)["y_true"]
        .agg([("windows", "size"), ("events", "sum"), ("event_rate", "mean")])
        .reset_index()
    )
    result.insert(0, "model", model)
    result["validation_cut_low"] = cut_low
    result["validation_cut_high"] = cut_high
    return result


def build_first_event_table(reference: pd.DataFrame, sofa_csv: Path, horizon: int) -> pd.DataFrame:
    stays = set(reference.loc[reference["y_true"] == 1, "stay_id"].unique())
    sofa_parts = []
    for chunk in pd.read_csv(
        sofa_csv,
        usecols=["stay_id", "sofa_hour", "sofa_score"],
        chunksize=500_000,
    ):
        part = chunk[chunk["stay_id"].isin(stays)]
        if not part.empty:
            sofa_parts.append(part)
    sofa = pd.concat(sofa_parts, ignore_index=True).sort_values(["stay_id", "sofa_hour"])
    score_groups = {stay: group.set_index("sofa_hour")["sofa_score"] for stay, group in sofa.groupby("stay_id")}
    rows = []
    positive = reference[reference["y_true"] == 1].sort_values(["stay_id", "sofa_hour"])
    for stay_id, group in positive.groupby("stay_id", sort=False):
        alert_hour = int(group["sofa_hour"].iloc[0])
        scores = score_groups.get(stay_id)
        if scores is None or alert_hour not in scores.index or pd.isna(scores.loc[alert_hour]):
            continue
        baseline = float(scores.loc[alert_hour])
        future = scores.loc[(scores.index > alert_hour) & (scores.index <= alert_hour + horizon)]
        hit = future[future >= baseline + 2]
        if hit.empty:
            continue
        first_row = group.iloc[0]
        rows.append(
            {
                "subject_id": first_row["subject_id"],
                "stay_id": stay_id,
                "event_hour": int(hit.index[0]),
                "window_start_hour": alert_hour,
            }
        )
    return pd.DataFrame(rows)


def lead_time_summary(
    test: pd.DataFrame,
    events: pd.DataFrame,
    threshold: float,
    model: str,
    horizon: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    event_lookup = events.set_index("stay_id")
    details = []
    for stay_id, event in event_lookup.iterrows():
        event_hour = int(event["event_hour"])
        windows = test[
            (test["stay_id"] == stay_id)
            & (test["sofa_hour"] < event_hour)
            & (test["sofa_hour"] >= event_hour - horizon)
        ]
        alerts = windows[windows["y_prob"] >= threshold]
        lead_time = float(event_hour - alerts["sofa_hour"].min()) if not alerts.empty else math.nan
        details.append(
            {
                "model": model,
                "subject_id": event["subject_id"],
                "stay_id": stay_id,
                "event_hour": event_hour,
                "detected": int(not alerts.empty),
                "lead_time_hours": lead_time,
            }
        )
    detail = pd.DataFrame(details)
    detected = detail["lead_time_hours"].dropna()
    summary = {
        "model": model,
        "events": int(len(detail)),
        "detected_events": int(detail["detected"].sum()) if not detail.empty else 0,
        "event_detection_rate": float(detail["detected"].mean()) if not detail.empty else math.nan,
        "lead_time_median_h": float(detected.median()) if not detected.empty else math.nan,
        "lead_time_q1_h": float(detected.quantile(0.25)) if not detected.empty else math.nan,
        "lead_time_q3_h": float(detected.quantile(0.75)) if not detected.empty else math.nan,
    }
    return summary, detail


def save_figures(
    models: dict[str, dict[str, pd.DataFrame]],
    calibration_frames: list[pd.DataFrame],
    dca: pd.DataFrame,
    risk: pd.DataFrame,
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(models), 3)))

    for kind in ["roc", "pr"]:
        fig, ax = plt.subplots(figsize=(8, 7))
        for color, (model, parts) in zip(colors, models.items()):
            y = parts["test"]["y_true"].to_numpy()
            p = parts["test"]["y_prob"].to_numpy()
            if kind == "roc":
                x, curve_y, _ = roc_curve(y, p)
                score = binary_metrics(y, p)["auroc"]
                ax.set(xlabel="False positive rate", ylabel="Sensitivity")
            else:
                curve_y, x, _ = precision_recall_curve(y, p)
                score = binary_metrics(y, p)["auprc"]
                ax.set(xlabel="Recall", ylabel="Precision")
            ax.plot(x, curve_y, color=color, label=f"{display_model_name(model)} ({score:.3f})")
        ax.set_title(f"6-hour {kind.upper()} curves")
        ax.legend(fontsize=7, ncol=2, loc="lower right" if kind == "roc" else "upper right")
        fig.tight_layout()
        for suffix in ["png", "pdf"]:
            fig.savefig(figures / f"{kind}_curves_6h.{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1)
    for color, table in zip(colors, calibration_frames):
        valid = table.dropna(subset=["mean_predicted_probability", "observed_event_rate"])
        if not valid.empty:
            ax.plot(
                valid["mean_predicted_probability"],
                valid["observed_event_rate"],
                marker="o",
                color=color,
                label=display_model_name(str(valid["model"].iloc[0])),
            )
    ax.set(xlabel="Predicted probability", ylabel="Observed event rate", title="6-hour calibration")
    calibration_points = pd.concat(calibration_frames, ignore_index=True)
    calibration_max = float(
        calibration_points[["mean_predicted_probability", "observed_event_rate"]]
        .max(numeric_only=True)
        .max()
    )
    calibration_limit = min(1.0, max(0.15, calibration_max * 1.08))
    ax.set_xlim(0, calibration_limit)
    ax.set_ylim(0, calibration_limit)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(figures / f"calibration_6h.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for color, (model, group) in zip(colors, dca.groupby("model", sort=False)):
        ax.plot(
            group["threshold"],
            group["net_benefit"],
            color=color,
            label=display_model_name(str(model)),
        )
    reference = dca.drop_duplicates("threshold").sort_values("threshold")
    ax.plot(reference["threshold"], reference["treat_all"], "--", color="gray", label="Treat all")
    ax.axhline(0, color="black", linewidth=1, label="Treat none")
    ax.set(xlabel="Risk threshold", ylabel="Net benefit", title="Decision Curve Analysis")
    model_upper = float(dca["net_benefit"].quantile(0.995))
    ax.set_ylim(-0.02, max(0.065, model_upper * 1.15))
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(figures / f"decision_curve_6h.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    plot_risk = risk.copy()
    plot_risk["model"] = plot_risk["model"].map(display_model_name)
    pivot = plot_risk.pivot(index="model", columns="risk_group", values="event_rate")
    pivot = pivot.reindex(columns=["low", "medium", "high"])
    ax = pivot.plot(kind="bar", figsize=(13, 6), color=["#4E79A7", "#F28E2B", "#E15759"])
    ax.set(ylabel="Observed event rate", title="Validation-defined risk strata on test set")
    ax.tick_params(axis="x", rotation=20)
    ax.figure.tight_layout()
    for suffix in ["png", "pdf"]:
        ax.figure.savefig(figures / f"risk_stratification_6h.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(ax.figure)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_models = find_prediction_pairs(Path(args.predictions_root), args.target_col)
    audit = validate_independent_splits(raw_models, Path(args.patient_split))
    (output_dir / "independent_test_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if audit["status"] != "passed":
        raise ValueError(f"Prediction audit failed: {audit['errors']}")

    if args.figures_only:
        calibration_tables = []
        calibrated_models = {}
        for model, parts in raw_models.items():
            val = parts["validation"].copy()
            test = parts["test"].copy()
            intercept, slope = calibration_intercept_slope(
                val["y_true"].to_numpy(dtype=np.int8),
                val["y_prob"].to_numpy(dtype=float),
            )
            val["y_prob"] = apply_platt_calibration(val["y_prob"].to_numpy(), intercept, slope)
            test["y_prob"] = apply_platt_calibration(test["y_prob"].to_numpy(), intercept, slope)
            calibrated_models[model] = {"validation": val, "test": test}
            calibration_tables.append(
                quantile_calibration_bins(
                    test["y_true"].to_numpy(dtype=np.int8),
                    test["y_prob"].to_numpy(dtype=float),
                    args.n_bins,
                    model,
                )
            )
        calibration = pd.concat(calibration_tables, ignore_index=True)
        calibration.to_csv(output_dir / "calibration_bins.csv", index=False)
        dca = pd.read_csv(output_dir / "decision_curve.csv")
        risk = pd.read_csv(output_dir / "risk_stratification.csv")
        save_figures(calibrated_models, calibration_tables, dca, risk, output_dir)
        print(f"Figures regenerated: {output_dir}")
        return

    specificities = tuple(float(value) for value in args.specificities.split(","))
    risk_quantiles = tuple(float(value) for value in args.risk_quantiles.split(","))
    summary_rows = []
    operating_rows = []
    bootstrap_tables = {}
    calibration_tables = []
    dca_tables = []
    risk_tables = []

    calibrated_models: dict[str, dict[str, pd.DataFrame]] = {}
    for model, parts in raw_models.items():
        val = parts["validation"].copy()
        test = parts["test"].copy()
        y_val = val["y_true"].to_numpy(dtype=np.int8)
        p_val_raw = val["y_prob"].to_numpy(dtype=float)
        y_test = test["y_true"].to_numpy(dtype=np.int8)
        p_test_raw = test["y_prob"].to_numpy(dtype=float)
        platt_intercept, platt_slope = calibration_intercept_slope(y_val, p_val_raw)
        p_val = apply_platt_calibration(p_val_raw, platt_intercept, platt_slope)
        p_test = apply_platt_calibration(p_test_raw, platt_intercept, platt_slope)
        val["y_prob_raw"] = p_val_raw
        test["y_prob_raw"] = p_test_raw
        val["y_prob"] = p_val
        test["y_prob"] = p_test
        calibrated_models[model] = {"validation": val, "test": test, "path": parts["path"]}
        thresholds = {
            specificity: threshold_at_specificity(y_val, p_val, specificity)
            for specificity in specificities
        }
        raw_point = binary_metrics(y_test, p_test_raw)
        point = binary_metrics(y_test, p_test)
        raw_point["ece"] = weighted_ece(y_test, p_test_raw, np.ones_like(p_test_raw))
        point["ece"] = weighted_ece(y_test, p_test, np.ones_like(p_test))
        intercept, slope = calibration_intercept_slope(y_test, p_test)
        raw_intercept, raw_slope = calibration_intercept_slope(y_test, p_test_raw)
        row = {
            "model": model,
            "evaluation_split": "independent_test",
            "patients": int(test["subject_id"].nunique()),
            "windows": int(len(test)),
            **point,
            "raw_brier": raw_point["brier"],
            "raw_ece": raw_point["ece"],
            "raw_log_loss": raw_point["log_loss"],
            "raw_calibration_intercept": raw_intercept,
            "raw_calibration_slope": raw_slope,
            "calibration_intercept": intercept,
            "calibration_slope": slope,
            "validation_platt_intercept": platt_intercept,
            "validation_platt_slope": platt_slope,
            "calibration_method": "validation_only_platt",
        }
        for specificity, threshold in thresholds.items():
            metrics = operating_metrics(y_test, p_test, threshold)
            tag = int(round(specificity * 100))
            for key, value in metrics.items():
                operating_rows.append(
                    {
                        "model": model,
                        "target_specificity": specificity,
                        "metric": key,
                        "value": value,
                    }
                )
            row[f"threshold_spec_{tag}"] = threshold
            for metric in ("sensitivity", "specificity", "ppv", "npv", "f1"):
                prefix = "observed_specificity" if metric == "specificity" else metric
                row[f"{prefix}_at_spec_{tag}"] = metrics[metric]

        bootstrap = patient_bootstrap(test, thresholds, args.bootstrap_reps, args.seed)
        bootstrap.insert(0, "model", model)
        bootstrap_tables[model] = bootstrap
        operating_ci_metrics = [
            f"{metric}_at_spec_{int(round(specificity * 100))}"
            for specificity in specificities
            for metric in ("sensitivity", "specificity", "ppv", "npv", "f1")
        ]
        for metric in ["auroc", "auprc", "brier", "ece", *operating_ci_metrics]:
            low, high = percentile_ci(bootstrap[metric])
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        summary_rows.append(row)

        bins = quantile_calibration_bins(y_test, p_test, args.n_bins, model)
        calibration_tables.append(bins)
        dca_tables.append(decision_curve(y_test, p_test, model, args.dca_max_threshold))
        risk_tables.append(risk_strata(val, test, model, risk_quantiles))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "advanced_metrics.csv", index=False)
    pd.DataFrame(operating_rows).to_csv(output_dir / "fixed_specificity_metrics.csv", index=False)
    bootstrap_all = pd.concat(bootstrap_tables.values(), ignore_index=True)
    bootstrap_all.to_csv(output_dir / "patient_cluster_bootstrap.csv.gz", index=False, compression="gzip")

    paired_rows = []
    for left, right in itertools.combinations(calibrated_models, 2):
        paired_metrics = [
            "auroc",
            "auprc",
            "brier",
            *[f"sensitivity_at_spec_{int(round(s*100))}" for s in specificities],
        ]
        for metric in paired_metrics:
            delta = bootstrap_tables[left][metric] - bootstrap_tables[right][metric]
            low, high = percentile_ci(delta)
            point_delta = float(
                summary.loc[summary["model"] == left, metric].iloc[0]
                - summary.loc[summary["model"] == right, metric].iloc[0]
            )
            p_value = 2 * min(float(np.mean(delta <= 0)), float(np.mean(delta >= 0)))
            paired_rows.append(
                {
                    "model_a": left,
                    "model_b": right,
                    "metric": metric,
                    "delta_a_minus_b": point_delta,
                    "cluster_bootstrap_ci95_low": low,
                    "cluster_bootstrap_ci95_high": high,
                    "two_sided_bootstrap_p": min(p_value, 1.0),
                    "bootstrap_unit": "subject_id",
                }
            )
    pd.DataFrame(paired_rows).to_csv(output_dir / "paired_model_comparisons.csv", index=False)

    calibration = pd.concat(calibration_tables, ignore_index=True)
    calibration.to_csv(output_dir / "calibration_bins.csv", index=False)
    dca = pd.concat(dca_tables, ignore_index=True)
    dca.to_csv(output_dir / "decision_curve.csv", index=False)
    risk = pd.concat(risk_tables, ignore_index=True)
    risk.to_csv(output_dir / "risk_stratification.csv", index=False)

    reference = next(iter(calibrated_models.values()))["test"]
    events = build_first_event_table(reference, Path(args.sofa_csv), args.horizon)
    lead_summaries = []
    lead_details = []
    for model, parts in calibrated_models.items():
        threshold = float(summary.loc[summary["model"] == model, "threshold_spec_90"].iloc[0])
        lead_summary, lead_detail = lead_time_summary(
            parts["test"], events, threshold, model, args.horizon
        )
        lead_summaries.append(lead_summary)
        lead_details.append(lead_detail)
    pd.DataFrame(lead_summaries).to_csv(output_dir / "lead_time_summary.csv", index=False)
    pd.concat(lead_details, ignore_index=True).to_csv(
        output_dir / "lead_time_events.csv.gz", index=False, compression="gzip"
    )

    save_figures(calibrated_models, calibration_tables, dca, risk, output_dir)
    config = {
        **vars(args),
        "models": list(calibrated_models),
        "bootstrap_unit": "subject_id",
        "calibration": "Platt scaling fit on validation only and applied unchanged to test",
        "calibration_curve_binning": "equal-frequency quantile bins on independent test",
    }
    (output_dir / "evaluation_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary[["model", "auroc", "auprc", "brier", "calibration_intercept", "calibration_slope"]].to_string(index=False))
    print(f"Advanced evaluation complete: {output_dir}")


if __name__ == "__main__":
    main()
