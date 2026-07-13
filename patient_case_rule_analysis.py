"""Extract patient-specific cross-rule firing for the prespecified TP/FP/FN cases."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from anfis_model import FEATURE_ORDER, explicit_temporal_input_order
from raw_rule_firing_analysis import load_frozen_model
from train_fnn import prepare_explicit_temporal_arrays


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "outputs/explicit_temporal_fnn_formal_6h/seed_42"
CASE_CSV = ROOT / "outputs/rule_evaluation_6h/selected_tp_fp_fn_cases.csv"
HOURLY_CSV = ROOT / "model_hourly_features_v3.csv"
OUTPUT = ROOT / "outputs/rule_evaluation_6h/patient_case_rules"
TARGET = "label_sofa_increase_ge2_6h"
TIME = "sofa_hour"
SPLIT = "subject_id"


def load_case_stays(cases: pd.DataFrame) -> pd.DataFrame:
    cache = OUTPUT / "selected_case_hourly_rows.csv.gz"
    if cache.exists():
        return pd.read_csv(cache)
    columns = ["stay_id", SPLIT, TIME, TARGET, *explicit_temporal_input_order(FEATURE_ORDER)]
    wanted = set(cases.stay_id.astype(np.int64))
    chunks = []
    for chunk in pd.read_csv(HOURLY_CSV, usecols=columns, chunksize=250_000):
        keep = chunk.stay_id.isin(wanted)
        if keep.any():
            chunks.append(chunk.loc[keep].copy())
    if not chunks:
        raise RuntimeError("None of the selected ICU stays were found in the hourly feature file.")
    result = pd.concat(chunks, ignore_index=True)
    result.to_csv(cache, index=False, compression="gzip")
    return result


def rule_text(config: dict) -> str:
    antecedents = " AND ".join(
        f"{feature} IS {term}" for feature, term in config["antecedents"]
    )
    return f"IF {antecedents} THEN deterioration risk increases"


def short_rule_name(config: dict) -> str:
    names = {
        "oxygenation_failure_with_support": "oxygenation failure",
        "hypoperfusion_pattern": "hypoperfusion",
        "multi_organ_dysfunction_pattern": "multi-organ dysfunction",
        "altered_consciousness_hypoxemia": "altered consciousness + hypoxemia",
        "respiratory_failure_pattern": "respiratory failure",
        "renal_failure_pattern": "renal failure",
    }
    return names.get(config["name"], config["name"].replace("_", " "))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "figures").mkdir(exist_ok=True)
    cases = pd.read_csv(CASE_CSV)
    frame = load_case_stays(cases)
    features, labels, stay_ids, _, time_values = prepare_explicit_temporal_arrays(
        frame, target_col=TARGET, time_col=TIME, split_col=SPLIT
    )

    config = json.loads((RUN_DIR / "train_config.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_frozen_model(RUN_DIR, config, device)
    weights = model.static_fnn.cross_rule_weights.detach().cpu().numpy()
    relative_hours = np.arange(-config["seq_length"] + 1, 1)

    result_rows = []
    figure, axes = plt.subplots(3, 1, figsize=(11, 7.8), sharex=True)
    colors = ["#0072B2", "#D55E00", "#009E73"]
    for axis, case in zip(axes, cases.itertuples(index=False)):
        matches = np.flatnonzero(
            (stay_ids.astype(np.int64) == int(case.stay_id))
            & (time_values == int(case.sofa_hour))
        )
        if len(matches) != 1:
            raise RuntimeError(f"Expected one target row for stay {case.stay_id}, hour {case.sofa_hour}.")
        target_index = int(matches[0])
        start = target_index - config["seq_length"] + 1
        if start < 0 or np.any(stay_ids[start : target_index + 1] != case.stay_id):
            raise RuntimeError(f"Incomplete observation window for stay {case.stay_id}.")

        x = torch.from_numpy(features[start : target_index + 1]).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(x)
        firing = output.raw_rule_firing[0].detach().cpu().numpy()
        attention = output.attention_weights[0].detach().cpu().numpy()
        selected_hour = int(np.argmax(attention))
        explanation_hour = len(relative_hours) - 1
        weighted = firing[explanation_hour] * np.abs(weights)
        top_indices = np.argsort(-weighted)[:3]

        for rank, rule_index in enumerate(top_indices, start=1):
            config_rule = model.static_fnn.rule_configs[int(rule_index)]
            result_rows.append({
                "case_type": case.case_type,
                "subject_id": int(case.subject_id),
                "stay_id": int(case.stay_id),
                "prediction_hour": int(case.sofa_hour),
                "outcome": int(case.y_true),
                "calibrated_probability": float(case.y_prob_calibrated),
                "rank": rank,
                "rule_index": int(rule_index),
                "rule": rule_text(config_rule),
                "explanation_basis": "current_prediction_hour",
                "explanation_relative_hour": int(relative_hours[explanation_hour]),
                "attention_selected_relative_hour": int(relative_hours[selected_hour]),
                "attention_weight": float(attention[selected_hour]),
                "raw_firing": float(firing[explanation_hour, rule_index]),
                "trained_rule_weight": float(weights[rule_index]),
                "absolute_weighted_firing": float(weighted[rule_index]),
                "active_at_0.10": bool(firing[explanation_hour, rule_index] >= 0.10),
            })
            axis.plot(
                relative_hours, firing[:, rule_index], color=colors[rank - 1], linewidth=2,
                label=f"R{rank}: {short_rule_name(config_rule)}",
            )
        axis.axhline(0.10, color="#666666", linestyle="--", linewidth=1, label="Activation 0.10")
        axis.axvline(relative_hours[selected_hour], color="#CC79A7", linestyle=":", linewidth=1.5)
        axis.set_title(
            f"{case.case_type}: stay {int(case.stay_id)}, calibrated risk {case.y_prob_calibrated:.3f}"
        )
        axis.set_ylabel("Raw firing")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.18)
        axis.legend(loc="upper left", fontsize=7.5, frameon=False, ncol=2)

    axes[-1].set_xlabel("Hours relative to prediction")
    figure.suptitle("Patient-specific cross-rule firing over the 24-hour observation window", y=0.995)
    figure.tight_layout()
    figure.savefig(OUTPUT / "figures/patient_specific_rule_firing.pdf", bbox_inches="tight")
    figure.savefig(OUTPUT / "figures/patient_specific_rule_firing.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    result = pd.DataFrame(result_rows)
    result.to_csv(OUTPUT / "patient_specific_activated_rules.csv", index=False)
    (OUTPUT / "analysis_config.json").write_text(
        json.dumps(
            {
                "checkpoint": str(RUN_DIR / "best_model.pt"),
                "case_source": str(CASE_CSV),
                "activation_threshold": 0.10,
                "ranking": "absolute trained rule weight multiplied by raw firing at the current prediction hour",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(result[["case_type", "rank", "rule", "raw_firing", "trained_rule_weight"]].to_string(index=False))


if __name__ == "__main__":
    main()
