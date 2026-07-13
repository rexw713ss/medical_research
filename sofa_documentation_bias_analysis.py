"""Test whether changing SOFA component availability drives the 6-hour outcome."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clinical_sensitivity_analyses import (
    KEYS,
    METRICS,
    PREDICTIONS,
    SOFA,
    evaluate_frame,
    load_predictions,
)


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs/sofa_documentation_bias_6h"
COMPONENTS = [
    "respiration_score", "coagulation_score", "liver_score",
    "cardiovascular_score", "cns_score", "renal_score",
]
DISPLAY = {
    "respiration_score": "Respiratory", "coagulation_score": "Coagulation",
    "liver_score": "Liver", "cardiovascular_score": "Cardiovascular",
    "cns_score": "Neurological", "renal_score": "Renal",
}


def load_component_sofa(stay_ids: set[int]) -> pd.DataFrame:
    cache = OUTPUT / "test_sofa_components.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    columns = ["subject_id", "stay_id", "sofa_hour", "sofa_score", "sofa_component_count", *COMPONENTS]
    chunks = []
    for chunk in pd.read_csv(SOFA, usecols=columns, chunksize=500_000):
        keep = chunk.stay_id.isin(stay_ids)
        if keep.any():
            chunks.append(chunk.loc[keep].copy())
    if not chunks:
        raise RuntimeError("No component-level SOFA rows found for test stays.")
    frame = pd.concat(chunks, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    frame.to_pickle(cache)
    return frame


def construct_labels_and_contributions(
    predictions: pd.DataFrame,
    sofa: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    prediction_lookup = {
        int(stay): set(group.sofa_hour.astype(int))
        for stay, group in predictions.groupby("stay_id", sort=False)
    }
    label_rows = []
    contribution_rows = []
    positive_total = 0
    positive_new_component = 0
    positive_more_components = 0
    positive_preserved_common = 0

    for stay_id, group in sofa.groupby("stay_id", sort=False):
        wanted_hours = prediction_lookup.get(int(stay_id))
        if not wanted_hours:
            continue
        group = group.sort_values("sofa_hour")
        hours = group.sofa_hour.to_numpy(dtype=np.int64)
        components = group[COMPONENTS].to_numpy(dtype=np.float64)
        masks = np.isfinite(components)
        filled = np.nan_to_num(components, nan=0.0)
        scores = group.sofa_score.to_numpy(dtype=np.float64)
        subjects = group.subject_id.to_numpy()
        hour_to_index = {int(hour): index for index, hour in enumerate(hours)}

        for hour in wanted_hours:
            index = hour_to_index.get(int(hour))
            future_indices = [hour_to_index.get(int(hour + offset)) for offset in range(1, 7)]
            if index is None or any(value is None for value in future_indices):
                continue
            future_indices = np.asarray(future_indices, dtype=np.int64)
            if not np.isfinite(scores[index]) or not np.any(np.isfinite(scores[future_indices])):
                continue
            index_mask = masks[index]
            future_masks = masks[future_indices]
            future_scores = scores[future_indices]
            max_local = int(np.nanargmax(future_scores))
            max_index = int(future_indices[max_local])
            primary_label = int(future_scores[max_local] - scores[index] >= 2)

            common_deltas = []
            for future_index in future_indices:
                common = index_mask & masks[future_index]
                if common.sum() >= 4:
                    common_deltas.append(float(filled[future_index, common].sum() - filled[index, common].sum()))
            common_label = float(max(common_deltas) >= 2) if common_deltas else np.nan
            same_mask = np.array_equal(index_mask, masks[max_index])
            same_mask_label = float(primary_label) if same_mask else np.nan
            stable_mask = bool(np.all(future_masks == index_mask[None, :]))
            stable_mask_label = float(primary_label) if stable_mask else np.nan
            label_rows.append(
                {
                    "subject_id": subjects[index], "stay_id": int(stay_id), "sofa_hour": int(hour),
                    "label_pairwise_common_components": common_label,
                    "label_same_mask_at_primary_max": same_mask_label,
                    "label_stable_mask_all_future_hours": stable_mask_label,
                    "primary_label_reconstructed": primary_label,
                    "index_component_count": int(index_mask.sum()),
                    "future_max_component_count": int(masks[max_index].sum()),
                    "new_component_at_future_max": bool(np.any(masks[max_index] & ~index_mask)),
                }
            )

            if primary_label == 1:
                positive_total += 1
                positive_new_component += int(np.any(masks[max_index] & ~index_mask))
                positive_more_components += int(masks[max_index].sum() > index_mask.sum())
                positive_preserved_common += int(common_label == 1)
                deltas = filled[max_index] - filled[index]
                comparable = index_mask & masks[max_index]
                positive_increase = np.where(comparable, np.maximum(deltas, 0), 0.0)
                for component_index, component in enumerate(COMPONENTS):
                    contribution_rows.append(
                        {
                            "subject_id": subjects[index], "stay_id": int(stay_id), "sofa_hour": int(hour),
                            "component": DISPLAY[component], "comparable": bool(comparable[component_index]),
                            "component_delta": float(deltas[component_index]) if comparable[component_index] else np.nan,
                            "positive_point_increase": float(positive_increase[component_index]),
                            "component_increased": bool(comparable[component_index] and deltas[component_index] > 0),
                        }
                    )

    audit = {
        "primary_positive_windows": positive_total,
        "positive_with_newly_observed_component_at_future_max": positive_new_component,
        "positive_with_new_component_fraction": positive_new_component / max(positive_total, 1),
        "positive_with_more_observed_components_at_future_max": positive_more_components,
        "positive_with_more_components_fraction": positive_more_components / max(positive_total, 1),
        "positive_preserved_under_pairwise_common_component_label": positive_preserved_common,
        "positive_preserved_under_common_component_fraction": positive_preserved_common / max(positive_total, 1),
    }
    return pd.DataFrame(label_rows), pd.DataFrame(contribution_rows), audit


def summarize_contributions(frame: pd.DataFrame, positive_windows: int) -> pd.DataFrame:
    rows = []
    total_points = frame.positive_point_increase.sum()
    for component, group in frame.groupby("component", sort=False):
        comparable = group[group.comparable]
        rows.append(
            {
                "component": component,
                "positive_windows": positive_windows,
                "comparable_windows": len(comparable),
                "comparable_fraction": len(comparable) / max(positive_windows, 1),
                "windows_with_component_increase": int(comparable.component_increased.sum()),
                "fraction_positive_windows_with_increase": comparable.component_increased.sum() / max(positive_windows, 1),
                "positive_sofa_points": group.positive_point_increase.sum(),
                "share_of_positive_component_point_increases": group.positive_point_increase.sum() / max(total_points, 1e-12),
                "median_delta_when_comparable": comparable.component_delta.median(),
            }
        )
    return pd.DataFrame(rows).sort_values("share_of_positive_component_point_increases", ascending=False)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "figures").mkdir(exist_ok=True)
    predictions, thresholds = load_predictions()
    sofa = load_component_sofa(set(predictions.stay_id.astype(int)))
    labels, contribution_detail, audit = construct_labels_and_contributions(predictions, sofa)
    labels.to_csv(OUTPUT / "component_availability_labels.csv.gz", index=False, compression="gzip")

    definitions = {
        "Pairwise common components (>=4)": "label_pairwise_common_components",
        "Same component mask at primary future maximum": "label_same_mask_at_primary_max",
        "Stable component mask across all six future hours": "label_stable_mask_all_future_hours",
    }
    rows = []
    bootstraps = []
    for label, column in definitions.items():
        evaluation = predictions[[*KEYS, "subject_id", "y_prob"]].merge(
            labels[[*KEYS, column]], on=KEYS, how="left"
        ).rename(columns={column: "y_true"})
        row, bootstrap = evaluate_frame(evaluation, label, thresholds)
        rows.append(row)
        bootstraps.append(bootstrap)
        print(f"{label}: {row['windows']:,} windows, prevalence {row['prevalence']:.4f}")
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(OUTPUT / "component_availability_sensitivity.csv", index=False)
    pd.concat(bootstraps, ignore_index=True).to_csv(
        OUTPUT / "component_availability_patient_bootstrap.csv.gz", index=False, compression="gzip"
    )

    contributions = summarize_contributions(contribution_detail, audit["primary_positive_windows"])
    contributions.to_csv(OUTPUT / "organ_component_contributions.csv", index=False)
    contribution_detail.to_csv(OUTPUT / "organ_component_contribution_details.csv.gz", index=False, compression="gzip")
    (OUTPUT / "documentation_bias_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    existing = pd.read_csv(ROOT / "outputs/clinical_sensitivity_analyses_6h/sofa_outcome_definition_sensitivity.csv")
    combined = pd.concat([existing, sensitivity], ignore_index=True, sort=False)
    combined.to_csv(OUTPUT / "complete_sofa_outcome_sensitivity.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].barh(sensitivity.definition, sensitivity.auroc, color="#0072B2")
    axes[0].set_xlim(0.5, 0.75)
    axes[0].set_xlabel("AUROC")
    axes[0].set_title("Component-availability sensitivity")
    axes[0].grid(axis="x", alpha=0.2)
    axes[1].barh(contributions.component, contributions.share_of_positive_component_point_increases, color="#D55E00")
    axes[1].set_xlabel("Share of positive component-point increases")
    axes[1].set_title("Organ contribution among primary positive windows")
    axes[1].grid(axis="x", alpha=0.2)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(OUTPUT / f"figures/sofa_documentation_bias_sensitivity.{suffix}", dpi=240, bbox_inches="tight")
    plt.close(figure)

    report = [
        "# SOFA Documentation-Bias Sensitivity", "",
        "The pairwise-common definition compares only components observed at both index and each future hour and requires at least four common components. The same-mask definition retains windows whose index mask equals the mask at the primary future SOFA maximum. The stable-mask definition requires the same mask at index and all six future hours.", "",
        sensitivity.to_csv(index=False), "", contributions.to_csv(index=False), "", json.dumps(audit, indent=2),
    ]
    (OUTPUT / "sofa_documentation_bias_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(contributions[["component", "share_of_positive_component_point_increases"]].to_string(index=False))


if __name__ == "__main__":
    main()
