"""Standardize frozen prediction files for the primary equal-sample comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


OUTPUT = Path("outputs/explicit_kg_tfnn_paired_comparison_6h/inputs")
TARGET = "label_sofa_increase_ge2_6h"
SOURCES = {
    "explicit_kg_tfnn": {
        "validation": Path("outputs/fnn_ablation_6h_equal_sample/seed_42/full/validation_predictions.csv.gz"),
        "test": Path("outputs/fnn_ablation_6h_equal_sample/seed_42/full/test_predictions.csv.gz"),
        "raw_column": "y_prob_raw",
    },
    "logistic_regression": {
        "validation": Path("outputs/fair_comparison_6h_equal_sample/interpretable_6h/protocol/logistic_regression/val_predictions.csv.gz"),
        "test": Path("outputs/fair_comparison_6h_equal_sample/interpretable_6h/protocol/logistic_regression/test_predictions.csv.gz"),
        "raw_column": "y_prob",
    },
    "ebm": {
        "validation": Path("outputs/fair_comparison_6h_equal_sample/interpretable_6h/protocol/ebm/val_predictions.csv.gz"),
        "test": Path("outputs/fair_comparison_6h_equal_sample/interpretable_6h/protocol/ebm/test_predictions.csv.gz"),
        "raw_column": "y_prob",
    },
    "xgboost": {
        "validation": Path("outputs/fair_comparison_6h_equal_sample/blackbox_6h/protocol/xgboost/val_predictions.csv.gz"),
        "test": Path("outputs/fair_comparison_6h_equal_sample/blackbox_6h/protocol/xgboost/test_predictions.csv.gz"),
        "raw_column": "y_prob",
    },
    "gru": {
        "validation": Path("outputs/fair_comparison_6h_equal_sample/blackbox_6h/sequence_raw/gru/val_predictions.csv.gz"),
        "test": Path("outputs/fair_comparison_6h_equal_sample/blackbox_6h/sequence_raw/gru/test_predictions.csv.gz"),
        "raw_column": "y_prob",
    },
    "xgboost_matched": {
        "validation": Path("outputs/feature_matched_baselines_6h_equal_sample/sequence_matched_summary/xgboost/val_predictions.csv.gz"),
        "test": Path("outputs/feature_matched_baselines_6h_equal_sample/sequence_matched_summary/xgboost/test_predictions.csv.gz"),
        "raw_column": "y_prob",
        "feature_set": "matched_24h_summary",
    },
    "lightgbm_matched": {
        "validation": Path("outputs/feature_matched_baselines_6h_equal_sample/sequence_matched_summary/lightgbm/val_predictions.csv.gz"),
        "test": Path("outputs/feature_matched_baselines_6h_equal_sample/sequence_matched_summary/lightgbm/test_predictions.csv.gz"),
        "raw_column": "y_prob",
        "feature_set": "matched_24h_summary",
    },
    "gru_matched": {
        "validation": Path("outputs/feature_matched_baselines_6h_equal_sample/sequence_matched/gru/val_predictions.csv.gz"),
        "test": Path("outputs/feature_matched_baselines_6h_equal_sample/sequence_matched/gru/test_predictions.csv.gz"),
        "raw_column": "y_prob",
        "feature_set": "matched_24h_sequence",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    audit = {"target_col": TARGET, "models": {}}
    reference_test = None
    for model, config in SOURCES.items():
        model_dir = OUTPUT / model
        model_dir.mkdir(parents=True, exist_ok=True)
        audit["models"][model] = {}
        for split in ("validation", "test"):
            source = config[split]
            frame = pd.read_csv(source)
            probability = config["raw_column"]
            required = {"subject_id", "stay_id", "sofa_hour", "y_true", probability}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"{source} missing {sorted(missing)}")
            output = frame[["subject_id", "stay_id", "sofa_hour", "y_true", probability]].copy()
            output = output.rename(columns={probability: "y_prob"})
            for column in ["subject_id", "stay_id", "sofa_hour"]:
                output[column] = pd.to_numeric(output[column], errors="raise").astype("int64")
            output["y_true"] = pd.to_numeric(output["y_true"], errors="raise").astype("int8")
            output["y_prob"] = pd.to_numeric(output["y_prob"], errors="raise").astype("float64")
            output["model"] = model
            output["feature_set"] = config.get(
                "feature_set",
                "explicit_temporal" if model == "explicit_kg_tfnn" else "comparator",
            )
            output["target_col"] = TARGET
            output["evaluation_split"] = "validation" if split == "validation" else "test"
            output = output.sort_values(["subject_id", "stay_id", "sofa_hour"]).reset_index(drop=True)
            destination = model_dir / ("val_predictions.csv.gz" if split == "validation" else "test_predictions.csv.gz")
            output.to_csv(destination, index=False, compression="gzip")
            keys = output[["subject_id", "stay_id", "sofa_hour", "y_true"]]
            key_hash = hashlib.sha256(pd.util.hash_pandas_object(keys, index=False).values.tobytes()).hexdigest()
            audit["models"][model][split] = {
                "source": str(source),
                "source_sha256": sha256(source),
                "rows": len(output),
                "patients": int(output["subject_id"].nunique()),
                "positive": int(output["y_true"].sum()),
                "key_outcome_sha256": key_hash,
                "probability_source": probability,
            }
            if split == "test":
                if reference_test is None:
                    reference_test = key_hash
                elif key_hash != reference_test:
                    raise ValueError(f"Test windows/outcomes do not match for {model}")
    audit["test_windows_outcomes_identical"] = True
    (OUTPUT.parent / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote standardized inputs to {OUTPUT}")


if __name__ == "__main__":
    main()
