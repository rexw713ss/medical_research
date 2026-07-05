"""Locked one-time evaluation for the frozen primary 6-hour final model."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from comparison_protocol import cohort_record
from project_config import PRIMARY_OUTCOME_COLUMN


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run locked final 6-hour test evaluation once.")
    parser.add_argument(
        "--run-dir",
        default="outputs/explicit_temporal_fnn_formal_6h/seed_42",
    )
    parser.add_argument("--output-dir", default="outputs/final_test_evaluation_6h")
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted matching lock.")
    return parser.parse_args()


def validate_prediction_cohort(
    prediction_path: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    frame = pd.read_csv(
        prediction_path,
        usecols=["stay_id", "sofa_hour", "y_true"],
    )
    actual = cohort_record(
        frame["stay_id"],
        frame["sofa_hour"],
        frame["y_true"],
        expected["split"],
        PRIMARY_OUTCOME_COLUMN,
    )
    for field in ("windows", "positive", "negative", "cohort_sha256"):
        if actual[field] != expected[field]:
            raise ValueError(
                f"Prediction cohort mismatch for {expected['split']} / {field}: "
                f"actual={actual[field]} expected={expected[field]}"
            )
    return actual


def markdown_report(summary: pd.Series, fixed: pd.DataFrame, manifest: dict[str, Any]) -> str:
    lines = [
        "# Final Frozen-Model Test Evaluation",
        "",
        f"- Primary outcome: `{PRIMARY_OUTCOME_COLUMN}`.",
        f"- Frozen checkpoint SHA-256: `{manifest['checkpoint_sha256']}`.",
        f"- Checkpoint epoch: {manifest['checkpoint_epoch']}.",
        f"- Test windows: {int(summary['windows']):,}; patients: {int(summary['patients']):,}.",
        f"- Bootstrap: {manifest['bootstrap_reps']} replicates clustered by `subject_id`.",
        "- Calibration and operating thresholds were fitted on validation only.",
        "",
        "## Performance",
        "",
        "| Metric | Estimate | Patient-clustered 95% CI |",
        "|---|---:|---:|",
    ]
    for label, key in (
        ("AUROC", "auroc"),
        ("AUPRC", "auprc"),
        ("Brier score", "brier"),
        ("ECE", "ece"),
    ):
        lines.append(
            f"| {label} | {summary[key]:.4f} | "
            f"{summary[f'{key}_ci95_low']:.4f}-{summary[f'{key}_ci95_high']:.4f} |"
        )
    lines.extend(
        [
            f"| Calibration intercept | {summary['calibration_intercept']:.3f} | - |",
            f"| Calibration slope | {summary['calibration_slope']:.3f} | - |",
            "",
            "## Fixed-Specificity Operating Points",
            "",
            "| Target specificity | Threshold | Observed specificity | Sensitivity | PPV | NPV | F1 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for specificity in sorted(fixed["target_specificity"].unique()):
        group = fixed[fixed["target_specificity"] == specificity].set_index("metric")["value"]
        tag = int(round(specificity * 100))

        def estimate_ci(metric: str, point_key: str | None = None) -> str:
            point_key = point_key or f"{metric}_at_spec_{tag}"
            return (
                f"{summary[point_key]:.4f} "
                f"({summary[f'{metric}_at_spec_{tag}_ci95_low']:.4f}-"
                f"{summary[f'{metric}_at_spec_{tag}_ci95_high']:.4f})"
            )

        lines.append(
            f"| {specificity:.0%} | {group['threshold']:.4f} | "
            f"{estimate_ci('specificity', f'observed_specificity_at_spec_{tag}')} | "
            f"{estimate_ci('sensitivity')} | {estimate_ci('ppv')} | "
            f"{estimate_ci('npv')} | {estimate_ci('f1')} |"
        )
    lines.extend(
        [
            "",
            "F1 at probability 0.5 is retained in `advanced_metrics.csv` for supplement use.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    lock_path = output_dir / "FINAL_TEST_LOCK.json"
    checkpoint_path = run_dir / "best_model.pt"
    config_path = run_dir / "train_config.json"
    summary_path = run_dir / "training_summary.json"
    cohort_path = run_dir / "cohort_audit.json"
    for path in (checkpoint_path, config_path, summary_path, cohort_path):
        if not path.exists():
            raise FileNotFoundError(path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    training = json.loads(summary_path.read_text(encoding="utf-8"))
    cohorts = json.loads(cohort_path.read_text(encoding="utf-8"))
    if config.get("target_col") != PRIMARY_OUTCOME_COLUMN:
        raise ValueError("Final evaluation is locked to the primary 6-hour outcome.")
    if config.get("comparison_mode") != "full":
        raise ValueError("Expected the frozen full-cohort final model.")

    checkpoint_hash = sha256_file(checkpoint_path)
    lock = {
        "status": "running",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_outcome": PRIMARY_OUTCOME_COLUMN,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": int(training["best_epoch"]),
        "bootstrap_reps": int(args.bootstrap_reps),
        "bootstrap_unit": "subject_id",
        "specificities": [0.90, 0.95],
        "calibration": "validation-only Platt scaling applied unchanged to test",
        "test_access_note": (
            "train_fnn.py produced the initial test_metrics.csv only after freezing this checkpoint; "
            "this locked run adds prespecified calibration, operating points and clustered uncertainty "
            "without changing or selecting the model."
        ),
    }
    if lock_path.exists():
        previous = json.loads(lock_path.read_text(encoding="utf-8"))
        if previous.get("status") == "complete":
            raise RuntimeError("Final test evaluation is already complete and locked.")
        if not args.resume or previous.get("checkpoint_sha256") != checkpoint_hash:
            raise RuntimeError("An incomplete final-test lock already exists; use --resume only if unchanged.")
        lock = previous
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")

    prediction_dir = output_dir / "predictions"
    raw_eval_dir = output_dir / "raw_model_evaluation"
    advanced_dir = output_dir / "advanced"
    test_prediction = prediction_dir / "test_predictions.csv.gz"
    val_prediction = prediction_dir / "val_predictions.csv.gz"
    if not (test_prediction.exists() and val_prediction.exists()):
        run(
            [
                sys.executable,
                "model_evaluation_report.py",
                "--sources",
                "fnn",
                "--fnn-run-dirs",
                str(run_dir),
                "--comparison-mode",
                "full",
                "--horizons",
                "6",
                "--save-predictions",
                "--prediction-output-dir",
                str(prediction_dir),
                "--output-dir",
                str(raw_eval_dir),
                "--device",
                args.device,
            ]
        )

    expected = {record["split"]: record for record in cohorts}
    cohort_audit = {
        "validation": validate_prediction_cohort(val_prediction, expected["validation"]),
        "test": validate_prediction_cohort(test_prediction, expected["test"]),
    }
    (output_dir / "prediction_cohort_audit.json").write_text(
        json.dumps(cohort_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not (advanced_dir / "advanced_metrics.csv").exists():
        run(
            [
                sys.executable,
                "advanced_model_evaluation.py",
                "--predictions-root",
                str(prediction_dir),
                "--target-col",
                PRIMARY_OUTCOME_COLUMN,
                "--horizon",
                "6",
                "--bootstrap-reps",
                str(args.bootstrap_reps),
                "--specificities",
                "0.90,0.95",
                "--output-dir",
                str(advanced_dir),
            ]
        )

    metrics = pd.read_csv(advanced_dir / "advanced_metrics.csv").iloc[0]
    fixed = pd.read_csv(advanced_dir / "fixed_specificity_metrics.csv")
    completed_lock = {
        **lock,
        **{
            "status": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "validation_predictions_sha256": sha256_file(val_prediction),
            "test_predictions_sha256": sha256_file(test_prediction),
            "advanced_metrics_sha256": sha256_file(advanced_dir / "advanced_metrics.csv"),
        },
    }
    (output_dir / "final_test_report.md").write_text(
        markdown_report(metrics, fixed, completed_lock),
        encoding="utf-8",
    )
    lock_path.write_text(
        json.dumps(completed_lock, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Final test evaluation complete and locked: {output_dir}")


if __name__ == "__main__":
    main()
