"""Audit that canonical 6-hour experiments use their declared formal cohorts.

The primary full-cohort model uses every eligible train, validation, and test
window. Equal-sample experiments are a prespecified fairness analysis: only
train/validation are sampled, while the independent test cohort remains full.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MIMIC_FULL = {
    "train_windows": 3_843_400,
    "validation_windows": 819_573,
    "test_windows": 830_839,
    "test_patients": 7_287,
    "test_stays": 9_894,
    "test_positive": 47_292,
}
EQUAL_SAMPLE = {
    "train_windows": 200_000,
    "validation_windows": 50_000,
    "test_windows": MIMIC_FULL["test_windows"],
    "test_positive": MIMIC_FULL["test_positive"],
}
EICU_FULL = {
    "windows": 6_215_890,
    "patients": 80_239,
    "stays": 99_262,
    "positive": 294_949,
}

LIMIT_KEYS = {
    "max_rows",
    "max_stays",
    "max_train_windows",
    "max_val_windows",
    "max_test_windows",
    "max_external_windows",
    "limit_train_batches",
    "limit_val_batches",
    "limit_test_batches",
    "rule_quality_batches",
    "sample_windows",
}

FORMAL_CONFIGS = [
    "outputs/explicit_temporal_fnn_tuning_6h/tuning_config.json",
    "outputs/explicit_temporal_fnn_formal_6h/seed_42/train_config.json",
    "outputs/final_test_evaluation_6h/raw_model_evaluation/evaluation_config.json",
    "outputs/feature_matched_baselines_6h_equal_sample/experiment_config.json",
    "outputs/fnn_ablation_6h_equal_sample/ablation_config.json",
    "outputs/missingness_ablation_6h_equal_sample/ablation_config.json",
    "outputs/explicit_temporal_observation_sensitivity_6h/experiment_config.json",
    "outputs/temporal_rule_extraction_6h/rule_extraction_config.json",
    "outputs/eicu_external_validation/eicu_preprocessing_config.json",
    "outputs/eicu_external_validation/final_frozen_model_evaluation/external_validation_config.json",
    "outputs/eicu_frozen_baseline_validation_6h/analysis_config.json",
    "outputs/posthoc_explainability_comparison_6h/analysis_config.json",
    "outputs/clinical_consistency_regularization_6h/analysis_config.json",
]


@dataclass
class Check:
    name: str
    passed: bool
    observed: Any
    expected: Any
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": self.observed,
            "expected": self.expected,
            "note": self.note,
        }


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def check_equal(checks: list[Check], name: str, observed: Any, expected: Any, note: str = "") -> None:
    checks.append(Check(name, observed == expected, observed, expected, note))


def cohort_rows(path: Path) -> dict[str, dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, dict):
        return data
    return {row["split"]: row for row in data}


def collect_settings(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            dotted = f"{prefix}.{key}" if prefix else key
            if key in LIMIT_KEYS or key in {"include_smoke", "allow_incomplete_cohort"}:
                found.append((dotted, child))
            found.extend(collect_settings(child, dotted))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(collect_settings(child, f"{prefix}[{index}]"))
    return found


def audit_limits(root: Path, checks: list[Check]) -> None:
    for raw_path in FORMAL_CONFIGS:
        path = root / raw_path
        if not path.exists():
            checks.append(Check(f"config exists: {raw_path}", False, "missing", "present"))
            continue
        invalid: list[dict[str, Any]] = []
        for key, value in collect_settings(load_json(path)):
            leaf = key.rsplit(".", 1)[-1]
            if leaf in LIMIT_KEYS and value not in (None, 0):
                invalid.append({"key": key, "value": value})
            elif leaf == "include_smoke" and value is not False:
                invalid.append({"key": key, "value": value})
            elif leaf == "allow_incomplete_cohort" and value is not False:
                invalid.append({"key": key, "value": value})
        checks.append(
            Check(
                f"no runtime truncation: {raw_path}",
                not invalid,
                invalid,
                [],
                "Zero and null mean unlimited; equal-sample membership is checked separately.",
            )
        )


def audit_primary_full_cohort(root: Path, checks: list[Check]) -> None:
    rows = cohort_rows(root / "outputs/explicit_temporal_fnn_formal_6h/seed_42/cohort_audit.json")
    check_equal(checks, "primary FNN full train windows", rows["train"]["windows"], MIMIC_FULL["train_windows"])
    check_equal(
        checks,
        "primary FNN full validation windows",
        rows["validation"]["windows"],
        MIMIC_FULL["validation_windows"],
    )
    check_equal(checks, "primary FNN full test windows", rows["test"]["windows"], MIMIC_FULL["test_windows"])
    check_equal(checks, "primary FNN test positives", rows["test"]["positive"], MIMIC_FULL["test_positive"])

    frozen = cohort_rows(root / "outputs/final_test_evaluation_6h/prediction_cohort_audit.json")
    check_equal(checks, "frozen final evaluation test windows", frozen["test"]["windows"], MIMIC_FULL["test_windows"])
    check_equal(checks, "frozen final evaluation test positives", frozen["test"]["positive"], MIMIC_FULL["test_positive"])


def audit_equal_sample_protocol(root: Path, checks: list[Check]) -> None:
    paired = load_json(root / "outputs/explicit_kg_tfnn_paired_comparison_6h/input_audit.json")
    check_equal(checks, "paired comparison identical test keys/outcomes", paired["test_windows_outcomes_identical"], True)
    for model, splits in paired["models"].items():
        check_equal(checks, f"paired {model} validation windows", splits["validation"]["rows"], EQUAL_SAMPLE["validation_windows"])
        check_equal(checks, f"paired {model} full test windows", splits["test"]["rows"], EQUAL_SAMPLE["test_windows"])
        check_equal(checks, f"paired {model} test patients", splits["test"]["patients"], MIMIC_FULL["test_patients"])
        check_equal(checks, f"paired {model} test positives", splits["test"]["positive"], EQUAL_SAMPLE["test_positive"])

    cohort_audits = [
        "outputs/explicit_temporal_fnn_tuning_6h/cohort_audit.json",
        "outputs/feature_matched_baselines_6h_equal_sample/cohort_audit.json",
        "outputs/feature_matched_baselines_6h_equal_sample/sequence_cohort_audit.json",
    ]
    for raw_path in cohort_audits:
        rows = cohort_rows(root / raw_path)
        check_equal(checks, f"{raw_path} train windows", rows["train"]["windows"], EQUAL_SAMPLE["train_windows"])
        check_equal(checks, f"{raw_path} validation windows", rows["validation"]["windows"], EQUAL_SAMPLE["validation_windows"])
        if "test" in rows:
            check_equal(checks, f"{raw_path} full test windows", rows["test"]["windows"], EQUAL_SAMPLE["test_windows"])

    variant_audits = [
        ("outputs/fnn_ablation_6h_equal_sample/seed_*/**/cohort_audit.json", 12),
        # The full comparator reuses the three matching full-model checkpoints
        # from fnn_ablation_6h_equal_sample, so this directory contains only
        # the two newly fitted variants per seed.
        ("outputs/missingness_ablation_6h_equal_sample/seed_*/**/cohort_audit.json", 6),
        ("outputs/explicit_temporal_observation_sensitivity_6h/seed_*/**/cohort_audit.json", 15),
    ]
    for pattern, expected_files in variant_audits:
        paths = sorted(root.glob(pattern))
        check_equal(checks, f"variant cohort-audit file count: {pattern}", len(paths), expected_files)
        invalid: list[str] = []
        for path in paths:
            rows = cohort_rows(path)
            observed = (
                rows.get("train", {}).get("windows"),
                rows.get("validation", {}).get("windows"),
                rows.get("test", {}).get("windows"),
                rows.get("test", {}).get("positive"),
            )
            expected = (
                EQUAL_SAMPLE["train_windows"],
                EQUAL_SAMPLE["validation_windows"],
                EQUAL_SAMPLE["test_windows"],
                EQUAL_SAMPLE["test_positive"],
            )
            if observed != expected:
                invalid.append(str(path.relative_to(root)))
        checks.append(Check(f"all variant cohorts match protocol: {pattern}", not invalid, invalid, []))

    missingness = load_json(root / "outputs/missingness_ablation_6h_equal_sample/evaluation/comparison_audit.json")
    check_equal(checks, "missingness comparison audit status", missingness["status"], "passed")
    check_equal(checks, "missingness comparison full test windows", missingness["test_windows"], EQUAL_SAMPLE["test_windows"])
    check_equal(checks, "missingness comparison identical windows/outcomes", missingness["identical_windows_and_outcomes"], True)

    observation_path = root / "outputs/explicit_temporal_observation_sensitivity_6h/observation_window_runs.csv"
    with observation_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    invalid_rows = [
        {"seed": row["seed"], "variant": row["variant"]}
        for row in rows
        if int(float(row["train_windows"])) != EQUAL_SAMPLE["train_windows"]
        or int(float(row["val_windows"])) != EQUAL_SAMPLE["validation_windows"]
        or int(float(row["test_windows"])) != EQUAL_SAMPLE["test_windows"]
        or int(float(row["test_positive"])) != EQUAL_SAMPLE["test_positive"]
    ]
    check_equal(checks, "observation-window run count", len(rows), 15)
    checks.append(Check("all observation-window runs use full test cohort", not invalid_rows, invalid_rows, []))


def audit_full_window_analyses(root: Path, checks: list[Check]) -> None:
    posthoc_config = load_json(root / "outputs/posthoc_explainability_comparison_6h/analysis_config.json")
    posthoc_audit = load_json(root / "outputs/posthoc_explainability_comparison_6h/formal_cohort_audit.json")
    check_equal(checks, "post-hoc XAI formal_full_data", posthoc_config["formal_full_data"], True)
    check_equal(checks, "post-hoc MIMIC windows", posthoc_audit["MIMIC-IV"]["processed_windows"], MIMIC_FULL["test_windows"])
    check_equal(checks, "post-hoc MIMIC patients", posthoc_audit["MIMIC-IV"]["processed_patients"], MIMIC_FULL["test_patients"])
    check_equal(checks, "post-hoc MIMIC positives", posthoc_audit["MIMIC-IV"]["processed_positive"], MIMIC_FULL["test_positive"])
    check_equal(checks, "post-hoc eICU windows", posthoc_audit["eICU-CRD"]["processed_windows"], EICU_FULL["windows"])
    check_equal(checks, "post-hoc eICU patients", posthoc_audit["eICU-CRD"]["processed_patients"], EICU_FULL["patients"])
    check_equal(checks, "post-hoc eICU positives", posthoc_audit["eICU-CRD"]["processed_positive"], EICU_FULL["positive"])
    check_equal(
        checks,
        "post-hoc all prediction windows reconstructed",
        all(row["all_prediction_windows_reconstructed"] for row in posthoc_audit.values()),
        True,
    )
    with (root / "outputs/posthoc_explainability_comparison_6h/explanation_quality_comparison.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        explanation_rows = list(csv.DictReader(handle))
    check_equal(checks, "post-hoc model rows", len(explanation_rows), 4)
    invalid_explanations = [
        row["model"]
        for row in explanation_rows
        if int(float(row["mimic_windows"])) != MIMIC_FULL["test_windows"]
        or int(float(row["eicu_windows"])) != EICU_FULL["windows"]
        or int(float(row["stability_pairs"])) != 3 * MIMIC_FULL["test_windows"]
    ]
    checks.append(Check("all post-hoc models use full cohorts", not invalid_explanations, invalid_explanations, []))

    complexity = load_json(
        root
        / "outputs/posthoc_explainability_comparison_6h/unified_explanation_complexity_audit.json"
    )
    check_equal(checks, "unified explanation complexity audit status", complexity["status"], "passed")
    check_equal(checks, "unified complexity formal_full_data", complexity["formal_full_data"], True)
    check_equal(
        checks,
        "unified complexity MIMIC windows",
        complexity["mimic_windows"],
        MIMIC_FULL["test_windows"],
    )
    check_equal(
        checks,
        "unified complexity eICU windows",
        complexity["eicu_windows"],
        EICU_FULL["windows"],
    )
    check_equal(
        checks,
        "unified complexity clinical-variable denominator",
        complexity["common_explanation_unit"],
        "13 harmonized clinical variables",
    )

    stale_sample_caches = [
        root / "outputs/posthoc_explainability_comparison_6h/mimic_explanation_sample.csv",
        root / "outputs/posthoc_explainability_comparison_6h/eicu_explanation_sample.csv",
        root / "outputs/posthoc_explainability_comparison_6h/mimic-iv_selected_hourly_rows.pkl",
        root / "outputs/posthoc_explainability_comparison_6h/eicu_selected_hourly_rows.pkl",
    ]
    existing_stale = [str(path.relative_to(root)) for path in stale_sample_caches if path.exists()]
    checks.append(Check("legacy sampled XAI caches removed", not existing_stale, existing_stale, []))

    consistency_config = load_json(root / "outputs/clinical_consistency_regularization_6h/analysis_config.json")
    consistency_audit = load_json(root / "outputs/clinical_consistency_regularization_6h/formal_cohort_audit.json")
    check_equal(checks, "consistency audit formal_full_data", consistency_config["formal_full_data"], True)
    check_equal(
        checks,
        "consistency windows per seed/variant",
        consistency_config["processed_windows_per_model"],
        MIMIC_FULL["test_windows"],
    )
    check_equal(checks, "consistency model passes", consistency_config["models_evaluated"], 6)
    check_equal(checks, "consistency reconstructed windows", consistency_audit["processed_windows"], MIMIC_FULL["test_windows"])
    with (root / "outputs/clinical_consistency_regularization_6h/consistency_metrics_by_seed.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    check_equal(checks, "consistency seed/variant rows", len(rows), 6)
    invalid = [row for row in rows if int(float(row["processed_windows"])) != MIMIC_FULL["test_windows"]]
    checks.append(Check("all consistency seed/variant passes use full test", not invalid, len(invalid), 0))

    raw_rule = load_json(root / "outputs/raw_rule_firing_6h/analysis_config.json")
    check_equal(checks, "raw-rule firing full test windows", raw_rule["windows"], MIMIC_FULL["test_windows"])


def audit_external(root: Path, checks: list[Check]) -> None:
    metrics = load_json(root / "outputs/eicu_external_validation/final_frozen_model_evaluation/external_metrics.json")
    stats = load_json(root / "outputs/eicu_external_validation/final_frozen_model_evaluation/external_inference_stats.json")
    check_equal(checks, "eICU external windows", metrics["windows"], EICU_FULL["windows"])
    check_equal(checks, "eICU external patients", metrics["patients"], EICU_FULL["patients"])
    check_equal(checks, "eICU external stays", metrics["stays"], EICU_FULL["stays"])
    check_equal(checks, "eICU external inference windows", stats["prediction_windows"], EICU_FULL["windows"])
    check_equal(checks, "eICU no fitting", metrics["no_eicu_fitting"], True)

    baseline_root = root / "outputs/eicu_frozen_baseline_validation_6h"
    baseline_audit = load_json(baseline_root / "formal_cohort_and_freeze_audit.json")
    external = baseline_audit["external_cohort"]
    check_equal(checks, "frozen baseline external audit status", baseline_audit["status"], "passed")
    check_equal(checks, "frozen baseline eICU windows", external["processed_windows"], EICU_FULL["windows"])
    check_equal(checks, "frozen baseline eICU patients", external["processed_patients"], EICU_FULL["patients"])
    check_equal(checks, "frozen baseline eICU stays", external["processed_stays"], EICU_FULL["stays"])
    check_equal(checks, "frozen baseline eICU positives", external["processed_positive"], EICU_FULL["positive"])
    check_equal(
        checks,
        "frozen baseline all eICU windows reconstructed",
        external["all_prediction_windows_reconstructed"],
        True,
    )
    check_equal(checks, "frozen baseline external subsampling disabled", baseline_audit["external_subsampling"], False)
    check_equal(checks, "frozen baseline no eICU model fitting", baseline_audit["eicu_model_refitting"], False)
    check_equal(checks, "frozen baseline no eICU calibration fitting", baseline_audit["eicu_calibration_fitting"], False)
    check_equal(checks, "frozen baseline no eICU threshold selection", baseline_audit["eicu_threshold_selection"], False)
    check_equal(checks, "frozen baseline bootstrap unit", baseline_audit["bootstrap_unit"], "subject_id")
    check_equal(checks, "frozen baseline bootstrap replicates", baseline_audit["bootstrap_reps"], 200)
    source_checks = baseline_audit["source_prediction_reproduction"]
    check_equal(checks, "frozen baseline source model checks", len(source_checks), 5)
    check_equal(
        checks,
        "all frozen baselines reproduce source predictions",
        all(record["passed"] for record in source_checks.values()),
        True,
    )
    with (baseline_root / "external_baseline_metrics.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        baseline_rows = list(csv.DictReader(handle))
    check_equal(checks, "frozen external comparison model rows", len(baseline_rows), 5)
    invalid_rows = [
        row["model"]
        for row in baseline_rows
        if int(float(row["windows"])) != EICU_FULL["windows"]
        or int(float(row["patients"])) != EICU_FULL["patients"]
        or int(float(row["stays"])) != EICU_FULL["stays"]
        or str(row["no_eicu_fitting"]).lower() != "true"
    ]
    checks.append(
        Check(
            "all frozen external comparator rows use full cohort",
            not invalid_rows,
            invalid_rows,
            [],
        )
    )


def audit_output_registry(root: Path, checks: list[Check]) -> None:
    smoke_dirs = [
        str(path.relative_to(root))
        for path in (root / "outputs").rglob("*")
        if path.is_dir() and "smoke" in path.name.lower()
    ]
    checks.append(Check("no smoke-test output directories", not smoke_dirs, smoke_dirs, []))


def write_report(output_dir: Path, checks: list[Check]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    failed = [check for check in checks if not check.passed]
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if not failed else "failed",
        "primary_outcome": "future 6-hour SOFA increase >=2",
        "scope_policy": {
            "primary_model": "all eligible train, validation, and test windows",
            "equal_sample_sensitivity": "prespecified 200,000 train and 50,000 validation windows; all 830,839 test windows",
            "external_validation": "all 6,215,890 eligible eICU prediction windows; no refitting or recalibration",
            "smoke_results_allowed_in_formal_outputs": False,
        },
        "expected_counts": {
            "mimic_full": MIMIC_FULL,
            "equal_sample": EQUAL_SAMPLE,
            "eicu_full": EICU_FULL,
        },
        "checks_passed": len(checks) - len(failed),
        "checks_failed": len(failed),
        "checks": [check.as_dict() for check in checks],
    }
    (output_dir / "formal_data_scope_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Formal Data-Scope Audit",
        "",
        f"- Status: **{payload['status'].upper()}**",
        f"- Checks passed: {payload['checks_passed']}",
        f"- Checks failed: {payload['checks_failed']}",
        "- Primary outcome: future 6-hour SOFA increase >=2",
        "",
        "## Locked Data Scopes",
        "",
        "| Analysis scope | Train | Validation | Test / external | Interpretation |",
        "|---|---:|---:|---:|---|",
        "| Primary full-cohort KG-TFNN | 3,843,400 | 819,573 | 830,839 | Every eligible MIMIC-IV window |",
        "| Equal-sample sensitivity | 200,000 | 50,000 | 830,839 | Prespecified fair train/validation subset; complete test |",
        "| Frozen eICU transport | NA | NA | 6,215,890 | Every eligible external window; no refitting/recalibration |",
        "| Frozen eICU equal-sample comparator transport | 200,000 | 50,000 | 6,215,890 | Five models; complete external cohort and source-only calibration |",
        "| Full post-hoc XAI | NA | NA | 830,839 MIMIC + 6,215,890 eICU | Every prediction-key window |",
        "| Consistency behavior | NA | NA | 830,839 per model | 3 seeds x 2 variants = 4,985,034 model-window evaluations |",
        "",
        "## Failed Checks",
        "",
    ]
    if failed:
        lines.extend(f"- `{check.name}`: observed `{check.observed}`, expected `{check.expected}`" for check in failed)
    else:
        lines.append("None. No smoke-test or runtime-truncated result is registered as canonical evidence.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "`equal_sample` is not a full-cohort training estimate and must remain labelled as a prespecified fairness sensitivity analysis. "
            "Its independent test evaluation is complete. Primary full-cohort and external estimates must be reported separately.",
            "",
        ]
    )
    (output_dir / "formal_data_scope_audit.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/formal_data_scope_audit_6h"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    checks: list[Check] = []
    audit_limits(root, checks)
    audit_primary_full_cohort(root, checks)
    audit_equal_sample_protocol(root, checks)
    audit_full_window_analyses(root, checks)
    audit_external(root, checks)
    audit_output_registry(root, checks)
    write_report(root / args.output_dir, checks)
    failed = [check for check in checks if not check.passed]
    print(f"Formal data-scope audit: {len(checks) - len(failed)} passed, {len(failed)} failed")
    if failed:
        for check in failed:
            print(f"FAILED: {check.name}: observed={check.observed!r}, expected={check.expected!r}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
