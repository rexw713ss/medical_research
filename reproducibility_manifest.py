"""Create a compact reproducibility manifest for the primary 6-hour study."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path


OUTPUT = Path("outputs/reproducibility_6h")
FILES = [
    "patient_split.csv",
    "comparison_protocol.json",
    "equal_sample_windows.csv.gz",
    "requirements.txt",
    "anfis_model.py",
    "train_fnn.py",
    "ablation_fnn_experiments.py",
    "advanced_model_evaluation.py",
    "clinical_sensitivity_analyses.py",
    "eicu_hospital_sensitivity.py",
    "blackbox_baselines.py",
    "raw_rule_firing_analysis.py",
    "expanded_experiment_reporting.py",
    "posthoc_explainability_comparison.py",
    "clinical_consistency_regularization_analysis.py",
    "sofa_documentation_bias_analysis.py",
    "build_supplementary_material.py",
    "outputs/explicit_temporal_fnn_formal_6h/seed_42/best_model.pt",
    "outputs/explicit_temporal_fnn_tuning_6h/best_params.json",
    "outputs/final_test_evaluation_6h/FINAL_TEST_LOCK.json",
    "outputs/explicit_kg_tfnn_paired_comparison_6h/input_audit.json",
    "outputs/clinical_sensitivity_analyses_6h/analysis_config.json",
    "outputs/eicu_hospital_sensitivity_6h/analysis_config.json",
    "outputs/missingness_ablation_6h_equal_sample/ablation_config.json",
    "outputs/feature_matched_baselines_6h_equal_sample/experiment_config.json",
    "outputs/raw_rule_firing_6h/analysis_config.json",
    "outputs/expanded_experiment_reporting_6h/reporting_config.json",
    "outputs/posthoc_explainability_comparison_6h/analysis_config.json",
    "outputs/posthoc_explainability_comparison_6h/explanation_quality_comparison.csv",
    "outputs/clinical_consistency_regularization_6h/analysis_config.json",
    "outputs/clinical_consistency_regularization_6h/consistency_metrics_by_seed.csv",
    "outputs/sofa_documentation_bias_6h/documentation_bias_audit.json",
    "outputs/sofa_documentation_bias_6h/complete_sofa_outcome_sensitivity.csv",
    "outputs/supplementary_material/table_s1_sofa_reconstruction.csv",
    "outputs/supplementary_material/table_s2_news2_fuzzy_mapping.csv",
]
PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "torch",
    "optuna",
    "xgboost",
    "lightgbm",
    "interpret",
    "pygam",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    files = {}
    missing = []
    for raw_path in FILES:
        path = Path(raw_path)
        if not path.exists():
            missing.append(raw_path)
            continue
        files[raw_path] = {"bytes": path.stat().st_size, "sha256": sha256(path)}

    packages = {}
    for package in PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    torch_environment = {}
    try:
        import torch

        torch_environment = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        torch_environment = {"torch_available": False}

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "study": "Primary 6-hour SOFA increase >=2 KG-TFNN",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch_environment": torch_environment,
        "files": files,
        "missing_expected_files": missing,
        "test_lock_policy": "Test outcomes were excluded from tuning, checkpoint selection, calibration, and threshold selection.",
        "bootstrap_clusters": {
            "internal_and_subgroups": "subject_id",
            "external_primary": "subject_id",
            "external_hospital_sensitivity": "hospital_id",
        },
    }
    (OUTPUT / "analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {OUTPUT / 'analysis_manifest.json'}")


if __name__ == "__main__":
    main()
