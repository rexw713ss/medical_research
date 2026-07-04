"""Shared helpers for merging SOFA deterioration labels into feature tables."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

import clinical_score_baselines as clinical


def horizon_from_target_col(target_col: str) -> int:
    match = re.search(r"label_sofa_increase_ge2_(\d+)h$", target_col)
    if match is None:
        raise ValueError(f"Cannot infer SOFA horizon from target column: {target_col}")
    return int(match.group(1))


def maybe_existing_usecols(csv_path: Path, usecols: list[str]) -> tuple[list[str], list[str]]:
    header = set(clinical.read_header(csv_path))
    existing = [col for col in usecols if col in header]
    missing = [col for col in usecols if col not in header]
    return existing, missing


def merge_sofa_targets(
    df: pd.DataFrame,
    target_cols: list[str],
    split_col: str,
    time_col: str,
    sofa_csv: str | Path | None = None,
    include_sofa_score: bool = False,
) -> pd.DataFrame:
    missing_targets = [col for col in target_cols if col not in df.columns]
    needs_sofa_score = include_sofa_score and "sofa_score" not in df.columns
    if not missing_targets and not needs_sofa_score:
        return df

    horizons = sorted({horizon_from_target_col(col) for col in missing_targets})
    sofa_path = clinical.find_sofa_csv(str(sofa_csv) if sofa_csv is not None else None)
    if sofa_path is None:
        raise FileNotFoundError("Cannot find sofa_scores_hourly.csv for SOFA label merge.")

    # Patient-level split 使用 subject_id，但逐小時 SOFA 必須以 stay_id 對齊，
    # 否則同一病人的多次 ICU stay 會在相同 sofa_hour 發生碰撞。
    join_col = "stay_id" if "stay_id" in df.columns else split_col
    allowed_stays: set[Any] | None = set(pd.unique(df[join_col]))
    print(f"Merging SOFA target reference: {sofa_path}")
    reference = clinical.load_sofa_reference(
        sofa_csv=sofa_path,
        horizons=horizons,
        split_col=join_col,
        time_col=time_col,
        allowed_stays=allowed_stays,
    )

    ref_cols = [join_col, time_col, *missing_targets]
    if needs_sofa_score:
        ref_cols.append("sofa_score")
    reference = reference[list(dict.fromkeys(ref_cols))]
    reference = reference.drop_duplicates(subset=[join_col, time_col])
    return df.merge(reference, on=[join_col, time_col], how="left", validate="many_to_one")
