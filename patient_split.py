"""建立並讀取全專案共用的 patient-level train/validation/test split。

正式實驗以 ``subject_id`` 為切分單位，確保同一病人的所有 ICU stays
只會出現在 train、validation 或 test 其中一組。切分時依病人在 6、12、24
小時 horizon 是否曾出現 SOFA 增加 >= 2 分進行分層，以降低三組事件率差異。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from project_config import MIMIC_DATA_DIR, PATIENT_SPLIT_CSV, SOFA_HOURLY_CSV


SPLIT_NAMES = ("train", "validation", "test")
HORIZONS = (6, 12, 24)
TARGET_COLUMNS = tuple(f"label_sofa_increase_ge2_{h}h" for h in HORIZONS)


def read_patient_split(path: str | Path = PATIENT_SPLIT_CSV) -> pd.DataFrame:
    """讀取 manifest 並檢查每位病人只屬於一個資料集。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"找不到 patient split manifest: {path}。請先執行 patient_split.py。"
        )

    manifest = pd.read_csv(path)
    required = {"subject_id", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Patient split manifest 缺少欄位: {sorted(missing)}")
    if manifest["subject_id"].isna().any():
        raise ValueError("Patient split manifest 含有空白 subject_id。")
    if manifest["subject_id"].duplicated().any():
        duplicates = manifest.loc[manifest["subject_id"].duplicated(), "subject_id"].head().tolist()
        raise ValueError(f"Patient split manifest 有重複 subject_id，例如: {duplicates}")

    invalid_splits = sorted(set(manifest["split"].dropna()) - set(SPLIT_NAMES))
    if invalid_splits:
        raise ValueError(f"Patient split manifest 含未知 split: {invalid_splits}")
    if manifest["split"].isna().any():
        raise ValueError("Patient split manifest 含有空白 split。")
    return manifest


def split_ids_for_values(
    values: Iterable[Any] | np.ndarray,
    manifest_path: str | Path = "patient_split.csv",
    require_complete: bool = True,
) -> tuple[set[Any], set[Any], set[Any]]:
    """取得目前資料中 train/validation/test 的 subject_id 集合。"""
    manifest = read_patient_split(manifest_path)
    available = set(pd.Series(values).dropna().unique().tolist())
    manifest_ids = set(manifest["subject_id"].tolist())
    missing = available - manifest_ids
    if require_complete and missing:
        examples = sorted(missing)[:5]
        raise ValueError(
            f"目前資料有 {len(missing):,} 位 subject_id 不在 manifest，例如: {examples}"
        )

    ids_by_split = {
        split: set(manifest.loc[manifest["split"] == split, "subject_id"].tolist()) & available
        for split in SPLIT_NAMES
    }
    empty = [split for split, ids in ids_by_split.items() if not ids]
    if empty:
        raise ValueError(
            f"目前載入的資料在 {empty} 沒有病人；測試子集請增加 --max-stays/--max-rows。"
        )
    return (
        ids_by_split["train"],
        ids_by_split["validation"],
        ids_by_split["test"],
    )


def attach_split(
    df: pd.DataFrame,
    manifest_path: str | Path = "patient_split.csv",
    patient_col: str = "subject_id",
    output_col: str = "dataset_split",
) -> pd.DataFrame:
    """將固定 split 欄位合併到逐小時資料，並拒絕未被分配的病人。"""
    if patient_col not in df.columns:
        raise ValueError(f"資料缺少 patient-level split 欄位: {patient_col}")
    manifest = read_patient_split(manifest_path)[["subject_id", "split"]].rename(
        columns={"subject_id": patient_col, "split": output_col}
    )
    result = df.merge(manifest, on=patient_col, how="left", validate="many_to_one")
    missing = result[output_col].isna()
    if missing.any():
        examples = result.loc[missing, patient_col].drop_duplicates().head().tolist()
        raise ValueError(
            f"有 {result.loc[missing, patient_col].nunique():,} 位病人不在 manifest，例如: {examples}"
        )
    return result


def aggregate_patient_outcomes(sofa_csv: Path, chunk_size: int) -> pd.DataFrame:
    """把逐小時標籤聚合成 patient-level ever-event 指標。"""
    usecols = ["subject_id", *TARGET_COLUMNS]
    chunk_summaries: list[pd.DataFrame] = []
    for chunk in pd.read_csv(sofa_csv, usecols=usecols, chunksize=chunk_size):
        for col in TARGET_COLUMNS:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        grouped = chunk.groupby("subject_id", sort=False)[list(TARGET_COLUMNS)]
        maxima = grouped.max().rename(columns={col: f"event_{h}h" for col, h in zip(TARGET_COLUMNS, HORIZONS)})
        valid = grouped.count().rename(columns={col: f"valid_hours_{h}h" for col, h in zip(TARGET_COLUMNS, HORIZONS)})
        positive = grouped.sum(min_count=1).rename(
            columns={col: f"positive_hours_{h}h" for col, h in zip(TARGET_COLUMNS, HORIZONS)}
        )
        chunk_summaries.append(maxima.join(valid).join(positive).reset_index())

    if not chunk_summaries:
        raise ValueError(f"SOFA CSV 沒有可用資料: {sofa_csv}")

    combined = pd.concat(chunk_summaries, ignore_index=True)
    event_cols = [f"event_{h}h" for h in HORIZONS]
    valid_cols = [f"valid_hours_{h}h" for h in HORIZONS]
    positive_cols = [f"positive_hours_{h}h" for h in HORIZONS]
    aggregation = {
        **{col: "max" for col in event_cols},
        **{col: "sum" for col in valid_cols},
        **{col: "sum" for col in positive_cols},
    }
    return combined.groupby("subject_id", as_index=False, sort=True).agg(aggregation)


def allocate_stratified_splits(
    patients: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> pd.DataFrame:
    """在每個 6/12/24h outcome 組合內做可重現的隨機切分。"""
    fractions = np.asarray([train_frac, val_frac, test_frac], dtype=float)
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1.0):
        raise ValueError("train/validation/test 比例皆須 > 0，且總和必須為 1。")

    event_cols = [f"event_{h}h" for h in HORIZONS]
    strata_values = patients[event_cols].fillna(-1).astype("int8").astype(str)
    patients = patients.copy()
    patients["stratum"] = strata_values.agg("|".join, axis=1)
    patients["split"] = ""
    rng = np.random.default_rng(seed)

    for _, indices in patients.groupby("stratum", sort=True).groups.items():
        shuffled = np.asarray(sorted(indices), dtype=np.int64)
        rng.shuffle(shuffled)
        n = len(shuffled)

        raw = fractions * n
        counts = np.floor(raw).astype(int)
        for idx in np.argsort(-(raw - counts))[: n - int(counts.sum())]:
            counts[idx] += 1

        if n >= 3:
            for idx in range(3):
                if counts[idx] == 0:
                    donor = int(np.argmax(counts))
                    counts[donor] -= 1
                    counts[idx] += 1

        train_end = counts[0]
        val_end = train_end + counts[1]
        patients.loc[shuffled[:train_end], "split"] = "train"
        patients.loc[shuffled[train_end:val_end], "split"] = "validation"
        patients.loc[shuffled[val_end:], "split"] = "test"

    if (patients["split"] == "").any():
        raise RuntimeError("部分病人未被分配 split。")
    return patients


def build_summary(manifest: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n_patients": int(len(manifest)),
        "n_icu_stays": int(manifest["n_icu_stays"].sum()),
        "splits": {},
    }
    for split in SPLIT_NAMES:
        part = manifest[manifest["split"] == split]
        split_summary: dict[str, Any] = {
            "patients": int(len(part)),
            "patient_fraction": float(len(part) / len(manifest)),
            "icu_stays": int(part["n_icu_stays"].sum()),
        }
        for horizon in HORIZONS:
            event = part[f"event_{horizon}h"]
            split_summary[f"event_{horizon}h_known_patients"] = int(event.notna().sum())
            split_summary[f"event_{horizon}h_patient_rate"] = (
                float(event.mean()) if event.notna().any() else None
            )
            valid_hours = float(part[f"valid_hours_{horizon}h"].sum())
            positive_hours = float(part[f"positive_hours_{horizon}h"].sum())
            split_summary[f"label_{horizon}h_valid_hours"] = int(valid_hours)
            split_summary[f"label_{horizon}h_positive_hours"] = int(positive_hours)
            split_summary[f"label_{horizon}h_hourly_rate"] = (
                positive_hours / valid_hours if valid_hours > 0 else None
            )
        summary["splits"][split] = split_summary
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="建立固定 patient-level 資料切分。")
    parser.add_argument("--sofa-csv", default=SOFA_HOURLY_CSV)
    parser.add_argument("--icustays-csv", default=f"{MIMIC_DATA_DIR}/icustays.csv.gz")
    parser.add_argument("--output", default=PATIENT_SPLIT_CSV)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sofa_csv = Path(args.sofa_csv)
    icustays_csv = Path(args.icustays_csv)
    if not sofa_csv.exists():
        raise FileNotFoundError(f"找不到 SOFA CSV: {sofa_csv}")
    if not icustays_csv.exists():
        raise FileNotFoundError(f"找不到 ICU stays CSV: {icustays_csv}")

    print(f"聚合 patient-level outcomes: {sofa_csv}")
    outcomes = aggregate_patient_outcomes(sofa_csv, args.chunk_size)
    stays = pd.read_csv(icustays_csv, usecols=["subject_id", "stay_id"])
    stay_counts = (
        stays.groupby("subject_id", as_index=False)["stay_id"]
        .nunique()
        .rename(columns={"stay_id": "n_icu_stays"})
    )
    patients = stay_counts.merge(outcomes, on="subject_id", how="left", validate="one_to_one")
    patients = allocate_stratified_splits(
        patients,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    column_order = [
        "subject_id",
        "split",
        "stratum",
        "n_icu_stays",
        *[f"event_{h}h" for h in HORIZONS],
        *[f"valid_hours_{h}h" for h in HORIZONS],
        *[f"positive_hours_{h}h" for h in HORIZONS],
    ]
    patients = patients[column_order].sort_values("subject_id").reset_index(drop=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patients.to_csv(output_path, index=False)

    summary = {
        "method": "patient-level multi-horizon stratified random split",
        "patient_id_column": "subject_id",
        "seed": args.seed,
        "fractions": {
            "train": args.train_frac,
            "validation": args.val_frac,
            "test": args.test_frac,
        },
        "source_sofa_csv": str(sofa_csv),
        "source_icustays_csv": str(icustays_csv),
        "checks": {
            "one_row_per_subject": bool(not patients["subject_id"].duplicated().any()),
            "all_patients_assigned": bool(patients["split"].isin(SPLIT_NAMES).all()),
            "no_split_overlap": True,
            "split_overlap_patients": 0,
            "all_icustays_patients_in_manifest": bool(
                set(stays["subject_id"].unique()) == set(patients["subject_id"].unique())
            ),
        },
        **build_summary(patients),
    }
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"完成: {output_path}")
    print(json.dumps(summary["splits"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
