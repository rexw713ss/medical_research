"""將 eICU-CRD 對齊為 MIMIC-IV 模型使用的 hourly predictors 與 SOFA labels。

所有事件時間均使用 ICU admission-relative offset，所有補值只允許同一 stay 內
forward fill。中間聚合檔可續跑，避免大型原始表因中斷而重讀。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd

from anfis_model import FEATURE_ORDER
from clinical_data_quality import filter_plausible
from preprocessing import add_measurement_process_features, leakage_free_forward_fill
from project_config import EICU_DATA_DIR
from sofa_score import add_future_labels, add_sofa_scores, normalize_fio2


MODEL_FEATURES = list(FEATURE_ORDER)
SOFA_SOURCE_FEATURES = [
    "pao2",
    "mechanical_vent",
    "dopamine",
    "dobutamine",
    "epinephrine",
    "norepinephrine",
    "urine_output",
]
LAB_MAP = {
    "pao2": "pao2",
    "platelets x 1000": "platelets",
    "total bilirubin": "bilirubin",
    "creatinine": "creatinine",
    "lactate": "lactate",
}
LAB_AGG = {
    "pao2": "min",
    "platelets": "min",
    "bilirubin": "max",
    "creatinine": "max",
    "lactate": "max",
}
PRESSORS = ("dopamine", "dobutamine", "epinephrine", "norepinephrine")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build harmonized eICU hourly features and SOFA labels.")
    parser.add_argument("--dataset-dir", default=EICU_DATA_DIR)
    parser.add_argument("--output-dir", default="outputs/eicu_external_validation")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument("--rolling-hours", type=int, default=24)
    parser.add_argument("--label-horizons", default="6")
    parser.add_argument("--min-components", type=int, default=4, choices=range(1, 7))
    parser.add_argument("--min-age", type=int, default=18)
    parser.add_argument("--max-stays", type=int, default=None)
    parser.add_argument("--max-chunks", type=int, default=None, help="Smoke-test limit per source table.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--write-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also write a portable gzip CSV in addition to the resumable pickle.",
    )
    return parser.parse_args()


def parse_horizons(raw: str) -> list[int]:
    values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("label horizons must be positive integers")
    return sorted(set(values))


def load_stays(dataset_dir: Path, min_age: int, max_stays: int | None) -> pd.DataFrame:
    columns = [
        "patientunitstayid",
        "patienthealthsystemstayid",
        "uniquepid",
        "hospitalid",
        "gender",
        "age",
        "ethnicity",
        "unitdischargeoffset",
        "unitdischargestatus",
    ]
    stays = pd.read_csv(dataset_dir / "patient.csv.gz", usecols=columns, low_memory=False)
    source_stays = len(stays)
    source_patients = int(stays["uniquepid"].nunique())
    stays["age_numeric"] = pd.to_numeric(stays["age"].replace("> 89", "90"), errors="coerce")
    stays["unitdischargeoffset"] = pd.to_numeric(stays["unitdischargeoffset"], errors="coerce")
    known_age = stays["age_numeric"].notna()
    valid_duration = stays["unitdischargeoffset"].notna() & stays["unitdischargeoffset"].ge(0)
    adult = known_age & stays["age_numeric"].ge(min_age)
    cohort_audit = {
        "minimum_age_years": int(min_age),
        "source_icu_stays": int(source_stays),
        "source_patients": source_patients,
        "excluded_missing_age_stays": int((~known_age).sum()),
        "excluded_age_below_minimum_stays": int((known_age & ~adult).sum()),
        "excluded_age_below_minimum_patients": int(
            stays.loc[known_age & ~adult, "uniquepid"].nunique()
        ),
        "excluded_missing_or_invalid_duration_stays": int((~valid_duration).sum()),
    }
    stays = stays[
        adult & valid_duration
    ].copy()
    stays = stays.sort_values("patientunitstayid", kind="mergesort")
    if max_stays is not None:
        stays = stays.head(max_stays).copy()
    stays = stays.rename(
        columns={
            "patientunitstayid": "stay_id",
            "patienthealthsystemstayid": "hadm_id",
            "uniquepid": "subject_id",
            "hospitalid": "hospital_id",
        }
    )
    stays["end_hour"] = np.floor(stays["unitdischargeoffset"] / 60.0).astype("int32")
    stays["stay_id"] = stays["stay_id"].astype("int64")
    stays["hadm_id"] = stays["hadm_id"].astype("int64")
    cohort_audit["eligible_adult_icu_stays"] = int(len(stays))
    cohort_audit["eligible_adult_patients"] = int(stays["subject_id"].nunique())
    stays = stays.reset_index(drop=True)
    stays.attrs["cohort_audit"] = cohort_audit
    return stays


def build_hourly_grid(stays: pd.DataFrame) -> pd.DataFrame:
    counts = stays["end_hour"].to_numpy(dtype=np.int64) + 1
    repeated = np.repeat(np.arange(len(stays), dtype=np.int64), counts)
    starts = np.repeat(np.cumsum(counts) - counts, counts)
    hours = np.arange(int(counts.sum()), dtype=np.int64) - starts
    columns = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "hospital_id",
        "gender",
        "age_numeric",
        "ethnicity",
        "unitdischargestatus",
    ]
    grid = stays.iloc[repeated][columns].reset_index(drop=True)
    grid["sofa_hour"] = hours.astype("int32")
    return grid


def valid_event_rows(
    chunk: pd.DataFrame,
    stay_end_minutes: pd.Series,
    stay_ids: set[int],
    id_col: str,
    offset_col: str,
) -> pd.DataFrame:
    chunk = chunk[chunk[id_col].isin(stay_ids)].copy()
    if chunk.empty:
        return chunk
    offset = pd.to_numeric(chunk[offset_col], errors="coerce")
    end = chunk[id_col].map(stay_end_minutes)
    keep = offset.notna() & end.notna() & (offset >= 0) & (offset <= end)
    chunk = chunk.loc[keep].copy()
    chunk["stay_id"] = chunk[id_col].astype("int64")
    chunk["sofa_hour"] = np.floor(offset.loc[keep] / 60.0).astype("int32")
    return chunk


def combine_chunk_aggregates(frames: list[pd.DataFrame], agg: dict[str, str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["stay_id", "sofa_hour", *agg])
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.groupby(["stay_id", "sofa_hour"], as_index=False).agg(agg)
    for column in agg:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    return frame


def extract_periodic_vitals(
    path: Path,
    stay_ids: set[int],
    stay_end_minutes: pd.Series,
    chunksize: int,
    max_chunks: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    usecols = [
        "patientunitstayid",
        "observationoffset",
        "temperature",
        "sao2",
        "heartrate",
        "respiration",
        "systemicsystolic",
        "systemicmean",
    ]
    rename = {
        "temperature": "temperature_c",
        "sao2": "spo2",
        "heartrate": "heart_rate",
        "respiration": "respiratory_rate",
        "systemicsystolic": "sbp_arterial",
        "systemicmean": "map_arterial",
    }
    agg = {
        "temperature_c": "mean",
        "spo2": "min",
        "heart_rate": "mean",
        "respiratory_rate": "mean",
        "sbp_arterial": "min",
        "map_arterial": "min",
    }
    frames = []
    source_rows = kept_rows = 0
    for chunk_index, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False)):
        if max_chunks is not None and chunk_index >= max_chunks:
            break
        source_rows += len(chunk)
        chunk = valid_event_rows(
            chunk,
            stay_end_minutes,
            stay_ids,
            "patientunitstayid",
            "observationoffset",
        )
        kept_rows += len(chunk)
        if chunk.empty:
            continue
        chunk = chunk.rename(columns=rename)
        for column in agg:
            feature = {
                "sbp_arterial": "sbp_arterial",
                "map_arterial": "map_arterial",
            }.get(column, column)
            chunk[column] = filter_plausible(chunk[column], feature)
        hourly = chunk.groupby(["stay_id", "sofa_hour"], as_index=False).agg(agg)
        frames.append(hourly)
    result = combine_chunk_aggregates(frames, agg)
    return result, {"source_rows_read": source_rows, "eligible_rows": kept_rows, "hourly_rows": len(result)}


def extract_aperiodic_vitals(
    path: Path,
    stay_ids: set[int],
    stay_end_minutes: pd.Series,
    chunksize: int,
    max_chunks: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    usecols = [
        "patientunitstayid",
        "observationoffset",
        "noninvasivesystolic",
        "noninvasivemean",
    ]
    agg = {"sbp_noninvasive": "min", "map_noninvasive": "min"}
    frames = []
    source_rows = kept_rows = 0
    for chunk_index, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False)):
        if max_chunks is not None and chunk_index >= max_chunks:
            break
        source_rows += len(chunk)
        chunk = valid_event_rows(
            chunk,
            stay_end_minutes,
            stay_ids,
            "patientunitstayid",
            "observationoffset",
        )
        kept_rows += len(chunk)
        if chunk.empty:
            continue
        chunk = chunk.rename(
            columns={
                "noninvasivesystolic": "sbp_noninvasive",
                "noninvasivemean": "map_noninvasive",
            }
        )
        for column in agg:
            chunk[column] = filter_plausible(chunk[column], column)
        frames.append(chunk.groupby(["stay_id", "sofa_hour"], as_index=False).agg(agg))
    result = combine_chunk_aggregates(frames, agg)
    return result, {"source_rows_read": source_rows, "eligible_rows": kept_rows, "hourly_rows": len(result)}


def extract_labs(
    path: Path,
    stay_ids: set[int],
    stay_end_minutes: pd.Series,
    chunksize: int,
    max_chunks: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    usecols = ["patientunitstayid", "labresultoffset", "labname", "labresult"]
    frames = []
    source_rows = relevant_rows = 0
    for chunk_index, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False)):
        if max_chunks is not None and chunk_index >= max_chunks:
            break
        source_rows += len(chunk)
        names = chunk["labname"].astype(str).str.strip().str.lower()
        chunk = chunk[names.isin(LAB_MAP)].copy()
        if chunk.empty:
            continue
        chunk["feature"] = names.loc[chunk.index].map(LAB_MAP)
        chunk = valid_event_rows(
            chunk,
            stay_end_minutes,
            stay_ids,
            "patientunitstayid",
            "labresultoffset",
        )
        relevant_rows += len(chunk)
        if chunk.empty:
            continue
        chunk["value"] = pd.to_numeric(chunk["labresult"], errors="coerce")
        for feature in LAB_AGG:
            mask = chunk["feature"].eq(feature)
            chunk.loc[mask, "value"] = filter_plausible(chunk.loc[mask, "value"], feature)
        hourly_parts = []
        for feature, method in LAB_AGG.items():
            selected = chunk[chunk["feature"].eq(feature)]
            if selected.empty:
                continue
            hourly_parts.append(
                selected.groupby(["stay_id", "sofa_hour"], as_index=False)["value"]
                .agg(method)
                .rename(columns={"value": feature})
            )
        if hourly_parts:
            hourly = hourly_parts[0]
            for part in hourly_parts[1:]:
                hourly = hourly.merge(part, on=["stay_id", "sofa_hour"], how="outer")
            frames.append(hourly)
    result = combine_chunk_aggregates(frames, LAB_AGG)
    return result, {"source_rows_read": source_rows, "relevant_rows": relevant_rows, "hourly_rows": len(result)}


def extract_nurse_charting(
    path: Path,
    stay_ids: set[int],
    stay_end_minutes: pd.Series,
    chunksize: int,
    max_chunks: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    usecols = [
        "patientunitstayid",
        "nursingchartoffset",
        "nursingchartcelltypevallabel",
        "nursingchartcelltypevalname",
        "nursingchartvalue",
    ]
    agg = {
        "gcs_total": "min",
        "heart_rate_nurse": "mean",
        "respiratory_rate_nurse": "mean",
        "spo2_nurse": "min",
        "temperature_c_nurse": "mean",
    }
    frames = []
    source_rows = relevant_rows = 0
    feature_rows = {column: 0 for column in agg}
    for chunk_index, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False)):
        if max_chunks is not None and chunk_index >= max_chunks:
            break
        source_rows += len(chunk)
        label = chunk["nursingchartcelltypevallabel"].astype(str).str.strip().str.lower()
        name = chunk["nursingchartcelltypevalname"].astype(str).str.strip().str.lower()
        gcs_mask = label.eq("gcs total") | name.eq("gcs total")
        heart_rate_mask = name.eq("heart rate")
        respiratory_rate_mask = name.eq("respiratory rate")
        spo2_mask = name.eq("o2 saturation")
        temperature_c_mask = name.eq("temperature (c)")
        temperature_f_mask = name.eq("temperature (f)")
        keep = (
            gcs_mask
            | heart_rate_mask
            | respiratory_rate_mask
            | spo2_mask
            | temperature_c_mask
            | temperature_f_mask
        )
        chunk = chunk[keep].copy()
        chunk["_name"] = name.loc[keep]
        chunk["_label"] = label.loc[keep]
        chunk = valid_event_rows(
            chunk,
            stay_end_minutes,
            stay_ids,
            "patientunitstayid",
            "nursingchartoffset",
        )
        relevant_rows += len(chunk)
        if chunk.empty:
            continue
        value = pd.to_numeric(chunk["nursingchartvalue"], errors="coerce")
        chunk["gcs_total"] = np.nan
        chunk["heart_rate_nurse"] = np.nan
        chunk["respiratory_rate_nurse"] = np.nan
        chunk["spo2_nurse"] = np.nan
        chunk["temperature_c_nurse"] = np.nan
        gcs = chunk["_label"].eq("gcs total") | chunk["_name"].eq("gcs total")
        chunk.loc[gcs, "gcs_total"] = filter_plausible(value.loc[gcs], "gcs_total")
        heart_rate = chunk["_name"].eq("heart rate")
        chunk.loc[heart_rate, "heart_rate_nurse"] = filter_plausible(
            value.loc[heart_rate], "heart_rate"
        )
        respiratory_rate = chunk["_name"].eq("respiratory rate")
        chunk.loc[respiratory_rate, "respiratory_rate_nurse"] = filter_plausible(
            value.loc[respiratory_rate], "respiratory_rate"
        )
        spo2 = chunk["_name"].eq("o2 saturation")
        chunk.loc[spo2, "spo2_nurse"] = filter_plausible(value.loc[spo2], "spo2")
        temperature_c = chunk["_name"].eq("temperature (c)")
        chunk.loc[temperature_c, "temperature_c_nurse"] = filter_plausible(
            value.loc[temperature_c], "temperature_c"
        )
        temperature_f = chunk["_name"].eq("temperature (f)")
        converted_f = (value.loc[temperature_f] - 32.0) * (5.0 / 9.0)
        chunk.loc[temperature_f, "temperature_c_nurse"] = filter_plausible(
            converted_f, "temperature_c"
        )
        for column in agg:
            feature_rows[column] += int(chunk[column].notna().sum())
        frames.append(chunk.groupby(["stay_id", "sofa_hour"], as_index=False).agg(agg))
    result = combine_chunk_aggregates(frames, agg)
    return result, {
        "source_rows_read": source_rows,
        "relevant_rows": relevant_rows,
        "valid_feature_rows": feature_rows,
        "hourly_rows": len(result),
    }


def extract_respiratory(
    path: Path,
    stay_ids: set[int],
    stay_end_minutes: pd.Series,
    chunksize: int,
    max_chunks: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    usecols = ["patientunitstayid", "respchartoffset", "respchartvaluelabel", "respchartvalue"]
    fio2_labels = {"fio2", "fio2 (%)", "set fraction of inspired oxygen (fio2)"}
    support_labels = {"peep", "peep/cpap"}
    frames = []
    source_rows = relevant_rows = fio2_valid = support_rows = 0
    for chunk_index, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False)):
        if max_chunks is not None and chunk_index >= max_chunks:
            break
        source_rows += len(chunk)
        labels = chunk["respchartvaluelabel"].astype(str).str.strip().str.lower()
        keep = labels.isin(fio2_labels | support_labels | {"rt vent on/off"})
        chunk = chunk[keep].copy()
        chunk["_label"] = labels.loc[keep]
        chunk = valid_event_rows(
            chunk,
            stay_end_minutes,
            stay_ids,
            "patientunitstayid",
            "respchartoffset",
        )
        relevant_rows += len(chunk)
        if chunk.empty:
            continue
        value_numeric = pd.to_numeric(chunk["respchartvalue"], errors="coerce")
        fio2_mask = chunk["_label"].isin(fio2_labels)
        chunk["fio2"] = np.nan
        chunk.loc[fio2_mask, "fio2"] = normalize_fio2(value_numeric.loc[fio2_mask])
        chunk["fio2"] = filter_plausible(chunk["fio2"], "fio2")
        fio2_valid += int(chunk["fio2"].notna().sum())

        support = pd.Series(0.0, index=chunk.index)
        peep_mask = chunk["_label"].isin(support_labels) & value_numeric.gt(0)
        vent_state = chunk["respchartvalue"].astype(str).str.strip().str.lower()
        state_mask = chunk["_label"].eq("rt vent on/off") & vent_state.isin({"start", "continued"})
        support.loc[peep_mask | state_mask] = 1.0
        support_rows += int((peep_mask | state_mask).sum())
        chunk["mechanical_vent"] = support
        frames.append(
            chunk.groupby(["stay_id", "sofa_hour"], as_index=False).agg(
                fio2=("fio2", "max"),
                mechanical_vent=("mechanical_vent", "max"),
            )
        )
    result = combine_chunk_aggregates(frames, {"fio2": "max", "mechanical_vent": "max"})
    return result, {
        "source_rows_read": source_rows,
        "relevant_rows": relevant_rows,
        "valid_fio2_rows": fio2_valid,
        "respiratory_support_rows": support_rows,
        "hourly_rows": len(result),
    }


def pressor_name(value: Any) -> str | None:
    lowered = str(value).strip().lower()
    return next((drug for drug in PRESSORS if lowered.startswith(drug)), None)


def normalized_pressor_rate(
    names: pd.Series,
    rates: pd.Series,
    weights: pd.Series,
) -> pd.Series:
    names = names.astype(str).str.lower()
    rates = pd.to_numeric(rates, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce")
    out = pd.Series(np.nan, index=names.index, dtype="float64")
    direct = names.str.contains(r"\(mcg/kg/min\)", regex=True)
    out.loc[direct] = rates.loc[direct]
    mg_kg_min = names.str.contains(r"\(mg/kg/min\)", regex=True)
    out.loc[mg_kg_min] = rates.loc[mg_kg_min] * 1000.0
    mcg_kg_hr = names.str.contains(r"\(mcg/kg/hr\)", regex=True)
    out.loc[mcg_kg_hr] = rates.loc[mcg_kg_hr] / 60.0
    valid_weight = weights.between(20.0, 400.0)
    mcg_min = names.str.contains(r"\(mcg/min\)", regex=True) & valid_weight
    out.loc[mcg_min] = rates.loc[mcg_min] / weights.loc[mcg_min]
    mg_min = names.str.contains(r"\(mg/min\)", regex=True) & valid_weight
    out.loc[mg_min] = rates.loc[mg_min] * 1000.0 / weights.loc[mg_min]
    mcg_hr = names.str.contains(r"\(mcg/hr\)", regex=True) & valid_weight
    out.loc[mcg_hr] = rates.loc[mcg_hr] / 60.0 / weights.loc[mcg_hr]
    mg_hr = names.str.contains(r"\(mg/hr\)", regex=True) & valid_weight
    out.loc[mg_hr] = rates.loc[mg_hr] * 1000.0 / 60.0 / weights.loc[mg_hr]
    return out


def extract_pressors(
    path: Path,
    stay_ids: set[int],
    stay_end_minutes: pd.Series,
    chunksize: int,
    max_chunks: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    usecols = ["patientunitstayid", "infusionoffset", "drugname", "drugrate", "patientweight"]
    frames = []
    source_rows = relevant_rows = convertible_rows = 0
    for chunk_index, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False)):
        if max_chunks is not None and chunk_index >= max_chunks:
            break
        source_rows += len(chunk)
        chunk["feature"] = chunk["drugname"].astype(str).map(pressor_name)
        chunk = chunk[chunk["feature"].notna()].copy()
        chunk = valid_event_rows(
            chunk,
            stay_end_minutes,
            stay_ids,
            "patientunitstayid",
            "infusionoffset",
        )
        relevant_rows += len(chunk)
        if chunk.empty:
            continue
        chunk["value"] = normalized_pressor_rate(
            chunk["drugname"], chunk["drugrate"], chunk["patientweight"]
        )
        for feature in PRESSORS:
            mask = chunk["feature"].eq(feature)
            chunk.loc[mask, "value"] = filter_plausible(chunk.loc[mask, "value"], feature)
        chunk = chunk[chunk["value"].notna() & chunk["value"].gt(0)].copy()
        convertible_rows += len(chunk)
        hourly_parts = []
        for feature in PRESSORS:
            selected = chunk[chunk["feature"].eq(feature)]
            if selected.empty:
                continue
            hourly_parts.append(
                selected.groupby(["stay_id", "sofa_hour"], as_index=False)["value"]
                .max()
                .rename(columns={"value": feature})
            )
        if hourly_parts:
            hourly = hourly_parts[0]
            for part in hourly_parts[1:]:
                hourly = hourly.merge(part, on=["stay_id", "sofa_hour"], how="outer")
            frames.append(hourly)
    result = combine_chunk_aggregates(frames, {drug: "max" for drug in PRESSORS})
    return result, {
        "source_rows_read": source_rows,
        "pressor_rows": relevant_rows,
        "convertible_positive_rows": convertible_rows,
        "excluded_unconvertible_or_zero_rows": relevant_rows - convertible_rows,
        "hourly_rows": len(result),
    }


def urinary_output_label(labels: pd.Series) -> pd.Series:
    labels = labels.astype(str).str.strip().str.lower()
    include = (
        labels.eq("urine")
        | labels.eq("voided amount")
        | labels.eq("indwelling catheter output")
        | labels.eq("condom catheter output")
        | labels.eq("or urine")
        | labels.str.contains("urine output", regex=False)
        | labels.str.contains("urinary catheter output", regex=False)
        | (labels.str.contains("urethral catheter", regex=False) & labels.str.contains("output", regex=False))
        | (labels.str.contains("nephrostomy", regex=False) & labels.str.contains("output", regex=False))
        | labels.str.startswith("output amt-urinary catheter")
        | labels.str.startswith("foley")
    )
    exclude = labels.str.contains(
        r"count|occurrence|incontinence|unmeasured|number of|24 hr|total|variance|color|appearance",
        regex=True,
    )
    return include & ~exclude


def extract_urine(
    path: Path,
    stay_ids: set[int],
    stay_end_minutes: pd.Series,
    chunksize: int,
    max_chunks: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    usecols = ["patientunitstayid", "intakeoutputoffset", "celllabel", "cellvaluenumeric"]
    frames = []
    source_rows = relevant_rows = plausible_rows = 0
    for chunk_index, chunk in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False)):
        if max_chunks is not None and chunk_index >= max_chunks:
            break
        source_rows += len(chunk)
        chunk = chunk[urinary_output_label(chunk["celllabel"])].copy()
        chunk = valid_event_rows(
            chunk,
            stay_end_minutes,
            stay_ids,
            "patientunitstayid",
            "intakeoutputoffset",
        )
        relevant_rows += len(chunk)
        if chunk.empty:
            continue
        chunk["urine_output"] = pd.to_numeric(chunk["cellvaluenumeric"], errors="coerce")
        chunk = chunk[chunk["urine_output"].between(0.0, 5000.0, inclusive="both")].copy()
        plausible_rows += len(chunk)
        frames.append(
            chunk.groupby(["stay_id", "sofa_hour"], as_index=False)["urine_output"].sum()
        )
    result = combine_chunk_aggregates(frames, {"urine_output": "sum"})
    return result, {
        "source_rows_read": source_rows,
        "urinary_label_rows": relevant_rows,
        "plausible_numeric_rows": plausible_rows,
        "hourly_rows": len(result),
    }


def cached_extract(
    name: str,
    output_dir: Path,
    force: bool,
    function: Callable[..., tuple[pd.DataFrame, dict[str, Any]]],
    *args: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache_path = output_dir / "intermediate" / f"{name}.pkl"
    stats_path = output_dir / "intermediate" / f"{name}_stats.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and stats_path.exists() and not force:
        print(f"Loading cached {name}: {cache_path}")
        return pd.read_pickle(cache_path), json.loads(stats_path.read_text(encoding="utf-8"))
    started = perf_counter()
    frame, stats = function(*args)
    stats["seconds"] = perf_counter() - started
    frame.to_pickle(cache_path)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{name}: {len(frame):,} hourly rows in {stats['seconds']:.1f}s")
    return frame, stats


def coalesce_model_features(frame: pd.DataFrame) -> pd.DataFrame:
    for feature, nurse_column in (
        ("heart_rate", "heart_rate_nurse"),
        ("respiratory_rate", "respiratory_rate_nurse"),
        ("spo2", "spo2_nurse"),
        ("temperature_c", "temperature_c_nurse"),
    ):
        frame[feature] = frame[feature].combine_first(frame[nurse_column])
    frame["sbp"] = frame.get("sbp_arterial").combine_first(frame.get("sbp_noninvasive"))
    frame["map"] = frame.get("map_arterial").combine_first(frame.get("map_noninvasive"))
    frame["pao2_fio2"] = np.where(
        frame["pao2"].notna() & frame["fio2"].notna() & frame["fio2"].gt(0),
        frame["pao2"] / frame["fio2"],
        np.nan,
    )
    return frame


def quality_report(
    frame: pd.DataFrame,
    extraction_stats: dict[str, Any],
    output_dir: Path,
    horizons: list[int],
    cohort_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    features = {}
    for feature in MODEL_FEATURES:
        missing_col = f"{feature}_is_missing"
        observed = int((frame[missing_col] == 0).sum())
        features[feature] = {
            "observed_hourly_rows": observed,
            "observed_fraction": observed / max(len(frame), 1),
            "stays_with_measurement": int(frame.loc[frame[missing_col].eq(0), "stay_id"].nunique()),
        }
    labels = {}
    for horizon in horizons:
        column = f"label_sofa_increase_ge2_{horizon}h"
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        labels[column] = {
            "valid_rows": int(len(values)),
            "positive_rows": int(values.sum()),
            "prevalence": float(values.mean()) if len(values) else None,
        }
    report = {
        "database": "eICU-CRD",
        "rows": len(frame),
        "stays": int(frame["stay_id"].nunique()),
        "patients": int(frame["subject_id"].nunique()),
        "hospitals": int(frame["hospital_id"].nunique()),
        "adult_cohort": cohort_audit or {},
        "features": features,
        "sofa_component_count": {
            str(int(key)): int(value)
            for key, value in frame["sofa_component_count"].value_counts().sort_index().items()
        },
        "labels": labels,
        "extraction": extraction_stats,
        "leakage_control": "within-stay current/past measurements only; no backward fill",
        "schema_version": "eicu_mimic_harmonized_v1",
    }
    (output_dir / "eicu_hourly_quality.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    horizons = parse_horizons(args.label_horizons)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_pickle = output_dir / "eicu_hourly_features.pkl"
    final_csv = output_dir / "eicu_hourly_features.csv.gz"
    if final_pickle.exists() and not args.force:
        print(f"Hourly harmonization already exists: {final_pickle}")
        return

    stays = load_stays(dataset_dir, args.min_age, args.max_stays)
    cohort_audit = dict(stays.attrs.get("cohort_audit", {}))
    print(f"Eligible adult ICU stays: {len(stays):,}")
    stay_ids = set(stays["stay_id"].tolist())
    stay_end_minutes = stays.set_index("stay_id")["unitdischargeoffset"]
    common = (stay_ids, stay_end_minutes, args.chunksize, args.max_chunks)
    stats: dict[str, Any] = {}

    periodic, stats["vital_periodic"] = cached_extract(
        "vital_periodic", output_dir, args.force, extract_periodic_vitals,
        dataset_dir / "vitalPeriodic.csv.gz", *common,
    )
    aperiodic, stats["vital_aperiodic"] = cached_extract(
        "vital_aperiodic", output_dir, args.force, extract_aperiodic_vitals,
        dataset_dir / "vitalAperiodic.csv.gz", *common,
    )
    labs, stats["labs"] = cached_extract(
        "labs", output_dir, args.force, extract_labs,
        dataset_dir / "lab.csv.gz", *common,
    )
    nurse_charting, stats["nurse_charting"] = cached_extract(
        "nurse_charting", output_dir, args.force, extract_nurse_charting,
        dataset_dir / "nurseCharting.csv.gz", *common,
    )
    respiratory, stats["respiratory"] = cached_extract(
        "respiratory", output_dir, args.force, extract_respiratory,
        dataset_dir / "respiratoryCharting.csv.gz", *common,
    )
    pressors, stats["pressors"] = cached_extract(
        "pressors", output_dir, args.force, extract_pressors,
        dataset_dir / "infusionDrug.csv.gz", *common,
    )
    urine, stats["urine"] = cached_extract(
        "urine", output_dir, args.force, extract_urine,
        dataset_dir / "intakeOutput.csv.gz", *common,
    )

    print("Building hourly grid and merging sources...")
    frame = build_hourly_grid(stays)
    for source in (periodic, aperiodic, labs, nurse_charting, respiratory, pressors, urine):
        frame = frame.merge(source, on=["stay_id", "sofa_hour"], how="left", validate="one_to_one")
    del periodic, aperiodic, labs, nurse_charting, respiratory, pressors, urine
    gc.collect()
    frame = coalesce_model_features(frame)

    # SOFA 需要 forward-filled FiO2，但模型 missingness 必須反映原始量測狀態。
    observed_mask = frame[MODEL_FEATURES].notna().copy()
    raw_fio2 = frame["fio2"].copy()
    raw_pao2_fio2 = frame["pao2_fio2"].copy()

    print("Calculating 24-hour SOFA and future labels...")
    frame = add_sofa_scores(frame, args.rolling_hours, args.min_components)
    frame = add_future_labels(frame, horizons)
    frame["fio2"] = raw_fio2
    frame["pao2_fio2"] = raw_pao2_fio2
    del raw_fio2, raw_pao2_fio2

    # 必須在 LOCF 前保存量測過程；這三組 channels 與 MIMIC explicit model 完全同序。
    frame = add_measurement_process_features(frame, MODEL_FEATURES, observed_mask)
    frame = leakage_free_forward_fill(frame, [*MODEL_FEATURES, "pao2"])
    frame["pao2_fio2"] = np.where(
        frame["pao2"].notna() & frame["fio2"].notna() & frame["fio2"].gt(0),
        frame["pao2"] / frame["fio2"],
        np.nan,
    )

    keep_columns = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "hospital_id",
        "gender",
        "age_numeric",
        "ethnicity",
        "unitdischargestatus",
        "sofa_hour",
        *MODEL_FEATURES,
        *[f"{feature}_is_missing" for feature in MODEL_FEATURES],
        *[f"{feature}_time_since_last_measurement_h" for feature in MODEL_FEATURES],
        "sofa_score",
        "sofa_score_assume_normal",
        "sofa_score_complete",
        "sofa_score_observed",
        "sofa_component_count",
        "respiration_score",
        "coagulation_score",
        "liver_score",
        "cardiovascular_score",
        "cns_score",
        "renal_score",
    ]
    for horizon in horizons:
        keep_columns.extend(
            [
                f"future_{horizon}h_max_sofa",
                f"sofa_increase_{horizon}h",
                f"label_sofa_increase_ge2_{horizon}h",
            ]
        )
    frame = frame[keep_columns]
    report = quality_report(frame, stats, output_dir, horizons, cohort_audit=cohort_audit)
    frame.to_pickle(final_pickle)
    if args.write_csv:
        print(f"Writing portable CSV: {final_csv}")
        frame.to_csv(final_csv, index=False, compression="gzip", encoding="utf-8")
    config = {
        **vars(args),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "label_horizons": horizons,
        "model_feature_order": MODEL_FEATURES,
        "output_pickle": str(final_pickle),
        "output_csv": str(final_csv) if args.write_csv else None,
        "quality_report": str(output_dir / "eicu_hourly_quality.json"),
        "rows": report["rows"],
    }
    (output_dir / "eicu_preprocessing_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Hourly eICU harmonization complete: {final_pickle}")


if __name__ == "__main__":
    main()
