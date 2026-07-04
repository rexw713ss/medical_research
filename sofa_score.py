"""SOFA score 與 SOFA increase outcome label 計算程式。

這支程式會從 MIMIC-IV ICU 資料建立逐小時 SOFA 分數：
1. 以 icustays 建立每個 ICU stay 的 hourly grid。
2. 從 chartevents/labevents/inputevents/outputevents 抽出 SOFA 需要的欄位。
3. 對每個小時回看過去 rolling-hours，預設 24 小時，取各器官系統 worst value。
4. 計算六個 SOFA component score 與總分。
5. 產生未來 6、12、24 小時內 SOFA score increase >= 2 的 outcome label。

注意：若 inputevents.csv.gz 或 outputevents.csv.gz 尚未放入 dataset/，
程式仍可執行，但 cardiovascular 與 renal component 會使用目前可用資料的簡化版。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_data_quality import filter_long_frame, filter_plausible
from project_config import MIMIC_DATA_DIR, SOFA_HOURLY_CSV


# chartevents 中可用於 SOFA 的 itemid。
# MAP、FiO2、GCS 與機械通氣紀錄會從這裡抽取。
CHARTEVENT_ITEMIDS = {
    220052: "map",  # Arterial Blood Pressure mean，動脈平均血壓。
    220181: "map",  # Non Invasive Blood Pressure mean，非侵入式平均血壓。
    223835: "fio2",  # Inspired O2 Fraction，吸入氧氣濃度。
    220739: "gcs_eye",
    223900: "gcs_verbal",
    223901: "gcs_motor",
    223848: "ventilator",  # Ventilator Type
    223849: "ventilator",  # Ventilator Mode
    229314: "ventilator",  # Ventilator Mode (Hamilton)
    225792: "ventilator",  # Invasive Ventilation
    225794: "ventilator",  # Non-invasive Ventilation
}

# labevents 中可用於 SOFA 的 itemid。
LABEVENT_ITEMIDS = {
    50821: "pao2",
    51265: "platelets",
    50885: "bilirubin",
    50912: "creatinine",
}

# inputevents 中的升壓劑；用於 cardiovascular SOFA。
INPUTEVENT_ITEMIDS = {
    221662: "dopamine",
    221653: "dobutamine",
    221289: "epinephrine",
    229617: "epinephrine",
    221906: "norepinephrine",
}

# outputevents 中的尿量相關 itemid；用於 renal SOFA。
OUTPUTEVENT_ITEMIDS = {
    226557: "urine_output",  # R Ureteral Stent
    226558: "urine_output",  # L Ureteral Stent
    226559: "urine_output",  # Foley
    226560: "urine_output",  # Void
    226561: "urine_output",  # Condom Cath
    226563: "urine_output",  # Suprapubic
    226564: "urine_output",  # R Nephrostomy
    226565: "urine_output",  # L Nephrostomy
    226566: "urine_output",  # Urine and GU Irrigant Out
    226567: "urine_output",  # Straight Cath
    226584: "urine_output",  # Ileoconduit
    226627: "urine_output",  # OR Urine
    226631: "urine_output",  # PACU Urine
    227489: "urine_output",  # GU Irrigant/Urine Volume Out
}

# 同一小時有多筆紀錄時，依照 SOFA worst-value 邏輯做聚合。
CHARTEVENT_AGG = {
    "map": "min",
    "fio2": "max",
    "gcs_eye": "min",
    "gcs_verbal": "min",
    "gcs_motor": "min",
}

# 實驗室數值同樣取 SOFA 計分所需的 worst direction。
LABEVENT_AGG = {
    "pao2": "min",
    "platelets": "min",
    "bilirubin": "max",
    "creatinine": "max",
}

# SOFA 六大器官系統分數欄位。
SCORE_COLUMNS = [
    "respiration_score",
    "coagulation_score",
    "liver_score",
    "cardiovascular_score",
    "cns_score",
    "renal_score",
]

# 後續計分流程需要存在的原始特徵欄位；缺少時會補成 NaN。
RAW_FEATURE_COLUMNS = [
    "map",
    "fio2",
    "gcs_eye",
    "gcs_verbal",
    "gcs_motor",
    "mechanical_vent",
    "pao2",
    "platelets",
    "bilirubin",
    "creatinine",
    "dopamine",
    "dobutamine",
    "epinephrine",
    "norepinephrine",
    "urine_output",
]


def parse_args() -> argparse.Namespace:
    """設定命令列參數，包含資料路徑、輸出檔與 label horizon。"""
    parser = argparse.ArgumentParser(
        description=(
            "Calculate hourly SOFA scores from the local MIMIC-IV ICU subset. "
            "The script uses only tables currently present in dataset/."
        )
    )
    parser.add_argument("--dataset-dir", default=MIMIC_DATA_DIR, help="Folder containing *.csv.gz files.")
    parser.add_argument("--output", default=SOFA_HOURLY_CSV, help="Output CSV path.")
    parser.add_argument("--chunksize", type=int, default=1_000_000, help="CSV chunk size.")
    parser.add_argument(
        "--rolling-hours",
        type=int,
        default=24,
        help="Lookback window used for SOFA worst values.",
    )
    parser.add_argument(
        "--label-horizons",
        nargs="*",
        type=int,
        default=[6, 12, 24],
        help=(
            "Future horizons for SOFA increase >= 2 labels. "
            "Pass --label-horizons with no values to disable labels."
        ),
    )
    parser.add_argument(
        "--max-stays",
        type=int,
        default=None,
        help="Optional small-run limit for testing.",
    )
    parser.add_argument(
        "--min-components",
        type=int,
        default=4,
        choices=range(1, 7),
        metavar="1-6",
        help="Minimum observed SOFA components required for primary labels.",
    )
    return parser.parse_args()


def load_icu_stays(dataset_dir: Path, max_stays: int | None) -> pd.DataFrame:
    """讀取 ICU stay 清單，並計算每個 stay 的最後一個可用小時。"""
    path = dataset_dir / "icustays.csv.gz"
    stays = pd.read_csv(
        path,
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "outtime"],
        parse_dates=["intime", "outtime"],
    )
    stays = stays.dropna(subset=["intime", "outtime"]).copy()
    stays = stays.sort_values(["subject_id", "hadm_id", "intime"])
    if max_stays is not None:
        # 小樣本測試用；正式分析時不要指定 max_stays。
        stays = stays.head(max_stays)

    # sofa_hour 從 ICU intime 開始，以整數小時計算。
    stays["end_hour"] = np.floor(
        (stays["outtime"] - stays["intime"]).dt.total_seconds() / 3600
    ).astype("int64")
    stays = stays[stays["end_hour"] >= 0].copy()

    for col in ["subject_id", "hadm_id", "stay_id"]:
        stays[col] = stays[col].astype("int64")

    return stays.reset_index(drop=True)


def build_hourly_grid(stays: pd.DataFrame) -> pd.DataFrame:
    """建立每個 ICU stay 的逐小時時間軸，作為所有事件資料的對齊基準。"""
    counts = stays["end_hour"].to_numpy(dtype=np.int64) + 1
    row_index = np.repeat(np.arange(len(stays)), counts)
    hour_start = np.repeat(np.cumsum(counts) - counts, counts)
    sofa_hour = np.arange(counts.sum(), dtype=np.int64) - hour_start

    grid = stays.iloc[row_index][["subject_id", "hadm_id", "stay_id", "intime", "outtime"]]
    grid = grid.reset_index(drop=True)
    grid["sofa_hour"] = sofa_hour
    grid["sofa_time"] = grid["intime"] + pd.to_timedelta(grid["sofa_hour"], unit="h")
    return grid


def add_sofa_hour(events: pd.DataFrame, stays: pd.DataFrame, join_cols: list[str]) -> pd.DataFrame:
    """將事件資料對齊到 ICU 入住後第幾小時 sofa_hour。"""
    lookup_cols = join_cols + ["stay_id", "intime", "outtime"]
    if "stay_id" in join_cols:
        lookup_cols = ["stay_id", "intime", "outtime"]

    # 只保留能對回 ICU stay 的事件，並排除 ICU stay 以外的紀錄。
    events = events.merge(stays[lookup_cols].drop_duplicates(), on=join_cols, how="inner")
    events = events[(events["charttime"] >= events["intime"]) & (events["charttime"] <= events["outtime"])]
    if events.empty:
        return events

    events["sofa_hour"] = np.floor(
        (events["charttime"] - events["intime"]).dt.total_seconds() / 3600
    ).astype("int64")
    return events


def normalize_fio2(values: pd.Series) -> pd.Series:
    """將 FiO2 統一成比例值。

    MIMIC 中 FiO2 可能記成 50，也可能記成 0.5；SOFA 的 PaO2/FiO2 ratio
    需要用 0.21 到 1.0 之間的比例值。
    """
    values = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=values.index, dtype="float64")

    fraction = (values >= 0.21) & (values <= 1.0)
    percent = (values > 1.0) & (values <= 100.0)
    out.loc[fraction] = values.loc[fraction]
    out.loc[percent] = values.loc[percent] / 100.0
    return out


def is_ventilator_value(value: object) -> bool:
    """根據文字紀錄判斷該小時是否有機械通氣。"""
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    if not text or text in {"none", "off", "standby", "not applicable", "other/remarks"}:
        return False
    return True


def aggregate_long(rows: list[pd.DataFrame], agg_map: dict[str, str]) -> pd.DataFrame:
    """將事件 long table 依 stay_id 與 sofa_hour 聚合成寬表。"""
    if not rows:
        return pd.DataFrame(columns=["stay_id", "sofa_hour"])

    long = pd.concat(rows, ignore_index=True)
    frames = []
    for feature, agg_func in agg_map.items():
        sub = long[long["feature"] == feature]
        if sub.empty:
            continue
        grouped = (
            sub.groupby(["stay_id", "sofa_hour"], as_index=False)["value"]
            .agg(agg_func)
            .rename(columns={"value": feature})
        )
        frames.append(grouped)

    if not frames:
        return pd.DataFrame(columns=["stay_id", "sofa_hour"])

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on=["stay_id", "sofa_hour"], how="outer")
    return out


def extract_chartevents(dataset_dir: Path, stays: pd.DataFrame, chunksize: int) -> pd.DataFrame:
    """從 chartevents 抽取 MAP、FiO2、GCS 與機械通氣資料。"""
    path = dataset_dir / "chartevents.csv.gz"
    stay_ids = set(stays["stay_id"].tolist())
    numeric_rows: list[pd.DataFrame] = []
    vent_rows: list[pd.DataFrame] = []
    wanted = set(CHARTEVENT_ITEMIDS)

    usecols = ["stay_id", "charttime", "itemid", "value", "valuenum"]
    # chartevents 檔案很大，使用 chunk 分批讀取。
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk = chunk[chunk["itemid"].isin(wanted) & chunk["stay_id"].isin(stay_ids)].copy()
        if chunk.empty:
            continue

        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = add_sofa_hour(chunk, stays, ["stay_id"])
        if chunk.empty:
            continue

        chunk["feature"] = chunk["itemid"].map(CHARTEVENT_ITEMIDS)

        numeric = chunk[chunk["feature"].isin(CHARTEVENT_AGG)].copy()
        if not numeric.empty:
            # MAP、FiO2、GCS 可從 valuenum 取得數值。
            numeric["value"] = pd.to_numeric(numeric["valuenum"], errors="coerce")
            is_fio2 = numeric["feature"] == "fio2"
            numeric.loc[is_fio2, "value"] = normalize_fio2(numeric.loc[is_fio2, "value"])
            numeric = filter_long_frame(numeric)
            numeric = numeric.dropna(subset=["value"])
            numeric_rows.append(numeric[["stay_id", "sofa_hour", "feature", "value"]])

        vent = chunk[chunk["feature"] == "ventilator"].copy()
        if not vent.empty:
            # 機械通氣多為文字類別，因此用 value 欄位判斷是否正在使用呼吸器。
            vent["mechanical_vent"] = vent["value"].map(is_ventilator_value).astype("int8")
            vent = (
                vent.groupby(["stay_id", "sofa_hour"], as_index=False)["mechanical_vent"]
                .max()
            )
            vent_rows.append(vent)

    numeric_out = aggregate_long(numeric_rows, CHARTEVENT_AGG)
    if vent_rows:
        vent_out = (
            pd.concat(vent_rows, ignore_index=True)
            .groupby(["stay_id", "sofa_hour"], as_index=False)["mechanical_vent"]
            .max()
        )
        numeric_out = numeric_out.merge(vent_out, on=["stay_id", "sofa_hour"], how="outer")
    else:
        numeric_out["mechanical_vent"] = np.nan

    return numeric_out


def extract_labevents(dataset_dir: Path, stays: pd.DataFrame, chunksize: int) -> pd.DataFrame:
    """從 labevents 抽取 PaO2、血小板、bilirubin、creatinine。"""
    path = dataset_dir / "labevents.csv.gz"
    stay_hadm = set(stays["hadm_id"].tolist())
    rows: list[pd.DataFrame] = []
    wanted = set(LABEVENT_ITEMIDS)

    usecols = ["subject_id", "hadm_id", "charttime", "itemid", "valuenum"]
    dtypes = {"subject_id": "int64", "hadm_id": "Int64", "itemid": "int64"}
    # labevents 沒有 stay_id，因此用 subject_id + hadm_id 接回 ICU stay。
    for chunk in pd.read_csv(path, usecols=usecols, dtype=dtypes, chunksize=chunksize):
        chunk = chunk[chunk["itemid"].isin(wanted) & chunk["hadm_id"].isin(stay_hadm)].copy()
        if chunk.empty:
            continue

        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime", "hadm_id", "valuenum"])
        chunk["hadm_id"] = chunk["hadm_id"].astype("int64")
        chunk = add_sofa_hour(chunk, stays, ["subject_id", "hadm_id"])
        if chunk.empty:
            continue

        chunk["feature"] = chunk["itemid"].map(LABEVENT_ITEMIDS)
        chunk = chunk.rename(columns={"valuenum": "value"})
        chunk = filter_long_frame(chunk)
        chunk = chunk.dropna(subset=["value"])
        rows.append(chunk[["stay_id", "sofa_hour", "feature", "value"]])

    return aggregate_long(rows, LABEVENT_AGG)


def normalize_vaso_rate(rate: pd.Series, rateuom: pd.Series, weight: pd.Series) -> pd.Series:
    """將升壓劑速率轉成 SOFA 使用的 mcg/kg/min。

    MIMIC 的 rate 單位可能是 mcg/kg/min、mg/kg/hour、mcg/min 等。
    若是未除以體重的單位，會使用 patientweight 轉成每公斤每分鐘劑量。
    """
    rate = pd.to_numeric(rate, errors="coerce")
    weight = pd.to_numeric(weight, errors="coerce")
    unit = rateuom.fillna("").astype(str).str.lower()

    out = pd.Series(np.nan, index=rate.index, dtype="float64")

    direct = unit.str.contains("mcg/kg/min", regex=False)
    mg_kg_min = unit.str.contains("mg/kg/min", regex=False)
    mcg_kg_hour = unit.str.contains("mcg/kg/hour", regex=False) | unit.str.contains(
        "mcg/kg/hr", regex=False
    )
    mg_kg_hour = unit.str.contains("mg/kg/hour", regex=False) | unit.str.contains(
        "mg/kg/hr", regex=False
    )
    mcg_min = unit.str.contains("mcg/min", regex=False) & ~direct
    mg_min = unit.str.contains("mg/min", regex=False) & ~mg_kg_min
    mcg_hour = unit.str.contains("mcg/hour", regex=False) | unit.str.contains("mcg/hr", regex=False)
    mg_hour = unit.str.contains("mg/hour", regex=False) | unit.str.contains("mg/hr", regex=False)
    has_weight = weight > 0

    out.loc[direct] = rate.loc[direct]
    out.loc[mg_kg_min] = rate.loc[mg_kg_min] * 1000.0
    out.loc[mcg_kg_hour] = rate.loc[mcg_kg_hour] / 60.0
    out.loc[mg_kg_hour] = rate.loc[mg_kg_hour] * 1000.0 / 60.0
    out.loc[mcg_min & has_weight] = rate.loc[mcg_min & has_weight] / weight.loc[mcg_min & has_weight]
    out.loc[mg_min & has_weight] = (
        rate.loc[mg_min & has_weight] * 1000.0 / weight.loc[mg_min & has_weight]
    )
    out.loc[mcg_hour & has_weight] = (
        rate.loc[mcg_hour & has_weight] / weight.loc[mcg_hour & has_weight] / 60.0
    )
    out.loc[mg_hour & has_weight] = (
        rate.loc[mg_hour & has_weight] * 1000.0 / weight.loc[mg_hour & has_weight] / 60.0
    )
    return out


def expand_interval_hours(events: pd.DataFrame) -> pd.DataFrame:
    """將 inputevents 的連續給藥區間展開成逐小時紀錄。"""
    if events.empty:
        return pd.DataFrame(columns=["stay_id", "sofa_hour", "feature", "value"])

    counts = (events["end_hour"] - events["start_hour"] + 1).astype("int64")
    counts = counts.clip(lower=0)
    events = events[counts > 0].copy()
    counts = counts[counts > 0].to_numpy(dtype=np.int64)
    if len(events) == 0:
        return pd.DataFrame(columns=["stay_id", "sofa_hour", "feature", "value"])

    repeated_index = np.repeat(np.arange(len(events)), counts)
    start_hours = events["start_hour"].to_numpy(dtype=np.int64)
    hour_offsets = np.concatenate([np.arange(count, dtype=np.int64) for count in counts])

    expanded = events.iloc[repeated_index][["stay_id", "feature", "value"]].reset_index(drop=True)
    expanded["sofa_hour"] = start_hours[repeated_index] + hour_offsets
    return expanded[["stay_id", "sofa_hour", "feature", "value"]]


def extract_inputevents(dataset_dir: Path, stays: pd.DataFrame, chunksize: int) -> pd.DataFrame:
    """從 inputevents 抽取升壓劑，供 cardiovascular SOFA 使用。"""
    path = dataset_dir / "inputevents.csv.gz"
    if not path.exists():
        # 使用者尚未放入 inputevents 時，回傳空表讓主流程可以繼續。
        return pd.DataFrame(columns=["stay_id", "sofa_hour"])

    header = set(pd.read_csv(path, nrows=0).columns)
    requested = [
        "stay_id",
        "starttime",
        "endtime",
        "itemid",
        "rate",
        "rateuom",
        "patientweight",
        "statusdescription",
    ]
    usecols = [col for col in requested if col in header]
    required = {"stay_id", "starttime", "endtime", "itemid", "rate"}
    if not required.issubset(usecols):
        missing = ", ".join(sorted(required - set(usecols)))
        print(f"Skipping inputevents: missing required columns: {missing}")
        return pd.DataFrame(columns=["stay_id", "sofa_hour"])

    stay_ids = set(stays["stay_id"].tolist())
    rows: list[pd.DataFrame] = []
    wanted = set(INPUTEVENT_ITEMIDS)
    lookup = stays[["stay_id", "intime", "outtime"]].drop_duplicates()

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk = chunk[chunk["itemid"].isin(wanted) & chunk["stay_id"].isin(stay_ids)].copy()
        if chunk.empty:
            continue

        if "statusdescription" in chunk.columns:
            # 排除被重寫、取消、暫停或停止的給藥紀錄。
            status = chunk["statusdescription"].fillna("").astype(str).str.lower()
            chunk = chunk[
                ~status.str.contains("rewritten|cancelled|canceled|paused|stopped", regex=True)
            ].copy()
            if chunk.empty:
                continue

        chunk["starttime"] = pd.to_datetime(chunk["starttime"], errors="coerce")
        chunk["endtime"] = pd.to_datetime(chunk["endtime"], errors="coerce")
        chunk = chunk.dropna(subset=["starttime", "endtime"])
        chunk = chunk.merge(lookup, on="stay_id", how="inner")
        chunk["starttime"] = chunk[["starttime", "intime"]].max(axis=1)
        chunk["endtime"] = chunk[["endtime", "outtime"]].min(axis=1)
        chunk = chunk[chunk["starttime"] <= chunk["endtime"]].copy()
        if chunk.empty:
            continue

        if "rateuom" not in chunk.columns:
            chunk["rateuom"] = ""
        if "patientweight" not in chunk.columns:
            chunk["patientweight"] = np.nan

        chunk["value"] = normalize_vaso_rate(
            chunk["rate"], chunk["rateuom"], chunk["patientweight"]
        )
        chunk = chunk.dropna(subset=["value"])
        chunk = chunk[chunk["value"] > 0].copy()
        if chunk.empty:
            continue

        chunk["feature"] = chunk["itemid"].map(INPUTEVENT_ITEMIDS)
        # 單位轉換後再排除不合理升壓劑劑量，避免單位錯誤放大 SOFA 分數。
        for feature in set(INPUTEVENT_ITEMIDS.values()):
            feature_mask = chunk["feature"].eq(feature)
            chunk.loc[feature_mask, "value"] = filter_plausible(
                chunk.loc[feature_mask, "value"], feature
            )
        chunk = chunk.dropna(subset=["value"])
        # 將給藥開始與結束時間換成 ICU stay 內的起迄小時。
        chunk["start_hour"] = np.floor(
            (chunk["starttime"] - chunk["intime"]).dt.total_seconds() / 3600
        ).astype("int64")
        end_seconds = (chunk["endtime"] - chunk["intime"]).dt.total_seconds() - 1
        chunk["end_hour"] = np.floor(end_seconds.clip(lower=0) / 3600).astype("int64")

        rows.append(expand_interval_hours(chunk[["stay_id", "start_hour", "end_hour", "feature", "value"]]))

    return aggregate_long(
        rows,
        {
            "dopamine": "max",
            "dobutamine": "max",
            "epinephrine": "max",
            "norepinephrine": "max",
        },
    )


def extract_outputevents(dataset_dir: Path, stays: pd.DataFrame, chunksize: int) -> pd.DataFrame:
    """從 outputevents 抽取尿量，供 renal SOFA 使用。"""
    path = dataset_dir / "outputevents.csv.gz"
    if not path.exists():
        # 使用者尚未放入 outputevents 時，回傳空表讓主流程可以繼續。
        return pd.DataFrame(columns=["stay_id", "sofa_hour"])

    header = set(pd.read_csv(path, nrows=0).columns)
    requested = ["stay_id", "charttime", "itemid", "value"]
    usecols = [col for col in requested if col in header]
    required = {"stay_id", "charttime", "itemid", "value"}
    if not required.issubset(usecols):
        missing = ", ".join(sorted(required - set(usecols)))
        print(f"Skipping outputevents: missing required columns: {missing}")
        return pd.DataFrame(columns=["stay_id", "sofa_hour"])

    stay_ids = set(stays["stay_id"].tolist())
    rows: list[pd.DataFrame] = []
    wanted = set(OUTPUTEVENT_ITEMIDS)

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk = chunk[chunk["itemid"].isin(wanted) & chunk["stay_id"].isin(stay_ids)].copy()
        if chunk.empty:
            continue

        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime", "value"])
        chunk = chunk[chunk["value"] >= 0].copy()
        chunk = add_sofa_hour(chunk, stays, ["stay_id"])
        if chunk.empty:
            continue

        urine = (
            # 同一小時可能有多個尿量來源，全部加總成該小時尿量。
            chunk.groupby(["stay_id", "sofa_hour"], as_index=False)["value"]
            .sum()
            .rename(columns={"value": "urine_output"})
        )
        rows.append(urine)

    if not rows:
        return pd.DataFrame(columns=["stay_id", "sofa_hour", "urine_output"])

    return (
        pd.concat(rows, ignore_index=True)
        .groupby(["stay_id", "sofa_hour"], as_index=False)["urine_output"]
        .sum()
    )


def rolling_by_stay(
    df: pd.DataFrame, source: str, target: str, window: int, method: str
) -> None:
    """在同一個 stay_id 內計算過去 window 小時的 rolling min/max/sum。"""
    grouped = df.groupby("stay_id", group_keys=False)[source]
    rolling = grouped.rolling(window=window, min_periods=1)
    if method == "min":
        result = rolling.min()
    elif method == "max":
        result = rolling.max()
    elif method == "sum":
        result = rolling.sum()
    else:
        raise ValueError(f"Unsupported rolling method: {method}")
    df[target] = result.reset_index(level=0, drop=True)


def score_respiration(ratio: pd.Series, mechanical_vent: pd.Series) -> pd.Series:
    """計算 respiratory SOFA：依 PaO2/FiO2 ratio 與是否機械通氣給分。"""
    score = pd.Series(np.nan, index=ratio.index, dtype="float64")
    has_ratio = ratio.notna()
    score.loc[has_ratio] = 0
    score.loc[has_ratio & (ratio < 400)] = 1
    score.loc[has_ratio & (ratio < 300)] = 2
    score.loc[has_ratio & (ratio < 200) & (mechanical_vent == 1)] = 3
    score.loc[has_ratio & (ratio < 100) & (mechanical_vent == 1)] = 4
    return score


def score_platelets(platelets: pd.Series) -> pd.Series:
    """計算 coagulation SOFA：血小板越低分數越高。"""
    score = pd.Series(np.nan, index=platelets.index, dtype="float64")
    present = platelets.notna()
    score.loc[present] = 0
    score.loc[present & (platelets < 150)] = 1
    score.loc[present & (platelets < 100)] = 2
    score.loc[present & (platelets < 50)] = 3
    score.loc[present & (platelets < 20)] = 4
    return score


def score_bilirubin(bilirubin: pd.Series) -> pd.Series:
    """計算 liver SOFA：bilirubin 越高分數越高。"""
    score = pd.Series(np.nan, index=bilirubin.index, dtype="float64")
    present = bilirubin.notna()
    score.loc[present] = 0
    score.loc[present & (bilirubin >= 1.2)] = 1
    score.loc[present & (bilirubin >= 2.0)] = 2
    score.loc[present & (bilirubin >= 6.0)] = 3
    score.loc[present & (bilirubin >= 12.0)] = 4
    return score


def score_cardiovascular(
    map_value: pd.Series,
    dopamine: pd.Series,
    dobutamine: pd.Series,
    epinephrine: pd.Series,
    norepinephrine: pd.Series,
) -> pd.Series:
    """計算 cardiovascular SOFA：整合 MAP 與升壓劑劑量。"""
    score = pd.Series(np.nan, index=map_value.index, dtype="float64")
    vaso_present = (
        dopamine.notna()
        | dobutamine.notna()
        | epinephrine.notna()
        | norepinephrine.notna()
    )
    present = map_value.notna() | vaso_present
    score.loc[present] = 0
    score.loc[present & (map_value < 70)] = 1

    # 缺少升壓劑紀錄時，不代表有完整用藥資訊；只有在判斷門檻時暫時視為 0。
    dopamine = dopamine.fillna(0)
    dobutamine = dobutamine.fillna(0)
    epinephrine = epinephrine.fillna(0)
    norepinephrine = norepinephrine.fillna(0)

    score.loc[present & ((dopamine > 0) & (dopamine <= 5) | (dobutamine > 0))] = np.maximum(
        score.loc[present & ((dopamine > 0) & (dopamine <= 5) | (dobutamine > 0))],
        2,
    )
    level_3 = (
        ((dopamine > 5) & (dopamine <= 15))
        | ((epinephrine > 0) & (epinephrine <= 0.1))
        | ((norepinephrine > 0) & (norepinephrine <= 0.1))
    )
    score.loc[present & level_3] = np.maximum(score.loc[present & level_3], 3)
    level_4 = (dopamine > 15) | (epinephrine > 0.1) | (norepinephrine > 0.1)
    score.loc[present & level_4] = 4
    return score


def score_gcs(gcs: pd.Series) -> pd.Series:
    """計算 CNS SOFA：GCS 越低分數越高。"""
    score = pd.Series(np.nan, index=gcs.index, dtype="float64")
    present = gcs.notna()
    score.loc[present] = 0
    score.loc[present & (gcs < 15)] = 1
    score.loc[present & (gcs <= 12)] = 2
    score.loc[present & (gcs <= 9)] = 3
    score.loc[present & (gcs < 6)] = 4
    return score


def score_creatinine(creatinine: pd.Series) -> pd.Series:
    """依 creatinine 計算 renal SOFA 的腎功能分數。"""
    score = pd.Series(np.nan, index=creatinine.index, dtype="float64")
    present = creatinine.notna()
    score.loc[present] = 0
    score.loc[present & (creatinine >= 1.2)] = 1
    score.loc[present & (creatinine >= 2.0)] = 2
    score.loc[present & (creatinine >= 3.5)] = 3
    score.loc[present & (creatinine >= 5.0)] = 4
    return score


def score_renal(creatinine: pd.Series, urine_output_24h: pd.Series) -> pd.Series:
    """計算 renal SOFA：creatinine 與 24 小時尿量兩者取較嚴重分數。"""
    creatinine_score = score_creatinine(creatinine)
    urine_score = pd.Series(np.nan, index=urine_output_24h.index, dtype="float64")
    urine_present = urine_output_24h.notna()
    urine_score.loc[urine_present] = 0
    urine_score.loc[urine_present & (urine_output_24h < 500)] = 3
    urine_score.loc[urine_present & (urine_output_24h < 200)] = 4

    return pd.concat([creatinine_score, urine_score], axis=1).max(axis=1, skipna=True)


def add_sofa_scores(
    df: pd.DataFrame,
    rolling_hours: int,
    minimum_components: int = 4,
) -> pd.DataFrame:
    """根據逐小時特徵計算 SOFA component scores 與總分。"""
    df = df.sort_values(["stay_id", "sofa_hour"]).reset_index(drop=True)

    for col in RAW_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["fio2"] = df.groupby("stay_id")["fio2"].ffill()
    df["mechanical_vent"] = df["mechanical_vent"].fillna(0)

    # 尿量要計算 rolling sum；缺值先補 0，但會另外記錄該 window 是否真的有觀察到尿量。
    urine_observed = df["urine_output"].notna().astype("int8")
    df["urine_output_for_sum"] = df["urine_output"].fillna(0)

    component_gcs = df[["gcs_eye", "gcs_verbal", "gcs_motor"]].sum(axis=1, min_count=3)
    if "gcs_total" in df.columns:
        # eICU 直接提供 total GCS；MIMIC 則沿用 eye/verbal/motor 三分項加總。
        df["gcs_total"] = pd.to_numeric(df["gcs_total"], errors="coerce").combine_first(
            component_gcs
        )
    else:
        df["gcs_total"] = component_gcs
    df["pao2_fio2"] = np.where(
        df["pao2"].notna() & df["fio2"].notna() & (df["fio2"] > 0),
        df["pao2"] / df["fio2"],
        np.nan,
    )

    # 各 component 都採用過去 rolling_hours 小時內的 worst value。
    rolling_by_stay(df, "pao2_fio2", "pao2_fio2_min_24h", rolling_hours, "min")
    rolling_by_stay(df, "mechanical_vent", "mechanical_vent_24h", rolling_hours, "max")
    rolling_by_stay(df, "platelets", "platelets_min_24h", rolling_hours, "min")
    rolling_by_stay(df, "bilirubin", "bilirubin_max_24h", rolling_hours, "max")
    rolling_by_stay(df, "map", "map_min_24h", rolling_hours, "min")
    rolling_by_stay(df, "gcs_total", "gcs_min_24h", rolling_hours, "min")
    rolling_by_stay(df, "creatinine", "creatinine_max_24h", rolling_hours, "max")
    rolling_by_stay(df, "dopamine", "dopamine_max_24h", rolling_hours, "max")
    rolling_by_stay(df, "dobutamine", "dobutamine_max_24h", rolling_hours, "max")
    rolling_by_stay(df, "epinephrine", "epinephrine_max_24h", rolling_hours, "max")
    rolling_by_stay(df, "norepinephrine", "norepinephrine_max_24h", rolling_hours, "max")
    rolling_by_stay(df, "urine_output_for_sum", "urine_output_sum_24h", rolling_hours, "sum")

    df["urine_observed"] = urine_observed
    rolling_by_stay(df, "urine_observed", "urine_observed_24h", rolling_hours, "max")
    df.loc[df["urine_observed_24h"] == 0, "urine_output_sum_24h"] = np.nan
    # 不足完整 rolling window 的早期小時，先不用尿量判 renal SOFA，避免假性高分。
    df.loc[df["sofa_hour"] < rolling_hours - 1, "urine_output_sum_24h"] = np.nan

    # 依六大器官系統分別計分，再加總成 SOFA score。
    df["respiration_score"] = score_respiration(df["pao2_fio2_min_24h"], df["mechanical_vent_24h"])
    df["coagulation_score"] = score_platelets(df["platelets_min_24h"])
    df["liver_score"] = score_bilirubin(df["bilirubin_max_24h"])
    df["cardiovascular_score"] = score_cardiovascular(
        df["map_min_24h"],
        df["dopamine_max_24h"],
        df["dobutamine_max_24h"],
        df["epinephrine_max_24h"],
        df["norepinephrine_max_24h"],
    )
    df["cns_score"] = score_gcs(df["gcs_min_24h"])
    df["renal_score"] = score_renal(df["creatinine_max_24h"], df["urine_output_sum_24h"])

    df["sofa_component_count"] = df[SCORE_COLUMNS].notna().sum(axis=1)
    df["sofa_score_observed"] = df[SCORE_COLUMNS].sum(axis=1, min_count=1)
    # 保留 missing-as-normal 與 complete-case 版本供敏感度分析。
    df["sofa_score_assume_normal"] = df[SCORE_COLUMNS].fillna(0).sum(axis=1)
    df["sofa_score_complete"] = df[SCORE_COLUMNS].sum(
        axis=1, min_count=len(SCORE_COLUMNS)
    )
    # 主要 outcome 僅使用至少有指定數量器官分數的時點。
    df["sofa_score"] = df["sofa_score_assume_normal"].where(
        df["sofa_component_count"] >= minimum_components
    )
    return df


def add_future_labels(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """產生未來 H 小時內 SOFA 是否上升至少 2 分的 binary label。"""
    if not horizons:
        return df

    df = df.sort_values(["stay_id", "sofa_hour"]).reset_index(drop=True)
    stay_end_hour = df.groupby("stay_id")["sofa_hour"].transform("max")
    remaining_hours = stay_end_hour - df["sofa_hour"]

    for horizon in horizons:
        # shift(-1) 代表 label 不包含當下這一小時，只看真正的未來視窗。
        future_max = (
            df.groupby("stay_id", group_keys=False)["sofa_score"]
            .apply(lambda s: s.shift(-1).iloc[::-1].rolling(horizon, min_periods=1).max().iloc[::-1])
        )
        increase_col = f"sofa_increase_{horizon}h"
        label_col = f"label_sofa_increase_ge2_{horizon}h"
        df[f"future_{horizon}h_max_sofa"] = future_max
        df[increase_col] = future_max - df["sofa_score"]
        label = (df[increase_col] >= 2).astype("Int64")
        invalid_score = df["sofa_score"].isna() | future_max.isna()
        # stay 結尾不足完整預測 horizon 的樣本不標 label，避免右側截尾造成誤標。
        label.loc[(remaining_hours < horizon) | invalid_score] = pd.NA
        df[label_col] = label

    return df


def write_sofa_quality_report(
    df: pd.DataFrame,
    output_path: Path,
    minimum_components: int,
) -> Path:
    """輸出 component completeness 與各 horizon 標籤盛行率。"""
    component_counts = {
        str(int(key)): int(value)
        for key, value in df["sofa_component_count"].value_counts().sort_index().items()
    }
    labels = {}
    for col in [name for name in df.columns if name.startswith("label_sofa_increase_ge2_")]:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        labels[col] = {
            "valid_rows": int(len(values)),
            "prevalence": float(values.mean()) if len(values) else None,
        }

    report_path = output_path.with_name(f"{output_path.stem}_quality.json")
    report_path.write_text(
        json.dumps(
            {
                "rows": len(df),
                "stays": int(df["stay_id"].nunique()),
                "minimum_components_for_primary_label": minimum_components,
                "component_count_distribution": component_counts,
                "labels": labels,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report_path


def main() -> None:
    """SOFA 計算主流程。"""
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output)

    stays = load_icu_stays(dataset_dir, args.max_stays)
    print(f"Loaded {len(stays):,} ICU stays.")

    # 先建立逐小時 grid，再把各來源資料 merge 到同一張表。
    grid = build_hourly_grid(stays)
    print(f"Built {len(grid):,} stay-hour rows.")

    print("Extracting chartevents...")
    char_features = extract_chartevents(dataset_dir, stays, args.chunksize)
    print(f"Chartevents feature rows: {len(char_features):,}")

    print("Extracting labevents...")
    lab_features = extract_labevents(dataset_dir, stays, args.chunksize)
    print(f"Labevents feature rows: {len(lab_features):,}")

    print("Extracting inputevents...")
    input_features = extract_inputevents(dataset_dir, stays, args.chunksize)
    print(f"Inputevents feature rows: {len(input_features):,}")

    print("Extracting outputevents...")
    output_features = extract_outputevents(dataset_dir, stays, args.chunksize)
    print(f"Outputevents feature rows: {len(output_features):,}")

    df = grid.merge(char_features, on=["stay_id", "sofa_hour"], how="left")
    df = df.merge(lab_features, on=["stay_id", "sofa_hour"], how="left")
    df = df.merge(input_features, on=["stay_id", "sofa_hour"], how="left")
    df = df.merge(output_features, on=["stay_id", "sofa_hour"], how="left")
    df = add_sofa_scores(df, args.rolling_hours, args.min_components)
    df = add_future_labels(df, args.label_horizons)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    quality_path = write_sofa_quality_report(df, output_path, args.min_components)
    print(f"Wrote {output_path}")
    print(f"Quality report: {quality_path}")

    notes = []
    if not (dataset_dir / "inputevents.csv.gz").exists():
        notes.append("cardiovascular score uses MAP only because inputevents.csv.gz is absent")
    if not (dataset_dir / "outputevents.csv.gz").exists():
        notes.append("renal score uses creatinine only because outputevents.csv.gz is absent")
    if notes:
        print("Note: dataset-limited SOFA calculation: " + "; ".join(notes) + ".")


if __name__ == "__main__":
    main()
