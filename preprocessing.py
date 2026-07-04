"""研究計畫二版用的資料前處理流程。

這支程式負責把 MIMIC-IV ICU 原始資料整理成「每個 ICU stay、每小時一列」的模型輸入表。
重點包含：
1. 以 stay_id 對齊資料，避免同一病人不同 ICU stay 被混在一起。
2. 將生命徵象與實驗室檢驗資料對齊到 ICU 入住後第幾小時 sofa_hour。
3. 只使用預測時間點以前的資料做 forward-fill，避免資料洩漏。
4. 產生過去 4、6、12 小時的 temporal features。
5. 若已先跑 sofa_score.py，會合併 SOFA score 與 SOFA increase label。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_data_quality import fahrenheit_to_celsius, filter_long_frame
from project_config import (
    DEFAULT_OBSERVATION_WINDOWS,
    MIMIC_DATA_DIR,
    PRIMARY_HOURLY_FEATURES,
    SOFA_HOURLY_CSV,
)
from sofa_score import (
    add_sofa_hour,
    build_hourly_grid,
    is_ventilator_value,
    load_icu_stays,
    normalize_fio2,
)


# 研究計畫二版的 chartevents 候選變數。
# key 是 MIMIC-IV itemid，value 是整理後輸出的標準欄位名稱。
CHARTEVENT_ITEMIDS = {
    220045: "heart_rate",
    220210: "respiratory_rate",
    220277: "spo2",
    220050: "sbp_arterial",
    220179: "sbp_noninvasive",
    220051: "dbp_arterial",
    220180: "dbp_noninvasive",
    220052: "map_arterial",
    220181: "map_noninvasive",
    223761: "temperature_c",
    223762: "temperature_c",
    223835: "fio2",
    220739: "gcs_eye",
    223900: "gcs_verbal",
    223901: "gcs_motor",
    223848: "mechanical_vent",  # Ventilator Type
    223849: "mechanical_vent",  # Ventilator Mode
    229314: "mechanical_vent",  # Ventilator Mode (Hamilton)
}

# 研究計畫二版的 labevents 候選變數。
# 這些欄位同時支援 SOFA 計分與模型特徵。
LABEVENT_ITEMIDS = {
    50813: "lactate",
    50821: "pao2",
    50885: "bilirubin",
    50912: "creatinine",
    51265: "platelets",
}

# 同一個 stay_id、同一小時內若有多筆紀錄，使用臨床上較保守或代表性的聚合方式。
# 例如 SpO2 取最低值、FiO2 取最高值、GCS 各分項取最低值。
CHARTEVENT_AGG = {
    "heart_rate": "mean",
    "respiratory_rate": "mean",
    "spo2": "min",
    "sbp_arterial": "min",
    "sbp_noninvasive": "min",
    "dbp_arterial": "min",
    "dbp_noninvasive": "min",
    "map_arterial": "min",
    "map_noninvasive": "min",
    "temperature_c": "mean",
    "fio2": "max",
    "gcs_eye": "min",
    "gcs_verbal": "min",
    "gcs_motor": "min",
    "mechanical_vent": "max",
}

# 實驗室檢驗也採用 SOFA/風險判斷常用的 worst-value 聚合方向。
LABEVENT_AGG = {
    "lactate": "max",
    "pao2": "min",
    "bilirubin": "max",
    "creatinine": "max",
    "platelets": "min",
}

# 模型主要使用的基礎特徵欄位；後續會為這些欄位產生 temporal features。
BASE_FEATURES = [
    "heart_rate",
    "respiratory_rate",
    "spo2",
    "sbp",
    "dbp",
    "map",
    "temperature_c",
    "fio2",
    "gcs_total",
    "mechanical_vent",
    "lactate",
    "pao2",
    "pao2_fio2",
    "bilirubin",
    "creatinine",
    "platelets",
]

# 異常範圍用於 abnormal duration/frequency。界值綜合 NEWS2、SOFA 與常用 ICU
# 參考範圍；目的為描述異常暴露，而不是建立新的 outcome label。
ABNORMAL_RANGES: dict[str, dict[str, float]] = {
    "heart_rate": {"low": 51.0, "high": 90.0},
    "respiratory_rate": {"low": 12.0, "high": 20.0},
    "spo2": {"low": 96.0},
    "sbp": {"low": 111.0, "high": 219.0},
    "dbp": {"low": 60.0, "high": 90.0},
    "map": {"low": 70.0},
    "temperature_c": {"low": 36.1, "high": 38.0},
    "fio2": {"high": 0.21},
    "gcs_total": {"low": 15.0},
    "mechanical_vent": {"high": 0.0},
    "lactate": {"high": 2.0},
    "pao2": {"low": 80.0},
    "pao2_fio2": {"low": 300.0},
    "bilirubin": {"high": 1.2},
    "creatinine": {"high": 1.2},
    "platelets": {"low": 150.0},
}

# 從 sofa_score.py 輸出的欄位中，需要合併回模型資料表的 SOFA 與 label 欄位。
SOFA_COLUMNS = [
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
    "label_sofa_increase_ge2_6h",
    "label_sofa_increase_ge2_12h",
    "label_sofa_increase_ge2_24h",
]


def parse_args() -> argparse.Namespace:
    """設定命令列參數，方便切換資料夾、輸出檔與 temporal window 長度。"""
    parser = argparse.ArgumentParser(
        description=(
            "Build a leakage-free, stay-level hourly feature table for the proposal v2. "
            "Run sofa_score.py first if you want SOFA outcome labels merged in."
        )
    )
    parser.add_argument("--dataset-dir", default=MIMIC_DATA_DIR, help="Folder containing *.csv.gz files.")
    parser.add_argument("--sofa-path", default=SOFA_HOURLY_CSV, help="SOFA output CSV path.")
    parser.add_argument("--output", default=PRIMARY_HOURLY_FEATURES, help="Output CSV path.")
    parser.add_argument("--chunksize", type=int, default=1_000_000, help="CSV chunk size.")
    parser.add_argument("--windows", nargs="*", type=int, default=list(DEFAULT_OBSERVATION_WINDOWS))
    parser.add_argument("--max-stays", type=int, default=None, help="Optional small-run limit.")
    parser.add_argument(
        "--no-temporal-features",
        action="store_true",
        help="Only output current hourly values and labels.",
    )
    return parser.parse_args()


def aggregate_long(rows: list[pd.DataFrame], agg_map: dict[str, str]) -> pd.DataFrame:
    """將 long format 的事件資料依 stay_id 與 sofa_hour 聚合成 wide format。

    原始 chartevents/labevents 是一筆事件一列；模型訓練較適合一小時一列、
    每個特徵一欄，因此這裡會依照 agg_map 指定的方式做 pivot-like 聚合。
    """
    if not rows:
        return pd.DataFrame(columns=["stay_id", "sofa_hour"])

    long = pd.concat(rows, ignore_index=True)
    frames = []
    for feature, agg_func in agg_map.items():
        sub = long[long["feature"] == feature]
        if sub.empty:
            continue
        frame = (
            sub.groupby(["stay_id", "sofa_hour"], as_index=False)["value"]
            .agg(agg_func)
            .rename(columns={"value": feature})
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["stay_id", "sofa_hour"])

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on=["stay_id", "sofa_hour"], how="outer")
    return out


def extract_chartevents(dataset_dir: Path, stays: pd.DataFrame, chunksize: int) -> pd.DataFrame:
    """讀取 chartevents，抽出生命徵象、GCS、FiO2 與機械通氣相關特徵。"""
    path = dataset_dir / "chartevents.csv.gz"
    stay_ids = set(stays["stay_id"].tolist())
    rows: list[pd.DataFrame] = []
    wanted = set(CHARTEVENT_ITEMIDS)
    usecols = ["stay_id", "charttime", "itemid", "value", "valuenum"]

    # chartevents 很大，使用 chunksize 分批讀取，避免一次吃掉太多記憶體。
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        # 只保留研究計畫需要的 itemid，以及本次 ICU stay 清單中的資料。
        chunk = chunk[chunk["itemid"].isin(wanted) & chunk["stay_id"].isin(stay_ids)].copy()
        if chunk.empty:
            continue

        # 將 charttime 轉成 ICU 入住後第幾小時 sofa_hour，並排除 ICU stay 外的紀錄。
        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk = add_sofa_hour(chunk, stays, ["stay_id"])
        if chunk.empty:
            continue

        chunk["feature"] = chunk["itemid"].map(CHARTEVENT_ITEMIDS)
        raw_value = chunk["value"].copy()
        vent = chunk["feature"].eq("mechanical_vent")
        chunk["value"] = pd.to_numeric(chunk["valuenum"], errors="coerce").astype("float64")
        # 機械通氣欄位有時是文字紀錄，若 valuenum 缺值則改用文字判斷是否正在通氣。
        # 呼吸器的 valuenum 是 mode code，不是二元狀態；一律由原始文字轉為 0/1。
        chunk.loc[vent, "value"] = raw_value.loc[vent].map(
            is_ventilator_value
        ).astype("float64")

        # 將華氏體溫轉成攝氏，與 itemid 223762 的攝氏紀錄合併。
        temperature_f = chunk["itemid"].eq(223761)
        chunk.loc[temperature_f, "value"] = fahrenheit_to_celsius(
            chunk.loc[temperature_f, "valuenum"]
        )

        fio2 = chunk["feature"].eq("fio2")
        # MIMIC 中 FiO2 可能以 50 或 0.5 表示，統一轉成 0.21 到 1.0 的比例。
        chunk.loc[fio2, "value"] = normalize_fio2(chunk.loc[fio2, "value"])
        # 聚合與 forward-fill 前先排除不可能數值，避免錯值向後傳播。
        chunk = filter_long_frame(chunk)
        chunk = chunk.dropna(subset=["value"])
        rows.append(chunk[["stay_id", "sofa_hour", "feature", "value"]])

    return aggregate_long(rows, CHARTEVENT_AGG)


def extract_labevents(dataset_dir: Path, stays: pd.DataFrame, chunksize: int) -> pd.DataFrame:
    """讀取 labevents，抽出 lactate、PaO2、bilirubin、creatinine、platelets。"""
    path = dataset_dir / "labevents.csv.gz"
    hadm_ids = set(stays["hadm_id"].tolist())
    rows: list[pd.DataFrame] = []
    wanted = set(LABEVENT_ITEMIDS)
    usecols = ["subject_id", "hadm_id", "charttime", "itemid", "valuenum"]
    dtypes = {"subject_id": "int64", "hadm_id": "Int64", "itemid": "int64"}

    # labevents 沒有 stay_id，因此先用 subject_id + hadm_id 接回 ICU stay。
    for chunk in pd.read_csv(path, usecols=usecols, dtype=dtypes, chunksize=chunksize):
        chunk = chunk[chunk["itemid"].isin(wanted) & chunk["hadm_id"].isin(hadm_ids)].copy()
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


def coalesce_clinical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """合併同類型欄位，形成模型可直接使用的臨床特徵。

    例如動脈血壓與非侵入血壓都可提供 SBP/MAP，這裡優先使用動脈血壓，
    若缺值再用非侵入血壓補上。
    """
    for col in CHARTEVENT_AGG | LABEVENT_AGG:
        if col not in df.columns:
            df[col] = np.nan

    df["sbp"] = df["sbp_arterial"].combine_first(df["sbp_noninvasive"])
    df["dbp"] = df["dbp_arterial"].combine_first(df["dbp_noninvasive"])
    df["map"] = df["map_arterial"].combine_first(df["map_noninvasive"])
    df["gcs_total"] = df[["gcs_eye", "gcs_verbal", "gcs_motor"]].sum(axis=1, min_count=3)
    # PaO2/FiO2 ratio 是 SOFA respiratory component 的核心指標，也可作為模型特徵。
    df["pao2_fio2"] = np.where(
        df["pao2"].notna() & df["fio2"].notna() & (df["fio2"] > 0),
        df["pao2"] / df["fio2"],
        np.nan,
    )
    return df


def abnormal_indicator(values: pd.Series, feature: str) -> pd.Series:
    """依預先定義的臨床界值回傳異常狀態；缺值不視為異常。"""
    limits = ABNORMAL_RANGES.get(feature)
    if limits is None:
        return pd.Series(False, index=values.index)
    numeric = pd.to_numeric(values, errors="coerce")
    abnormal = pd.Series(False, index=values.index)
    if "low" in limits:
        abnormal |= numeric < limits["low"]
    if "high" in limits:
        abnormal |= numeric > limits["high"]
    return abnormal & numeric.notna()


def add_measurement_process_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    observed_mask: pd.DataFrame,
) -> pd.DataFrame:
    """加入當下缺值與距離上次真實量測的時間，且不使用未來資料。"""
    stay_ids = df["stay_id"]
    hours = pd.to_numeric(df["sofa_hour"], errors="coerce").astype("float32")
    columns: dict[str, pd.Series] = {}
    for col in feature_cols:
        observed = observed_mask[col].astype(bool)
        columns[f"{col}_is_missing"] = (~observed).astype("uint8")
        last_observed_hour = hours.where(observed).groupby(stay_ids, sort=False).ffill()
        time_since = hours - last_observed_hour
        # 尚未量測者以 ICU 入住後已等待的小時數表示，不使用未來第一次量測。
        time_since = time_since.where(last_observed_hour.notna(), hours + 1.0)
        columns[f"{col}_time_since_last_measurement_h"] = time_since.astype("float32")
    return pd.concat([df, pd.DataFrame(columns, index=df.index)], axis=1)


def leakage_free_forward_fill(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """只在同一個 stay_id 內做 forward-fill，避免用未來資料補現在。

    這裡刻意不使用 backward-fill，因為 bfill 會把預測時間點之後才出現的檢查值
    補回較早時間點，造成資料洩漏。
    """
    df = df.sort_values(["stay_id", "sofa_hour"]).reset_index(drop=True)
    df[feature_cols] = df.groupby("stay_id", group_keys=False)[feature_cols].ffill()
    return df


def add_temporal_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    windows: list[int],
    observed_mask: pd.DataFrame,
) -> pd.DataFrame:
    """根據過去數小時的歷史資料產生 temporal features。

    每個 window 產生 mean、min、max、std、slope、change、異常持續時間與
    真實量測中的異常頻率。observed_mask 保留補值前的狀態，避免把 LOCF
    複製值誤當成新量測。
    """
    df = df.sort_values(["stay_id", "sofa_hour"]).reset_index(drop=True)
    stay_ids = df["stay_id"]

    # Short-term change 定義為目前小時相對前一小時 LOCF 狀態的變化。
    # 這與 window-level change（目前值減去觀察窗起點）分開保留。
    short_term_cols: dict[str, pd.Series] = {}
    for col in feature_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        previous = values.groupby(stay_ids, sort=False).shift(1)
        short_term_cols[f"{col}_short_term_change"] = (values - previous).astype("float32")
    if short_term_cols:
        df = pd.concat([df, pd.DataFrame(short_term_cols, index=df.index)], axis=1)

    for window in windows:
        temporal_cols: dict[str, pd.Series] = {}
        for col in feature_cols:
            # rolling 只往回看目前時間點以前的資料，不會看未來。
            values = pd.to_numeric(df[col], errors="coerce")
            grouped_values = values.groupby(stay_ids, sort=False)
            rolling = grouped_values.rolling(window=window, min_periods=1)
            temporal_cols[f"{col}_w{window}h_mean"] = rolling.mean().reset_index(level=0, drop=True).astype("float32")
            temporal_cols[f"{col}_w{window}h_min"] = rolling.min().reset_index(level=0, drop=True).astype("float32")
            temporal_cols[f"{col}_w{window}h_max"] = rolling.max().reset_index(level=0, drop=True).astype("float32")
            temporal_cols[f"{col}_w{window}h_std"] = (
                rolling.std().reset_index(level=0, drop=True).fillna(0).astype("float32")
            )
            prior = grouped_values.shift(window - 1)
            change = values - prior
            temporal_cols[f"{col}_w{window}h_change"] = change.astype("float32")
            temporal_cols[f"{col}_w{window}h_slope"] = (
                change / max(window - 1, 1)
            ).astype("float32")

            abnormal_filled = abnormal_indicator(values, col).astype("float32")
            abnormal_duration = (
                abnormal_filled.groupby(stay_ids, sort=False)
                .rolling(window=window, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )
            temporal_cols[f"{col}_w{window}h_abnormal_duration"] = abnormal_duration.astype("float32")

            observed = observed_mask[col].astype(bool)
            observed_count = (
                observed.astype("float32")
                .groupby(stay_ids, sort=False)
                .rolling(window=window, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )
            abnormal_observed = (abnormal_indicator(values, col) & observed).astype("float32")
            abnormal_count = (
                abnormal_observed.groupby(stay_ids, sort=False)
                .rolling(window=window, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )
            frequency = abnormal_count.div(observed_count.where(observed_count > 0))
            temporal_cols[f"{col}_w{window}h_abnormal_frequency"] = frequency.astype("float32")

        if temporal_cols:
            # 每次只保留一個 window 的暫存欄位，控制大型 hourly table 的峰值記憶體。
            df = pd.concat([df, pd.DataFrame(temporal_cols, index=df.index)], axis=1)
    return df


def merge_sofa_labels(df: pd.DataFrame, sofa_path: Path) -> pd.DataFrame:
    """合併 sofa_score.py 產生的 SOFA 分數與未來 6/12/24 小時惡化標籤。"""
    if not sofa_path.exists():
        print(f"SOFA file not found: {sofa_path}. Output will not include SOFA labels.")
        return df

    sofa_header = pd.read_csv(sofa_path, nrows=0).columns.tolist()
    cols = ["stay_id", "sofa_hour"] + [col for col in SOFA_COLUMNS if col in sofa_header]
    sofa = pd.read_csv(sofa_path, usecols=cols)
    return df.merge(sofa, on=["stay_id", "sofa_hour"], how="left")


def write_quality_report(
    df: pd.DataFrame,
    feature_cols: list[str],
    output_path: Path,
) -> Path:
    """輸出重建後的覆蓋率與數值範圍，供論文資料品質表使用。"""
    feature_report = {}
    for col in feature_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        observed = int(values.notna().sum())
        feature_report[col] = {
            "observed_rows": observed,
            "observed_fraction": observed / max(len(df), 1),
            "minimum": float(values.min()) if observed else None,
            "maximum": float(values.max()) if observed else None,
        }

    label_report = {}
    for col in [name for name in SOFA_COLUMNS if name.startswith("label_") and name in df]:
        values = pd.to_numeric(df[col], errors="coerce")
        valid = values.dropna()
        label_report[col] = {
            "valid_rows": int(len(valid)),
            "prevalence": float(valid.mean()) if len(valid) else None,
        }

    report_path = output_path.with_name(f"{output_path.stem}_quality.json")
    report_path.write_text(
        json.dumps(
            {
                "rows": len(df),
                "stays": int(df["stay_id"].nunique()),
                "features": feature_report,
                "labels": label_report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report_path


def write_feature_manifest(
    output_path: Path,
    feature_cols: list[str],
    windows: list[int],
) -> Path:
    """記錄 temporal feature 定義，供研究方法與敏感度分析重現。"""
    manifest_path = output_path.with_name(f"{output_path.stem}_feature_manifest.json")
    manifest = {
        "observation_windows_hours": windows,
        "base_features": feature_cols,
        "measurement_process_features": {
            "is_missing": "1 when no raw measurement exists at the current ICU hour; computed before LOCF",
            "time_since_last_measurement_h": "hours since the most recent raw measurement; no future backfill",
        },
        "rolling_features": {
            "short_term_change": "current LOCF value minus the previous ICU hour",
            "mean": "rolling mean including current hour",
            "min": "rolling minimum including current hour",
            "max": "rolling maximum including current hour",
            "std": "rolling sample standard deviation; one observation is set to zero",
            "change": "current LOCF value minus the value at the beginning of the window",
            "slope": "change divided by window_hours - 1",
            "abnormal_duration": "number of hourly LOCF states outside the predefined clinical range",
            "abnormal_frequency": "fraction of actual measurements outside the predefined clinical range",
        },
        "abnormal_ranges": ABNORMAL_RANGES,
        "leakage_control": "all features use current/past hours within stay_id only",
        "schema_version": "temporal_features_v2",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest_path


def main() -> None:
    """前處理主流程：讀 ICU stay、抽特徵、補值、產 temporal features、合併 label。"""
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    sofa_path = Path(args.sofa_path)
    output_path = Path(args.output)

    stays = load_icu_stays(dataset_dir, args.max_stays)
    print(f"Loaded {len(stays):,} ICU stays.")

    # 先建立每個 ICU stay 的逐小時時間軸，後續所有事件都對齊到這個 grid。
    grid = build_hourly_grid(stays)
    print(f"Built {len(grid):,} hourly rows.")

    print("Extracting chart features...")
    chart_features = extract_chartevents(dataset_dir, stays, args.chunksize)
    print(f"Chart feature rows: {len(chart_features):,}")

    print("Extracting lab features...")
    lab_features = extract_labevents(dataset_dir, stays, args.chunksize)
    print(f"Lab feature rows: {len(lab_features):,}")

    df = grid.merge(chart_features, on=["stay_id", "sofa_hour"], how="left")
    df = df.merge(lab_features, on=["stay_id", "sofa_hour"], how="left")
    df = coalesce_clinical_columns(df)
    df = df.sort_values(["stay_id", "sofa_hour"]).reset_index(drop=True)

    feature_cols = [col for col in BASE_FEATURES if col in df.columns]
    # 必須在任何補值之前保存真實量測狀態，才能建立 leakage-free missingness features。
    observed_mask = df[feature_cols].notna().copy()
    df = add_measurement_process_features(df, feature_cols, observed_mask)
    # 呼吸器事件未出現時視為未使用；缺值狀態已由上一步獨立保存。
    df["mechanical_vent"] = df["mechanical_vent"].fillna(0)
    df = leakage_free_forward_fill(df, feature_cols)
    # forward-fill 之後 FiO2 或 PaO2 可能新增可用值，因此重新計算 PaO2/FiO2 ratio。
    df["pao2_fio2"] = np.where(
        df["pao2"].notna() & df["fio2"].notna() & (df["fio2"] > 0),
        df["pao2"] / df["fio2"],
        np.nan,
    )

    if not args.no_temporal_features:
        print(f"Adding temporal features for windows: {args.windows}")
        df = add_temporal_features(df, feature_cols, args.windows, observed_mask)

    df = merge_sofa_labels(df, sofa_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    quality_path = write_quality_report(df, feature_cols, output_path)
    manifest_path = write_feature_manifest(output_path, feature_cols, args.windows)
    print(f"Wrote {output_path}")
    print(f"Quality report: {quality_path}")
    print(f"Feature manifest: {manifest_path}")
    print("Imputation is forward-fill within stay_id only; no backward fill is used.")


if __name__ == "__main__":
    main()
