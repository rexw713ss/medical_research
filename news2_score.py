"""NEWS2 自動評分與未來惡化標記。

這支程式會讀取新版 preprocessing 產生的 model_hourly_features_v3.csv，
或舊版 extracted_vitals_sample.csv，計算每列的 NEWS2 分數，並建立：

1. NEWS2_Score：目前時間點的 NEWS2 分數。
2. future_6h_max_NEWS2：未來 6 列/6 小時內的最高 NEWS2 分數。
3. Deterioration_Label：若未來 6 小時內 NEWS2 最高分 >= 7，標記為 1，否則為 0。

注意：
- 若輸入是 model_hourly_features_v3.csv，資料已經是一小時一列，因此 6 列可視為 6 小時。
- 若輸入是 extracted_vitals_sample.csv，資料是不規則事件列，6 列不一定等於 6 小時。
- 補值只使用同一個 stay_id 或 subject_id 內的 forward-fill，不使用 backward-fill。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from project_config import PRIMARY_HOURLY_FEATURES


DEFAULT_INPUTS = [
    Path(PRIMARY_HOURLY_FEATURES),
    Path("extracted_vitals_sample.csv"),
]

NEWS2_COLUMNS = [
    "RespRate",
    "SpO2",
    "FiO2",
    "Temperature_C",
    "SBP",
    "HeartRate",
    "GCS_Total",
]

INPUT_CANDIDATES = {
    "HeartRate": ["HeartRate", "heart_rate", "Heart Rate"],
    "RespRate": ["RespRate", "respiratory_rate", "Respiratory Rate"],
    "SpO2": ["SpO2", "spo2", "O2 saturation pulseoxymetry"],
    "Temperature_C": ["Temperature_C", "temperature_c", "Temperature Celsius"],
    "SBP": [
        "SBP",
        "sbp",
        "Arterial Blood Pressure systolic",
        "Non Invasive Blood Pressure systolic",
    ],
    "FiO2": ["FiO2", "fio2", "Inspired O2 Fraction"],
    "GCS_Total": ["GCS_Total", "gcs_total"],
    "GCS_Eye": ["GCS - Eye Opening", "gcs_eye"],
    "GCS_Verbal": ["GCS - Verbal Response", "gcs_verbal"],
    "GCS_Motor": ["GCS - Motor Response", "gcs_motor"],
}


def parse_args() -> argparse.Namespace:
    """設定命令列參數。"""
    parser = argparse.ArgumentParser(description="計算 NEWS2 分數並建立未來惡化標記。")
    parser.add_argument("--input", default=None, help="輸入 CSV；預設自動尋找正式 hourly features。")
    parser.add_argument("--output", default="news2_scores.csv", help="輸出 CSV。")
    parser.add_argument("--future-hours", type=int, default=6, help="未來惡化視窗，預設 6 小時。")
    parser.add_argument("--threshold", type=int, default=7, help="NEWS2 高風險門檻，預設 >= 7。")
    parser.add_argument(
        "--keep-incomplete-labels",
        action="store_true",
        help="保留 stay 結尾不足完整未來視窗的列；預設會丟棄。",
    )
    return parser.parse_args()


def choose_input_path(input_arg: str | None) -> Path:
    """選擇 NEWS2 輸入檔。"""
    if input_arg:
        path = Path(input_arg)
        if not path.exists():
            raise FileNotFoundError(f"找不到指定輸入檔：{path}")
        return path

    for path in DEFAULT_INPUTS:
        if path.exists():
            return path

    candidates = ", ".join(str(path) for path in DEFAULT_INPUTS)
    raise FileNotFoundError(f"找不到 NEWS2 輸入檔，請先產生其中之一：{candidates}")


def read_needed_columns(input_path: Path) -> pd.DataFrame:
    """只讀 NEWS2 需要的欄位，避免把大型 preprocessing 檔案整個載入記憶體。"""
    header = pd.read_csv(input_path, nrows=0).columns.tolist()
    needed = {"subject_id", "stay_id", "charttime", "sofa_time", "sofa_hour"}
    for candidates in INPUT_CANDIDATES.values():
        needed.update(candidates)

    usecols = [col for col in header if col in needed]
    if "subject_id" not in usecols and "stay_id" not in usecols:
        raise ValueError("輸入檔至少需要 subject_id 或 stay_id，才能分組建立時序標籤。")

    if "charttime" not in usecols and "sofa_time" not in usecols:
        raise ValueError("輸入檔至少需要 charttime 或 sofa_time，才能排序時序資料。")

    return pd.read_csv(input_path, usecols=usecols)


def ensure_charttime(df: pd.DataFrame) -> pd.DataFrame:
    """若新版 preprocessing 使用 sofa_time，將它轉成 charttime 供後續通用流程使用。"""
    df = df.copy()
    if "charttime" not in df.columns and "sofa_time" in df.columns:
        df["charttime"] = df["sofa_time"]
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    return df.dropna(subset=["charttime"])


def first_existing(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """依候選欄位順序取第一個存在且非缺值的數值。"""
    result = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in candidates:
        if col in df.columns:
            result = result.combine_first(pd.to_numeric(df[col], errors="coerce"))
    return result


def normalize_fio2_percent(values: pd.Series) -> pd.Series:
    """將 FiO2 統一成百分比。

    MIMIC 中 FiO2 可能記成 0.5，也可能記成 50。
    NEWS2 只需要判斷是否大於室內空氣約 21%。
    """
    values = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        np.where((values > 0) & (values <= 1.0), values * 100.0, values),
        index=values.index,
        dtype="float64",
    )


def prepare_news2_columns(df: pd.DataFrame) -> pd.DataFrame:
    """把 MIMIC 原始欄位或 preprocessing 欄位整理成 NEWS2 標準欄位。"""
    df = df.copy()
    df["HeartRate"] = first_existing(df, INPUT_CANDIDATES["HeartRate"])
    df["RespRate"] = first_existing(df, INPUT_CANDIDATES["RespRate"])
    df["SpO2"] = first_existing(df, INPUT_CANDIDATES["SpO2"])
    df["Temperature_C"] = first_existing(df, INPUT_CANDIDATES["Temperature_C"])
    df["SBP"] = first_existing(df, INPUT_CANDIDATES["SBP"])
    df["FiO2"] = normalize_fio2_percent(first_existing(df, INPUT_CANDIDATES["FiO2"]))

    df["GCS_Total"] = first_existing(df, INPUT_CANDIDATES["GCS_Total"])
    if df["GCS_Total"].isna().all():
        gcs_eye = first_existing(df, INPUT_CANDIDATES["GCS_Eye"])
        gcs_verbal = first_existing(df, INPUT_CANDIDATES["GCS_Verbal"])
        gcs_motor = first_existing(df, INPUT_CANDIDATES["GCS_Motor"])
        df["GCS_Total"] = pd.concat([gcs_eye, gcs_verbal, gcs_motor], axis=1).sum(
            axis=1,
            min_count=3,
        )

    return df


def group_columns(df: pd.DataFrame) -> list[str]:
    """NEWS2 時序分組；有 stay_id 就以 ICU stay 為單位，否則退回 subject_id。"""
    return ["stay_id"] if "stay_id" in df.columns else ["subject_id"]


def sort_columns(df: pd.DataFrame) -> list[str]:
    """排序欄位；新版 hourly table 優先使用 sofa_hour，舊資料使用 charttime。"""
    cols = group_columns(df)
    if "sofa_hour" in df.columns:
        return cols + ["sofa_hour"]
    return cols + ["charttime"]


def leakage_safe_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """只在同一 ICU stay 或 subject_id 內 forward-fill，不使用 bfill。"""
    df = df.sort_values(sort_columns(df)).reset_index(drop=True)
    groups = group_columns(df)
    df[NEWS2_COLUMNS] = df.groupby(groups, group_keys=False)[NEWS2_COLUMNS].ffill()
    return df


def calculate_news2(row: pd.Series) -> int:
    """依 NEWS2 Scale 1 門檻計算單列資料的 NEWS2 分數。"""
    score = 0

    rr = row.get("RespRate", np.nan)
    if pd.notna(rr):
        if rr <= 8 or rr >= 25:
            score += 3
        elif 21 <= rr <= 24:
            score += 2
        elif 9 <= rr <= 11:
            score += 1

    spo2 = row.get("SpO2", np.nan)
    if pd.notna(spo2):
        if spo2 <= 91:
            score += 3
        elif 92 <= spo2 <= 93:
            score += 2
        elif 94 <= spo2 <= 95:
            score += 1

    fio2 = row.get("FiO2", np.nan)
    if pd.notna(fio2) and fio2 > 21:
        score += 2

    temp = row.get("Temperature_C", np.nan)
    if pd.notna(temp):
        if temp <= 35.0:
            score += 3
        elif temp >= 39.1:
            score += 2
        elif 35.1 <= temp <= 36.0 or 38.1 <= temp <= 39.0:
            score += 1

    sbp = row.get("SBP", np.nan)
    if pd.notna(sbp):
        if sbp <= 90 or sbp >= 220:
            score += 3
        elif 91 <= sbp <= 100:
            score += 2
        elif 101 <= sbp <= 110:
            score += 1

    hr = row.get("HeartRate", np.nan)
    if pd.notna(hr):
        if hr <= 40 or hr >= 131:
            score += 3
        elif 111 <= hr <= 130:
            score += 2
        elif 41 <= hr <= 50 or 91 <= hr <= 110:
            score += 1

    gcs = row.get("GCS_Total", np.nan)
    if pd.notna(gcs) and gcs < 15:
        score += 3

    return score


def add_future_news2_label(
    df: pd.DataFrame,
    future_hours: int,
    threshold: int,
) -> pd.DataFrame:
    """建立未來 future_hours 小時 NEWS2 惡化標記。

    這裡會先 shift(-1)，所以標籤只看真正未來，不包含當下這一列。
    """
    df = df.sort_values(sort_columns(df)).reset_index(drop=True)
    groups = group_columns(df)
    future_col = f"future_{future_hours}h_max_NEWS2"

    df[future_col] = (
        df.groupby(groups)["NEWS2_Score"]
        .apply(lambda s: s.shift(-1).iloc[::-1].rolling(future_hours, min_periods=1).max().iloc[::-1])
        .reset_index(level=list(range(len(groups))), drop=True)
    )

    label = (df[future_col] >= threshold).astype("Int64")
    future_available_rows = df.groupby(groups)["charttime"].transform("count") - df.groupby(groups).cumcount() - 1
    label.loc[future_available_rows < future_hours] = pd.NA

    df["Deterioration_Label"] = label
    df["NEWS2_High_Risk_Label"] = label
    return df


def main() -> None:
    """NEWS2 主流程。"""
    args = parse_args()
    input_path = choose_input_path(args.input)
    output_path = Path(args.output)

    print(f"讀取 NEWS2 輸入檔：{input_path}")
    df = read_needed_columns(input_path)
    df = ensure_charttime(df)

    if "sofa_hour" not in df.columns:
        print("提醒：輸入檔不是 hourly table，future-hours 會以未來列數近似，不一定等於真實小時。")

    print("整理 NEWS2 欄位並進行 leakage-free forward-fill...")
    df = prepare_news2_columns(df)
    df = leakage_safe_imputation(df)

    print("正在計算每列 NEWS2 分數...")
    df["NEWS2_Score"] = df.apply(calculate_news2, axis=1)

    print(f"正在建立未來 {args.future_hours} 小時 NEWS2 惡化標記...")
    df = add_future_news2_label(df, args.future_hours, args.threshold)

    if not args.keep_incomplete_labels:
        before = len(df)
        df = df.dropna(subset=["Deterioration_Label"]).copy()
        df["Deterioration_Label"] = df["Deterioration_Label"].astype(int)
        df["NEWS2_High_Risk_Label"] = df["NEWS2_High_Risk_Label"].astype(int)
        print(f"已移除結尾不足未來視窗的列數：{before - len(df):,}")

    preview_cols = [
        "subject_id",
        "stay_id",
        "charttime",
        "sofa_hour",
        "NEWS2_Score",
        f"future_{args.future_hours}h_max_NEWS2",
        "Deterioration_Label",
    ]
    preview_cols = [col for col in preview_cols if col in df.columns]
    print(df[preview_cols].head(15))

    print("\n總資料筆數:", len(df))
    print("惡化標籤分佈:")
    print(df["Deterioration_Label"].value_counts(normalize=True, dropna=False) * 100)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"已輸出：{output_path}")


if __name__ == "__main__":
    main()
