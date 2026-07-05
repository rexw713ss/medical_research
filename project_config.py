"""專案共用路徑與正式資料版本。

所有命令預期由專案根目錄執行。集中管理這些常數可避免不同模型默默讀到
舊版 hourly table 或錯誤的原始資料目錄。
"""

from __future__ import annotations


MIMIC_DATA_DIR = "dataset/MIMIC-IV"
EICU_DATA_DIR = "dataset/e-ICU"

SOFA_HOURLY_CSV = "sofa_scores_hourly.csv"
PRIMARY_HOURLY_FEATURES = "model_hourly_features_v3.csv"

PATIENT_SPLIT_CSV = "patient_split.csv"
COMPARISON_PROTOCOL_JSON = "comparison_protocol.json"
EQUAL_SAMPLE_WINDOWS_CSV = "equal_sample_windows.csv.gz"

DEFAULT_OBSERVATION_WINDOWS = (4, 6, 12, 24)

PRIMARY_PREDICTION_HORIZON = 6
PRIMARY_OUTCOME_COLUMN = "label_sofa_increase_ge2_6h"
SECONDARY_PREDICTION_HORIZONS = (12, 24)
SECONDARY_OUTCOME_COLUMNS = tuple(
    f"label_sofa_increase_ge2_{horizon}h"
    for horizon in SECONDARY_PREDICTION_HORIZONS
)
ALL_PREDICTION_HORIZONS = (
    PRIMARY_PREDICTION_HORIZON,
    *SECONDARY_PREDICTION_HORIZONS,
)

# 不帶 --horizons 的正式分析只執行 primary outcome。
DEFAULT_PREDICTION_HORIZONS = (PRIMARY_PREDICTION_HORIZON,)
