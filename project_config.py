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
DEFAULT_PREDICTION_HORIZONS = (6, 12, 24)
