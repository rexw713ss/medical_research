"""Temporal feature schema shared by preprocessing and baseline experiments."""

from __future__ import annotations

import re
from collections.abc import Iterable


ROLLING_STATISTICS = (
    "mean",
    "min",
    "max",
    "std",
    "slope",
    "change",
    "abnormal_duration",
    "abnormal_frequency",
)

MEASUREMENT_PROCESS_SUFFIXES = (
    "is_missing",
    "time_since_last_measurement_h",
    "short_term_change",
)


def parse_observation_windows(raw: str | Iterable[int]) -> list[int]:
    if isinstance(raw, str):
        windows = [int(value.strip()) for value in raw.split(",") if value.strip()]
    else:
        windows = [int(value) for value in raw]
    if not windows or any(value <= 0 for value in windows):
        raise ValueError("Observation windows 必須是正整數。")
    if len(windows) != len(set(windows)):
        raise ValueError("Observation windows 不可重複。")
    return windows


def temporal_feature_window(column: str, base_features: list[str]) -> int | None:
    for feature in base_features:
        match = re.match(
            rf"^{re.escape(feature)}_w(\d+)h_({'|'.join(ROLLING_STATISTICS)})$",
            column,
        )
        if match:
            return int(match.group(1))
    return None


def is_measurement_process_feature(column: str, base_features: list[str]) -> bool:
    return any(
        column == f"{feature}_{suffix}"
        for feature in base_features
        for suffix in MEASUREMENT_PROCESS_SUFFIXES
    )


def temporal_columns_for_window(
    columns: list[str],
    base_features: list[str],
    window: int,
    include_measurement_process: bool = True,
) -> list[str]:
    selected = []
    for column in columns:
        if include_measurement_process and is_measurement_process_feature(column, base_features):
            selected.append(column)
            continue
        if temporal_feature_window(column, base_features) == window:
            selected.append(column)
    feature_position = {feature: index for index, feature in enumerate(base_features)}
    return sorted(
        selected,
        key=lambda column: (
            next(
                (position for feature, position in feature_position.items() if column.startswith(f"{feature}_")),
                999,
            ),
            column,
        ),
    )


def build_observation_window_feature_sets(
    columns: list[str],
    base_features: list[str],
    windows: list[int],
) -> dict[str, list[str]]:
    current = [feature for feature in base_features if feature in columns]
    output = {"static": current}
    for window in windows:
        temporal = temporal_columns_for_window(columns, base_features, window)
        output[f"temporal_w{window}h"] = [*current, *temporal]
    return output
