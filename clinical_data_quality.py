"""Shared clinical-unit normalization and plausibility filters."""

from __future__ import annotations

import numpy as np
import pandas as pd


# These broad bounds remove impossible unit/charting errors without clipping
# clinically extreme values to an artificial limit.
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "heart_rate": (20.0, 300.0),
    "respiratory_rate": (1.0, 80.0),
    "spo2": (50.0, 100.0),
    "sbp_arterial": (40.0, 300.0),
    "sbp_noninvasive": (40.0, 300.0),
    "dbp_arterial": (20.0, 200.0),
    "dbp_noninvasive": (20.0, 200.0),
    "map_arterial": (20.0, 250.0),
    "map_noninvasive": (20.0, 250.0),
    "map": (20.0, 250.0),
    "temperature_c": (25.0, 45.0),
    "fio2": (0.21, 1.0),
    "gcs_eye": (1.0, 4.0),
    "gcs_verbal": (1.0, 5.0),
    "gcs_motor": (1.0, 6.0),
    "gcs_total": (3.0, 15.0),
    "mechanical_vent": (0.0, 1.0),
    "lactate": (0.0, 30.0),
    "pao2": (20.0, 800.0),
    "bilirubin": (0.0, 80.0),
    "creatinine": (0.1, 30.0),
    "platelets": (1.0, 2000.0),
    "dopamine": (0.0, 100.0),
    "dobutamine": (0.0, 100.0),
    "epinephrine": (0.0, 10.0),
    "norepinephrine": (0.0, 10.0),
}


def fahrenheit_to_celsius(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    return (values - 32.0) * (5.0 / 9.0)


def filter_plausible(values: pd.Series, feature: str) -> pd.Series:
    """Set values outside a broad clinical range to NaN; never clip."""
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    bounds = PLAUSIBLE_RANGES.get(feature)
    if bounds is None:
        return numeric
    lower, upper = bounds
    return numeric.where(numeric.between(lower, upper, inclusive="both"), np.nan)


def filter_long_frame(
    frame: pd.DataFrame,
    feature_col: str = "feature",
    value_col: str = "value",
) -> pd.DataFrame:
    """Apply feature-specific plausibility filters to a long-format table."""
    frame = frame.copy()
    for feature in frame[feature_col].dropna().unique():
        mask = frame[feature_col].eq(feature)
        frame.loc[mask, value_col] = filter_plausible(
            frame.loc[mask, value_col], str(feature)
        )
    return frame
