"""Load every eligible prediction window for formal streaming analyses."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from anfis_model import FEATURE_ORDER, explicit_temporal_input_order
from train_fnn import ICUWindowDataset, prepare_explicit_temporal_arrays


WINDOW_ID_MULTIPLIER = 100_000


@dataclass
class FormalWindowData:
    """Prepared hourly arrays and exact eligible window starts for one database."""

    database: str
    features: np.ndarray
    labels: np.ndarray
    stay_ids: np.ndarray
    subject_ids: np.ndarray
    hours: np.ndarray
    window_starts: np.ndarray
    expected_windows: int
    expected_patients: int
    expected_stays: int
    expected_positive: int
    prediction_path: Path

    @property
    def target_indices(self) -> np.ndarray:
        return self.window_starts + 23

    def audit_record(self) -> dict[str, object]:
        target_indices = self.target_indices
        return {
            "database": self.database,
            "prediction_path": str(self.prediction_path),
            "prediction_sha256": sha256_file(self.prediction_path),
            "expected_windows": self.expected_windows,
            "processed_windows": int(len(self.window_starts)),
            "expected_patients": self.expected_patients,
            "processed_patients": int(pd.Series(self.subject_ids[target_indices]).nunique()),
            "expected_stays": self.expected_stays,
            "processed_stays": int(np.unique(self.stay_ids[target_indices]).size),
            "expected_positive": self.expected_positive,
            "processed_positive": int(np.sum(self.labels[target_indices] == 1)),
            "all_prediction_windows_reconstructed": len(self.window_starts) == self.expected_windows,
        }


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def window_ids(stay_ids: np.ndarray, hours: np.ndarray) -> np.ndarray:
    return (
        stay_ids.astype(np.int64, copy=False) * WINDOW_ID_MULTIPLIER
        + hours.astype(np.int64, copy=False)
    )


def _read_prediction_keys(
    prediction_path: Path,
    split_col: str,
    time_col: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    required = [split_col, "stay_id", time_col, "y_true"]
    frame = pd.read_csv(prediction_path, usecols=required)
    if frame.empty:
        raise ValueError(f"Prediction file is empty: {prediction_path}")

    for column in ("stay_id", time_col, "y_true"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["stay_id"] = frame["stay_id"].astype(np.int64)
    frame[time_col] = frame[time_col].astype(np.int64)
    frame["y_true"] = frame["y_true"].astype(np.int8)
    if not set(frame["y_true"].unique()).issubset({0, 1}):
        raise ValueError(f"Prediction outcomes must be binary: {prediction_path}")

    ids = window_ids(frame["stay_id"].to_numpy(), frame[time_col].to_numpy())
    if np.unique(ids).size != len(ids):
        raise ValueError(f"Duplicate stay-hour prediction keys: {prediction_path}")
    order = np.argsort(ids, kind="mergesort")
    return frame, ids[order], frame["y_true"].to_numpy()[order]


def _load_hourly_rows(
    hourly_path: Path,
    database: str,
    columns: list[str],
    wanted_stays: set[int],
    chunk_size: int,
) -> pd.DataFrame:
    if hourly_path.suffix.lower() == ".pkl":
        source = pd.read_pickle(hourly_path)
        missing = set(columns) - set(source.columns)
        if missing:
            raise ValueError(f"{database} hourly data are missing columns: {sorted(missing)}")
        keep = source["stay_id"].isin(wanted_stays)
        frame = source.loc[keep, columns].copy()
        del source
        return frame

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(hourly_path, usecols=columns, chunksize=chunk_size):
        keep = chunk["stay_id"].isin(wanted_stays)
        if keep.any():
            chunks.append(chunk.loc[keep].copy())
    if not chunks:
        raise ValueError(f"No {database} hourly rows matched prediction stays.")
    return pd.concat(chunks, ignore_index=True)


def load_formal_window_data(
    *,
    database: str,
    hourly_path: Path,
    prediction_path: Path,
    target_col: str,
    time_col: str,
    split_col: str,
    seq_length: int = 24,
    chunk_size: int = 250_000,
) -> FormalWindowData:
    """Reconstruct and verify every window listed in a formal prediction file."""

    if seq_length != 24:
        raise ValueError("Formal helper currently requires the locked 24-hour observation window.")
    predictions, expected_ids, expected_labels = _read_prediction_keys(
        prediction_path, split_col, time_col
    )
    wanted_stays = set(predictions["stay_id"].astype(np.int64))
    columns = [
        "stay_id",
        split_col,
        time_col,
        target_col,
        *explicit_temporal_input_order(FEATURE_ORDER),
    ]
    hourly = _load_hourly_rows(
        hourly_path, database, columns, wanted_stays, chunk_size
    )
    features, labels, stay_ids, subject_ids, hours = prepare_explicit_temporal_arrays(
        hourly,
        target_col=target_col,
        time_col=time_col,
        split_col=split_col,
    )
    del hourly

    dataset = ICUWindowDataset(
        features=features,
        labels=labels,
        stay_ids=stay_ids,
        split_values=subject_ids,
        time_values=hours,
        allowed_split_values=set(pd.unique(subject_ids)),
        seq_length=seq_length,
        allowed_window_ids=expected_ids,
        require_all_window_ids=True,
    )
    target_indices = dataset.window_starts + seq_length - 1
    actual_ids = window_ids(stay_ids[target_indices], hours[target_indices])
    actual_order = np.argsort(actual_ids, kind="mergesort")
    if not np.array_equal(actual_ids[actual_order], expected_ids):
        raise ValueError(f"{database} reconstructed keys differ from formal predictions.")
    actual_labels = labels[target_indices][actual_order].astype(np.int8)
    if not np.array_equal(actual_labels, expected_labels):
        mismatch = int(np.sum(actual_labels != expected_labels))
        raise ValueError(f"{database} has {mismatch:,} reconstructed outcome mismatches.")

    result = FormalWindowData(
        database=database,
        features=features,
        labels=labels,
        stay_ids=stay_ids,
        subject_ids=subject_ids,
        hours=hours,
        window_starts=dataset.window_starts,
        expected_windows=len(predictions),
        expected_patients=int(predictions[split_col].nunique()),
        expected_stays=int(predictions["stay_id"].nunique()),
        expected_positive=int(predictions["y_true"].sum()),
        prediction_path=prediction_path,
    )
    audit = result.audit_record()
    if not audit["all_prediction_windows_reconstructed"]:
        raise ValueError(f"{database} did not reconstruct every formal prediction window.")
    del predictions
    return result


def iter_window_batches(
    data: FormalWindowData,
    batch_size: int,
    seq_length: int = 24,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield sequence batches plus target stay/hour keys in deterministic order."""

    offsets = np.arange(seq_length, dtype=np.int64)
    for start in range(0, len(data.window_starts), batch_size):
        starts = data.window_starts[start : start + batch_size]
        row_indices = starts[:, None] + offsets[None, :]
        target_indices = starts + seq_length - 1
        yield (
            data.features[row_indices],
            data.stay_ids[target_indices].astype(np.int64, copy=False),
            data.hours[target_indices].astype(np.int64, copy=False),
        )
