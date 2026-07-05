"""所有模型共用的公平比較條件與 equal-sample window manifest。

Window 的唯一鍵為 ``stay_id + sofa_hour``。正式 protocol 固定：
1. patient split 讀取 patient_split.csv；
2. predictors 使用 anfis_model.FEATURE_ORDER 的 13 個來源變數；
3. 所有模型在相同的 24 小時 target eligibility 上比較；
4. test 永遠使用完整 eligible windows；
5. equal-sample 模式只固定抽樣 train/validation，所有模型讀同一份名單。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from anfis_model import FEATURE_ORDER
from patient_split import HORIZONS, SPLIT_NAMES, read_patient_split
from project_config import (
    COMPARISON_PROTOCOL_JSON,
    EQUAL_SAMPLE_WINDOWS_CSV,
    PATIENT_SPLIT_CSV,
    PRIMARY_OUTCOME_COLUMN,
    SECONDARY_OUTCOME_COLUMNS,
    SOFA_HOURLY_CSV,
)


WINDOW_ID_FACTOR = 100_000
COMPARISON_MODES = ("full", "equal_sample")
DEFAULT_PROTOCOL_PATH = COMPARISON_PROTOCOL_JSON
DEFAULT_EQUAL_SAMPLE_PATH = EQUAL_SAMPLE_WINDOWS_CSV


def horizon_from_target_col(target_col: str) -> int:
    match = re.search(r"label_sofa_increase_ge2_(\d+)h$", target_col)
    if match is None:
        raise ValueError(f"無法由 outcome 欄位判斷 horizon: {target_col}")
    return int(match.group(1))


def encode_window_ids(stay_ids: Iterable[Any], hours: Iterable[Any]) -> np.ndarray:
    """把 stay_id 與 ICU hour 編成無碰撞的 int64 window ID。"""
    stay = np.asarray(stay_ids, dtype=np.int64)
    hour = np.asarray(hours, dtype=np.int64)
    if stay.shape != hour.shape:
        raise ValueError("stay_id 與 sofa_hour 長度不同。")
    if np.any(hour < 0) or np.any(hour >= WINDOW_ID_FACTOR):
        raise ValueError(f"sofa_hour 必須位於 0 到 {WINDOW_ID_FACTOR - 1}。")
    return stay * WINDOW_ID_FACTOR + hour


def decode_window_ids(window_ids: Iterable[Any]) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(window_ids, dtype=np.int64)
    return ids // WINDOW_ID_FACTOR, ids % WINDOW_ID_FACTOR


def window_id_membership(window_ids: Iterable[Any], allowed_window_ids: np.ndarray) -> np.ndarray:
    """對已排序 manifest 做 searchsorted membership，避免逐 stay 重複排序。"""
    values = np.asarray(window_ids, dtype=np.int64)
    allowed = np.asarray(allowed_window_ids, dtype=np.int64)
    if len(allowed) == 0:
        return np.zeros(values.shape, dtype=bool)
    if len(allowed) > 1 and np.any(allowed[1:] < allowed[:-1]):
        allowed = np.sort(allowed)
    positions = np.searchsorted(allowed, values)
    within = positions < len(allowed)
    matched = np.zeros(values.shape, dtype=bool)
    matched[within] = allowed[positions[within]] == values[within]
    return matched


def load_protocol(path: str | Path = DEFAULT_PROTOCOL_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 comparison protocol: {path}。請先執行 comparison_protocol.py。")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("predictors") != FEATURE_ORDER:
        raise ValueError("Comparison protocol 的 predictor 順序與 FEATURE_ORDER 不一致。")
    if int(protocol.get("sequence_length_hours", 0)) != 24:
        raise ValueError("正式 comparison protocol 必須使用 24 小時回看。")
    return protocol


def validate_comparison_args(
    mode: str,
    protocol_path: str | Path,
    target_col: str,
    seq_length: int,
) -> dict[str, Any]:
    if mode not in COMPARISON_MODES:
        raise ValueError(f"未知 comparison mode: {mode}")
    protocol = load_protocol(protocol_path)
    if seq_length != int(protocol["sequence_length_hours"]):
        raise ValueError(
            f"公平比較固定 seq_length={protocol['sequence_length_hours']}，目前收到 {seq_length}。"
        )
    if target_col not in protocol["outcomes"]:
        raise ValueError(f"Comparison protocol 未定義 outcome: {target_col}")
    return protocol


def load_equal_sample_window_ids(
    manifest_path: str | Path,
    target_col: str,
    split: str,
) -> np.ndarray:
    if split == "test":
        raise ValueError("test set 不可抽樣；正式 protocol 一律使用完整 eligible test windows。")
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 equal-sample window manifest: {path}")
    selected = pd.read_csv(
        path,
        usecols=["target_col", "split", "window_id"],
        dtype={"target_col": "string", "split": "string", "window_id": "int64"},
    )
    selected = selected[(selected["target_col"] == target_col) & (selected["split"] == split)]
    if selected.empty:
        raise ValueError(f"Equal-sample manifest 沒有 {target_col} / {split} windows。")
    ids = np.sort(selected["window_id"].to_numpy(dtype=np.int64, copy=True))
    if len(ids) != len(np.unique(ids)):
        raise ValueError(f"Equal-sample manifest 的 {target_col} / {split} 有重複 window。")
    return ids


def window_ids_for_mode(
    mode: str,
    equal_sample_path: str | Path,
    target_col: str,
    split: str,
) -> np.ndarray | None:
    """full 或 test 回傳 None，代表保留所有 eligible windows。"""
    if mode == "full" or split == "test":
        return None
    return load_equal_sample_window_ids(equal_sample_path, target_col, split)


def comparison_eligible_mask(
    df: pd.DataFrame,
    target_col: str,
    history_col: str = "_history_index",
    seq_length: int = 24,
) -> pd.Series:
    if history_col not in df.columns:
        raise ValueError(f"資料缺少 history index: {history_col}")
    target = pd.to_numeric(df[target_col], errors="coerce")
    return (df[history_col] >= seq_length - 1) & target.notna()


def filter_frame_to_comparison_windows(
    df: pd.DataFrame,
    target_col: str,
    time_col: str,
    split: str,
    mode: str,
    equal_sample_path: str | Path,
    seq_length: int = 24,
    require_all_window_ids: bool = True,
) -> pd.DataFrame:
    """套用共同 24h eligibility，equal-sample 時再套固定 target-window 名單。"""
    eligible = df.loc[comparison_eligible_mask(df, target_col, seq_length=seq_length)].copy()
    selected_ids = window_ids_for_mode(mode, equal_sample_path, target_col, split)
    if selected_ids is not None:
        frame_ids = encode_window_ids(eligible["stay_id"], eligible[time_col])
        eligible = eligible.loc[window_id_membership(frame_ids, selected_ids)].copy()
        found = np.sort(encode_window_ids(eligible["stay_id"], eligible[time_col]))
        if require_all_window_ids and not np.array_equal(found, selected_ids):
            missing = len(selected_ids) - len(found)
            raise ValueError(
                f"目前資料缺少 {missing:,} 個 {target_col} / {split} equal-sample windows；"
                "正式比較不可搭配 --max-rows 或 --max-stays。"
            )
    return eligible


def cohort_fingerprint(
    stay_ids: Iterable[Any],
    hours: Iterable[Any],
    labels: Iterable[Any],
) -> str:
    """依 window ID 排序後建立 cohort SHA-256，供跨模型核對。"""
    ids = encode_window_ids(stay_ids, hours)
    y = np.asarray(labels, dtype=np.int8)
    if ids.shape != y.shape:
        raise ValueError("Window IDs 與 labels 長度不同。")
    order = np.argsort(ids, kind="mergesort")
    payload = np.empty((len(ids), 2), dtype="<i8")
    payload[:, 0] = ids[order]
    payload[:, 1] = y[order]
    return hashlib.sha256(payload.tobytes()).hexdigest()


def cohort_record(
    stay_ids: Iterable[Any],
    hours: Iterable[Any],
    labels: Iterable[Any],
    split: str,
    target_col: str,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.float64)
    finite = np.isfinite(y)
    y = y[finite].astype(np.int8)
    stay = np.asarray(stay_ids)[finite]
    time = np.asarray(hours)[finite]
    return {
        "split": split,
        "target_col": target_col,
        "horizon_hours": horizon_from_target_col(target_col),
        "windows": int(len(y)),
        "positive": int((y == 1).sum()),
        "negative": int((y == 0).sum()),
        "prevalence": float(y.mean()) if len(y) else None,
        "cohort_sha256": cohort_fingerprint(stay, time, y) if len(y) else None,
    }


def write_cohort_audit(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_cohort_records(
    records: list[dict[str, Any]],
    protocol: dict[str, Any],
    mode: str,
    allow_incomplete: bool = False,
) -> None:
    """確認模型實際使用的 windows 與 protocol 的數量、正類數及 fingerprint 相同。"""
    if allow_incomplete:
        return
    expected_rows = protocol["full_cohort"] if mode == "full" else protocol["equal_sample_cohort"]
    expected = {(row["target_col"], row["split"]): row for row in expected_rows}
    for actual in records:
        key = (actual["target_col"], actual["split"])
        if key not in expected:
            raise ValueError(f"Protocol 找不到 cohort 定義: {key}")
        reference = expected[key]
        for field in ["windows", "positive", "negative", "cohort_sha256"]:
            if actual.get(field) != reference.get(field):
                raise ValueError(
                    f"Cohort 不一致 {key} / {field}: actual={actual.get(field)}, "
                    f"expected={reference.get(field)}"
                )


def _stratified_sample_indices(labels: np.ndarray, n: int, seed: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("Equal-sample size 必須大於 0。")
    if n > len(labels):
        raise ValueError(f"要求抽樣 {n:,} windows，但 eligible windows 只有 {len(labels):,}。")
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    n_pos = int(round(n * len(pos) / len(labels)))
    n_pos = min(max(n_pos, 1), len(pos), n - 1)
    n_neg = n - n_pos
    if n_neg > len(neg):
        n_neg = len(neg)
        n_pos = n - n_neg
    rng = np.random.default_rng(seed)
    selected = np.concatenate(
        [rng.choice(pos, size=n_pos, replace=False), rng.choice(neg, size=n_neg, replace=False)]
    )
    rng.shuffle(selected)
    return selected


def _frame_record(frame: pd.DataFrame, target_col: str, split: str) -> dict[str, Any]:
    return cohort_record(
        frame["stay_id"].to_numpy(),
        frame["sofa_hour"].to_numpy(),
        frame[target_col].to_numpy(),
        split,
        target_col,
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="建立公平模型比較與 equal-sample windows。")
    parser.add_argument("--sofa-csv", default=SOFA_HOURLY_CSV)
    parser.add_argument("--patient-split", default=PATIENT_SPLIT_CSV)
    parser.add_argument("--output", default=DEFAULT_EQUAL_SAMPLE_PATH)
    parser.add_argument("--protocol-output", default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--train-windows", type=int, default=200_000)
    parser.add_argument("--validation-windows", type=int, default=50_000)
    parser.add_argument("--seq-length", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seq_length != 24:
        raise ValueError("正式公平比較固定使用 24 小時回看。")
    sofa_path = Path(args.sofa_csv)
    if not sofa_path.exists():
        raise FileNotFoundError(f"找不到 SOFA CSV: {sofa_path}")

    target_cols = [f"label_sofa_increase_ge2_{h}h" for h in HORIZONS]
    print(f"讀取 target windows: {sofa_path}")
    frame = pd.read_csv(
        sofa_path,
        usecols=["subject_id", "stay_id", "sofa_hour", *target_cols],
    )
    split_manifest = read_patient_split(args.patient_split)
    split_map = split_manifest.set_index("subject_id")["split"]
    frame["split"] = frame["subject_id"].map(split_map)
    if frame["split"].isna().any():
        raise ValueError("SOFA 資料含有未分配 patient split 的 subject_id。")
    frame = frame[frame["sofa_hour"] >= args.seq_length - 1].copy()

    sample_frames: list[pd.DataFrame] = []
    full_records: list[dict[str, Any]] = []
    equal_records: list[dict[str, Any]] = []
    sample_sizes = {"train": args.train_windows, "validation": args.validation_windows}

    for horizon_index, target_col in enumerate(target_cols):
        valid = frame[target_col].notna()
        for split_index, split in enumerate(SPLIT_NAMES):
            cohort = frame.loc[
                valid & frame["split"].eq(split),
                ["stay_id", "sofa_hour", target_col],
            ].copy()
            cohort[target_col] = cohort[target_col].astype("int8")
            full_records.append(_frame_record(cohort, target_col, split))

            if split == "test":
                equal_record = dict(full_records[-1])
                equal_record["selection"] = "full_test"
                equal_records.append(equal_record)
                continue

            n = sample_sizes[split]
            selected_idx = _stratified_sample_indices(
                cohort[target_col].to_numpy(dtype=np.int8),
                n=n,
                seed=args.seed + horizon_index * 100 + split_index,
            )
            selected = cohort.iloc[selected_idx].copy()
            selected = selected.rename(columns={target_col: "y_true"})
            selected["target_col"] = target_col
            selected["horizon_hours"] = HORIZONS[horizon_index]
            selected["split"] = split
            selected["window_id"] = encode_window_ids(selected["stay_id"], selected["sofa_hour"])
            sample_frames.append(selected)
            record = cohort_record(
                selected["stay_id"], selected["sofa_hour"], selected["y_true"], split, target_col
            )
            record["selection"] = "equal_sample"
            equal_records.append(record)

    samples = pd.concat(sample_frames, ignore_index=True)
    samples = samples[
        ["target_col", "horizon_hours", "split", "window_id", "stay_id", "sofa_hour", "y_true"]
    ].sort_values(["horizon_hours", "split", "window_id"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples.to_csv(output_path, index=False, compression="gzip")

    protocol: dict[str, Any] = {
        "protocol_name": "fair_patient_level_24h_comparison_v1",
        "patient_split": str(args.patient_split),
        "patient_id_column": "subject_id",
        "window_key": ["stay_id", "sofa_hour"],
        "sequence_length_hours": args.seq_length,
        "predictors": FEATURE_ORDER,
        "outcomes": target_cols,
        "analysis_outcome_roles": {
            "primary": PRIMARY_OUTCOME_COLUMN,
            "secondary": list(SECONDARY_OUTCOME_COLUMNS),
        },
        "modes": {
            "full": "all eligible train/validation/test windows",
            "equal_sample": {
                "train_windows_per_horizon": args.train_windows,
                "validation_windows_per_horizon": args.validation_windows,
                "test": "all eligible test windows",
                "manifest": str(output_path),
            },
        },
        "imputation": "within-stay forward fill, then fixed clinical defaults; no backward fill",
        "seed": args.seed,
        "full_cohort": full_records,
        "equal_sample_cohort": equal_records,
        "equal_sample_manifest_sha256": sha256_file(output_path),
    }
    critical = {
        key: protocol[key]
        for key in [
            "protocol_name",
            "patient_split",
            "window_key",
            "sequence_length_hours",
            "predictors",
            "outcomes",
            "modes",
            "imputation",
            "seed",
            "full_cohort",
            "equal_sample_cohort",
            "equal_sample_manifest_sha256",
        ]
    }
    protocol["protocol_sha256"] = hashlib.sha256(
        json.dumps(critical, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    protocol_path = Path(args.protocol_output)
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Equal-sample windows: {output_path} ({len(samples):,} rows)")
    print(f"Comparison protocol: {protocol_path}")
    for record in equal_records:
        print(
            f"{record['horizon_hours']}h {record['split']}: "
            f"{record['windows']:,} windows, prevalence={record['prevalence']:.4f}"
        )


if __name__ == "__main__":
    main()
