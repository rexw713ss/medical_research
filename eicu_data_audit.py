"""稽核 eICU 外部驗證資料是否足以重建 predictors 與 SOFA outcome。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from project_config import EICU_DATA_DIR


REQUIRED_TABLES: dict[str, set[str]] = {
    "patient.csv.gz": {
        "patientunitstayid",
        "patienthealthsystemstayid",
        "uniquepid",
        "age",
        "unitdischargeoffset",
    },
    "vitalPeriodic.csv.gz": {
        "patientunitstayid",
        "observationoffset",
        "temperature",
        "sao2",
        "heartrate",
        "respiration",
        "systemicsystolic",
        "systemicmean",
    },
    "vitalAperiodic.csv.gz": {
        "patientunitstayid",
        "observationoffset",
        "noninvasivesystolic",
        "noninvasivemean",
    },
    "lab.csv.gz": {
        "patientunitstayid",
        "labresultoffset",
        "labname",
        "labresult",
        "labmeasurenamesystem",
    },
    "nurseCharting.csv.gz": {
        "patientunitstayid",
        "nursingchartoffset",
        "nursingchartcelltypevallabel",
        "nursingchartcelltypevalname",
        "nursingchartvalue",
    },
    "respiratoryCharting.csv.gz": {
        "patientunitstayid",
        "respchartoffset",
        "respchartvaluelabel",
        "respchartvalue",
    },
    "infusionDrug.csv.gz": {
        "patientunitstayid",
        "infusionoffset",
        "drugname",
        "drugrate",
        "patientweight",
    },
    "intakeOutput.csv.gz": {
        "patientunitstayid",
        "intakeoutputoffset",
        "celllabel",
        "cellvaluenumeric",
    },
}

LAB_TERMS = {
    "pao2": ("pao2",),
    "platelets": ("platelets x 1000", "platelets"),
    "bilirubin": ("total bilirubin",),
    "creatinine": ("creatinine",),
    "lactate": ("lactate",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit local eICU files for external validation.")
    parser.add_argument("--dataset-dir", default=EICU_DATA_DIR)
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=2_000_000,
        help="Maximum rows sampled from label-based tables; use 0 for schema-only audit.",
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--output-dir", default="outputs/eicu_readiness")
    return parser.parse_args()


def sampled_value_counts(
    path: Path,
    columns: list[str],
    sample_rows: int,
    chunksize: int,
) -> tuple[dict[str, Counter[str]], int]:
    counters = {column: Counter() for column in columns}
    if sample_rows <= 0:
        return counters, 0
    rows = 0
    for chunk in pd.read_csv(path, usecols=columns, chunksize=chunksize, low_memory=False):
        remaining = sample_rows - rows
        if remaining <= 0:
            break
        chunk = chunk.head(remaining)
        rows += len(chunk)
        for column in columns:
            values = chunk[column].dropna().astype(str).str.strip().str.lower()
            counters[column].update(values)
        if rows >= sample_rows:
            break
    return counters, rows


def count_matches(counter: Counter[str], terms: tuple[str, ...], *, startswith: bool = False) -> int:
    if startswith:
        return sum(count for value, count in counter.items() if value.startswith(terms))
    return sum(count for value, count in counter.items() if any(term in value for term in terms))


def patient_summary(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(
        path,
        usecols=[
            "patientunitstayid",
            "patienthealthsystemstayid",
            "uniquepid",
            "age",
            "unitdischargeoffset",
        ],
        low_memory=False,
    )
    age = pd.to_numeric(frame["age"].replace("> 89", "90"), errors="coerce")
    duration_hours = pd.to_numeric(frame["unitdischargeoffset"], errors="coerce") / 60.0
    return {
        "icu_stays": int(frame["patientunitstayid"].nunique()),
        "health_system_stays": int(frame["patienthealthsystemstayid"].nunique()),
        "patients": int(frame["uniquepid"].nunique()),
        "adult_stays": int((age >= 18).sum()),
        "stays_at_least_24h": int((duration_hours >= 24).sum()),
        "adult_stays_at_least_24h": int(((age >= 18) & (duration_hours >= 24)).sum()),
        "median_icu_hours": float(duration_hours.median()),
    }


def build_markdown(report: dict[str, Any]) -> str:
    cohort = report["cohort"]
    lines = [
        "# eICU 外部驗證資料就緒稽核",
        "",
        f"產生時間：{report['generated_at_utc']}",
        "",
        f"整體狀態：**{'READY' if report['ready_for_preprocessing'] else 'NOT READY'}**",
        "",
        "## Cohort",
        "",
        f"- ICU stays：{cohort.get('icu_stays', 0):,}",
        f"- Unique patients：{cohort.get('patients', 0):,}",
        f"- Adult stays with at least 24 h：{cohort.get('adult_stays_at_least_24h', 0):,}",
        "",
        "## Predictor 與 outcome 來源",
        "",
        "| 訊號 | 狀態 | 來源 | Sample matched rows |",
        "|---|---:|---|---:|",
    ]
    for signal, item in report["signals"].items():
        lines.append(
            f"| `{signal}` | {'OK' if item['available'] else 'Missing'} | "
            f"{item['source']} | {item.get('sample_matches', 0):,} |"
        )
    lines.extend(
        [
            "",
            "## SOFA components",
            "",
            "| Component | 狀態 |",
            "|---|---:|",
        ]
    )
    for component, available in report["sofa_components"].items():
        lines.append(f"| {component} | {'OK' if available else 'Missing'} |")
    lines.extend(
        [
            "",
            "## 外部驗證約束",
            "",
            "- eICU 使用 `uniquepid` 作為 patient-level identifier、`patientunitstayid` 作為 stay identifier。",
            "- 所有事件時間以 ICU admission-relative offset 對齊至整點，不使用未來量測補值。",
            "- MIMIC-IV 訓練完成的模型與 calibration 必須原封不動套用至 eICU；不得以 eICU test outcome 重新調參。",
            "- 升壓藥只採可換算為 mcg/kg/min 的紀錄；無法確認濃度的 mL/hr 不可直接當劑量。",
            "- 尿量必須使用逐筆 urinary output 欄位，排除 `outputtotal` 與其他 drain/chest-tube output。",
            "- 此稽核證明資料來源齊全，不代表 MIMIC/eICU 單位與定義已完成 harmonization。",
            "",
            f"Schema fingerprint：`{report['schema_fingerprint_sha256']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table_report: dict[str, Any] = {}
    schema_errors = []
    for name, required in REQUIRED_TABLES.items():
        path = dataset_dir / name
        if not path.exists():
            table_report[name] = {"exists": False, "missing_columns": sorted(required)}
            schema_errors.append(f"missing table: {name}")
            continue
        columns = list(pd.read_csv(path, nrows=0).columns)
        missing = sorted(required - set(columns))
        table_report[name] = {
            "exists": True,
            "bytes": path.stat().st_size,
            "columns": columns,
            "missing_columns": missing,
        }
        if missing:
            schema_errors.append(f"{name}: missing columns {missing}")

    cohort = {}
    if table_report.get("patient.csv.gz", {}).get("exists") and not table_report["patient.csv.gz"]["missing_columns"]:
        cohort = patient_summary(dataset_dir / "patient.csv.gz")

    sampled: dict[str, Any] = {}
    sample_specs = {
        "lab.csv.gz": ["labname"],
        "nurseCharting.csv.gz": ["nursingchartcelltypevallabel", "nursingchartcelltypevalname"],
        "respiratoryCharting.csv.gz": ["respchartvaluelabel"],
        "infusionDrug.csv.gz": ["drugname"],
        "intakeOutput.csv.gz": ["celllabel"],
    }
    counters: dict[str, dict[str, Counter[str]]] = {}
    for name, columns in sample_specs.items():
        if not table_report.get(name, {}).get("exists") or table_report[name]["missing_columns"]:
            continue
        values, rows = sampled_value_counts(
            dataset_dir / name,
            columns,
            args.sample_rows,
            args.chunksize,
        )
        counters[name] = values
        sampled[name] = {"rows": rows}

    lab_counter = counters.get("lab.csv.gz", {}).get("labname", Counter())
    nurse_counter = Counter()
    for counter in counters.get("nurseCharting.csv.gz", {}).values():
        nurse_counter.update(counter)
    respiratory_counter = counters.get("respiratoryCharting.csv.gz", {}).get(
        "respchartvaluelabel", Counter()
    )
    drug_counter = counters.get("infusionDrug.csv.gz", {}).get("drugname", Counter())
    io_counter = counters.get("intakeOutput.csv.gz", {}).get("celllabel", Counter())

    schema_ok = not schema_errors
    direct_signals = {
        "heart_rate": ("vitalPeriodic.csv.gz", "heartrate"),
        "respiratory_rate": ("vitalPeriodic.csv.gz", "respiration"),
        "spo2": ("vitalPeriodic.csv.gz", "sao2"),
        "temperature_c": ("vitalPeriodic.csv.gz", "temperature"),
        "sbp": ("vitalAperiodic.csv.gz", "noninvasivesystolic"),
        "map": ("vitalAperiodic.csv.gz", "noninvasivemean"),
    }
    signals: dict[str, dict[str, Any]] = {}
    for signal, (table, column) in direct_signals.items():
        available = table_report.get(table, {}).get("exists", False) and column in table_report[table].get("columns", [])
        signals[signal] = {"available": bool(available), "source": f"{table}:{column}", "sample_matches": 0}

    for signal, terms in LAB_TERMS.items():
        count = count_matches(lab_counter, terms)
        signals[signal] = {
            "available": count > 0 if args.sample_rows > 0 else schema_ok,
            "source": "lab.csv.gz:labname",
            "sample_matches": count,
        }

    gcs_count = count_matches(nurse_counter, ("gcs total", "glasgow coma score"))
    fio2_count = count_matches(respiratory_counter, ("fio2", "fraction of inspired oxygen"))
    urine_count = count_matches(
        io_counter,
        ("urine", "foley", "voided amount", "urinary catheter", "nephrostomy", "ureteral stent"),
    )
    signals["gcs_total"] = {
        "available": gcs_count > 0 if args.sample_rows > 0 else schema_ok,
        "source": "nurseCharting.csv.gz:GCS labels",
        "sample_matches": gcs_count,
    }
    signals["fio2"] = {
        "available": fio2_count > 0 if args.sample_rows > 0 else schema_ok,
        "source": "respiratoryCharting.csv.gz:FiO2 labels",
        "sample_matches": fio2_count,
    }
    signals["pao2_fio2"] = {
        "available": signals["pao2"]["available"] and signals["fio2"]["available"],
        "source": "derived from PaO2 / FiO2",
        "sample_matches": min(signals["pao2"]["sample_matches"], fio2_count),
    }
    signals["urine_output"] = {
        "available": urine_count > 0 if args.sample_rows > 0 else schema_ok,
        "source": "intakeOutput.csv.gz:urinary output labels",
        "sample_matches": urine_count,
    }

    for drug in ("dopamine", "dobutamine", "epinephrine", "norepinephrine"):
        count = count_matches(drug_counter, (drug,), startswith=True)
        signals[drug] = {
            "available": count > 0 if args.sample_rows > 0 else schema_ok,
            "source": "infusionDrug.csv.gz:drugname",
            "sample_matches": count,
        }

    sofa_components = {
        "respiration": signals["pao2_fio2"]["available"],
        "coagulation": signals["platelets"]["available"],
        "liver": signals["bilirubin"]["available"],
        "cardiovascular": signals["map"]["available"] and all(signals[name]["available"] for name in ("dopamine", "dobutamine", "epinephrine", "norepinephrine")),
        "cns": signals["gcs_total"]["available"],
        "renal": signals["creatinine"]["available"] and signals["urine_output"]["available"],
    }
    required_predictors = (
        "heart_rate",
        "respiratory_rate",
        "spo2",
        "fio2",
        "temperature_c",
        "sbp",
        "gcs_total",
        "map",
        "pao2_fio2",
        "platelets",
        "bilirubin",
        "creatinine",
        "lactate",
    )

    schema_payload = {
        name: {"bytes": item.get("bytes"), "columns": item.get("columns", [])}
        for name, item in table_report.items()
    }
    fingerprint = hashlib.sha256(
        json.dumps(schema_payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "schema_fingerprint_sha256": fingerprint,
        "schema_errors": schema_errors,
        "tables": table_report,
        "sampled_tables": sampled,
        "cohort": cohort,
        "signals": signals,
        "required_predictors": list(required_predictors),
        "sofa_components": sofa_components,
        "ready_for_preprocessing": (
            schema_ok
            and all(signals[name]["available"] for name in required_predictors)
            and all(sofa_components.values())
        ),
    }

    json_path = output_dir / "eicu_data_audit.json"
    md_path = output_dir / "eicu_data_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(report), encoding="utf-8")
    print(f"eICU readiness: {'READY' if report['ready_for_preprocessing'] else 'NOT READY'}")
    print(f"JSON: {json_path}")
    print(f"Markdown: {md_path}")
    if schema_errors:
        raise SystemExit("; ".join(schema_errors))


if __name__ == "__main__":
    main()
