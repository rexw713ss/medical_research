"""依 comparison_protocol.json 批次執行公平模型比較。

預設執行 equal-sample comparison：所有模型使用相同 200,000 train windows、
50,000 validation windows 與完整 test windows。使用 ``--mode full`` 可另跑
所有 eligible windows 的 full-cohort comparison。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from comparison_protocol import load_protocol
from project_config import (
    COMPARISON_PROTOCOL_JSON,
    DEFAULT_PREDICTION_HORIZONS,
    EQUAL_SAMPLE_WINDOWS_CSV,
    PATIENT_SPLIT_CSV,
)


DEFAULT_FAMILIES = ["clinical", "interpretable", "blackbox", "fnn", "evaluation"]


def parse_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run protocol-controlled fair model comparison.")
    parser.add_argument("--mode", choices=["full", "equal_sample"], default="equal_sample")
    parser.add_argument(
        "--horizons",
        default=",".join(map(str, DEFAULT_PREDICTION_HORIZONS)),
        help="Primary defaults to 6; explicitly pass 12,24 for secondary analyses.",
    )
    parser.add_argument(
        "--families",
        default=",".join(DEFAULT_FAMILIES),
        help="clinical,interpretable,blackbox,fnn,ablation,evaluation",
    )
    parser.add_argument("--protocol", default=COMPARISON_PROTOCOL_JSON)
    parser.add_argument("--equal-sample-windows", default=EQUAL_SAMPLE_WINDOWS_CSV)
    parser.add_argument("--patient-split", default=PATIENT_SPLIT_CSV)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--fnn-epochs", type=int, default=20)
    parser.add_argument("--rnn-epochs", type=int, default=20)
    parser.add_argument("--ablation-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def target_col(horizon: int) -> str:
    return f"label_sofa_increase_ge2_{horizon}h"


def shared_args(args: argparse.Namespace) -> list[str]:
    return [
        "--comparison-mode", args.mode,
        "--comparison-protocol", args.protocol,
        "--equal-sample-windows", args.equal_sample_windows,
        "--split-manifest", args.patient_split,
    ]


def command_record(name: str, command: list[str], output_dir: Path) -> dict[str, Any]:
    return {"name": name, "command": command, "output_dir": str(output_dir), "status": "pending"}


def build_commands(
    args: argparse.Namespace,
    output_root: Path,
    horizons: list[int],
    families: set[str],
) -> tuple[list[dict[str, Any]], list[Path]]:
    commands: list[dict[str, Any]] = []
    fnn_dirs: list[Path] = []
    shared = shared_args(args)

    if "clinical" in families:
        out = output_root / "clinical"
        command = [
            args.python, "clinical_score_baselines.py",
            "--horizons", ",".join(map(str, horizons)),
            "--comparison-mode", args.mode,
            "--comparison-protocol", args.protocol,
            "--equal-sample-windows", args.equal_sample_windows,
            "--split-manifest", args.patient_split,
            "--save-predictions",
            "--output-dir", str(out),
        ]
        commands.append(command_record("clinical", command, out))

    for horizon in horizons:
        target = target_col(horizon)
        if "interpretable" in families:
            out = output_root / f"interpretable_{horizon}h"
            command = [
                args.python, "interpretable_baselines.py",
                "--target-col", target,
                "--feature-set", "protocol",
                "--models", "all",
                "--save-predictions",
                "--output-dir", str(out),
                *shared,
            ]
            commands.append(command_record(f"interpretable_{horizon}h", command, out))

        if "blackbox" in families:
            out = output_root / f"blackbox_{horizon}h"
            command = [
                args.python, "blackbox_baselines.py",
                "--target-col", target,
                "--feature-set", "protocol",
                "--models", "all",
                "--sequence-epochs", str(args.rnn_epochs),
                "--sequence-batch-size", str(args.batch_size),
                "--device", args.device,
                "--save-predictions",
                "--output-dir", str(out),
                *shared,
            ]
            commands.append(command_record(f"blackbox_{horizon}h", command, out))

        if "fnn" in families:
            out = output_root / f"fnn_{horizon}h"
            fnn_dirs.append(out)
            command = [
                args.python, "train_fnn.py",
                "--target-col", target,
                "--epochs", str(args.fnn_epochs),
                "--batch-size", str(args.batch_size),
                "--device", args.device,
                "--output-dir", str(out),
                *shared,
            ]
            commands.append(command_record(f"fnn_{horizon}h", command, out))

        if "ablation" in families:
            out = output_root / f"ablation_{horizon}h"
            command = [
                args.python, "ablation_fnn_experiments.py",
                "--target-col", target,
                "--epochs", str(args.ablation_epochs),
                "--batch-size", str(args.batch_size),
                "--device", args.device,
                "--output-dir", str(out),
                *shared,
            ]
            commands.append(command_record(f"ablation_{horizon}h", command, out))

    if "evaluation" in families:
        out = output_root / "evaluation"
        command = [
            args.python, "model_evaluation_report.py",
            "--sources", "all",
            "--outputs-root", str(output_root),
            "--horizons", ",".join(map(str, horizons)),
            "--comparison-mode", args.mode,
            "--comparison-protocol", args.protocol,
            "--equal-sample-windows", args.equal_sample_windows,
            "--split-manifest", args.patient_split,
            "--device", args.device,
            "--output-dir", str(out),
        ]
        if fnn_dirs:
            command.extend(["--fnn-run-dirs", ",".join(map(str, fnn_dirs))])
        commands.append(command_record("evaluation", command, out))

    return commands, fnn_dirs


def save_run_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    horizons = [int(item) for item in parse_list(args.horizons)]
    invalid = [h for h in horizons if target_col(h) not in protocol["outcomes"]]
    if invalid:
        raise ValueError(f"Protocol 未定義 horizons: {invalid}")

    families = set(parse_list(args.families))
    valid_families = {"clinical", "interpretable", "blackbox", "fnn", "ablation", "evaluation"}
    unknown = families - valid_families
    if unknown:
        raise ValueError(f"未知 families: {sorted(unknown)}")

    run_name = datetime.now().strftime(f"fair_{args.mode}_%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) if args.output_dir else Path("outputs") / "fair_comparison" / run_name
    output_root.mkdir(parents=True, exist_ok=True)
    commands, _ = build_commands(args, output_root, horizons, families)
    state: dict[str, Any] = {
        "mode": args.mode,
        "horizons": horizons,
        "families": sorted(families),
        "protocol": args.protocol,
        "protocol_sha256": protocol["protocol_sha256"],
        "predictors": protocol["predictors"],
        "sequence_length_hours": protocol["sequence_length_hours"],
        "commands": commands,
    }
    state_path = output_root / "fair_comparison_run.json"
    save_run_state(state_path, state)

    for index, record in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {record['name']}")
        print(subprocess.list2cmdline(record["command"]))
        if args.dry_run:
            record["status"] = "dry_run"
            continue
        record["status"] = "running"
        save_run_state(state_path, state)
        try:
            subprocess.run(record["command"], cwd=Path.cwd(), check=True)
        except subprocess.CalledProcessError as exc:
            record["status"] = "failed"
            record["returncode"] = exc.returncode
            save_run_state(state_path, state)
            raise
        record["status"] = "completed"
        save_run_state(state_path, state)

    save_run_state(state_path, state)
    print(f"Fair comparison {'plan' if args.dry_run else 'run'} complete: {output_root}")


if __name__ == "__main__":
    main()
