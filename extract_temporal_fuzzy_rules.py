"""Extract data-supported temporal fuzzy rules from a frozen FNN checkpoint.

候選規則只由模型參數與預先定義的 temporal signals 產生。排名不使用 test
outcome；event rate 僅在規則排名固定後作描述，避免 outcome-driven rule selection。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from anfis_model import (
    EXPLICIT_TEMPORAL_SIGNAL_NAMES,
    FEATURE_ORDER,
    TemporalAttentionFNN,
    clinical_rule_priors,
    expert_feature_config,
)
from patient_split import split_ids_for_values
from train_fnn import (
    ICUWindowDataset,
    choose_device,
    load_training_frame,
    prepare_explicit_temporal_arrays,
)


FEATURE_LABELS = {
    "heart_rate": "heart rate",
    "respiratory_rate": "respiratory rate",
    "spo2": "SpO2",
    "fio2": "FiO2 requirement",
    "temperature_c": "temperature",
    "sbp": "systolic blood pressure",
    "gcs_total": "GCS",
    "map": "MAP",
    "pao2_fio2": "PaO2/FiO2 ratio",
    "platelets": "platelet count",
    "bilirubin": "bilirubin",
    "creatinine": "creatinine",
    "lactate": "lactate",
}

TEMPORAL_SIGNALS = (
    "risk_slope",
    "short_term_change",
    "window_change",
    "abnormal_duration",
    "abnormal_frequency",
)

TEMPORAL_THRESHOLDS = {
    "risk_slope": 0.02,
    "short_term_change": 0.05,
    "window_change": 0.10,
    "abnormal_duration": 0.50,
    "abnormal_frequency": 0.50,
}


@dataclass
class CandidateRule:
    rule_id: str
    antecedents: tuple[tuple[str, str], ...]
    temporal_feature: str
    temporal_signal: str
    cross_rule_index: int | None
    raw_weight: float
    clinical_concordance: float
    support: int = 0
    positives: int = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def term_label(term: str) -> str:
    return term.replace("_", " ")


def temporal_phrase(feature: str, signal: str) -> str:
    label = FEATURE_LABELS[feature]
    phrases = {
        "risk_slope": f"{label}-related fuzzy risk is progressively increasing",
        "short_term_change": f"{label}-related fuzzy risk increased during the last hour",
        "window_change": f"{label}-related fuzzy risk increased over the 24-hour window",
        "abnormal_duration": f"{label} abnormality persisted across the observation window",
        "abnormal_frequency": f"{label} measurements were frequently abnormal",
    }
    return phrases[signal]


def rule_text(rule: CandidateRule) -> str:
    static_parts = [
        f"{FEATURE_LABELS[feature]} IS {term_label(term)}"
        for feature, term in rule.antecedents
    ]
    conditions = " AND ".join([*static_parts, temporal_phrase(rule.temporal_feature, rule.temporal_signal)])
    return f"IF {conditions} THEN deterioration risk IS high"


def trained_term_weight(model: TemporalAttentionFNN, feature: str, term: str) -> float:
    index = model.static_fnn.term_index[feature][term]
    return float(model.static_fnn.rule_weights[feature][index].detach().cpu().item())


def initial_term_weight(model: TemporalAttentionFNN, feature: str, term: str) -> float:
    index = model.static_fnn.term_index[feature][term]
    values = getattr(model.static_fnn, f"initial_rule_weights__{feature}")
    return float(values[index].detach().cpu().item())


def candidate_concordance(
    model: TemporalAttentionFNN,
    antecedents: tuple[tuple[str, str], ...],
    temporal_feature: str,
    temporal_signal: str,
    cross_rule_index: int | None,
) -> float:
    checks = []
    for feature, term in antecedents:
        checks.append(
            initial_term_weight(model, feature, term) > 0
            and trained_term_weight(model, feature, term) >= 0
        )
    signal_index = EXPLICIT_TEMPORAL_SIGNAL_NAMES.index(temporal_signal)
    feature_index = FEATURE_ORDER.index(temporal_feature)
    temporal_weight = model.explicit_temporal_weights[feature_index, signal_index]
    checks.append(float(temporal_weight.detach().cpu().item()) >= 0)
    if cross_rule_index is not None:
        checks.append(
            float(model.static_fnn.cross_rule_weights[cross_rule_index].detach().cpu().item()) >= 0
        )
    return float(np.mean(checks)) if checks else math.nan


def candidate_weight(
    model: TemporalAttentionFNN,
    antecedents: tuple[tuple[str, str], ...],
    temporal_feature: str,
    temporal_signal: str,
    cross_rule_index: int | None,
) -> float:
    static_weight = float(np.mean([trained_term_weight(model, *item) for item in antecedents]))
    feature_index = FEATURE_ORDER.index(temporal_feature)
    signal_index = EXPLICIT_TEMPORAL_SIGNAL_NAMES.index(temporal_signal)
    temporal_weight = float(
        model.explicit_temporal_weights[feature_index, signal_index].detach().cpu().item()
    )
    temporal_weight *= float(model.explicit_temporal_scale) / math.sqrt(len(FEATURE_ORDER))
    cross_weight = 0.0
    if cross_rule_index is not None:
        cross_weight = float(
            model.static_fnn.cross_rule_weights[cross_rule_index].detach().cpu().item()
        ) * float(model.static_fnn.rule_score_scale)
    return max(static_weight + temporal_weight + cross_weight, 0.0)


def strongest_temporal_signals(
    model: TemporalAttentionFNN,
    feature: str,
    count: int,
) -> list[str]:
    feature_index = FEATURE_ORDER.index(feature)
    weighted = []
    for signal in TEMPORAL_SIGNALS:
        signal_index = EXPLICIT_TEMPORAL_SIGNAL_NAMES.index(signal)
        value = float(
            model.explicit_temporal_weights[feature_index, signal_index].detach().cpu().item()
        )
        weighted.append((signal, value))
    return [signal for signal, _ in sorted(weighted, key=lambda item: item[1], reverse=True)[:count]]


def build_candidates(model: TemporalAttentionFNN) -> list[CandidateRule]:
    candidates = []
    for feature in FEATURE_ORDER:
        configs = expert_feature_config[feature]
        high_risk = [config for config in configs if float(config["weight"]) > 0]
        if not high_risk:
            continue
        term = max(high_risk, key=lambda config: trained_term_weight(model, feature, config["name"]))[
            "name"
        ]
        antecedents = ((feature, term),)
        for signal in strongest_temporal_signals(model, feature, count=2):
            candidates.append(
                CandidateRule(
                    rule_id=f"single::{feature}::{term}::{signal}",
                    antecedents=antecedents,
                    temporal_feature=feature,
                    temporal_signal=signal,
                    cross_rule_index=None,
                    raw_weight=candidate_weight(model, antecedents, feature, signal, None),
                    clinical_concordance=candidate_concordance(
                        model, antecedents, feature, signal, None
                    ),
                )
            )

    for rule_index, rule in enumerate(clinical_rule_priors):
        antecedents = tuple((str(feature), str(term)) for feature, term in rule["antecedents"])
        choices = []
        for feature, _ in antecedents:
            signal = strongest_temporal_signals(model, feature, count=1)[0]
            signal_index = EXPLICIT_TEMPORAL_SIGNAL_NAMES.index(signal)
            feature_index = FEATURE_ORDER.index(feature)
            weight = float(
                model.explicit_temporal_weights[feature_index, signal_index].detach().cpu().item()
            )
            choices.append((feature, signal, weight))
        temporal_feature, signal, _ = max(choices, key=lambda item: item[2])
        candidates.append(
            CandidateRule(
                rule_id=f"cross::{rule['name']}::{temporal_feature}::{signal}",
                antecedents=antecedents,
                temporal_feature=temporal_feature,
                temporal_signal=signal,
                cross_rule_index=rule_index,
                raw_weight=candidate_weight(
                    model,
                    antecedents,
                    temporal_feature,
                    signal,
                    rule_index,
                ),
                clinical_concordance=candidate_concordance(
                    model,
                    antecedents,
                    temporal_feature,
                    signal,
                    rule_index,
                ),
            )
        )
    return candidates


def candidate_mask(
    candidate: CandidateRule,
    model: TemporalAttentionFNN,
    output,
    membership_threshold: float,
    cross_membership_threshold: float,
) -> torch.Tensor:
    mask = torch.ones(output.logits.shape[0], dtype=torch.bool, device=output.logits.device)
    threshold = (
        cross_membership_threshold
        if candidate.cross_rule_index is not None
        else membership_threshold
    )
    for feature, term in candidate.antecedents:
        term_index = model.static_fnn.term_index[feature][term]
        mask &= output.memberships[feature][:, -1, term_index] >= threshold
    feature_index = FEATURE_ORDER.index(candidate.temporal_feature)
    signal_index = EXPLICIT_TEMPORAL_SIGNAL_NAMES.index(candidate.temporal_signal)
    temporal_value = output.explicit_temporal_features[:, feature_index, signal_index]
    mask &= temporal_value >= TEMPORAL_THRESHOLDS[candidate.temporal_signal]
    return mask


def extract_rules(
    model: TemporalAttentionFNN,
    dataset: ICUWindowDataset | Subset,
    device: torch.device,
    batch_size: int,
    membership_threshold: float,
    cross_membership_threshold: float,
) -> tuple[pd.DataFrame, float]:
    candidates = build_candidates(model)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    total = 0
    positives_total = 0
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            output = model(batch_x)
            total += int(batch_y.numel())
            positives_total += int(batch_y.sum().item())
            for candidate in candidates:
                active = candidate_mask(
                    candidate,
                    model,
                    output,
                    membership_threshold,
                    cross_membership_threshold,
                )
                candidate.support += int(active.sum().item())
                candidate.positives += int(batch_y[active].sum().item())

    prevalence = positives_total / max(total, 1)
    max_weight = max((candidate.raw_weight for candidate in candidates), default=1.0)
    rows = []
    for candidate in candidates:
        support_fraction = candidate.support / max(total, 1)
        positive_rate = candidate.positives / candidate.support if candidate.support else math.nan
        normalized_weight = candidate.raw_weight / max(max_weight, 1e-8)
        rows.append(
            {
                "rule_id": candidate.rule_id,
                "extracted_temporal_fuzzy_rule": rule_text(candidate),
                "rule_weight": normalized_weight,
                "raw_model_weight": candidate.raw_weight,
                "support": candidate.support,
                "support_fraction": support_fraction,
                "positive": candidate.positives,
                "positive_rate": positive_rate,
                "positive_rate_lift": positive_rate / prevalence if candidate.support else math.nan,
                "clinical_concordance": candidate.clinical_concordance,
                "ranking_score": normalized_weight * math.sqrt(support_fraction),
                "temporal_feature": candidate.temporal_feature,
                "temporal_signal": candidate.temporal_signal,
                "membership_threshold": (
                    cross_membership_threshold
                    if candidate.cross_rule_index is not None
                    else membership_threshold
                ),
                "temporal_threshold": TEMPORAL_THRESHOLDS[candidate.temporal_signal],
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["ranking_score", "rule_weight"], ascending=False
    ).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame, prevalence


def write_markdown(
    path: Path,
    rules: pd.DataFrame,
    checkpoint_hash: str,
    test_windows: int,
    prevalence: float,
    top_k: int,
) -> None:
    selected = rules[rules["support"] > 0].head(top_k)
    cross_rules = rules[
        rules["rule_id"].astype(str).str.startswith("cross::") & (rules["support"] > 0)
    ].head(5)
    lines = [
        "# Extracted Temporal Fuzzy Rules",
        "",
        "這些規則由 frozen full-cohort 6-hour FNN 與實際 MIMIC-IV test windows 萃取，並非假想規則。",
        "候選規則與排名只使用模型參數及 support；positive rate 未參與規則挑選。",
        "",
        f"- Frozen checkpoint SHA-256: `{checkpoint_hash}`",
        f"- Test windows: {test_windows:,}",
        f"- Overall deterioration rate: {prevalence:.2%}",
        "- Rule weight: model-derived weight normalized to the largest candidate rule.",
        "- Clinical concordance: fraction of static, temporal and cross-rule directions aligned with guideline priors.",
        "",
        "## Main-Text Cross-Feature Examples",
        "",
        "| Rank | Extracted temporal fuzzy rule | Rule weight | Support | Positive rate | Clinical concordance |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for rank, (_, row) in enumerate(cross_rules.iterrows(), start=1):
        lines.append(
            f"| {rank} | {row['extracted_temporal_fuzzy_rule']} | "
            f"{row['rule_weight']:.3f} | n={int(row['support']):,} | "
            f"{row['positive_rate']:.1%} | {row['clinical_concordance']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Cross-feature antecedents use fuzzy membership >= 0.35; single-feature rules use >= 0.50.",
            "",
            "## Overall Top Model-Supported Rules",
            "",
        "| Rank | Extracted temporal fuzzy rule | Rule weight | Support | Positive rate | Clinical concordance |",
        "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in selected.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['extracted_temporal_fuzzy_rule']} | "
            f"{row['rule_weight']:.3f} | n={int(row['support']):,} | "
            f"{row['positive_rate']:.1%} | {row['clinical_concordance']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Method Note",
            "",
            "`risk_slope`, `short_term_change`, and `window_change` describe changes in learned fuzzy risk, "
            "not raw-value slopes. Persistence and frequency are computed from the model's differentiable "
            "abnormality probability over the 24-hour observation window.",
            "",
            "完整候選規則、support、event counts、lift 與 thresholds 見 `extracted_temporal_rules.csv`。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract actual temporal fuzzy rules.")
    parser.add_argument("--run-dir", default="outputs/explicit_temporal_fnn_formal_6h/seed_42")
    parser.add_argument("--output-dir", default="outputs/temporal_rule_extraction_6h")
    parser.add_argument("--markdown", default="docs/extracted_temporal_rules_6h.md")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--membership-threshold", type=float, default=0.50)
    parser.add_argument("--cross-membership-threshold", type=float, default=0.35)
    parser.add_argument("--min-support", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=0, help="Smoke test only.")
    parser.add_argument("--max-stays", type=int, default=0, help="Smoke test only.")
    parser.add_argument("--max-test-windows", type=int, default=0, help="Smoke test only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
    checkpoint_path = run_dir / "best_model.pt"
    checkpoint_hash = sha256_file(checkpoint_path)
    if not config.get("explicit_temporal_features", False):
        raise ValueError("Rule extraction requires an explicit-temporal FNN checkpoint.")

    input_order = config["input_order"]
    df = load_training_frame(
        csv_path=Path(config["csv"]),
        feature_cols=input_order,
        target_col=config["target_col"],
        time_col=config["time_col"],
        split_col=config["split_col"],
        max_rows=args.max_rows or None,
        max_stays=args.max_stays or None,
        chunk_size=config.get("chunk_size", 500_000),
        sofa_csv=config.get("sofa_csv"),
    )
    features, labels, stay_ids, split_values, time_values = prepare_explicit_temporal_arrays(
        df,
        config["target_col"],
        config["time_col"],
        config["split_col"],
    )
    _, _, test_ids = split_ids_for_values(split_values, config["split_manifest"])
    dataset: ICUWindowDataset | Subset = ICUWindowDataset(
        features=features,
        labels=labels,
        stay_ids=stay_ids,
        split_values=split_values,
        time_values=time_values,
        allowed_split_values=test_ids,
        seq_length=config["seq_length"],
    )
    if args.max_test_windows and len(dataset) > args.max_test_windows:
        rng = np.random.default_rng(42)
        indices = np.sort(rng.choice(len(dataset), args.max_test_windows, replace=False))
        dataset = Subset(dataset, indices.tolist())

    device = choose_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TemporalAttentionFNN(
        seq_length=config["seq_length"],
        attention_hidden=config["attention_hidden"],
        threshold=config["threshold"],
        rule_score_scale=config["rule_score_scale"],
        use_explicit_temporal_features=True,
        explicit_temporal_scale=config["explicit_temporal_scale"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    rules, prevalence = extract_rules(
        model,
        dataset,
        device,
        args.batch_size,
        args.membership_threshold,
        args.cross_membership_threshold,
    )
    eligible = rules[rules["support"] >= args.min_support].copy()
    eligible["rank"] = np.arange(1, len(eligible) + 1)
    rules.to_csv(output_dir / "all_candidate_temporal_rules.csv", index=False)
    eligible.to_csv(output_dir / "extracted_temporal_rules.csv", index=False)
    write_markdown(
        output_dir / "extracted_temporal_rules.md",
        eligible,
        checkpoint_hash,
        len(dataset),
        prevalence,
        args.top_k,
    )
    if args.markdown:
        write_markdown(
            Path(args.markdown),
            eligible,
            checkpoint_hash,
            len(dataset),
            prevalence,
            args.top_k,
        )
    (output_dir / "rule_extraction_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "checkpoint_sha256": checkpoint_hash,
                "test_windows": len(dataset),
                "test_prevalence": prevalence,
                "eligible_rules": len(eligible),
                "ranking_uses_outcome": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Extracted {len(eligible)} supported temporal fuzzy rules: {output_dir}")


if __name__ == "__main__":
    main()
