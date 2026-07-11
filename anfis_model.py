"""Knowledge-Guided Temporal Fuzzy Neural Network.

本檔案對齊研究計畫二版的模型設計：
1. 使用 NEWS2 / SOFA 臨床門檻初始化 fuzzy membership functions。
2. 建立 feature-level fuzzy scoring 與 cross-feature IF-THEN rule layer。
3. 加入 temporal attention，融合過去多小時的風險軌跡。
4. 輸出 raw logits 與 probability，方便使用 BCEWithLogitsLoss 訓練。
5. 提供 clinical consistency、rule sparsity、rule drift 等 regularization loss。

模型輸入預期為：
    x_seq: (Batch, Seq_Length, Num_Features)

其中 Num_Features 的順序必須等於 FEATURE_ORDER。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# 與正式 model_hourly_features_v3.csv 對齊的預設特徵順序。
FEATURE_ORDER = [
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
]


EXPLICIT_TEMPORAL_SIGNAL_NAMES = [
    "mean_risk",
    "min_risk",
    "max_risk",
    "risk_std",
    "risk_slope",
    "short_term_change",
    "window_change",
    "abnormal_duration",
    "abnormal_frequency",
    "missing_fraction",
    "current_missing",
    "time_since_last_measurement",
]


def explicit_temporal_input_order(
    feature_order: list[str] | None = None,
) -> list[str]:
    """回傳 explicit temporal FNN 的固定輸入欄位順序。"""
    features = FEATURE_ORDER if feature_order is None else feature_order
    return [
        *features,
        *(f"{feature}_is_missing" for feature in features),
        *(f"{feature}_time_since_last_measurement_h" for feature in features),
    ]


# NEWS2 + SOFA guided membership function initialization。
# weight 是該 fuzzy linguistic term 的臨床異常程度初始分數。
expert_feature_config = {
    "heart_rate": [
        {"name": "very_low", "center": 35.0, "sigma": 6.0, "weight": 3.0},
        {"name": "normal", "center": 70.0, "sigma": 15.0, "weight": 0.0},
        {"name": "mild_high", "center": 100.0, "sigma": 8.0, "weight": 1.0},
        {"name": "high", "center": 120.0, "sigma": 8.0, "weight": 2.0},
        {"name": "critical_high", "center": 140.0, "sigma": 10.0, "weight": 3.0},
    ],
    "respiratory_rate": [
        {"name": "very_low", "center": 7.0, "sigma": 2.0, "weight": 3.0},
        {"name": "mild_low", "center": 10.0, "sigma": 1.5, "weight": 1.0},
        {"name": "normal", "center": 16.0, "sigma": 4.0, "weight": 0.0},
        {"name": "high", "center": 22.5, "sigma": 2.0, "weight": 2.0},
        {"name": "critical_high", "center": 28.0, "sigma": 3.0, "weight": 3.0},
    ],
    "spo2": [
        {"name": "critical_low", "center": 88.0, "sigma": 2.0, "weight": 3.0},
        {"name": "low", "center": 92.5, "sigma": 1.2, "weight": 2.0},
        {"name": "mild_low", "center": 94.5, "sigma": 1.0, "weight": 1.0},
        {"name": "normal", "center": 97.0, "sigma": 2.5, "weight": 0.0},
    ],
    "fio2": [
        {"name": "room_air", "center": 0.21, "sigma": 0.03, "weight": 0.0},
        {"name": "supplemental_o2", "center": 0.40, "sigma": 0.12, "weight": 2.0},
        {"name": "high_support", "center": 0.80, "sigma": 0.18, "weight": 2.0},
    ],
    "temperature_c": [
        {"name": "very_low", "center": 34.5, "sigma": 0.6, "weight": 3.0},
        {"name": "mild_low", "center": 35.6, "sigma": 0.5, "weight": 1.0},
        {"name": "normal", "center": 37.0, "sigma": 0.8, "weight": 0.0},
        {"name": "fever", "center": 38.5, "sigma": 0.5, "weight": 1.0},
        {"name": "high_fever", "center": 39.5, "sigma": 0.6, "weight": 2.0},
    ],
    "sbp": [
        {"name": "very_low", "center": 85.0, "sigma": 8.0, "weight": 3.0},
        {"name": "low", "center": 95.0, "sigma": 5.0, "weight": 2.0},
        {"name": "mild_low", "center": 105.0, "sigma": 5.0, "weight": 1.0},
        {"name": "normal", "center": 130.0, "sigma": 18.0, "weight": 0.0},
        {"name": "very_high", "center": 225.0, "sigma": 15.0, "weight": 3.0},
    ],
    "gcs_total": [
        {"name": "severely_altered", "center": 5.0, "sigma": 2.0, "weight": 4.0},
        {"name": "altered", "center": 12.0, "sigma": 2.0, "weight": 3.0},
        {"name": "normal", "center": 15.0, "sigma": 0.8, "weight": 0.0},
    ],
    "map": [
        {"name": "low", "center": 60.0, "sigma": 8.0, "weight": 1.0},
        {"name": "normal", "center": 85.0, "sigma": 12.0, "weight": 0.0},
    ],
    "pao2_fio2": [
        {"name": "critical_low", "center": 80.0, "sigma": 25.0, "weight": 4.0},
        {"name": "very_low", "center": 150.0, "sigma": 35.0, "weight": 3.0},
        {"name": "low", "center": 250.0, "sigma": 45.0, "weight": 2.0},
        {"name": "mild_low", "center": 350.0, "sigma": 45.0, "weight": 1.0},
        {"name": "normal", "center": 450.0, "sigma": 80.0, "weight": 0.0},
    ],
    "platelets": [
        {"name": "critical_low", "center": 15.0, "sigma": 8.0, "weight": 4.0},
        {"name": "very_low", "center": 40.0, "sigma": 12.0, "weight": 3.0},
        {"name": "low", "center": 80.0, "sigma": 18.0, "weight": 2.0},
        {"name": "mild_low", "center": 130.0, "sigma": 25.0, "weight": 1.0},
        {"name": "normal", "center": 220.0, "sigma": 70.0, "weight": 0.0},
    ],
    "bilirubin": [
        {"name": "normal", "center": 0.8, "sigma": 0.4, "weight": 0.0},
        {"name": "mild_high", "center": 1.5, "sigma": 0.5, "weight": 1.0},
        {"name": "high", "center": 3.0, "sigma": 1.0, "weight": 2.0},
        {"name": "very_high", "center": 8.0, "sigma": 2.0, "weight": 3.0},
        {"name": "critical_high", "center": 15.0, "sigma": 4.0, "weight": 4.0},
    ],
    "creatinine": [
        {"name": "normal", "center": 0.8, "sigma": 0.3, "weight": 0.0},
        {"name": "mild_high", "center": 1.5, "sigma": 0.4, "weight": 1.0},
        {"name": "high", "center": 2.5, "sigma": 0.6, "weight": 2.0},
        {"name": "very_high", "center": 4.0, "sigma": 0.8, "weight": 3.0},
        {"name": "critical_high", "center": 6.0, "sigma": 1.2, "weight": 4.0},
    ],
    "lactate": [
        {"name": "normal", "center": 1.2, "sigma": 0.5, "weight": 0.0},
        {"name": "elevated", "center": 2.5, "sigma": 0.7, "weight": 1.0},
        {"name": "high", "center": 4.0, "sigma": 1.0, "weight": 2.0},
        {"name": "severe", "center": 8.0, "sigma": 2.0, "weight": 3.0},
    ],
}


# Clinical rule priors。這層形成跨特徵 IF-THEN fuzzy rules。
clinical_rule_priors = [
    {
        "name": "respiratory_failure_pattern",
        "antecedents": [("spo2", "critical_low"), ("respiratory_rate", "critical_high")],
        "weight": 5.0,
    },
    {
        "name": "shock_pattern",
        "antecedents": [("sbp", "very_low"), ("heart_rate", "critical_high")],
        "weight": 5.0,
    },
    {
        "name": "oxygenation_failure_with_support",
        "antecedents": [("pao2_fio2", "very_low"), ("fio2", "high_support")],
        "weight": 4.0,
    },
    {
        "name": "hypoperfusion_pattern",
        "antecedents": [("lactate", "high"), ("map", "low")],
        "weight": 4.0,
    },
    {
        "name": "multi_organ_dysfunction_pattern",
        "antecedents": [("creatinine", "high"), ("platelets", "low"), ("bilirubin", "high")],
        "weight": 5.0,
    },
    {
        "name": "altered_consciousness_hypoxemia",
        "antecedents": [("gcs_total", "altered"), ("spo2", "low")],
        "weight": 4.0,
    },
]


@dataclass
class TemporalFNNOutput:
    """模型 forward 的結構化輸出。"""

    logits: torch.Tensor
    probabilities: torch.Tensor
    attention_weights: torch.Tensor
    hourly_risk_scores: torch.Tensor
    feature_risks: torch.Tensor
    rule_activations: torch.Tensor
    raw_rule_firing: torch.Tensor
    memberships: dict[str, torch.Tensor]
    explicit_temporal_features: torch.Tensor | None = None
    explicit_temporal_contributions: torch.Tensor | None = None


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    """將正數初始化值轉成 raw parameter，使 softplus(raw) 約等於原值。"""
    value = torch.clamp(value, min=1e-4)
    return torch.log(torch.expm1(value))


class ExpertGuidedStaticFNN(nn.Module):
    """單一時間點的 expert-guided fuzzy neural network。

    此層負責：
    1. Fuzzification：將連續數值轉成 fuzzy membership。
    2. Feature-level rule prior：依臨床權重計算每個特徵的異常分數。
    3. Cross-feature rule layer：計算 IF-THEN 規則 firing strength。
    """

    def __init__(
        self,
        feature_configs: dict[str, list[dict]],
        rule_configs: list[dict] | None = None,
        rule_score_scale: float = 0.2,
    ) -> None:
        super().__init__()
        self.feature_names = list(feature_configs.keys())
        self.rule_configs = rule_configs or []
        # Cross-feature IF-THEN rules can overlap with feature-level NEWS2/SOFA scores.
        # Keep this scale small initially so rule priors act as a clinical boost instead
        # of double-counting the same abnormal physiology too aggressively.
        self.rule_score_scale = rule_score_scale

        self.centers = nn.ParameterDict()
        self.raw_sigmas = nn.ParameterDict()
        self.rule_weights = nn.ParameterDict()

        self.term_names: dict[str, list[str]] = {}
        self.term_index: dict[str, dict[str, int]] = {}

        for feat, configs in feature_configs.items():
            self.term_names[feat] = [config["name"] for config in configs]
            self.term_index[feat] = {name: idx for idx, name in enumerate(self.term_names[feat])}

            center_init = torch.tensor([config["center"] for config in configs], dtype=torch.float32)
            sigma_init = torch.tensor([config["sigma"] for config in configs], dtype=torch.float32)
            weight_init = torch.tensor([config["weight"] for config in configs], dtype=torch.float32)

            self.centers[feat] = nn.Parameter(center_init)
            self.raw_sigmas[feat] = nn.Parameter(_inverse_softplus(sigma_init))
            self.rule_weights[feat] = nn.Parameter(weight_init)

            self.register_buffer(f"initial_centers__{feat}", center_init.clone())
            self.register_buffer(f"initial_sigmas__{feat}", sigma_init.clone())
            self.register_buffer(f"initial_rule_weights__{feat}", weight_init.clone())

        self.cross_rule_weights = nn.Parameter(
            torch.tensor([rule["weight"] for rule in self.rule_configs], dtype=torch.float32)
        )
        if self.rule_configs:
            self.register_buffer("initial_cross_rule_weights", self.cross_rule_weights.detach().clone())
        else:
            self.register_buffer("initial_cross_rule_weights", torch.empty(0))

        self._validate_rule_configs()

    def _validate_rule_configs(self) -> None:
        """確認 rule prior 指到的 feature 與 linguistic term 都存在。"""
        for rule in self.rule_configs:
            for feat, term in rule["antecedents"]:
                if feat not in self.term_index:
                    raise ValueError(f"Rule {rule['name']} 使用不存在的 feature: {feat}")
                if term not in self.term_index[feat]:
                    raise ValueError(f"Rule {rule['name']} 使用不存在的 term: {feat}.{term}")

    def sigma(self, feat: str) -> torch.Tensor:
        """取得正值 sigma，避免訓練時 sigma 變成 0 或負值。"""
        return F.softplus(self.raw_sigmas[feat]) + 1e-4

    def forward(self, x_t: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """計算單一時間點的 fuzzy risk。

        Args:
            x_t: (Batch, Num_Features)

        Returns:
            dict containing static_risk_score, feature_risks, rule_activations, memberships.
        """
        if x_t.dim() != 2:
            raise ValueError(f"x_t 應為 (Batch, Num_Features)，目前收到 {tuple(x_t.shape)}")
        if x_t.shape[1] != len(self.feature_names):
            raise ValueError(
                f"輸入特徵數 {x_t.shape[1]} 與模型設定 {len(self.feature_names)} 不一致；"
                f"順序應為 {self.feature_names}"
            )

        batch_size = x_t.shape[0]
        memberships: dict[str, torch.Tensor] = {}
        feature_risk_list = []

        for feature_idx, feat in enumerate(self.feature_names):
            x_i = x_t[:, feature_idx].unsqueeze(1)
            sigma = self.sigma(feat)
            mu = torch.exp(-0.5 * ((x_i - self.centers[feat]) / sigma) ** 2)
            mu_normalized = mu / (mu.sum(dim=1, keepdim=True) + 1e-8)
            memberships[feat] = mu_normalized

            feature_score = torch.sum(mu_normalized * self.rule_weights[feat], dim=1)
            feature_risk_list.append(feature_score)

        feature_risks = torch.stack(feature_risk_list, dim=1)
        additive_risk_score = feature_risks.sum(dim=1)

        if self.rule_configs:
            activations = []
            for rule in self.rule_configs:
                antecedent_values = []
                for feat, term in rule["antecedents"]:
                    term_idx = self.term_index[feat][term]
                    antecedent_values.append(memberships[feat][:, term_idx])
                stacked = torch.stack(antecedent_values, dim=1)
                activations.append(torch.prod(stacked, dim=1))

            rule_firing = torch.stack(activations, dim=1)
            rule_activations = rule_firing / (rule_firing.sum(dim=1, keepdim=True) + 1e-8)
            cross_rule_score = torch.sum(rule_activations * self.cross_rule_weights, dim=1)
        else:
            rule_firing = torch.empty(batch_size, 0, device=x_t.device)
            rule_activations = torch.empty(batch_size, 0, device=x_t.device)
            cross_rule_score = torch.zeros(batch_size, device=x_t.device)

        static_risk_score = additive_risk_score + (self.rule_score_scale * cross_rule_score)

        return {
            "static_risk_score": static_risk_score,
            "feature_risks": feature_risks,
            "rule_activations": rule_activations,
            "raw_rule_firing": rule_firing,
            "memberships": memberships,
        }

    def sparsity_loss(self) -> torch.Tensor:
        """Rule sparsity loss，鼓勵規則權重簡潔。"""
        l1 = torch.zeros((), device=self.cross_rule_weights.device)
        for weights in self.rule_weights.values():
            l1 = l1 + torch.mean(torch.abs(weights))
        if self.cross_rule_weights.numel() > 0:
            l1 = l1 + torch.mean(torch.abs(self.cross_rule_weights))
        return l1

    def drift_loss(self) -> torch.Tensor:
        """Rule drift loss，衡量訓練後參數偏離初始臨床知識的程度。"""
        drift = torch.zeros((), device=self.cross_rule_weights.device)
        count = 0
        for feat in self.feature_names:
            initial_centers = getattr(self, f"initial_centers__{feat}")
            initial_sigmas = getattr(self, f"initial_sigmas__{feat}")
            initial_weights = getattr(self, f"initial_rule_weights__{feat}")

            drift = drift + F.mse_loss(self.centers[feat], initial_centers)
            drift = drift + F.mse_loss(self.sigma(feat), initial_sigmas)
            drift = drift + F.mse_loss(self.rule_weights[feat], initial_weights)
            count += 3

        if self.cross_rule_weights.numel() > 0:
            drift = drift + F.mse_loss(self.cross_rule_weights, self.initial_cross_rule_weights)
            count += 1
        return drift / max(count, 1)

    def nonnegative_weight_loss(self) -> torch.Tensor:
        """避免 clinical risk weight 被訓練成負值。"""
        penalty = torch.zeros((), device=self.cross_rule_weights.device)
        for weights in self.rule_weights.values():
            penalty = penalty + torch.mean(F.relu(-weights))
        if self.cross_rule_weights.numel() > 0:
            penalty = penalty + torch.mean(F.relu(-self.cross_rule_weights))
        return penalty


class TemporalAttentionFNN(nn.Module):
    """加入 temporal attention 的完整 Knowledge-Guided Temporal FNN。"""

    def __init__(
        self,
        feature_configs: dict[str, list[dict]] = expert_feature_config,
        rule_configs: list[dict] | None = clinical_rule_priors,
        seq_length: int = 24,
        attention_hidden: int = 32,
        threshold: float = 7.0,
        rule_score_scale: float = 0.2,
        use_explicit_temporal_features: bool = False,
        explicit_temporal_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.feature_names = list(feature_configs.keys())
        self.use_explicit_temporal_features = use_explicit_temporal_features
        self.explicit_temporal_scale = explicit_temporal_scale
        self.temporal_signal_names = list(EXPLICIT_TEMPORAL_SIGNAL_NAMES)
        self.static_fnn = ExpertGuidedStaticFNN(
            feature_configs,
            rule_configs,
            rule_score_scale=rule_score_scale,
        )

        self.attention_net = nn.Sequential(
            nn.Linear(1, attention_hidden),
            nn.Tanh(),
            nn.Linear(attention_hidden, 1),
        )
        self.bias = nn.Parameter(torch.tensor([-threshold], dtype=torch.float32))

        if self.use_explicit_temporal_features:
            # 小幅 clinical boost 初始化，避免與 hourly fuzzy risk 重複加分。
            initial_signal_weights = torch.tensor(
                [0.05, 0.02, 0.10, 0.03, 0.12, 0.12, 0.12, 0.20, 0.20, 0.02, 0.02, 0.02],
                dtype=torch.float32,
            )
            initial_weights = initial_signal_weights.repeat(len(self.feature_names), 1)
            self.explicit_temporal_weights = nn.Parameter(initial_weights)
            self.register_buffer(
                "initial_explicit_temporal_weights",
                initial_weights.detach().clone(),
            )
        else:
            self.register_parameter("explicit_temporal_weights", None)

    def _explicit_temporal_features(
        self,
        feature_risks: torch.Tensor,
        missing: torch.Tensor,
        time_since_last: torch.Tensor,
    ) -> torch.Tensor:
        """從 fuzzy risk 軌跡建立明確且 leakage-free 的 temporal features。"""
        _, seq_length, _ = feature_risks.shape
        mean_risk = feature_risks.mean(dim=1) / 4.0
        min_risk = feature_risks.min(dim=1).values / 4.0
        max_risk = feature_risks.max(dim=1).values / 4.0
        risk_std = feature_risks.std(dim=1, unbiased=False) / 2.0

        if seq_length > 1:
            hours = torch.arange(
                seq_length,
                dtype=feature_risks.dtype,
                device=feature_risks.device,
            )
            centered = hours - hours.mean()
            slope = torch.sum(feature_risks * centered.view(1, -1, 1), dim=1)
            slope = slope / (torch.sum(centered.square()) + 1e-8)
            short_change = feature_risks[:, -1, :] - feature_risks[:, -2, :]
            window_change = feature_risks[:, -1, :] - feature_risks[:, 0, :]
        else:
            slope = torch.zeros_like(mean_risk)
            short_change = torch.zeros_like(mean_risk)
            window_change = torch.zeros_like(mean_risk)

        # 0.5 約代表至少輕度異常；平滑門檻可讓 membership parameters 接收梯度。
        abnormal_probability = torch.sigmoid((feature_risks - 0.5) / 0.20)
        abnormal_duration = abnormal_probability.mean(dim=1)
        observed = 1.0 - missing.clamp(0.0, 1.0)
        observed_count = observed.sum(dim=1)
        abnormal_frequency = torch.sum(abnormal_probability * observed, dim=1)
        abnormal_frequency = abnormal_frequency / observed_count.clamp_min(1.0)
        abnormal_frequency = torch.where(
            observed_count > 0,
            abnormal_frequency,
            torch.zeros_like(abnormal_frequency),
        )

        missing_fraction = missing.clamp(0.0, 1.0).mean(dim=1)
        current_missing = missing[:, -1, :].clamp(0.0, 1.0)
        time_since = torch.log1p(time_since_last[:, -1, :].clamp(0.0, 168.0))
        time_since = time_since / math.log(169.0)

        return torch.stack(
            [
                mean_risk,
                min_risk,
                max_risk,
                risk_std,
                torch.tanh(slope),
                torch.tanh(short_change),
                torch.tanh(window_change),
                abnormal_duration,
                abnormal_frequency,
                missing_fraction,
                current_missing,
                time_since,
            ],
            dim=-1,
        )

    def temporal_sparsity_loss(self) -> torch.Tensor:
        if self.explicit_temporal_weights is None:
            return self.bias.new_tensor(0.0)
        return torch.mean(torch.abs(self.explicit_temporal_weights))

    def temporal_drift_loss(self) -> torch.Tensor:
        if self.explicit_temporal_weights is None:
            return self.bias.new_tensor(0.0)
        return F.mse_loss(
            self.explicit_temporal_weights,
            self.initial_explicit_temporal_weights,
        )

    def temporal_nonnegative_loss(self) -> torch.Tensor:
        if self.explicit_temporal_weights is None:
            return self.bias.new_tensor(0.0)
        return torch.mean(F.relu(-self.explicit_temporal_weights))

    def forward(self, x_seq: torch.Tensor) -> TemporalFNNOutput:
        """Forward pass。

        Args:
            x_seq: (Batch, Seq_Length, Num_Features)
        """
        if x_seq.dim() != 3:
            raise ValueError(f"x_seq 應為 (Batch, Seq_Length, Num_Features)，目前收到 {tuple(x_seq.shape)}")

        batch_size, seq_length, num_features = x_seq.shape
        if seq_length != self.seq_length:
            raise ValueError(f"輸入序列長度 {seq_length} 與模型設定 seq_length={self.seq_length} 不一致")
        base_feature_count = len(self.feature_names)
        expected_features = base_feature_count * (3 if self.use_explicit_temporal_features else 1)
        if num_features != expected_features:
            raise ValueError(
                f"輸入特徵數 {num_features} 與模型設定 {len(self.feature_names)} 不一致；"
                f"順序應為 {self.feature_names}"
            )

        raw_x = x_seq[:, :, :base_feature_count]
        flat_x = raw_x.reshape(batch_size * seq_length, base_feature_count)
        static_out = self.static_fnn(flat_x)

        hourly_risk_scores = static_out["static_risk_score"].reshape(batch_size, seq_length)
        feature_risks = static_out["feature_risks"].reshape(
            batch_size,
            seq_length,
            base_feature_count,
        )
        rule_activations = static_out["rule_activations"].reshape(
            batch_size,
            seq_length,
            -1,
        )
        raw_rule_firing = static_out["raw_rule_firing"].reshape(
            batch_size,
            seq_length,
            -1,
        )
        memberships = {
            feat: values.reshape(batch_size, seq_length, values.shape[-1])
            for feat, values in static_out["memberships"].items()
        }

        attention_logits = self.attention_net(hourly_risk_scores.unsqueeze(-1)).squeeze(-1)
        attention_weights = F.softmax(attention_logits, dim=1)

        temporal_risk_score = torch.sum(hourly_risk_scores * attention_weights, dim=1)
        explicit_features = None
        explicit_contributions = None
        if self.use_explicit_temporal_features:
            missing_start = base_feature_count
            time_since_start = base_feature_count * 2
            missing = x_seq[:, :, missing_start:time_since_start]
            time_since_last = x_seq[:, :, time_since_start:]
            explicit_features = self._explicit_temporal_features(
                feature_risks,
                missing,
                time_since_last,
            )
            explicit_contributions = explicit_features * self.explicit_temporal_weights.unsqueeze(0)
            explicit_score = explicit_contributions.sum(dim=(1, 2)) / math.sqrt(base_feature_count)
            temporal_risk_score = temporal_risk_score + self.explicit_temporal_scale * explicit_score
        logits = temporal_risk_score + self.bias.squeeze(0)
        probabilities = torch.sigmoid(logits)

        return TemporalFNNOutput(
            logits=logits,
            probabilities=probabilities,
            attention_weights=attention_weights,
            hourly_risk_scores=hourly_risk_scores,
            feature_risks=feature_risks,
            rule_activations=rule_activations,
            raw_rule_firing=raw_rule_firing,
            memberships=memberships,
            explicit_temporal_features=explicit_features,
            explicit_temporal_contributions=explicit_contributions,
        )


class NeuroSymbolicLoss(nn.Module):
    """Prediction loss + clinical consistency + sparsity + drift regularization."""

    def __init__(
        self,
        lambda_cons: float = 0.1,
        lambda_sparse: float = 0.01,
        lambda_drift: float = 0.01,
        lambda_nonnegative: float = 0.05,
    ) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.lambda_cons = lambda_cons
        self.lambda_sparse = lambda_sparse
        self.lambda_drift = lambda_drift
        self.lambda_nonnegative = lambda_nonnegative

    @staticmethod
    def clinical_consistency_loss(output: TemporalFNNOutput) -> torch.Tensor:
        """若 feature risk 上升但 hourly total risk 下降，給予懲罰。"""
        if output.hourly_risk_scores.shape[1] < 2:
            return output.hourly_risk_scores.new_tensor(0.0)
        delta_total = output.hourly_risk_scores[:, 1:] - output.hourly_risk_scores[:, :-1]
        delta_feature = output.feature_risks[:, 1:, :] - output.feature_risks[:, :-1, :]

        penalty = F.relu(delta_feature) * F.relu(-delta_total.unsqueeze(-1))
        return penalty.mean()

    def forward(
        self,
        output: TemporalFNNOutput,
        targets: torch.Tensor,
        model: TemporalAttentionFNN,
    ) -> dict[str, torch.Tensor]:
        targets = targets.float().view_as(output.logits)
        pred_loss = self.bce(output.logits, targets)
        zero = pred_loss.new_tensor(0.0)
        cons_loss = self.clinical_consistency_loss(output) if self.lambda_cons > 0 else zero
        sparse_loss = (
            model.static_fnn.sparsity_loss() + model.temporal_sparsity_loss()
            if self.lambda_sparse > 0
            else zero
        )
        drift_loss = (
            model.static_fnn.drift_loss() + model.temporal_drift_loss()
            if self.lambda_drift > 0
            else zero
        )
        nonnegative_loss = (
            model.static_fnn.nonnegative_weight_loss() + model.temporal_nonnegative_loss()
            if self.lambda_nonnegative > 0
            else zero
        )

        total_loss = (
            pred_loss
            + self.lambda_cons * cons_loss
            + self.lambda_sparse * sparse_loss
            + self.lambda_drift * drift_loss
            + self.lambda_nonnegative * nonnegative_loss
        )

        return {
            "total": total_loss,
            "prediction": pred_loss,
            "clinical_consistency": cons_loss,
            "rule_sparsity": sparse_loss,
            "rule_drift": drift_loss,
            "nonnegative_weights": nonnegative_loss,
        }


def extract_top_rules(
    model: TemporalAttentionFNN,
    output: TemporalFNNOutput,
    sample_index: int = 0,
    hour_index: int | None = None,
    top_k: int = 5,
) -> list[dict]:
    """萃取單一樣本最活躍的 IF-THEN rule，供 rule evaluation 使用。"""
    if output.rule_activations.shape[-1] == 0:
        return []

    if hour_index is None:
        hour_index = int(torch.argmax(output.attention_weights[sample_index]).item())

    activations = output.rule_activations[sample_index, hour_index]
    top_values, top_indices = torch.topk(activations, k=min(top_k, activations.numel()))

    rules = []
    for value, idx in zip(top_values.detach().cpu(), top_indices.detach().cpu()):
        rule = model.static_fnn.rule_configs[int(idx)]
        rules.append(
            {
                "rule_name": rule["name"],
                "antecedents": rule["antecedents"],
                "activation": float(value),
                "weight": float(model.static_fnn.cross_rule_weights[int(idx)].detach().cpu()),
                "hour_index": hour_index,
            }
        )
    return rules


def smoke_test() -> None:
    """快速確認模型、loss、rule extraction 都能執行。"""
    torch.manual_seed(7)
    model = TemporalAttentionFNN(seq_length=24)
    criterion = NeuroSymbolicLoss(lambda_cons=0.5, lambda_sparse=0.01, lambda_drift=0.01)

    batch_size = 4
    num_features = len(model.feature_names)
    x_seq = torch.zeros(batch_size, 24, num_features)

    # 先用正常值填滿，再讓前兩個樣本後 8 小時變成高風險狀態。
    normal_values = torch.tensor(
        [75.0, 16.0, 98.0, 21.0, 37.0, 125.0, 15.0, 85.0, 450.0, 220.0, 0.8, 0.8, 1.2]
    )
    high_risk_values = torch.tensor(
        [135.0, 28.0, 88.0, 80.0, 39.2, 85.0, 10.0, 60.0, 120.0, 45.0, 3.5, 2.6, 4.2]
    )
    x_seq[:] = normal_values
    x_seq[:2, 16:, :] = high_risk_values

    targets = torch.tensor([1.0, 1.0, 0.0, 0.0])
    output = model(x_seq)
    losses = criterion(output, targets, model)

    print("Feature order:", model.feature_names)
    print("Probabilities:", output.probabilities.detach().cpu().numpy())
    print("Attention sums:", output.attention_weights.sum(dim=1).detach().cpu().numpy())
    print("Loss:", {key: round(value.item(), 4) for key, value in losses.items()})
    print("Top rules sample 0:", extract_top_rules(model, output, sample_index=0, top_k=3))


if __name__ == "__main__":
    smoke_test()
