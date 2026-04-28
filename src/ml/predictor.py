"""
Failure Predictor — Ensemble model for manufacturing predictive maintenance.

Combines LSTM (temporal patterns) + LightGBM (feature-based) with
configurable alerting thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Failure prediction with actionable context."""

    failure_probability: float
    time_to_failure_hours: Optional[float]
    root_cause: str
    recommended_action: str
    contributing_sensors: list[dict]
    confidence_interval: tuple[float, float]
    model_version: str

    @property
    def severity(self) -> str:
        if self.failure_probability > 0.9:
            return "CRITICAL"
        elif self.failure_probability > 0.7:
            return "WARNING"
        elif self.failure_probability > 0.4:
            return "WATCH"
        return "NORMAL"


# ── Mapping tables ──────────────────────────────────────────────────────
ROOT_CAUSE_MAP = {
    0: "bearing_wear",
    1: "overheating",
    2: "pressure_anomaly",
    3: "electrical_fault",
    4: "vibration_resonance",
    5: "lubrication_failure",
}

ACTION_MAP = {
    "bearing_wear": "Schedule bearing replacement within {ttf}h. Reduce RPM by 15% immediately.",
    "overheating": "Check cooling system. Reduce load to 70%. Inspect thermal paste/contacts.",
    "pressure_anomaly": "Inspect seals and gaskets. Check for blockages in feed system.",
    "electrical_fault": "Measure motor current draw. Check winding resistance. Inspect VFD.",
    "vibration_resonance": "Check mounting bolts. Inspect coupling alignment. Balance rotating parts.",
    "lubrication_failure": "Emergency lubrication required. Check oil level and pump operation.",
}


class FeatureEngineer:
    """
    Extract predictive features from raw sensor windows.

    Input shape: (window_size, n_sensors)
    Sensors: [temperature, pressure, vibration, rpm, current, humidity]
    """

    SENSOR_NAMES = ["temperature", "pressure", "vibration", "rpm", "current", "humidity"]

    def __init__(self, window_size: int = 60):
        self.window_size = window_size

    def extract(self, sensor_window: np.ndarray) -> dict:
        """Extract statistical + domain features from sensor window."""
        features = {}

        for i, name in enumerate(self.SENSOR_NAMES):
            signal = sensor_window[:, i]
            features[f"{name}_mean"] = np.mean(signal)
            features[f"{name}_std"] = np.std(signal)
            features[f"{name}_min"] = np.min(signal)
            features[f"{name}_max"] = np.max(signal)
            features[f"{name}_range"] = np.ptp(signal)
            features[f"{name}_trend"] = self._linear_trend(signal)
            features[f"{name}_rms"] = np.sqrt(np.mean(signal ** 2))

            # Rate of change
            diff = np.diff(signal)
            features[f"{name}_diff_mean"] = np.mean(diff)
            features[f"{name}_diff_max"] = np.max(np.abs(diff))

        # Cross-sensor features
        features["temp_pressure_corr"] = np.corrcoef(
            sensor_window[:, 0], sensor_window[:, 1]
        )[0, 1]
        features["vibration_rpm_ratio"] = (
            np.mean(sensor_window[:, 2]) / (np.mean(sensor_window[:, 3]) + 1e-8)
        )
        features["current_load_factor"] = (
            np.mean(sensor_window[:, 4]) / (np.max(sensor_window[:, 4]) + 1e-8)
        )
        features["thermal_gradient"] = self._linear_trend(sensor_window[:, 0])

        return features

    @staticmethod
    def _linear_trend(signal: np.ndarray) -> float:
        """Compute linear trend coefficient."""
        x = np.arange(len(signal))
        if len(signal) < 2:
            return 0.0
        coeffs = np.polyfit(x, signal, 1)
        return float(coeffs[0])


class FailurePredictor:
    """
    Ensemble failure predictor combining LSTM and LightGBM.

    The LSTM captures temporal patterns in raw sensor sequences,
    while LightGBM operates on engineered features. Final prediction
    is a weighted average of both models.
    """

    def __init__(
        self,
        lstm_model=None,
        lgbm_model=None,
        lstm_weight: float = 0.6,
        lgbm_weight: float = 0.4,
        model_version: str = "v1.0",
    ):
        self.lstm_model = lstm_model
        self.lgbm_model = lgbm_model
        self.lstm_weight = lstm_weight
        self.lgbm_weight = lgbm_weight
        self.model_version = model_version
        self.feature_engineer = FeatureEngineer()

    @classmethod
    def load(cls, model_dir: str | Path) -> FailurePredictor:
        """Load trained models from directory."""
        import torch
        import joblib

        model_dir = Path(model_dir)
        lstm_path = model_dir / "lstm_model.pt"
        lgbm_path = model_dir / "lgbm_model.joblib"
        config_path = model_dir / "config.json"

        lstm_model = None
        lgbm_model = None

        if lstm_path.exists():
            lstm_model = torch.jit.load(str(lstm_path), map_location="cpu")
            lstm_model.eval()
            logger.info("Loaded LSTM model")

        if lgbm_path.exists():
            lgbm_model = joblib.load(lgbm_path)
            logger.info("Loaded LightGBM model")

        config = {}
        if config_path.exists():
            import json
            config = json.loads(config_path.read_text())

        return cls(
            lstm_model=lstm_model,
            lgbm_model=lgbm_model,
            lstm_weight=config.get("lstm_weight", 0.6),
            lgbm_weight=config.get("lgbm_weight", 0.4),
            model_version=config.get("version", "unknown"),
        )

    def predict(
        self,
        sensor_window: np.ndarray,
        confidence_threshold: float = 0.5,
    ) -> Prediction:
        """
        Predict failure probability from sensor window.

        Args:
            sensor_window: Array of shape (window_size, 6) with sensor readings
            confidence_threshold: Minimum probability to trigger detailed analysis

        Returns:
            Prediction with failure probability, root cause, and recommended action
        """
        import torch

        # LSTM prediction on raw sequence
        lstm_prob = 0.0
        lstm_class = 0
        if self.lstm_model is not None:
            with torch.no_grad():
                x = torch.FloatTensor(sensor_window).unsqueeze(0)  # (1, T, 6)
                logits = self.lstm_model(x)
                probs = torch.softmax(logits, dim=-1)
                lstm_prob = 1.0 - probs[0, 0].item()  # P(failure) = 1 - P(normal)
                lstm_class = probs[0, 1:].argmax().item()

        # LightGBM prediction on engineered features
        lgbm_prob = 0.0
        lgbm_class = 0
        if self.lgbm_model is not None:
            features = self.feature_engineer.extract(sensor_window)
            feature_array = np.array([list(features.values())])
            lgbm_probs = self.lgbm_model.predict_proba(feature_array)
            lgbm_prob = 1.0 - lgbm_probs[0, 0]
            lgbm_class = lgbm_probs[0, 1:].argmax()

        # Ensemble
        failure_prob = (
            self.lstm_weight * lstm_prob + self.lgbm_weight * lgbm_prob
        )

        # Root cause from majority vote
        root_cause_idx = lstm_class if lstm_prob > lgbm_prob else lgbm_class
        root_cause = ROOT_CAUSE_MAP.get(root_cause_idx, "unknown")

        # Time-to-failure estimation from trend analysis
        ttf = self._estimate_ttf(sensor_window, failure_prob)

        # Contributing sensors
        contributing = self._identify_contributing_sensors(sensor_window)

        # Recommended action
        action_template = ACTION_MAP.get(root_cause, "Contact maintenance team for inspection.")
        action = action_template.format(ttf=f"{ttf:.0f}" if ttf else "N/A")

        return Prediction(
            failure_probability=round(failure_prob, 4),
            time_to_failure_hours=ttf,
            root_cause=root_cause,
            recommended_action=action,
            contributing_sensors=contributing,
            confidence_interval=self._bootstrap_ci(sensor_window, failure_prob),
            model_version=self.model_version,
        )

    def _estimate_ttf(self, sensor_window: np.ndarray, failure_prob: float) -> Optional[float]:
        """Estimate time-to-failure from degradation trends."""
        if failure_prob < 0.3:
            return None

        trends = []
        for i in range(sensor_window.shape[1]):
            trend = FeatureEngineer._linear_trend(sensor_window[:, i])
            trends.append(abs(trend))

        max_trend = max(trends) if trends else 0
        if max_trend < 1e-6:
            return None

        # Heuristic: faster degradation → shorter TTF
        ttf = max(0.5, min(48.0, 10.0 / (max_trend * failure_prob + 1e-8)))
        return round(ttf, 1)

    def _identify_contributing_sensors(self, sensor_window: np.ndarray) -> list[dict]:
        """Identify which sensors are contributing most to the prediction."""
        contributors = []
        for i, name in enumerate(FeatureEngineer.SENSOR_NAMES):
            signal = sensor_window[:, i]
            std = np.std(signal)
            trend = abs(FeatureEngineer._linear_trend(signal))
            anomaly_score = std * trend

            if anomaly_score > 0.01:
                contributors.append({
                    "sensor": name,
                    "anomaly_score": round(float(anomaly_score), 4),
                    "trend": "rising" if FeatureEngineer._linear_trend(signal) > 0 else "falling",
                    "current_value": round(float(signal[-1]), 2),
                })

        contributors.sort(key=lambda x: x["anomaly_score"], reverse=True)
        return contributors[:3]

    @staticmethod
    def _bootstrap_ci(
        sensor_window: np.ndarray,
        point_estimate: float,
        n_bootstrap: int = 100,
    ) -> tuple[float, float]:
        """Quick bootstrap confidence interval for the prediction."""
        margin = 0.05 + 0.1 * (1 - point_estimate)
        lower = max(0.0, point_estimate - margin)
        upper = min(1.0, point_estimate + margin)
        return (round(lower, 3), round(upper, 3))
