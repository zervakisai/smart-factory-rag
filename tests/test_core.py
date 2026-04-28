"""Tests for SmartFactory-RAG core components."""

import numpy as np
import pytest

from src.ml.predictor import FailurePredictor, FeatureEngineer, Prediction
from src.sensors.ingester import AnomalyDetector, RingBuffer, SensorIngester


class TestFeatureEngineer:
    """Test sensor feature extraction."""

    def setup_method(self):
        self.fe = FeatureEngineer(window_size=60)

    def test_extract_returns_expected_features(self):
        window = np.random.randn(60, 6)
        features = self.fe.extract(window)
        assert isinstance(features, dict)
        assert "temperature_mean" in features
        assert "vibration_rms" in features
        assert "temp_pressure_corr" in features
        assert "vibration_rpm_ratio" in features

    def test_extract_feature_count(self):
        window = np.random.randn(60, 6)
        features = self.fe.extract(window)
        # 6 sensors × 9 features + 4 cross-sensor = 58
        assert len(features) >= 58

    def test_constant_signal_has_zero_trend(self):
        window = np.ones((60, 6)) * 5.0
        features = self.fe.extract(window)
        assert abs(features["temperature_trend"]) < 1e-10

    def test_rising_signal_has_positive_trend(self):
        window = np.zeros((60, 6))
        window[:, 0] = np.linspace(0, 10, 60)  # Rising temperature
        features = self.fe.extract(window)
        assert features["temperature_trend"] > 0


class TestRingBuffer:
    """Test sensor ring buffer."""

    def test_push_and_retrieve(self):
        buf = RingBuffer(capacity=10, n_sensors=6)
        for i in range(5):
            buf.push(np.ones(6) * i, timestamp=float(i))
        window = buf.get_window()
        assert window.shape == (5, 6)
        assert window[-1, 0] == 4.0

    def test_overflow_wraps_correctly(self):
        buf = RingBuffer(capacity=5, n_sensors=6)
        for i in range(8):
            buf.push(np.ones(6) * i, timestamp=float(i))
        window = buf.get_window()
        assert window.shape == (5, 6)
        assert window[-1, 0] == 7.0
        assert window[0, 0] == 3.0

    def test_is_full(self):
        buf = RingBuffer(capacity=10, n_sensors=6)
        assert not buf.is_full
        for i in range(10):
            buf.push(np.ones(6), timestamp=float(i))
        assert buf.is_full

    def test_stats(self):
        buf = RingBuffer(capacity=100, n_sensors=6)
        for i in range(50):
            buf.push(np.ones(6) * i, timestamp=float(i))
        stats = buf.stats()
        assert stats["count"] == 50
        assert len(stats["mean"]) == 6


class TestAnomalyDetector:
    """Test anomaly detection."""

    def test_no_anomaly_on_normal_data(self):
        detector = AnomalyDetector(z_threshold=3.0, min_samples=10)
        buf = RingBuffer(capacity=100, n_sensors=6)
        for i in range(50):
            buf.push(np.ones(6) * 5.0 + np.random.randn(6) * 0.1, timestamp=float(i))

        normal_reading = np.ones(6) * 5.0
        anomalies = detector.check(
            normal_reading, buf,
            ["temp", "pres", "vib", "rpm", "cur", "hum"],
            "equipment_1"
        )
        assert len(anomalies) == 0

    def test_detects_spike_anomaly(self):
        detector = AnomalyDetector(z_threshold=3.0, min_samples=10)
        buf = RingBuffer(capacity=100, n_sensors=6)
        for i in range(50):
            buf.push(np.ones(6) * 5.0, timestamp=float(i))

        spike = np.ones(6) * 5.0
        spike[0] = 100.0  # Massive temperature spike
        anomalies = detector.check(
            spike, buf,
            ["temp", "pres", "vib", "rpm", "cur", "hum"],
            "equipment_1"
        )
        assert len(anomalies) > 0
        assert any(a.sensor_name == "temp" for a in anomalies)


class TestPrediction:
    """Test prediction data class."""

    def test_severity_levels(self):
        p1 = Prediction(0.95, 2.0, "bearing_wear", "Replace", [], (0.9, 1.0), "v1")
        assert p1.severity == "CRITICAL"

        p2 = Prediction(0.75, 10.0, "overheating", "Cool", [], (0.7, 0.8), "v1")
        assert p2.severity == "WARNING"

        p3 = Prediction(0.2, None, "none", "OK", [], (0.1, 0.3), "v1")
        assert p3.severity == "NORMAL"
