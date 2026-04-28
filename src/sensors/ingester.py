"""
Sensor Ingester — Real-time MQTT telemetry ingestion with anomaly detection.

Features:
    - Async MQTT subscription with reconnection
    - Ring buffer for sliding window analysis
    - Z-score + IQR anomaly detection
    - Event-driven callback system
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SensorReading:
    """Single sensor reading with metadata."""

    equipment_id: str
    sensor_name: str
    value: float
    timestamp: float
    unit: str
    metadata: dict = field(default_factory=dict)


@dataclass
class AnomalyEvent:
    """Detected anomaly event."""

    equipment_id: str
    anomaly_type: str
    sensor_name: str
    current_value: float
    expected_range: tuple[float, float]
    severity: str  # "low", "medium", "high", "critical"
    timestamp: float
    window_stats: dict

    def __str__(self) -> str:
        return (
            f"[{self.severity.upper()}] {self.equipment_id}/{self.sensor_name}: "
            f"{self.current_value:.2f} (expected {self.expected_range[0]:.2f}–{self.expected_range[1]:.2f})"
        )


class RingBuffer:
    """
    Fixed-size ring buffer for sensor time series with efficient statistics.
    """

    def __init__(self, capacity: int = 1000, n_sensors: int = 6):
        self.capacity = capacity
        self.n_sensors = n_sensors
        self._buffer = np.zeros((capacity, n_sensors))
        self._timestamps = np.zeros(capacity)
        self._write_idx = 0
        self._count = 0

    def push(self, values: np.ndarray, timestamp: float) -> None:
        """Add a sensor reading to the buffer."""
        self._buffer[self._write_idx % self.capacity] = values
        self._timestamps[self._write_idx % self.capacity] = timestamp
        self._write_idx += 1
        self._count = min(self._count + 1, self.capacity)

    def get_window(self, size: Optional[int] = None) -> np.ndarray:
        """Get the last N readings as a contiguous array."""
        if self._count == 0:
            return np.zeros((0, self.n_sensors))

        size = min(size or self._count, self._count)
        if self._count < self.capacity:
            return self._buffer[:self._count][-size:]

        end = self._write_idx % self.capacity
        if end >= size:
            return self._buffer[end - size:end]
        else:
            return np.vstack([
                self._buffer[self.capacity - (size - end):],
                self._buffer[:end],
            ])

    @property
    def is_full(self) -> bool:
        return self._count >= self.capacity

    def stats(self) -> dict:
        """Compute running statistics."""
        window = self.get_window()
        if len(window) == 0:
            return {}
        return {
            "mean": np.mean(window, axis=0).tolist(),
            "std": np.std(window, axis=0).tolist(),
            "min": np.min(window, axis=0).tolist(),
            "max": np.max(window, axis=0).tolist(),
            "count": self._count,
        }


class AnomalyDetector:
    """
    Real-time anomaly detection using Z-score and IQR methods.
    """

    def __init__(
        self,
        z_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        min_samples: int = 30,
    ):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.min_samples = min_samples

    def check(
        self,
        current_values: np.ndarray,
        buffer: RingBuffer,
        sensor_names: list[str],
        equipment_id: str,
    ) -> list[AnomalyEvent]:
        """Check for anomalies in the current reading."""
        if buffer._count < self.min_samples:
            return []

        window = buffer.get_window()
        anomalies = []
        stats = buffer.stats()

        for i, (value, name) in enumerate(zip(current_values, sensor_names)):
            col = window[:, i]
            mean = np.mean(col)
            std = np.std(col)

            # Z-score check
            if std > 0:
                z_score = abs(value - mean) / std
                if z_score > self.z_threshold:
                    severity = self._classify_severity(z_score)
                    anomalies.append(AnomalyEvent(
                        equipment_id=equipment_id,
                        anomaly_type="z_score_violation",
                        sensor_name=name,
                        current_value=value,
                        expected_range=(mean - 2 * std, mean + 2 * std),
                        severity=severity,
                        timestamp=time.time(),
                        window_stats={
                            "mean": float(mean),
                            "std": float(std),
                            "z_score": float(z_score),
                        },
                    ))

            # IQR check
            q1, q3 = np.percentile(col, [25, 75])
            iqr = q3 - q1
            lower_fence = q1 - self.iqr_multiplier * iqr
            upper_fence = q3 + self.iqr_multiplier * iqr

            if value < lower_fence or value > upper_fence:
                anomalies.append(AnomalyEvent(
                    equipment_id=equipment_id,
                    anomaly_type="iqr_outlier",
                    sensor_name=name,
                    current_value=value,
                    expected_range=(lower_fence, upper_fence),
                    severity="medium",
                    timestamp=time.time(),
                    window_stats={
                        "q1": float(q1),
                        "q3": float(q3),
                        "iqr": float(iqr),
                    },
                ))

        return anomalies

    @staticmethod
    def _classify_severity(z_score: float) -> str:
        if z_score > 5.0:
            return "critical"
        elif z_score > 4.0:
            return "high"
        elif z_score > 3.0:
            return "medium"
        return "low"


AnomalyCallback = Callable[[AnomalyEvent], Coroutine[Any, Any, None]]


class SensorIngester:
    """
    Production sensor ingestion pipeline with MQTT.

    Features:
        - Async MQTT subscription with auto-reconnect
        - Per-equipment ring buffers
        - Real-time anomaly detection
        - Event-driven callback system
    """

    SENSOR_NAMES = ["temperature", "pressure", "vibration", "rpm", "current", "humidity"]

    def __init__(
        self,
        mqtt_broker: str = "mqtt://localhost:1883",
        topics: Optional[list[str]] = None,
        buffer_size: int = 1000,
        anomaly_z_threshold: float = 3.0,
    ):
        self.mqtt_broker = mqtt_broker
        self.topics = topics or ["factory/#"]
        self.buffer_size = buffer_size
        self.detector = AnomalyDetector(z_threshold=anomaly_z_threshold)
        self._buffers: dict[str, RingBuffer] = {}
        self._anomaly_callbacks: list[AnomalyCallback] = []
        self._running = False
        self._stats = {
            "messages_received": 0,
            "anomalies_detected": 0,
            "start_time": None,
        }

    def on_anomaly(self, func: AnomalyCallback) -> AnomalyCallback:
        """Decorator to register anomaly callback."""
        self._anomaly_callbacks.append(func)
        return func

    def _get_buffer(self, equipment_id: str) -> RingBuffer:
        """Get or create ring buffer for equipment."""
        if equipment_id not in self._buffers:
            self._buffers[equipment_id] = RingBuffer(
                capacity=self.buffer_size,
                n_sensors=len(self.SENSOR_NAMES),
            )
        return self._buffers[equipment_id]

    async def _process_message(self, topic: str, payload: bytes) -> None:
        """Process a single MQTT message."""
        try:
            data = json.loads(payload)
            equipment_id = data.get("equipment_id", topic.split("/")[-2])
            timestamp = data.get("timestamp", time.time())

            # Extract sensor values
            values = np.array([
                data.get(name, 0.0) for name in self.SENSOR_NAMES
            ])

            # Push to buffer
            buffer = self._get_buffer(equipment_id)
            buffer.push(values, timestamp)
            self._stats["messages_received"] += 1

            # Check for anomalies
            anomalies = self.detector.check(
                values, buffer, self.SENSOR_NAMES, equipment_id
            )

            for anomaly in anomalies:
                self._stats["anomalies_detected"] += 1
                logger.warning(f"Anomaly detected: {anomaly}")
                for callback in self._anomaly_callbacks:
                    await callback(anomaly)

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Failed to process message on {topic}: {e}")

    async def start(self) -> None:
        """Start the MQTT ingestion loop."""
        import aiomqtt

        self._running = True
        self._stats["start_time"] = time.time()
        logger.info(f"Starting sensor ingestion from {self.mqtt_broker}")

        while self._running:
            try:
                # Parse broker URL
                broker_host = self.mqtt_broker.replace("mqtt://", "").split(":")[0]
                broker_port = int(self.mqtt_broker.split(":")[-1]) if ":" in self.mqtt_broker.replace("mqtt://", "") else 1883

                async with aiomqtt.Client(broker_host, broker_port) as client:
                    for topic in self.topics:
                        await client.subscribe(topic)
                    logger.info(f"Subscribed to {self.topics}")

                    async for message in client.messages:
                        await self._process_message(
                            str(message.topic),
                            message.payload,
                        )

            except Exception as e:
                logger.error(f"MQTT connection error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def stop(self) -> None:
        """Gracefully stop ingestion."""
        self._running = False
        logger.info(f"Ingestion stopped. Stats: {self._stats}")

    def get_window(self, equipment_id: str, size: int = 60) -> Optional[np.ndarray]:
        """Get recent sensor window for an equipment."""
        buffer = self._buffers.get(equipment_id)
        if buffer is None:
            return None
        return buffer.get_window(size)

    def get_stats(self) -> dict:
        """Get ingestion statistics."""
        stats = self._stats.copy()
        if stats["start_time"]:
            stats["uptime_seconds"] = time.time() - stats["start_time"]
            stats["messages_per_second"] = (
                stats["messages_received"] / stats["uptime_seconds"]
                if stats["uptime_seconds"] > 0 else 0
            )
        stats["active_equipment"] = list(self._buffers.keys())
        return stats
