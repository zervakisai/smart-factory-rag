"""
SmartFactory-RAG API — Unified gateway for RAG, predictions, and sensor data.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Schemas ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    rerank: bool = True
    language: Optional[str] = None  # "el", "en", or auto-detect

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    confidence: float
    retrieval_time_ms: float
    generation_time_ms: float

class PredictRequest(BaseModel):
    equipment_id: str
    window_size: int = Field(default=60, ge=10, le=500)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

class PredictResponse(BaseModel):
    equipment_id: str
    failure_probability: float
    severity: str
    time_to_failure_hours: Optional[float]
    root_cause: str
    recommended_action: str
    contributing_sensors: list[dict]

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    components: dict


# ── Application ─────────────────────────────────────────────────────────

_start_time = time.time()
_rag_engine = None
_predictor = None
_ingester = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize components on startup."""
    global _rag_engine, _predictor, _ingester

    from src.rag.engine import RAGEngine
    from src.ml.predictor import FailurePredictor
    from src.sensors.ingester import SensorIngester

    logger.info("Initializing SmartFactory-RAG components...")

    try:
        _rag_engine = RAGEngine(index_path="./data/index")
        logger.info("RAG engine initialized")
    except Exception as e:
        logger.warning(f"RAG engine not available: {e}")

    try:
        _predictor = FailurePredictor.load("./models/latest")
        logger.info("Failure predictor loaded")
    except Exception as e:
        logger.warning(f"Predictor not available: {e}")

    _ingester = SensorIngester(
        mqtt_broker="mqtt://localhost:1883",
        topics=["factory/#"],
    )

    yield

    if _ingester:
        await _ingester.stop()


app = FastAPI(
    title="SmartFactory-RAG",
    description="Intelligent Manufacturing Assistant — RAG + Predictive Maintenance + Sensor Fusion",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ───────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """System health check with component status."""
    return HealthResponse(
        status="healthy",
        uptime_seconds=round(time.time() - _start_time, 1),
        components={
            "rag_engine": "ready" if _rag_engine else "unavailable",
            "predictor": "ready" if _predictor else "unavailable",
            "sensor_ingester": "running" if _ingester and _ingester._running else "stopped",
        },
    )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query equipment manuals and documentation using natural language.

    Returns an answer with traceable sources (document, page, paragraph).
    """
    if _rag_engine is None:
        raise HTTPException(503, "RAG engine not initialized. Index documents first.")

    result = _rag_engine.query(
        question=request.question,
        top_k=request.top_k,
        rerank=request.rerank,
    )

    return QueryResponse(
        answer=result.answer,
        sources=[
            {
                "document": s.document,
                "page": s.page,
                "paragraph": s.paragraph,
                "excerpt": s.chunk_text,
                "relevance_score": s.score,
            }
            for s in result.sources
        ],
        confidence=result.confidence,
        retrieval_time_ms=round(result.retrieval_time_ms, 1),
        generation_time_ms=round(result.generation_time_ms, 1),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict_failure(request: PredictRequest):
    """
    Predict equipment failure from recent sensor data.

    Uses ensemble of LSTM + LightGBM models on the sensor window.
    """
    if _predictor is None:
        raise HTTPException(503, "Predictor not loaded. Train models first.")

    if _ingester is None:
        raise HTTPException(503, "Sensor ingester not running.")

    sensor_window = _ingester.get_window(
        request.equipment_id,
        size=request.window_size,
    )

    if sensor_window is None or len(sensor_window) < 10:
        raise HTTPException(
            404,
            f"Insufficient sensor data for equipment '{request.equipment_id}'. "
            f"Need at least 10 readings.",
        )

    prediction = _predictor.predict(
        sensor_window=sensor_window,
        confidence_threshold=request.confidence_threshold,
    )

    return PredictResponse(
        equipment_id=request.equipment_id,
        failure_probability=prediction.failure_probability,
        severity=prediction.severity,
        time_to_failure_hours=prediction.time_to_failure_hours,
        root_cause=prediction.root_cause,
        recommended_action=prediction.recommended_action,
        contributing_sensors=prediction.contributing_sensors,
    )


@app.get("/sensors/{equipment_id}/stats")
async def sensor_stats(equipment_id: str):
    """Get real-time sensor statistics for an equipment."""
    if _ingester is None:
        raise HTTPException(503, "Sensor ingester not running.")

    buffer = _ingester._buffers.get(equipment_id)
    if buffer is None:
        raise HTTPException(404, f"No data for equipment '{equipment_id}'")

    return {
        "equipment_id": equipment_id,
        "stats": buffer.stats(),
        "sensor_names": _ingester.SENSOR_NAMES,
    }


@app.get("/sensors/status")
async def ingestion_status():
    """Get sensor ingestion pipeline status."""
    if _ingester is None:
        return {"status": "not_initialized"}
    return _ingester.get_stats()


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
