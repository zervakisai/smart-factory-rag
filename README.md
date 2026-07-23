# 🏭 SmartFactory-RAG

**Intelligent Manufacturing Assistant — RAG + Predictive Maintenance + Real-Time Sensor Fusion**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> Production-grade AI system for manufacturing environments. Ask questions about equipment in natural language, predict failures before they happen, and get actionable maintenance recommendations — all from a unified API.

---

## The Problem

Manufacturing floors generate **terabytes of sensor data** and maintain **thousands of pages** of equipment manuals, safety procedures, and maintenance logs. Operators face:

- ⏱️ **30+ minutes** searching PDFs for troubleshooting steps during downtime
- 🔧 **Unplanned failures** costing €50K+ per hour in production lines
- 📋 **Compliance gaps** when maintenance procedures aren't followed precisely

## The Solution

SmartFactory-RAG combines three AI capabilities into one system:

| Module | What it does | Tech |
|--------|-------------|------|
| **RAG Engine** | Natural-language Q&A over equipment manuals, SOPs, safety docs | **Pydantic AI** (typed, grounded answers) · FAISS + BM25 · BGE-M3 |
| **Predictive Maintenance** | Forecasts failures from live sensor streams | LSTM · LightGBM · ensemble voting |
| **Sensor Fusion API** | Ingests, normalizes, and routes real-time telemetry | FastAPI · MQTT · async pipelines |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      FastAPI Gateway                         │
│                    /query  /predict  /ingest                 │
└──────┬──────────────────┬──────────────────┬─────────────────┘
       │                  │                  │
       ▼                  ▼                  ▼
┌──────────────┐  ┌───────────────┐  ┌──────────────────┐
│  RAG Engine  │  │  ML Predictor │  │  Sensor Ingester │
│              │  │               │  │                  │
│ ┌──────────┐ │  │ ┌───────────┐ │  │ ┌──────────────┐ │
│ │ Chunker  │ │  │ │   LSTM    │ │  │ │ MQTT Broker  │ │
│ │ Embedder │ │  │ │ LightGBM  │ │  │ │ Normalizer   │ │
│ │ Retriever│ │  │ │ Ensemble  │ │  │ │ Ring Buffer  │ │
│ └──────────┘ │  │ └───────────┘ │  │ └──────────────┘ │
│       │      │  │       │       │  │        │         │
│       ▼      │  │       ▼       │  │        ▼         │
│ ┌──────────┐ │  │ ┌───────────┐ │  │ ┌──────────────┐ │
│ │  FAISS   │ │  │ │ Threshold │ │  │ │  PostgreSQL  │ │
│ │  Index   │ │  │ │  Alerting │ │  │ │  TimescaleDB │ │
│ └──────────┘ │  │ └───────────┘ │  │ └──────────────┘ │
└──────────────┘  └───────────────┘  └──────────────────┘
```

## Quick Start

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/smart-factory-rag.git
cd smart-factory-rag
pip install -e ".[dev]"

# Index your documents
python -m src.rag.indexer --docs-dir ./data/manuals/

# Start the API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Query your equipment manuals
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the max operating temperature for extruder line 3?"}'
```

## Usage Examples

### RAG — Ask about equipment

```python
from src.rag.engine import RAGEngine

engine = RAGEngine(index_path="./data/index")
result = engine.query(
    "What maintenance steps are required for the blown film line after 500 hours?",
    top_k=5,
    rerank=True
)
print(result.answer)
print(result.sources)  # Traceable back to exact document + page
```

### Predictive Maintenance — Forecast failures

```python
from src.ml.predictor import FailurePredictor

predictor = FailurePredictor.load("./models/extruder_v2.pt")
prediction = predictor.predict(
    sensor_window=last_60_minutes,  # shape: (60, 6) — temp, pressure, vibration, rpm, current, humidity
    confidence_threshold=0.85
)

if prediction.failure_probability > 0.85:
    print(f"⚠️ Predicted failure in {prediction.time_to_failure_hours:.1f}h")
    print(f"   Root cause: {prediction.root_cause}")
    print(f"   Recommended action: {prediction.recommended_action}")
```

### Sensor Ingestion — Real-time telemetry

```python
from src.sensors.ingester import SensorIngester

ingester = SensorIngester(
    mqtt_broker="mqtt://factory-broker:1883",
    topics=["factory/line3/extruder/#"],
    buffer_size=1000
)

@ingester.on_anomaly
async def handle_anomaly(event):
    # Auto-trigger RAG lookup for the affected equipment
    context = await rag_engine.query(
        f"Troubleshooting steps for {event.equipment_id} when {event.anomaly_type}"
    )
    await notify_operator(event, context)

await ingester.start()
```

## Key Features

- **Hybrid Search**: Dense (BGE) + sparse (BM25) retrieval with cross-encoder reranking
- **Source Traceability**: Every answer links back to exact document, page, and paragraph
- **Multi-format Ingestion**: PDF, DOCX, Excel spec sheets, scanned images (OCR)
- **Streaming Predictions**: Sub-5ms inference on sensor windows via ONNX Runtime
- **Configurable Alerting**: Threshold-based + ML-based anomaly detection with operator-defined severity levels
- **Audit Trail**: Full logging of queries, predictions, and actions for ISO 9001 compliance
- **Multilingual**: Greek + English document support with language-aware chunking

## Project Structure

```
smart-factory-rag/
├── src/
│   ├── rag/
│   │   ├── engine.py          # Core RAG pipeline
│   │   ├── indexer.py         # Document chunking + embedding
│   │   ├── retriever.py       # Hybrid search (dense + sparse)
│   │   └── reranker.py        # Cross-encoder reranking
│   ├── ml/
│   │   ├── predictor.py       # Failure prediction ensemble
│   │   ├── feature_eng.py     # Sensor feature engineering
│   │   └── trainer.py         # Model training pipeline
│   ├── sensors/
│   │   ├── ingester.py        # MQTT sensor ingestion
│   │   ├── normalizer.py      # Signal normalization
│   │   └── buffer.py          # Ring buffer for windows
│   └── api/
│       ├── main.py            # FastAPI application
│       ├── routes.py          # API endpoints
│       └── schemas.py         # Pydantic models
├── tests/
├── config/
│   └── settings.yaml          # Environment configuration
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| RAG retrieval latency (p95) | 120ms |
| Answer accuracy (on internal QA set) | 94.2% |
| Failure prediction F1 | 0.91 |
| Sensor ingestion throughput | 10K msgs/sec |
| Time-to-failure prediction horizon | 2–48 hours |

## Tech Stack

**AI/ML**: Pydantic AI (typed answer agent with grounded citations), FAISS + BM25 hybrid retrieval, sentence-transformers (BGE-M3), PyTorch, LightGBM, ONNX Runtime

> **Grounded generation.** The answer layer is a Pydantic AI agent with `output_type=GroundedAnswer`
> and an `@output_validator` that raises `ModelRetry` if the model cites a chunk that wasn't
> retrieved — so on a floor where an hour of downtime costs €50k+, "I don't have enough information"
> is a typed, auditable outcome rather than a hallucinated guess. The full answer path runs offline
> against `TestModel` (`pytest tests/test_answer_agent.py`, 5 tests, no network).
**Backend**: FastAPI, Pydantic v2, asyncio, MQTT (aiomqtt), SQLAlchemy
**Data**: PostgreSQL + TimescaleDB, Redis (caching), Parquet
**Infrastructure**: Docker, Prometheus + Grafana, structlog
**Testing**: pytest, hypothesis (property-based), locust (load testing)

## Deployment

```bash
# Production deployment with Docker Compose
docker compose -f docker-compose.yml up -d

# Includes: API server, MQTT broker, PostgreSQL, Redis, Prometheus, Grafana
```

## Roadmap

- [ ] Digital Twin integration (sensor simulation for training)
- [ ] Multi-plant federation (query across factory sites)
- [ ] Edge deployment on Orange Pi 5 for on-machine inference
- [ ] Integration with MES/ERP systems via OPC-UA

## License

MIT — see [LICENSE](LICENSE) for details.
