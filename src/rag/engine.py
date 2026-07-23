"""
SmartFactory-RAG Engine — Hybrid retrieval with source traceability.

Combines dense (BGE-M3) + sparse (BM25) retrieval with cross-encoder
reranking for manufacturing document Q&A.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Source:
    """Traceable source reference."""

    document: str
    page: int
    paragraph: int
    chunk_text: str
    score: float

    def __str__(self) -> str:
        return f"{self.document} (p.{self.page}, §{self.paragraph}) [score={self.score:.3f}]"


@dataclass
class RAGResult:
    """Result from RAG query with full traceability."""

    answer: str
    sources: list[Source]
    confidence: float
    retrieval_time_ms: float
    generation_time_ms: float
    query: str
    metadata: dict = field(default_factory=dict)

    @property
    def total_time_ms(self) -> float:
        return self.retrieval_time_ms + self.generation_time_ms


class RAGEngine:
    """
    Production RAG engine for manufacturing document Q&A.

    Features:
        - Hybrid search: dense embeddings (BGE-M3) + sparse (BM25)
        - Cross-encoder reranking for precision
        - Source traceability to exact document/page/paragraph
        - Configurable fusion weights
        - Query expansion for manufacturing terminology
    """

    def __init__(
        self,
        index_path: str | Path,
        model_name: str = "BAAI/bge-m3",
        reranker_name: str = "BAAI/bge-reranker-v2-m3",
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        device: str = "auto",
        llm_model: str | None = None,
    ):
        from ..config import MODEL

        self.index_path = Path(index_path)
        self.model_name = model_name  # embedder
        self.reranker_name = reranker_name
        self.model = llm_model or MODEL  # Pydantic AI model string for answer generation
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.device = self._resolve_device(device)

        self._dense_index = None
        self._sparse_index = None
        self._reranker = None
        self._embedder = None
        self._chunk_store: list[dict] = []

        self._load_indices()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load_indices(self) -> None:
        """Load FAISS dense index and BM25 sparse index from disk."""
        import faiss
        import pickle

        dense_path = self.index_path / "dense.index"
        sparse_path = self.index_path / "sparse.pkl"
        chunks_path = self.index_path / "chunks.pkl"

        if dense_path.exists():
            self._dense_index = faiss.read_index(str(dense_path))
            logger.info(f"Loaded dense index: {self._dense_index.ntotal} vectors")

        if sparse_path.exists():
            with open(sparse_path, "rb") as f:
                self._sparse_index = pickle.load(f)
            logger.info("Loaded sparse (BM25) index")

        if chunks_path.exists():
            with open(chunks_path, "rb") as f:
                self._chunk_store = pickle.load(f)
            logger.info(f"Loaded {len(self._chunk_store)} chunks")

    def _get_embedder(self):
        """Lazy-load the embedding model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Loaded embedder: {self.model_name} on {self.device}")
        return self._embedder

    def _get_reranker(self):
        """Lazy-load the cross-encoder reranker."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.reranker_name, device=self.device)
            logger.info(f"Loaded reranker: {self.reranker_name}")
        return self._reranker

    def _dense_search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Dense retrieval using FAISS."""
        embedder = self._get_embedder()
        query_vec = embedder.encode([query], normalize_embeddings=True)
        scores, indices = self._dense_index.search(query_vec.astype(np.float32), top_k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]

    def _sparse_search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Sparse retrieval using BM25."""
        tokenized_query = query.lower().split()
        scores = self._sparse_index.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[tuple[int, float]],
        sparse_results: list[tuple[int, float]],
        k: int = 60,
    ) -> list[tuple[int, float]]:
        """Fuse dense and sparse results using Reciprocal Rank Fusion."""
        fused_scores: dict[int, float] = {}

        for rank, (idx, _) in enumerate(dense_results):
            fused_scores[idx] = fused_scores.get(idx, 0) + self.dense_weight / (k + rank + 1)

        for rank, (idx, _) in enumerate(sparse_results):
            fused_scores[idx] = fused_scores.get(idx, 0) + self.sparse_weight / (k + rank + 1)

        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results

    def _rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """Rerank candidates using cross-encoder."""
        reranker = self._get_reranker()
        pairs = [(query, c["chunk_text"]) for c in candidates]
        scores = reranker.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)

        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[:top_k]

    def _expand_query(self, query: str) -> str:
        """Expand query with manufacturing domain synonyms."""
        expansions = {
            "extruder": "extruder extrusion screw barrel",
            "temperature": "temperature thermal heat cooling",
            "pressure": "pressure PSI bar gauge",
            "vibration": "vibration oscillation frequency resonance",
            "maintenance": "maintenance service repair overhaul PM",
            "failure": "failure fault breakdown malfunction defect",
            "film": "film blown cast sheet thickness gauge",
            "die": "die head lip gap tooling",
        }
        expanded = query
        for term, expansion in expansions.items():
            if term.lower() in query.lower():
                expanded = f"{expanded} {expansion}"
        return expanded

    def query(
        self,
        question: str,
        top_k: int = 5,
        rerank: bool = True,
        expand_query: bool = True,
    ) -> RAGResult:
        """
        Query the manufacturing knowledge base.

        Args:
            question: Natural language question
            top_k: Number of results to return
            rerank: Whether to apply cross-encoder reranking
            expand_query: Whether to expand query with domain synonyms

        Returns:
            RAGResult with answer, sources, and metadata
        """
        import time

        search_query = self._expand_query(question) if expand_query else question

        # Retrieval phase
        t0 = time.perf_counter()
        dense_results = self._dense_search(search_query, top_k=top_k * 4)
        sparse_results = self._sparse_search(search_query, top_k=top_k * 4)
        fused = self._reciprocal_rank_fusion(dense_results, sparse_results)

        # Get candidate chunks
        candidates = []
        for idx, score in fused[:top_k * 3]:
            if idx < len(self._chunk_store):
                chunk = self._chunk_store[idx].copy()
                chunk["fusion_score"] = score
                candidates.append(chunk)

        # Rerank
        if rerank and candidates:
            candidates = self._rerank(question, candidates, top_k)
        else:
            candidates = candidates[:top_k]

        retrieval_time = (time.perf_counter() - t0) * 1000

        # Generation phase — bounded judgement over the retrieved chunks (Pydantic AI).
        t1 = time.perf_counter()
        grounded = self._answer(question, candidates)
        generation_time = (time.perf_counter() - t1) * 1000

        # Build traceable sources
        sources = [
            Source(
                document=c["document"],
                page=c["page"],
                paragraph=c.get("paragraph", 0),
                chunk_text=c["chunk_text"][:200],
                score=c.get("rerank_score", c.get("fusion_score", 0)),
            )
            for c in candidates
        ]

        return RAGResult(
            answer=grounded.answer,
            sources=sources,
            # Confidence stays grounded in retrieval scores; the model's self-assessment is metadata.
            confidence=self._compute_confidence(candidates),
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            query=question,
            metadata={
                "cited_source_ids": [c.source_id for c in grounded.citations],
                "insufficient_context": grounded.insufficient_context,
                "safety_critical": grounded.safety_critical,
                "model_confidence": grounded.confidence,
            },
        )

    def _answer(self, question: str, candidates: list[dict]):
        """Run the typed Pydantic AI answer agent, safely from sync or async callers.

        `query()` is sync but is invoked from an async FastAPI endpoint, so an event loop may
        already be running — in that case run the agent's coroutine on a worker thread rather than
        calling `asyncio.run` (which would raise inside a running loop).
        """
        from .answer import GroundedAnswer, run_answer_agent

        async def _coro():
            return await run_answer_agent(question, candidates, model=self.model)

        try:
            import asyncio

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(_coro())

            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(lambda: asyncio.run(_coro())).result()
        except Exception as e:  # never let a generation failure sink the whole query
            logger.exception("Answer generation failed: %s", e)
            return GroundedAnswer(
                answer="Answer generation is unavailable right now.",
                citations=[],
                insufficient_context=True,
                safety_critical=False,
                confidence=0.0,
            )

    @staticmethod
    def _compute_confidence(candidates: list[dict]) -> float:
        """Compute answer confidence from retrieval scores."""
        if not candidates:
            return 0.0
        scores = [c.get("rerank_score", c.get("fusion_score", 0)) for c in candidates]
        top_score = max(scores)
        score_gap = top_score - (scores[1] if len(scores) > 1 else 0)
        return min(1.0, (top_score * 0.7 + score_gap * 0.3))
