"""Grounded answer-agent tests — all via TestModel/override, never a live model."""

from __future__ import annotations

import asyncio

import pytest
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.test import TestModel

from src.rag.answer import GroundedAnswer, answer_agent, run_answer_agent

CHUNKS = [
    {"document": "extruder_manual.pdf", "page": 12, "chunk_text": "Barrel zone 3 setpoint is 210 °C."},
    {"document": "extruder_manual.pdf", "page": 47, "chunk_text": "Shut down before clearing a die blockage."},
]


def _args(answer="ok", citations=None, insufficient=False, safety=False, confidence=0.8):
    return {
        "answer": answer,
        "citations": citations or [],
        "insufficient_context": insufficient,
        "safety_critical": safety,
        "confidence": confidence,
    }


def test_returns_typed_grounded_answer():
    good = _args("Set zone 3 to 210 °C [Source 1].",
                 citations=[{"source_id": 1, "document": "extruder_manual.pdf", "page": 12}],
                 safety=False, confidence=0.9)
    with answer_agent.override(model=TestModel(custom_output_args=good)):
        out = asyncio.run(run_answer_agent("What is the zone 3 setpoint?", CHUNKS))
    assert isinstance(out, GroundedAnswer)
    assert out.citations[0].source_id == 1
    assert out.confidence == 0.9


def test_out_of_range_citation_is_retried_to_exhaustion():
    bad = _args("invented", citations=[{"source_id": 99, "document": "ghost", "page": 1}])
    with answer_agent.override(model=TestModel(custom_output_args=bad)):
        with pytest.raises(UnexpectedModelBehavior, match="Exceeded maximum"):
            asyncio.run(run_answer_agent("q", CHUNKS))


def test_answering_without_citations_is_rejected():
    # context present, insufficient_context=False, but no citations → must retry
    with answer_agent.override(model=TestModel(custom_output_args=_args("claim", citations=[]))):
        with pytest.raises(UnexpectedModelBehavior, match="Exceeded maximum"):
            asyncio.run(run_answer_agent("q", CHUNKS))


def test_insufficient_context_needs_no_citation():
    args = _args("The manual does not cover this.", citations=[], insufficient=True, confidence=0.3)
    with answer_agent.override(model=TestModel(custom_output_args=args)):
        out = asyncio.run(run_answer_agent("Unrelated question?", CHUNKS))
    assert out.insufficient_context is True
    assert out.citations == []


def test_no_chunks_allows_empty_citations():
    args = _args("No context was retrieved.", citations=[], insufficient=True, confidence=0.1)
    with answer_agent.override(model=TestModel(custom_output_args=args)):
        out = asyncio.run(run_answer_agent("q", []))
    assert out.insufficient_context is True
