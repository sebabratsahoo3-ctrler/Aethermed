"""
tests/conftest.py

Shared pytest fixtures for the AetherMed 3.0 test suite.

pytest auto-discovers fixtures defined in ``conftest.py`` files without
requiring an import in each test module, so this is where cross-cutting
fixtures (dummy patient states, temporal sequences, PK/PD metrics,
organ-toxicity risk sets, and a session-scoped RAG index) live. Individual
test files add their own module-local fixtures where a fixture is only
relevant to one component.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from src.rag.build_corpus import build_corpus
from src.rag.generate_report import OrganToxicityRisk, PatientState, PKPDMetrics


# --------------------------------------------------------------------------
# Core dummy digital-twin fixtures
# --------------------------------------------------------------------------

@pytest.fixture
def dummy_patient_state() -> PatientState:
    """A single well-formed, valid patient state for use across test modules."""
    return PatientState(
        patient_id="patient-0001",
        age_years=64.0,
        sex="F",
        weight_kg=68.5,
        egfr_ml_min=72.0,
        comorbidities=["atrial_fibrillation", "hypertension"],
        concomitant_drugs=["amiodarone"],
        raw_vector=[0.1, 0.2, 0.3, 0.4],
    )


@pytest.fixture
def dummy_pkpd_metrics() -> PKPDMetrics:
    """A single well-formed PK/PD metrics summary for Warfarin."""
    return PKPDMetrics(
        drug="Warfarin",
        cmax=2.1,
        cmin_trough=0.6,
        auc_24h=32.4,
        time_in_therapeutic_range_pct=78.0,
        half_life_hours=40.0,
        unit="mcg/mL",
    )


@pytest.fixture
def dummy_organ_risks() -> list[OrganToxicityRisk]:
    """A representative set of organ toxicity risk scores across bands."""
    return [
        OrganToxicityRisk(organ="renal", risk_score=0.78, confidence=0.91,
                           contributing_factors=["low_egfr", "concomitant_nephrotoxin"]),
        OrganToxicityRisk(organ="cardiac", risk_score=0.42, confidence=0.85),
        OrganToxicityRisk(organ="hepatic", risk_score=0.10, confidence=0.95),
    ]


@pytest.fixture
def time_steps_24h() -> np.ndarray:
    """A standard 24-hour simulation horizon at 15-minute resolution (97 points)."""
    return np.linspace(0.0, 24.0, 97)


@pytest.fixture
def time_steps_short() -> np.ndarray:
    """A short 8-point time grid, useful for fast/edge-case ODE integration tests."""
    return np.linspace(0.0, 4.0, 8)


# --------------------------------------------------------------------------
# RAG-specific fixtures (session-scoped: building the index is expensive)
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rag_index_dir() -> Iterator[Path]:
    """Build a throwaway Chroma index in a temp directory for the whole test session.

    This avoids mutating the real ``data/processed/chroma_db`` used by the
    application, and avoids rebuilding the (expensive) embedding index
    once per test. Skips the whole session's RAG tests if the required
    ML dependencies (``sentence-transformers``, ``chromadb``) are not
    installed in the test environment.
    """
    pytest.importorskip("chromadb", reason="chromadb not installed; skipping RAG tests")
    pytest.importorskip(
        "sentence_transformers", reason="sentence-transformers not installed; skipping RAG tests"
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="aethermed_test_chroma_"))
    try:
        build_corpus(persist_dir=tmp_dir, reset=True)
        yield tmp_dir
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)