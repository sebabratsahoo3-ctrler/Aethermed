"""
src/rag

AetherMed 3.0 - Retrieval-Augmented Generation (RAG) Module
--------------------------------------------------------------
This package implements the clinical literature RAG stack that grounds
AetherMed's digital-twin PK/PD/toxicity simulation outputs in retrievable
source evidence:

    - ``build_corpus``: builds/chunks/embeds the local clinical corpus and
      persists it into a Chroma vector store.
    - ``generate_report``: defines :class:`ClinicalReportGenerator`, which
      retrieves grounding literature and assembles structured, cited
      clinical trial simulation reports, including a citation-coverage /
      faithfulness check.
    - ``eval_rag``: retrieval quality (Precision@k, Recall@k, MRR) and
      report faithfulness (citation coverage) evaluation utilities.

Typical usage
~~~~~~~~~~~~~
    from src.rag import build_corpus, ClinicalReportGenerator, eval_rag

    # 1. One-time (or on corpus update) index build:
    build_corpus.build_corpus()

    # 2. Generate a grounded report from digital-twin outputs:
    generator = ClinicalReportGenerator()
    result = generator.generate_report(patient, pkpd, organ_risks)
    print(result["report_text"])

    # 3. Evaluate retrieval quality:
    summary = eval_rag.run_retrieval_evaluation(k=3)
"""

from __future__ import annotations

from src.rag import build_corpus, eval_rag
from src.rag.generate_report import (
    CitationCoverageResult,
    ClinicalReportGenerator,
    OrganToxicityRisk,
    PatientState,
    PKPDMetrics,
    RetrievedChunk,
)

__all__ = [
    # sub-modules exposed as entrypoints
    "build_corpus",
    "eval_rag",
    # primary report-generation API
    "ClinicalReportGenerator",
    "PatientState",
    "PKPDMetrics",
    "OrganToxicityRisk",
    "RetrievedChunk",
    "CitationCoverageResult",
]

__version__ = "3.0.0"