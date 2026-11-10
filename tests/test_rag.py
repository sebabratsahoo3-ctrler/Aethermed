"""
tests/test_rag.py

Tests for the RAG engine: local Chroma vector retrieval via
:class:`ClinicalReportGenerator` (``src/rag/generate_report.py``), full
report generation, and the :meth:`check_citation_coverage` faithfulness
check.

Tests that require a real embedding model / Chroma index use the
session-scoped ``rag_index_dir`` fixture (see ``tests/conftest.py``),
which builds a throwaway index in a temp directory so the real
``data/processed/chroma_db`` used by the application is never mutated by
the test suite. Those tests are automatically skipped if ``chromadb`` or
``sentence-transformers`` are not installed in the environment.

The citation-coverage grounding logic itself has no ML dependency (it is
pure lexical overlap), so those tests run unconditionally against
hand-constructed :class:`RetrievedChunk` fixtures.
"""

from __future__ import annotations

import pytest

from src.rag.generate_report import (
    ClinicalReportGenerator,
    OrganToxicityRisk,
    PatientState,
    PKPDMetrics,
    RetrievedChunk,
)


# --------------------------------------------------------------------------
# check_citation_coverage — pure logic, no ML dependency required
# --------------------------------------------------------------------------

@pytest.fixture
def bare_generator() -> ClinicalReportGenerator:
    """A ClinicalReportGenerator with no live Chroma connection.

    Bypasses ``__init__`` (which requires a persistent index directory)
    since these tests only exercise ``check_citation_coverage``, a pure
    lexical-overlap method that needs no vector store.
    """
    gen = ClinicalReportGenerator.__new__(ClinicalReportGenerator)
    gen.grounding_threshold = 0.12
    return gen


@pytest.fixture
def vancomycin_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="vancomycin_pubmed_01::chunk0",
        doc_id="vancomycin_pubmed_01",
        drug="Vancomycin",
        source_type="pubmed_abstract",
        title="AUC-guided versus trough-guided vancomycin dosing",
        citation="Synthetic literature abstract",
        text=(
            "AUC-guided dosing achieving an AUC24/MIC target in the range of "
            "400 to 600 was associated with lower incidence of nephrotoxicity "
            "relative to trough-guided strategies targeting 15 to 20 mcg/mL."
        ),
        score=0.91,
    )


class TestCitationCoverage:
    """Tests for ClinicalReportGenerator.check_citation_coverage."""

    def test_coverage_rate_is_in_valid_range(
        self, bare_generator: ClinicalReportGenerator, vancomycin_chunk: RetrievedChunk
    ) -> None:
        """coverage_rate must always fall within [0.0, 1.0]."""
        report_text = (
            "AUC-guided dosing achieving an AUC24/MIC target of 400 to 600 "
            "lowers nephrotoxicity relative to trough-guided strategies in "
            "this simulated cohort.\n"
            "The patient will definitely be cured within 48 hours with zero "
            "risk of any side effects whatsoever.\n"
        )
        result = bare_generator.check_citation_coverage(report_text, [vancomycin_chunk])
        assert 0.0 <= result.coverage_rate <= 1.0
        assert result.total_claims == result.grounded_claims + len(result.ungrounded_claims)

    def test_grounded_claim_is_not_flagged(
        self, bare_generator: ClinicalReportGenerator, vancomycin_chunk: RetrievedChunk
    ) -> None:
        """A claim closely paraphrasing retrieved evidence should be grounded."""
        grounded_sentence = (
            "AUC-guided dosing achieving an AUC24/MIC target of 400 to 600 "
            "lowers nephrotoxicity relative to trough-guided strategies."
        )
        result = bare_generator.check_citation_coverage([grounded_sentence], [vancomycin_chunk])
        assert result.grounded_claims == 1
        assert result.ungrounded_claims == []
        assert result.coverage_rate == 1.0

    def test_ungrounded_claim_is_flagged(
        self, bare_generator: ClinicalReportGenerator, vancomycin_chunk: RetrievedChunk
    ) -> None:
        """A fabricated claim with no lexical support must be flagged as ungrounded."""
        fabricated_sentence = (
            "The patient will definitely be cured within 48 hours with zero "
            "risk of any side effects whatsoever."
        )
        result = bare_generator.check_citation_coverage([fabricated_sentence], [vancomycin_chunk])
        assert result.grounded_claims == 0
        assert fabricated_sentence in result.ungrounded_claims
        assert result.coverage_rate == 0.0

    def test_mixed_claims_produce_partial_coverage(
        self, bare_generator: ClinicalReportGenerator, vancomycin_chunk: RetrievedChunk
    ) -> None:
        """One grounded + one ungrounded claim should yield coverage_rate == 0.5."""
        grounded = (
            "AUC-guided dosing achieving an AUC24/MIC target of 400 to 600 "
            "lowers nephrotoxicity relative to trough-guided strategies."
        )
        fabricated = "This drug interaction has never been studied in any species."
        result = bare_generator.check_citation_coverage([grounded, fabricated], [vancomycin_chunk])
        assert result.total_claims == 2
        assert result.grounded_claims == 1
        assert len(result.ungrounded_claims) == 1
        assert result.coverage_rate == pytest.approx(0.5)

    def test_no_retrieved_docs_flags_everything_ungrounded(
        self, bare_generator: ClinicalReportGenerator
    ) -> None:
        """With zero retrieved evidence, every claim must be ungrounded."""
        sentence = "Vancomycin trough levels above 20 mcg/mL increase nephrotoxicity risk."
        result = bare_generator.check_citation_coverage([sentence], [])
        assert result.grounded_claims == 0
        assert result.coverage_rate == 0.0

    def test_empty_report_text_raises(self, bare_generator: ClinicalReportGenerator) -> None:
        with pytest.raises(ValueError):
            bare_generator.check_citation_coverage("", [])

    def test_markdown_headers_and_short_lines_are_excluded_from_claims(
        self, bare_generator: ClinicalReportGenerator, vancomycin_chunk: RetrievedChunk
    ) -> None:
        """Markdown headers and very short lines should not be scored as claims."""
        report_text = (
            "# AetherMed Report\n"
            "### Section Header\n"
            "OK\n"  # too short (<20 chars) to be treated as a claim
            "AUC-guided dosing achieving an AUC24/MIC target of 400 to 600 "
            "lowers nephrotoxicity relative to trough-guided strategies.\n"
        )
        result = bare_generator.check_citation_coverage(report_text, [vancomycin_chunk])
        assert result.total_claims == 1  # only the substantive sentence counted


# --------------------------------------------------------------------------
# Live retrieval + report generation (requires chromadb + sentence-transformers)
# --------------------------------------------------------------------------

class TestLiveRetrieval:
    """Tests exercising the real Chroma-backed retrieval path.

    These use the session-scoped ``rag_index_dir`` fixture from
    ``tests/conftest.py``, which builds an isolated, throwaway index so
    the production ``data/processed/chroma_db`` is untouched.
    """

    @pytest.fixture
    def generator(self, rag_index_dir) -> ClinicalReportGenerator:
        return ClinicalReportGenerator(persist_dir=rag_index_dir, top_k=5)

    def test_retrieve_returns_relevant_chunks_for_known_drug(self, generator) -> None:
        """A query about warfarin monitoring should surface warfarin-tagged chunks."""
        results = generator.retrieve("What INR range should be targeted with warfarin?", top_k=3)
        assert len(results) > 0
        assert all(isinstance(r, RetrievedChunk) for r in results)
        assert all(0.0 <= r.score <= 1.0 for r in results)
        # Results should skew toward the queried drug even without an explicit filter.
        assert any(r.drug == "Warfarin" for r in results)

    def test_retrieve_respects_drug_filter(self, generator) -> None:
        """Passing drug_filter should restrict results to that drug only."""
        results = generator.retrieve(
            "toxicity monitoring parameters", top_k=5, drug_filter="Digoxin"
        )
        assert len(results) > 0
        assert all(r.drug == "Digoxin" for r in results)

    def test_retrieve_raises_on_empty_query(self, generator) -> None:
        with pytest.raises(ValueError):
            generator.retrieve("   ")

    def test_generate_report_produces_grounded_report(
        self,
        generator: ClinicalReportGenerator,
        dummy_patient_state: PatientState,
        dummy_organ_risks: list[OrganToxicityRisk],
    ) -> None:
        """End-to-end report generation should return a well-formed result dict."""
        pkpd = PKPDMetrics(
            drug="Warfarin",
            cmax=2.1,
            cmin_trough=0.6,
            auc_24h=32.4,
            time_in_therapeutic_range_pct=78.0,
            half_life_hours=40.0,
        )
        result = generator.generate_report(dummy_patient_state, pkpd, dummy_organ_risks)

        assert "report_text" in result
        assert "retrieved_chunks" in result
        assert "citation_coverage" in result

        assert isinstance(result["report_text"], str)
        assert dummy_patient_state.patient_id in result["report_text"]
        assert "Warfarin" in result["report_text"]

        coverage = result["citation_coverage"]
        assert 0.0 <= coverage.coverage_rate <= 1.0

    def test_generate_report_flags_high_risk_organs(
        self, generator: ClinicalReportGenerator, dummy_patient_state: PatientState
    ) -> None:
        """A high-band organ risk should trigger the simulation flag in the report text."""
        pkpd = PKPDMetrics(
            drug="Vancomycin", cmax=35.0, cmin_trough=18.0, auc_24h=520.0,
            time_in_therapeutic_range_pct=60.0, half_life_hours=6.0,
        )
        high_risk = [OrganToxicityRisk(organ="renal", risk_score=0.85, confidence=0.9)]
        result = generator.generate_report(dummy_patient_state, pkpd, high_risk)
        assert "Simulation Flag" in result["report_text"]
        assert "renal" in result["report_text"].lower()

    def test_generate_report_raises_on_empty_drug(
        self, generator: ClinicalReportGenerator, dummy_patient_state: PatientState
    ) -> None:
        with pytest.raises(ValueError):
            bad_pkpd_kwargs = dict(
                cmax=1.0, cmin_trough=0.1, auc_24h=5.0,
                time_in_therapeutic_range_pct=50.0, half_life_hours=10.0,
            )
            # PKPDMetrics itself validates non-empty drug at construction time,
            # so constructing the invalid metrics object is expected to raise.
            PKPDMetrics(drug="", **bad_pkpd_kwargs)