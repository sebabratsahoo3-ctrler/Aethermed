"""
tests/test_schema.py

Validates the core AetherMed 3.0 data schemas defined in
``src/rag/generate_report.py``: :class:`PatientState`,
:class:`PKPDMetrics`, :class:`OrganToxicityRisk`, and
:class:`RetrievedChunk`.

These are plain (frozen) ``dataclasses`` with ``__post_init__``
validation rather than Pydantic models, so "validation error" here means
a ``ValueError`` raised at construction time. Each schema's happy path
and its documented edge-case invariants (negative age, negative
clearance/exposure values, out-of-bound risk scores and similarity
scores) are covered.
"""

from __future__ import annotations

import pytest

from src.rag.generate_report import (
    OrganToxicityRisk,
    PatientState,
    PKPDMetrics,
    RetrievedChunk,
)


# --------------------------------------------------------------------------
# PatientState
# --------------------------------------------------------------------------

class TestPatientStateSchema:
    """Validation behavior for :class:`PatientState`."""

    def test_valid_patient_state_constructs(self) -> None:
        """A well-formed patient state should construct without error."""
        patient = PatientState(
            patient_id="p-001",
            age_years=55.0,
            sex="M",
            weight_kg=80.0,
            egfr_ml_min=95.0,
        )
        assert patient.patient_id == "p-001"
        assert patient.age_years == 55.0
        assert patient.comorbidities == []
        assert patient.concomitant_drugs == []

    def test_defaults_are_independent_between_instances(self) -> None:
        """Mutable defaults (comorbidities/concomitant_drugs) must not be shared."""
        p1 = PatientState(patient_id="p1", age_years=40, sex="F", weight_kg=60, egfr_ml_min=90)
        p2 = PatientState(patient_id="p2", age_years=40, sex="F", weight_kg=60, egfr_ml_min=90)
        assert p1.comorbidities is not p2.comorbidities
        assert p1.concomitant_drugs is not p2.concomitant_drugs

    def test_negative_age_raises(self) -> None:
        """Negative age is physiologically invalid and must raise ValueError."""
        with pytest.raises(ValueError, match="age_years"):
            PatientState(patient_id="p1", age_years=-1.0, sex="F", weight_kg=70, egfr_ml_min=90)

    def test_implausibly_high_age_raises(self) -> None:
        """Age far beyond plausible human lifespan should raise ValueError."""
        with pytest.raises(ValueError, match="age_years"):
            PatientState(patient_id="p1", age_years=250.0, sex="F", weight_kg=70, egfr_ml_min=90)

    def test_zero_age_is_valid_boundary(self) -> None:
        """Age of exactly 0 (e.g. neonatal simulation) is a valid boundary value."""
        patient = PatientState(patient_id="p1", age_years=0.0, sex="F", weight_kg=3.2, egfr_ml_min=40)
        assert patient.age_years == 0.0

    def test_negative_weight_raises(self) -> None:
        """Non-positive body weight must raise ValueError."""
        with pytest.raises(ValueError, match="weight_kg"):
            PatientState(patient_id="p1", age_years=40, sex="F", weight_kg=-70, egfr_ml_min=90)

    def test_zero_weight_raises(self) -> None:
        """Weight must be strictly positive."""
        with pytest.raises(ValueError, match="weight_kg"):
            PatientState(patient_id="p1", age_years=40, sex="F", weight_kg=0.0, egfr_ml_min=90)

    def test_negative_egfr_raises(self) -> None:
        """Negative eGFR (renal clearance proxy) is not physiologically valid."""
        with pytest.raises(ValueError, match="egfr_ml_min"):
            PatientState(patient_id="p1", age_years=40, sex="F", weight_kg=70, egfr_ml_min=-5.0)

    def test_zero_egfr_is_valid_boundary(self) -> None:
        """eGFR of 0 (e.g. anuric/dialysis-dependent patient) is a valid boundary value."""
        patient = PatientState(patient_id="p1", age_years=70, sex="M", weight_kg=75, egfr_ml_min=0.0)
        assert patient.egfr_ml_min == 0.0

    def test_empty_patient_id_raises(self) -> None:
        """An empty/blank patient_id must raise ValueError."""
        with pytest.raises(ValueError, match="patient_id"):
            PatientState(patient_id="  ", age_years=40, sex="F", weight_kg=70, egfr_ml_min=90)

    def test_patient_state_is_immutable(self) -> None:
        """PatientState is a frozen dataclass; attribute mutation should fail."""
        patient = PatientState(patient_id="p1", age_years=40, sex="F", weight_kg=70, egfr_ml_min=90)
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            patient.age_years = 41  # type: ignore[misc]


# --------------------------------------------------------------------------
# PKPDMetrics
# --------------------------------------------------------------------------

class TestPKPDMetricsSchema:
    """Validation behavior for :class:`PKPDMetrics`."""

    def test_valid_pkpd_metrics_constructs(self) -> None:
        metrics = PKPDMetrics(
            drug="Vancomycin",
            cmax=30.0,
            cmin_trough=12.0,
            auc_24h=480.0,
            time_in_therapeutic_range_pct=65.0,
            half_life_hours=6.0,
        )
        assert metrics.drug == "Vancomycin"
        assert metrics.unit == "mcg/mL"  # default

    def test_negative_cmax_raises(self) -> None:
        """Negative maximum concentration is physically impossible."""
        with pytest.raises(ValueError, match="cmax"):
            PKPDMetrics(
                drug="Digoxin", cmax=-0.5, cmin_trough=0.0, auc_24h=5.0,
                time_in_therapeutic_range_pct=50.0, half_life_hours=36.0,
            )

    def test_negative_cmin_trough_raises(self) -> None:
        """Negative trough concentration is physically impossible."""
        with pytest.raises(ValueError, match="cmin_trough"):
            PKPDMetrics(
                drug="Digoxin", cmax=1.0, cmin_trough=-0.1, auc_24h=5.0,
                time_in_therapeutic_range_pct=50.0, half_life_hours=36.0,
            )

    def test_trough_exceeding_cmax_raises(self) -> None:
        """Trough concentration cannot exceed the simulated Cmax."""
        with pytest.raises(ValueError, match="cmin_trough"):
            PKPDMetrics(
                drug="Digoxin", cmax=1.0, cmin_trough=2.0, auc_24h=5.0,
                time_in_therapeutic_range_pct=50.0, half_life_hours=36.0,
            )

    def test_negative_auc_raises(self) -> None:
        """Negative AUC (a clearance/exposure-derived quantity) is invalid."""
        with pytest.raises(ValueError, match="auc_24h"):
            PKPDMetrics(
                drug="Warfarin", cmax=2.0, cmin_trough=0.5, auc_24h=-10.0,
                time_in_therapeutic_range_pct=50.0, half_life_hours=40.0,
            )

    @pytest.mark.parametrize("bad_pct", [-0.01, 100.01, -50.0, 150.0])
    def test_out_of_bound_time_in_range_pct_raises(self, bad_pct: float) -> None:
        """time_in_therapeutic_range_pct must stay within [0, 100]."""
        with pytest.raises(ValueError, match="time_in_therapeutic_range_pct"):
            PKPDMetrics(
                drug="Warfarin", cmax=2.0, cmin_trough=0.5, auc_24h=30.0,
                time_in_therapeutic_range_pct=bad_pct, half_life_hours=40.0,
            )

    @pytest.mark.parametrize("boundary_pct", [0.0, 100.0])
    def test_time_in_range_pct_boundaries_are_valid(self, boundary_pct: float) -> None:
        """0 and 100 are valid inclusive boundary values."""
        metrics = PKPDMetrics(
            drug="Warfarin", cmax=2.0, cmin_trough=0.5, auc_24h=30.0,
            time_in_therapeutic_range_pct=boundary_pct, half_life_hours=40.0,
        )
        assert metrics.time_in_therapeutic_range_pct == boundary_pct

    def test_non_positive_half_life_raises(self) -> None:
        """Half-life must be strictly positive (zero/negative is non-physical)."""
        with pytest.raises(ValueError, match="half_life_hours"):
            PKPDMetrics(
                drug="Warfarin", cmax=2.0, cmin_trough=0.5, auc_24h=30.0,
                time_in_therapeutic_range_pct=50.0, half_life_hours=0.0,
            )

    def test_empty_drug_name_raises(self) -> None:
        with pytest.raises(ValueError, match="drug"):
            PKPDMetrics(
                drug="", cmax=2.0, cmin_trough=0.5, auc_24h=30.0,
                time_in_therapeutic_range_pct=50.0, half_life_hours=40.0,
            )


# --------------------------------------------------------------------------
# OrganToxicityRisk
# --------------------------------------------------------------------------

class TestOrganToxicityRiskSchema:
    """Validation behavior for :class:`OrganToxicityRisk`."""

    def test_valid_organ_toxicity_risk_constructs(self) -> None:
        risk = OrganToxicityRisk(organ="renal", risk_score=0.5, confidence=0.9)
        assert risk.organ == "renal"
        assert risk.risk_band == "moderate"

    @pytest.mark.parametrize("bad_score", [-0.01, 1.01, -1.0, 2.0, float("inf")])
    def test_out_of_bound_risk_score_raises(self, bad_score: float) -> None:
        """risk_score outside [0.0, 1.0] must raise ValueError."""
        with pytest.raises(ValueError, match="risk_score"):
            OrganToxicityRisk(organ="renal", risk_score=bad_score)

    @pytest.mark.parametrize("boundary_score", [0.0, 1.0])
    def test_risk_score_boundaries_are_valid(self, boundary_score: float) -> None:
        risk = OrganToxicityRisk(organ="hepatic", risk_score=boundary_score)
        assert risk.risk_score == boundary_score

    @pytest.mark.parametrize("bad_conf", [-0.5, 1.5])
    def test_out_of_bound_confidence_raises(self, bad_conf: float) -> None:
        with pytest.raises(ValueError, match="confidence"):
            OrganToxicityRisk(organ="cardiac", risk_score=0.5, confidence=bad_conf)

    def test_confidence_is_optional(self) -> None:
        """Confidence defaults to None and should not trigger validation."""
        risk = OrganToxicityRisk(organ="cardiac", risk_score=0.5)
        assert risk.confidence is None

    @pytest.mark.parametrize(
        "score,expected_band",
        [(0.0, "low"), (0.32, "low"), (0.33, "moderate"), (0.65, "moderate"),
         (0.66, "high"), (0.99, "high"), (1.0, "high")],
    )
    def test_risk_band_thresholds(self, score: float, expected_band: str) -> None:
        """risk_band should correctly bucket scores at threshold boundaries."""
        risk = OrganToxicityRisk(organ="renal", risk_score=score)
        assert risk.risk_band == expected_band


# --------------------------------------------------------------------------
# RetrievedChunk
# --------------------------------------------------------------------------

class TestRetrievedChunkSchema:
    """Validation behavior for :class:`RetrievedChunk`."""

    def _base_kwargs(self) -> dict:
        return dict(
            chunk_id="doc1::chunk0",
            doc_id="doc1",
            drug="Warfarin",
            source_type="fda_label",
            title="Warfarin Label",
            citation="FDA Prescribing Information",
            text="Sample retrieved passage text.",
        )

    def test_valid_retrieved_chunk_constructs(self) -> None:
        chunk = RetrievedChunk(score=0.87, **self._base_kwargs())
        assert chunk.score == 0.87

    @pytest.mark.parametrize("bad_score", [-0.001, 1.001, -5.0, 10.0])
    def test_out_of_bound_score_raises(self, bad_score: float) -> None:
        """Similarity score outside [0.0, 1.0] must raise ValueError."""
        with pytest.raises(ValueError, match="score"):
            RetrievedChunk(score=bad_score, **self._base_kwargs())

    @pytest.mark.parametrize("boundary_score", [0.0, 1.0])
    def test_score_boundaries_are_valid(self, boundary_score: float) -> None:
        chunk = RetrievedChunk(score=boundary_score, **self._base_kwargs())
        assert chunk.score == boundary_score