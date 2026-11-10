"""
backend/schemas.py
==================

Pydantic schemas for the AetherMed 4.0 FastAPI backend.
Defines strongly-typed inputs and outputs for health, cohort analysis,
simulation, and in-silico population trials.
"""

from __future__ import annotations

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class TDMPoint(BaseModel):
    time_h: float = Field(..., description="Time of blood draw post-initiation in hours")
    concentration: float = Field(..., description="Measured serum drug concentration")


class PatientTwinInput(BaseModel):
    patient_id: str = Field(default="TWIN-8821")
    age: float = Field(default=65.0, ge=0.0, le=120.0)
    gender: str = Field(default="female")
    weight: float = Field(default=72.0, gt=0.0)
    serum_creatinine: float = Field(default=1.2, gt=0.0)
    egfr: Optional[float] = Field(default=None)
    alt: float = Field(default=35.0, ge=0.0)
    ast: float = Field(default=38.0, ge=0.0)
    potassium: float = Field(default=4.25, ge=1.5, le=9.0)
    albumin: float = Field(default=4.10, ge=1.0, le=7.0)
    cyp2c9_status: str = Field(default="*1/*1")
    vkorc1_status: str = Field(default="G/A")
    comorbidities: List[str] = Field(default_factory=lambda: ["hypertension"])
    concomitant_drugs: List[str] = Field(default_factory=lambda: ["amiodarone"])
    drug: str = Field(default="Warfarin")
    dose_mg: float = Field(default=5.0, ge=0.0)
    interval_h: int = Field(default=24, ge=4, le=72)
    duration_h: int = Field(default=72, ge=24, le=168)
    loading_dose_mg: Optional[float] = None
    tdm_observations: List[TDMPoint] = Field(default_factory=list)


class PKResponse(BaseModel):
    drug: str
    display_unit: str
    time_hours: List[float]
    c_central_neural: List[float]
    c_central_physics: List[float]
    c_peripheral: List[float]
    c_cql_recommended: Optional[List[float]] = None
    cmax: float
    cmin_trough: float
    auc_24h: float
    time_in_therapeutic_range_pct: float
    half_life_hours: float
    clearance_l_h: float
    v_central_l: float
    therapeutic_min: float
    therapeutic_max: float
    mass_conservation_error: float
    active_model_source: str


class OrganRiskItem(BaseModel):
    organ: str
    risk_score: float
    risk_band: str
    confidence: float
    interval_lower: float
    interval_upper: float


class GraphNode(BaseModel):
    id: str
    type: str
    label: str
    risk_score: Optional[float] = None
    features: List[float] = Field(default_factory=list)


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    attention_weight: float = 1.0


class GraphPayload(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]


class RLResponse(BaseModel):
    baseline_dose_mg: float
    cql_recommended_dose_mg: float
    dose_delta_mg: float
    analytical_target_dose_mg: float
    recommendation_rationale: str
    ope_metrics: Dict[str, Any]


class SafetyOODResponse(BaseModel):
    mahalanobis_distance: float
    threshold: float
    is_ood: bool
    severity: str
    feature_attributions: List[Dict[str, Any]]
    dosing_alerts: List[str]
    is_dose_unsafe: bool


class CitationItem(BaseModel):
    citation_index: int
    chunk_id: str
    source_title: str
    source_type: str
    drug: str
    similarity_score: float
    citation: str
    text_snippet: str
    supported_claims: List[str]


class RAGResponse(BaseModel):
    report_markdown: str
    grounding_coverage_pct: float
    executive_verdict: str
    structured_citations: List[CitationItem]


class SimulationResponse(BaseModel):
    patient_id: str
    drug: str
    pk_results: PKResponse
    organ_risks: List[OrganRiskItem]
    overall_risk: float
    graph_topology: GraphPayload
    rl_policy: RLResponse
    safety_ood: SafetyOODResponse
    rag_report: RAGResponse
    bayesian_tdm_applied: bool = False
    bayesian_cl_multiplier: float = 1.0
    bayesian_v1_multiplier: float = 1.0


class CohortSummary(BaseModel):
    total_twins: int
    mean_age: float
    mean_weight: float
    mean_creatinine: float
    mean_egfr: float
    mean_alt: float
    mean_ast: float
    distributions: Dict[str, Any]
    correlation_matrix: Dict[str, Dict[str, float]]
    sample_patients: List[Dict[str, Any]]


class TrialFilter(BaseModel):
    drug: str = Field(default="Vancomycin")
    standard_dose_mg: float = Field(default=1000.0)
    interval_h: int = Field(default=12)
    min_age: float = Field(default=18.0)
    max_age: float = Field(default=90.0)
    max_egfr: Optional[float] = Field(default=60.0, description="Optional renal impairment filter")
    sample_size: int = Field(default=100, ge=10, le=500)


class TrialResponse(BaseModel):
    drug: str
    sample_size: int
    filter_description: str
    standard_physician_regimen: str
    aethermed_cql_regimen: str
    standard_mean_ttr_pct: float
    cql_mean_ttr_pct: float
    ttr_gain_pct: float
    standard_nephrotoxicity_incidence_pct: float
    cql_nephrotoxicity_incidence_pct: float
    toxicity_relative_reduction_pct: float
    population_pk_summary: Dict[str, Any]
    patient_outcomes_sample: List[Dict[str, Any]]
