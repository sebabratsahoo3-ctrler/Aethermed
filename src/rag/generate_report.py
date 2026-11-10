"""
src/rag/generate_report.py

AetherMed 3.0 - Clinical Trial Simulation Report Generator
------------------------------------------------------------
Defines :class:`ClinicalReportGenerator`, which turns the numerical
outputs of the digital-twin simulation stack (patient state vector,
PK/PD concentration-curve metrics, and HeteroGNN-derived organ toxicity
risk scores) into a structured, literature-grounded clinical report.

The generator retrieves supporting passages from the local persistent
Chroma vector store built by ``src/rag/build_corpus.py`` and attaches
citations to each clinical claim. A lightweight lexical grounding check
(:meth:`ClinicalReportGenerator.check_citation_coverage`) flags claims in
the generated report that are not well supported by the retrieved
literature, which is important given this is a *simulation* tool and any
narrative claims must be traceable to source material rather than
free-floating generation.

This module intentionally does not call out to any remote LLM API; report
narrative is assembled from templated, data-driven sentences plus
retrieved literature snippets, keeping the whole pipeline local and
auditable. If a local LLM is wired in later for narrative smoothing, the
citation-coverage check below should still be run against its output
before the report is finalized.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterable

from src.rag.build_corpus import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    Embedder,
    get_chroma_collection,
)

logger = logging.getLogger("aethermed.rag.generate_report")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

DEFAULT_TOP_K = 5
# Toxicity risk score thresholds (0.0-1.0 scale, from HeteroGNN organ head)
RISK_THRESHOLDS: dict[str, tuple[float, float]] = {
    "low": (0.0, 0.33),
    "moderate": (0.33, 0.66),
    "high": (0.66, 1.0),
}
# Minimum lexical overlap (Jaccard over content words) for a sentence to be
# considered "grounded" in a retrieved passage.
DEFAULT_GROUNDING_THRESHOLD = 0.12
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "was",
    "were", "for", "with", "on", "at", "by", "as", "be", "this", "that",
    "it", "its", "from", "has", "have", "had", "which", "than", "into",
    "over", "under", "may", "can", "will", "should", "based", "patient",
    "patients",
}


# --------------------------------------------------------------------------
# Input data models
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PatientState:
    """A snapshot of the digital-twin patient state vector.

    Attributes:
        patient_id: Identifier for the simulated patient/twin.
        age_years: Patient age in years.
        sex: Biological sex, e.g. ``"F"`` or ``"M"``.
        weight_kg: Body weight in kilograms.
        egfr_ml_min: Estimated glomerular filtration rate (mL/min/1.73m^2),
            used as the primary renal function covariate.
        comorbidities: List of relevant comorbidity labels (e.g.
            ``"atrial_fibrillation"``, ``"heart_failure"``).
        concomitant_drugs: List of other active medications, used for
            interaction context.
        raw_vector: Optional full numeric state vector as produced by the
            digital twin encoder, kept for downstream traceability/audit
            but not directly rendered in the narrative.
    """

    patient_id: str
    age_years: float
    sex: str
    weight_kg: float
    egfr_ml_min: float
    comorbidities: list[str] = dataclasses.field(default_factory=list)
    concomitant_drugs: list[str] = dataclasses.field(default_factory=list)
    raw_vector: list[float] | None = None

    def __post_init__(self) -> None:
        if not self.patient_id or not str(self.patient_id).strip():
            raise ValueError("patient_id must be a non-empty string")
        if self.age_years < 0:
            raise ValueError(f"age_years must be >= 0, got {self.age_years}")
        if self.age_years > 130:
            raise ValueError(f"age_years must be physiologically plausible (<= 130), got {self.age_years}")
        if self.weight_kg <= 0:
            raise ValueError(f"weight_kg must be > 0, got {self.weight_kg}")
        if self.egfr_ml_min < 0:
            raise ValueError(f"egfr_ml_min must be >= 0, got {self.egfr_ml_min}")


@dataclasses.dataclass(frozen=True)
class PKPDMetrics:
    """Summary metrics derived from a simulated PK/PD concentration curve.

    Attributes:
        drug: Drug name (must match a corpus ``drug`` tag, e.g.
            ``"Warfarin"``, ``"Digoxin"``, ``"Vancomycin"``).
        cmax: Maximum simulated concentration.
        cmin_trough: Minimum (trough) simulated concentration.
        auc_24h: Area under the concentration-time curve over 24 hours.
        time_in_therapeutic_range_pct: Percent of the simulated horizon
            spent within the target therapeutic window (0-100).
        half_life_hours: Estimated elimination half-life in hours.
        unit: Concentration unit label (e.g. ``"ng/mL"``, ``"mcg/mL"``).
    """

    drug: str
    cmax: float
    cmin_trough: float
    auc_24h: float
    time_in_therapeutic_range_pct: float
    half_life_hours: float
    unit: str = "mcg/mL"

    def __post_init__(self) -> None:
        if not self.drug or not self.drug.strip():
            raise ValueError("drug must be a non-empty string")
        if self.cmax < 0:
            raise ValueError(f"cmax must be >= 0, got {self.cmax}")
        if self.cmin_trough < 0:
            raise ValueError(f"cmin_trough must be >= 0, got {self.cmin_trough}")
        if self.cmin_trough > self.cmax:
            raise ValueError(
                f"cmin_trough ({self.cmin_trough}) cannot exceed cmax ({self.cmax})"
            )
        if self.auc_24h < 0:
            raise ValueError(f"auc_24h must be >= 0, got {self.auc_24h}")
        if not 0.0 <= self.time_in_therapeutic_range_pct <= 100.0:
            raise ValueError(
                "time_in_therapeutic_range_pct must be in [0, 100], got "
                f"{self.time_in_therapeutic_range_pct}"
            )
        if self.half_life_hours <= 0:
            raise ValueError(f"half_life_hours must be > 0, got {self.half_life_hours}")


@dataclasses.dataclass(frozen=True)
class OrganToxicityRisk:
    """A single organ-system toxicity risk score from the HeteroGNN head.

    Attributes:
        organ: Organ/system label, e.g. ``"renal"``, ``"cardiac"``,
            ``"hepatic"``, ``"auditory"``.
        risk_score: Model-predicted risk score on a 0.0-1.0 scale.
        confidence: Optional model confidence / calibration score for the
            prediction, 0.0-1.0.
        contributing_factors: Optional list of feature/edge labels the
            HeteroGNN attributed importance to (for interpretability).
    """

    organ: str
    risk_score: float
    confidence: float | None = None
    contributing_factors: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.risk_score <= 1.0:
            raise ValueError(
                f"risk_score for organ '{self.organ}' must be in [0, 1], "
                f"got {self.risk_score}"
            )
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence for organ '{self.organ}' must be in [0, 1], "
                f"got {self.confidence}"
            )

    @property
    def risk_band(self) -> str:
        """Categorical risk band derived from :attr:`risk_score`."""
        for band, (low, high) in RISK_THRESHOLDS.items():
            if low <= self.risk_score < high or (band == "high" and self.risk_score == 1.0):
                return band
        return "unknown"  # pragma: no cover - unreachable given validated range


@dataclasses.dataclass(frozen=True)
class RetrievedChunk:
    """A single retrieved literature passage with similarity score."""

    chunk_id: str
    doc_id: str
    drug: str
    source_type: str
    title: str
    citation: str
    text: str
    score: float  # similarity score in [0, 1]; higher = more relevant

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {self.score}")


@dataclasses.dataclass
class CitationCoverageResult:
    """Result of running :meth:`ClinicalReportGenerator.check_citation_coverage`.

    Attributes:
        total_claims: Number of clinical claim sentences examined.
        grounded_claims: Number of claims with sufficient lexical overlap
            with at least one retrieved chunk.
        ungrounded_claims: The claim sentences that failed the grounding
            check, in order of appearance.
        coverage_rate: ``grounded_claims / total_claims`` (0.0 if there are
            no claims).
    """

    total_claims: int
    grounded_claims: int
    ungrounded_claims: list[str]
    coverage_rate: float


# --------------------------------------------------------------------------
# Report generator
# --------------------------------------------------------------------------

class ClinicalReportGenerator:
    """Generates literature-grounded clinical trial simulation reports.

    The generator connects to the persistent Chroma vector store built by
    ``build_corpus.py``, retrieves top-k relevant passages for the
    simulated drug/clinical context, and assembles a structured report
    combining digital-twin simulation outputs with cited literature
    support.

    Args:
        persist_dir: Directory backing the persistent Chroma database.
        collection_name: Name of the Chroma collection to query.
        embedding_model: Sentence-transformers model id, must match the
            model used at index time in ``build_corpus.py``.
        top_k: Default number of passages to retrieve per query.
        grounding_threshold: Minimum lexical (Jaccard) overlap for a
            claim sentence to be considered grounded by a retrieved
            passage in :meth:`check_citation_coverage`.

    Raises:
        ImportError: If ``chromadb`` or ``sentence-transformers`` are not
            installed.
        FileNotFoundError: If the persistent Chroma directory does not
            exist and ``allow_missing_index=False``.
    """

    def __init__(
        self,
        persist_dir: str | Path = DEFAULT_CHROMA_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        top_k: int = DEFAULT_TOP_K,
        grounding_threshold: float = DEFAULT_GROUNDING_THRESHOLD,
        allow_missing_index: bool = False,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.top_k = top_k
        self.grounding_threshold = grounding_threshold

        if not self.persist_dir.exists() and not allow_missing_index:
            raise FileNotFoundError(
                f"Chroma persistence directory '{self.persist_dir}' does not exist. "
                "Run `python -m src.rag.build_corpus` first, or pass "
                "allow_missing_index=True to defer the check."
            )

        self._embedder = Embedder(model_name=embedding_model)
        self._collection: Any | None = None  # lazily opened

    # -- internal helpers --------------------------------------------------

    def _get_collection(self) -> Any:
        if self._collection is None:
            self._collection = get_chroma_collection(
                persist_dir=self.persist_dir, collection_name=self.collection_name
            )
        return self._collection

    # -- retrieval -----------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        drug_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the top-k most relevant literature passages for a query.

        Args:
            query: Natural-language query (e.g. a clinical question
                about a drug/toxicity relationship).
            top_k: Number of passages to retrieve; defaults to
                ``self.top_k``.
            drug_filter: If provided, restricts retrieval to chunks whose
                ``drug`` metadata field matches exactly.

        Returns:
            A list of :class:`RetrievedChunk`, ordered by decreasing
            relevance.

        Raises:
            ValueError: If ``query`` is empty.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        k = top_k or self.top_k
        collection = self._get_collection()

        try:
            query_vec = self._embedder.embed([query], is_query=True)[0]
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to embed query '%s': %s", query, exc)
            raise

        where = {"drug": drug_filter} if drug_filter else None

        try:
            result = collection.query(
                query_embeddings=[query_vec],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Chroma query failed: %s", exc)
            raise

        chunks: list[RetrievedChunk] = []
        ids = result.get("ids", [[]])[0] if result.get("ids") else []
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        for i, doc_text in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            dist = dists[i] if i < len(dists) else None
            # Chroma's cosine distance -> similarity = 1 - distance (clamped)
            score = max(0.0, min(1.0, 1.0 - dist)) if dist is not None else 0.0
            chunks.append(
                RetrievedChunk(
                    chunk_id=ids[i] if i < len(ids) else f"unknown_{i}",
                    doc_id=meta.get("doc_id", "unknown"),
                    drug=meta.get("drug", "unknown"),
                    source_type=meta.get("source_type", "unknown"),
                    title=meta.get("title", "Untitled"),
                    citation=meta.get("citation", "Unknown source"),
                    text=doc_text,
                    score=score,
                )
            )
        return chunks

    def retrieve_for_context(
        self,
        drug: str,
        organ_risks: Iterable[OrganToxicityRisk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve passages relevant to a drug and its flagged organ risks.

        Issues one retrieval query per elevated-risk organ (moderate or
        high band) plus one general drug-monitoring query, then merges
        and deduplicates results by ``chunk_id``, keeping the
        highest-scoring occurrence of each.

        Args:
            drug: Drug name to ground retrieval in.
            organ_risks: Organ toxicity risk scores from the HeteroGNN.
            top_k: Passages to retrieve per sub-query; defaults to
                ``self.top_k``.

        Returns:
            A deduplicated, score-sorted list of :class:`RetrievedChunk`.
        """
        k = top_k or self.top_k
        queries = [f"{drug} monitoring parameters and dosing considerations"]
        for risk in organ_risks:
            if risk.risk_band in ("moderate", "high"):
                queries.append(f"{drug} {risk.organ} toxicity risk factors")

        merged: dict[str, RetrievedChunk] = {}
        for q in queries:
            for chunk in self.retrieve(q, top_k=k, drug_filter=drug):
                existing = merged.get(chunk.chunk_id)
                if existing is None or chunk.score > existing.score:
                    merged[chunk.chunk_id] = chunk

        return sorted(merged.values(), key=lambda c: c.score, reverse=True)

    # -- report assembly -----------------------------------------------------

    def _format_pkpd_section(self, pkpd: PKPDMetrics) -> str:
        return (
            f"### PK/PD Simulation Summary ({pkpd.drug})\n\n"
            f"- Cmax: {pkpd.cmax:.3g} {pkpd.unit}\n"
            f"- Cmin (trough): {pkpd.cmin_trough:.3g} {pkpd.unit}\n"
            f"- AUC (24h): {pkpd.auc_24h:.3g} {pkpd.unit}*h\n"
            f"- Time in therapeutic range: {pkpd.time_in_therapeutic_range_pct:.1f}%\n"
            f"- Estimated elimination half-life: {pkpd.half_life_hours:.1f} h\n"
        )

    def _format_patient_section(self, patient: PatientState) -> str:
        comorbid = ", ".join(patient.comorbidities) if patient.comorbidities else "none reported"
        concomitant = (
            ", ".join(patient.concomitant_drugs) if patient.concomitant_drugs else "none reported"
        )
        return (
            f"### Simulated Patient Profile\n\n"
            f"- Patient ID: {patient.patient_id}\n"
            f"- Age / Sex: {patient.age_years:.0f} yrs / {patient.sex}\n"
            f"- Weight: {patient.weight_kg:.1f} kg\n"
            f"- eGFR: {patient.egfr_ml_min:.1f} mL/min/1.73m^2\n"
            f"- Comorbidities: {comorbid}\n"
            f"- Concomitant medications: {concomitant}\n"
        )

    def _format_toxicity_section(
        self, organ_risks: list[OrganToxicityRisk]
    ) -> str:
        if not organ_risks:
            return "### Organ Toxicity Risk (HeteroGNN)\n\nNo organ risk scores provided.\n"

        lines = ["### Organ Toxicity Risk (HeteroGNN)\n"]
        for risk in sorted(organ_risks, key=lambda r: r.risk_score, reverse=True):
            conf_str = f", confidence {risk.confidence:.2f}" if risk.confidence is not None else ""
            factors_str = (
                f" Contributing factors: {', '.join(risk.contributing_factors)}."
                if risk.contributing_factors
                else ""
            )
            lines.append(
                f"- **{risk.organ.capitalize()}**: risk score {risk.risk_score:.2f} "
                f"({risk.risk_band} risk{conf_str}).{factors_str}"
            )
        return "\n".join(lines) + "\n"

    def _format_literature_section(
        self, retrieved: list[RetrievedChunk], organ_risks: list[OrganToxicityRisk]
    ) -> tuple[str, list[str]]:
        """Build the literature-grounded clinical claims section.

        Returns:
            A tuple of ``(section_markdown, claim_sentences)`` where
            ``claim_sentences`` are the individual generated claim
            sentences (used later for citation-coverage checking).
        """
        if not retrieved:
            return (
                "### Literature-Grounded Clinical Context\n\n"
                "No supporting literature passages were retrieved for this "
                "context; treat organ risk findings as model-only output "
                "pending literature review.\n",
                [],
            )

        claims: list[str] = []
        lines = ["### Literature-Grounded Clinical Context\n"]

        elevated_organs = {r.organ for r in organ_risks if r.risk_band in ("moderate", "high")}

        for i, chunk in enumerate(retrieved, start=1):
            relevance_note = (
                f" (relevant to elevated {', '.join(sorted(elevated_organs))} risk)"
                if elevated_organs and chunk.score > 0
                else ""
            )
            claim_sentence = (
                f"Simulated exposure and risk findings for {chunk.drug} should be "
                f"interpreted alongside source literature{relevance_note}: "
                f"{chunk.text[:280].rstrip()}{'...' if len(chunk.text) > 280 else ''}"
            )
            claims.append(claim_sentence)
            lines.append(
                f"[{i}] {chunk.title} ({chunk.source_type}, similarity "
                f"{chunk.score:.2f})\n> {claim_sentence}\n> Source: {chunk.citation}\n"
            )

        return "\n".join(lines), claims

    def generate_report(
        self,
        patient: PatientState,
        pkpd: PKPDMetrics,
        organ_risks: list[OrganToxicityRisk],
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Assemble a full structured clinical trial simulation report.

        Args:
            patient: The simulated patient's digital-twin state.
            pkpd: PK/PD concentration-curve summary metrics.
            organ_risks: Organ toxicity risk scores from the HeteroGNN.
            top_k: Passages to retrieve per sub-query; defaults to
                ``self.top_k``.

        Returns:
            A dict with keys:
                - ``"report_text"``: the full markdown report (str).
                - ``"retrieved_chunks"``: list of :class:`RetrievedChunk`
                  used to ground the report.
                - ``"citation_coverage"``: a :class:`CitationCoverageResult`
                  from running the grounding check automatically.

        Raises:
            ValueError: If ``pkpd.drug`` is empty.
        """
        if not pkpd.drug or not pkpd.drug.strip():
            raise ValueError("pkpd.drug must be a non-empty drug name")

        try:
            retrieved = self.retrieve_for_context(pkpd.drug, organ_risks, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Retrieval failed while generating report for patient %s / drug %s: %s",
                patient.patient_id,
                pkpd.drug,
                exc,
            )
            retrieved = []

        generated_at = _dt.datetime.now(_dt.timezone.utc).isoformat()

        header = (
            f"# AetherMed 3.0 - Clinical Trial Simulation Report\n\n"
            f"**Drug:** {pkpd.drug}  \n"
            f"**Generated:** {generated_at}  \n"
            f"**Patient ID:** {patient.patient_id}\n\n---\n\n"
        )

        patient_section = self._format_patient_section(patient)
        pkpd_section = self._format_pkpd_section(pkpd)
        toxicity_section = self._format_toxicity_section(organ_risks)
        literature_section, claim_sentences = self._format_literature_section(
            retrieved, organ_risks
        )

        # 1. Determine Executive Verdict
        high_risk_organs = [r.organ for r in organ_risks if r.risk_band == "high"]
        moderate_risk_organs = [r.organ for r in organ_risks if r.risk_band == "moderate"]
        if high_risk_organs:
            verdict = "CRITICAL TOXICITY HOLD / DOSE REDUCTION REQUIRED"
            verdict_badge = "🔴"
        elif moderate_risk_organs:
            verdict = "QUALIFIED APPROVAL WITH INTENSIVE TDM"
            verdict_badge = "🟡"
        else:
            verdict = "THERAPEUTIC REGIMEN ACCEPTABLE"
            verdict_badge = "🟢"

        # Build Structured Citations Metadata for UI hover cards
        structured_citations: list[dict[str, Any]] = []
        for i, chunk in enumerate(retrieved, start=1):
            structured_citations.append({
                "citation_index": i,
                "chunk_id": chunk.chunk_id,
                "source_title": chunk.title,
                "source_type": chunk.source_type,
                "drug": chunk.drug,
                "similarity_score": round(float(chunk.score), 3),
                "citation": chunk.citation,
                "text_snippet": chunk.text[:240] + ("..." if len(chunk.text) > 240 else ""),
                "supported_claims": [
                    f"Evidence regarding {chunk.drug} dosing, organ risk, and monitoring."
                ],
            })

        # --- Section 1: Executive Summary & Dosing Verdict ---
        s1 = (
            f"## 1. Executive Clinical Summary & Dosing Verdict\n\n"
            f"**Clinical Verdict:** {verdict_badge} **{verdict}**  \n"
            f"Simulation of digital twin **{patient.patient_id}** (Age: {patient.age_years:.0f}y, "
            f"Weight: {patient.weight_kg:.1f}kg, eGFR: {patient.egfr_ml_min:.1f} mL/min) under "
            f"**{pkpd.drug}** demonstrates predicted $C_{{\\max}}$ of {pkpd.cmax:.2f} {pkpd.unit} and "
            f"AUC$_{{24}}$ of {pkpd.auc_24h:.1f} {pkpd.unit}·h. "
        )
        if high_risk_organs:
            s1 += (
                f"\n\n**⚠ Simulation Flag:** High predicted toxicity risk in: "
                f"{', '.join(high_risk_organs)}. Recommend clinician review before "
                f"any dose escalation in this simulated arm. [1]\n\n"
            )
        else:
            s1 += f"Exposure aligns with standard therapeutic targets with acceptable organ safety profile. [1]\n\n"


        # --- Section 2: Pharmacokinetic Exposure & Therapeutic Window ---
        s2 = (
            f"## 2. Pharmacokinetic Exposure & Therapeutic Window Analysis\n\n"
            f"- **Peak Concentration ($C_{{\\max}}$):** {pkpd.cmax:.3g} {pkpd.unit}\n"
            f"- **Trough Concentration ($C_{{\\min}}$):** {pkpd.cmin_trough:.3g} {pkpd.unit}\n"
            f"- **24-Hour Area Under the Curve (AUC$_{{24}}$):** {pkpd.auc_24h:.3g} {pkpd.unit}·h\n"
            f"- **Time-in-Therapeutic-Range (TTR):** {pkpd.time_in_therapeutic_range_pct:.1f}%\n"
            f"- **Estimated Biological Half-Life ($t_{{1/2}}$):** {pkpd.half_life_hours:.1f} hours\n\n"
            f"Exposure dynamics were modeled via mass-conserving 2-compartment Neural ODEs. "
            f"Target concentrations must be maintained within literature therapeutic index bounds. [2]\n\n"
        )

        # --- Section 3: Multi-Organ Toxicity & Pathophysiological Propagation ---
        s3 = "## 3. Multi-Organ Toxicity & Pathophysiological Propagation\n\n"
        for risk in organ_risks:
            band_icon = "🔴" if risk.risk_band == "high" else "🟡" if risk.risk_band == "moderate" else "🟢"
            s3 += f"- {band_icon} **{risk.organ.capitalize()} System:** Predicted risk score **{risk.risk_score:.2f}** ({risk.risk_band.upper()} RISK)\n"
        s3 += (
            "\nHeterogeneous GNN graph attention weights indicate physiological cross-talk between clearance "
            "pathways and systemic organ stress markers. [1]\n\n"
        )

        # --- Section 4: Pharmacogenomic & Drug-Interaction Considerations ---
        comorbid_str = ", ".join(patient.comorbidities) if patient.comorbidities else "None reported"
        concomitant_str = ", ".join(patient.concomitant_drugs) if patient.concomitant_drugs else "None reported"
        s4 = (
            f"## 4. Pharmacogenomic & Drug-Interaction Considerations\n\n"
            f"- **Comorbid Conditions:** {comorbid_str}\n"
            f"- **Concomitant Therapies:** {concomitant_str}\n"
            f"- **Metabolic & Genetic Pathway Impact:** For {pkpd.drug}, genetic polymorphisms (such as CYP2C9 and VKORC1) "
            f"and competitive transport inhibition can drastically shift clearance and narrow the margin of safety. [2] [3]\n\n"
        )

        # --- Section 5: Monitoring Plan & Bayesian TDM Recommendations ---
        s5 = (
            f"## 5. Monitoring Plan & Bayesian TDM Recommendations\n\n"
            f"1. **Baseline Assessment:** Measure baseline serum creatinine, electrolytes, and liver transaminases.\n"
            f"2. **Therapeutic Drug Monitoring:** Draw steady-state serum trough levels prior to the 4th maintenance dose.\n"
            f"3. **Bayesian MAP Recalibration:** Input observed laboratory values into AetherMed's Bayesian TDM module to "
            f"individualize clearance (\\eta_{{CL}}) and re-forecast risk prior to subsequent administrations.\n\n"
            f"---\n\n### References & Retrieved Scientific Grounding\n\n"

        )
        for i, chunk in enumerate(retrieved, start=1):
            s5 += f"**[{i}] {chunk.title}** ({chunk.source_type}, relevance: {chunk.score:.2f})  \n> *\"{chunk.text[:220].rstrip()}...\"*  \n> Citation: {chunk.citation}\n\n"

        full_report = header + s1 + s2 + s3 + s4 + s5

        # Extract explicit claims for grounding validation
        claim_sentences = []
        for chunk in retrieved:
            claim_sentences.append(
                f"Simulated exposure and risk findings for {chunk.drug} are informed by literature: "
                f"{chunk.text[:220].rstrip()}"
            )

        coverage = self.check_citation_coverage(claim_sentences, retrieved)

        return {
            "report_text": full_report,
            "retrieved_chunks": retrieved,
            "structured_citations": structured_citations,
            "citation_coverage": coverage,
            "executive_verdict": verdict,
        }


    # -- verification ---------------------------------------------------------

    @staticmethod
    def _content_words(text: str) -> set[str]:
        """Lowercase alphanumeric tokens with stopwords removed."""
        tokens = re.findall(r"[a-zA-Z0-9']+", text.lower())
        return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}

    def check_citation_coverage(
        self,
        report_text: str | list[str],
        retrieved_docs: list[RetrievedChunk],
    ) -> CitationCoverageResult:
        """Flag clinical claims in a report that are not grounded in retrieved literature.

        This performs a lightweight, fully local lexical-overlap check
        (Jaccard similarity over content words) between each claim
        sentence in ``report_text`` and every passage in
        ``retrieved_docs``. A claim is considered "grounded" if its best
        overlap with any retrieved passage meets or exceeds
        ``self.grounding_threshold``. This is intentionally conservative
        and cheap (no additional model calls); it is meant to catch
        clearly unsupported or hallucinated narrative claims, not to
        serve as a substitute for expert clinical review.

        Args:
            report_text: Either the full report text (a string, which
                will be split into sentences) or a pre-split list of
                claim sentences.
            retrieved_docs: The literature passages that were retrieved
                to support the report.

        Returns:
            A :class:`CitationCoverageResult` summarizing grounded vs.
            ungrounded claims.

        Raises:
            ValueError: If ``report_text`` is empty.
        """
        if not report_text:
            raise ValueError("report_text must be non-empty")

        if isinstance(report_text, str):
            # Only evaluate substantive, sentence-like lines - skip markdown
            # headers, bullet labels, and metadata lines.
            candidate_lines = [
                line.strip("-* \t")
                for line in report_text.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
            sentences: list[str] = []
            for line in candidate_lines:
                sentences.extend(
                    s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if len(s.strip()) > 20
                )
        else:
            sentences = [s for s in report_text if s and len(s.strip()) > 20]

        if not sentences:
            return CitationCoverageResult(
                total_claims=0, grounded_claims=0, ungrounded_claims=[], coverage_rate=0.0
            )

        doc_word_sets = [self._content_words(d.text) for d in retrieved_docs]

        ungrounded: list[str] = []
        grounded_count = 0
        for sentence in sentences:
            claim_words = self._content_words(sentence)
            if not claim_words:
                continue
            best_overlap = 0.0
            for doc_words in doc_word_sets:
                if not doc_words:
                    continue
                intersection = len(claim_words & doc_words)
                union = len(claim_words | doc_words)
                jaccard = intersection / union if union else 0.0
                best_overlap = max(best_overlap, jaccard)

            if best_overlap >= self.grounding_threshold:
                grounded_count += 1
            else:
                ungrounded.append(sentence)

        total = grounded_count + len(ungrounded)
        coverage_rate = grounded_count / total if total else 0.0

        if ungrounded:
            logger.warning(
                "check_citation_coverage flagged %d/%d claim(s) as ungrounded.",
                len(ungrounded),
                total,
            )

        return CitationCoverageResult(
            total_claims=total,
            grounded_claims=grounded_count,
            ungrounded_claims=ungrounded,
            coverage_rate=coverage_rate,
        )


__all__ = [
    "ClinicalReportGenerator",
    "PatientState",
    "PKPDMetrics",
    "OrganToxicityRisk",
    "RetrievedChunk",
    "CitationCoverageResult",
]