"""
src/rag/eval_rag.py

AetherMed 3.0 - RAG Retrieval & Grounding Evaluation
-------------------------------------------------------
Evaluates the retrieval quality of the local Chroma vector store built by
``src/rag/build_corpus.py`` against a small benchmark clinical QA
dataset, and evaluates the faithfulness of generated reports using the
:meth:`ClinicalReportGenerator.check_citation_coverage` grounding check.

Metrics implemented:
    - Precision@k
    - Recall@k
    - Mean Reciprocal Rank (MRR), reported alongside precision/recall as
      it is standard/cheap to compute and useful for debugging ranking
      quality.
    - Citation coverage / faithfulness rate (fraction of generated claim
      sentences considered "grounded" by retrieved evidence).

The benchmark QA set is a small, hand-authored set of clinical questions
about Warfarin, Digoxin, and Vancomycin with relevance judgments
(``relevant_doc_ids``) pointing at the ``doc_id`` values defined in the
seed corpus in ``build_corpus.py``. This keeps evaluation fully offline
and reproducible; a larger/external benchmark can be substituted via
``load_benchmark(path=...)``.

Usage
~~~~~
    python -m src.rag.eval_rag --k 3

    # or programmatically
    from src.rag.eval_rag import run_retrieval_evaluation
    results = run_retrieval_evaluation(k=3)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path
from typing import Any

from src.rag.build_corpus import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
)
from src.rag.generate_report import ClinicalReportGenerator, RetrievedChunk

logger = logging.getLogger("aethermed.rag.eval_rag")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

DEFAULT_K = 3


# --------------------------------------------------------------------------
# Benchmark data model
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class BenchmarkQuery:
    """A single labeled retrieval-evaluation example.

    Attributes:
        qid: Unique identifier for the query.
        question: The natural-language clinical question.
        drug: Drug the question pertains to (used as a retrieval filter,
            matching the ``drug`` metadata field in the corpus).
        relevant_doc_ids: Set of ``doc_id`` values (from the seed corpus
            in ``build_corpus.py``) considered relevant/correct sources
            for this question. A retrieved chunk counts as relevant if
            its ``doc_id`` is in this set (chunk-level granularity is
            collapsed to document-level for judgment, since relevance
            was authored at the document level).
    """

    qid: str
    question: str
    drug: str
    relevant_doc_ids: frozenset[str]


def _default_benchmark() -> list[BenchmarkQuery]:
    """Built-in benchmark clinical QA set with document-level relevance judgments."""
    return [
        BenchmarkQuery(
            qid="q1",
            question="What INR range should be targeted during warfarin therapy?",
            drug="Warfarin",
            relevant_doc_ids=frozenset({"warfarin_label_01"}),
        ),
        BenchmarkQuery(
            qid="q2",
            question="Which genetic factors affect warfarin dose requirements?",
            drug="Warfarin",
            relevant_doc_ids=frozenset({"warfarin_label_01", "warfarin_pubmed_01"}),
        ),
        BenchmarkQuery(
            qid="q3",
            question="What drug interactions increase bleeding risk with warfarin?",
            drug="Warfarin",
            relevant_doc_ids=frozenset({"warfarin_label_02"}),
        ),
        BenchmarkQuery(
            qid="q4",
            question="How often should INR be monitored when starting warfarin?",
            drug="Warfarin",
            relevant_doc_ids=frozenset({"warfarin_dailymed_01"}),
        ),
        BenchmarkQuery(
            qid="q5",
            question="What serum digoxin concentration range is considered therapeutic?",
            drug="Digoxin",
            relevant_doc_ids=frozenset({"digoxin_label_01"}),
        ),
        BenchmarkQuery(
            qid="q6",
            question="What electrolyte disturbances increase digoxin toxicity risk?",
            drug="Digoxin",
            relevant_doc_ids=frozenset({"digoxin_dailymed_01"}),
        ),
        BenchmarkQuery(
            qid="q7",
            question="How does renal function affect digoxin dosing and clearance?",
            drug="Digoxin",
            relevant_doc_ids=frozenset({"digoxin_label_01", "digoxin_pubmed_01"}),
        ),
        BenchmarkQuery(
            qid="q8",
            question="What AUC/MIC target is recommended for vancomycin in serious MRSA infection?",
            drug="Vancomycin",
            relevant_doc_ids=frozenset({"vancomycin_label_01", "vancomycin_pubmed_01"}),
        ),
        BenchmarkQuery(
            qid="q9",
            question="Is AUC-guided or trough-guided vancomycin dosing associated with less nephrotoxicity?",
            drug="Vancomycin",
            relevant_doc_ids=frozenset({"vancomycin_pubmed_01"}),
        ),
        BenchmarkQuery(
            qid="q10",
            question="What causes vancomycin flushing syndrome and how is it managed?",
            drug="Vancomycin",
            relevant_doc_ids=frozenset({"vancomycin_label_01"}),
        ),
    ]


def load_benchmark(path: str | Path | None = None) -> list[BenchmarkQuery]:
    """Load the benchmark clinical QA dataset.

    Args:
        path: Optional path to a JSON file containing a list of objects
            with keys ``qid``, ``question``, ``drug``,
            ``relevant_doc_ids`` (list of strings). If omitted, the
            built-in benchmark is used.

    Returns:
        A list of :class:`BenchmarkQuery`.

    Raises:
        ValueError: If a benchmark entry is missing required fields.
        FileNotFoundError: If ``path`` is provided but does not exist.
    """
    if path is None:
        return _default_benchmark()

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {file_path}")

    raw = json.loads(file_path.read_text(encoding="utf-8"))
    required = {"qid", "question", "drug", "relevant_doc_ids"}
    queries: list[BenchmarkQuery] = []
    for entry in raw:
        missing = required - entry.keys()
        if missing:
            raise ValueError(f"Benchmark entry missing required fields: {sorted(missing)}")
        queries.append(
            BenchmarkQuery(
                qid=entry["qid"],
                question=entry["question"],
                drug=entry["drug"],
                relevant_doc_ids=frozenset(entry["relevant_doc_ids"]),
            )
        )
    return queries


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def precision_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: frozenset[str], k: int) -> float:
    """Compute Precision@k for a single query.

    Args:
        retrieved_doc_ids: Ranked list of retrieved ``doc_id`` values
            (deduplicated document-level ids, best rank first).
        relevant_doc_ids: Set of ground-truth relevant ``doc_id`` values.
        k: Cutoff rank.

    Returns:
        Precision@k as a float in [0, 1]. Returns 0.0 if ``k <= 0``.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for d in top_k if d in relevant_doc_ids)
    return hits / len(top_k)


def recall_at_k(retrieved_doc_ids: list[str], relevant_doc_ids: frozenset[str], k: int) -> float:
    """Compute Recall@k for a single query.

    Args:
        retrieved_doc_ids: Ranked list of retrieved ``doc_id`` values.
        relevant_doc_ids: Set of ground-truth relevant ``doc_id`` values.
        k: Cutoff rank.

    Returns:
        Recall@k as a float in [0, 1]. Returns 0.0 if there are no
        relevant documents for the query (undefined recall treated as 0
        to keep aggregate averaging well-defined; callers wanting to
        exclude such queries can filter on ``len(relevant_doc_ids) == 0``
        beforehand).
    """
    if not relevant_doc_ids:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    hits = sum(1 for d in top_k if d in relevant_doc_ids)
    return hits / len(relevant_doc_ids)


def reciprocal_rank(retrieved_doc_ids: list[str], relevant_doc_ids: frozenset[str]) -> float:
    """Compute the reciprocal rank of the first relevant retrieved document.

    Args:
        retrieved_doc_ids: Ranked list of retrieved ``doc_id`` values.
        relevant_doc_ids: Set of ground-truth relevant ``doc_id`` values.

    Returns:
        ``1 / rank`` of the first relevant hit (1-indexed), or 0.0 if no
        relevant document was retrieved.
    """
    for idx, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1.0 / idx
    return 0.0


def _dedupe_doc_ids_by_rank(chunks: list[RetrievedChunk]) -> list[str]:
    """Collapse a ranked chunk list to a ranked, deduplicated doc_id list."""
    seen: list[str] = []
    for c in chunks:
        if c.doc_id not in seen:
            seen.append(c.doc_id)
    return seen


# --------------------------------------------------------------------------
# Retrieval evaluation
# --------------------------------------------------------------------------

@dataclasses.dataclass
class QueryEvalResult:
    """Per-query retrieval evaluation result."""

    qid: str
    question: str
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    retrieved_doc_ids: list[str]
    relevant_doc_ids: list[str]


@dataclasses.dataclass
class RetrievalEvalSummary:
    """Aggregate retrieval evaluation results across the benchmark."""

    k: int
    num_queries: int
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    per_query: list[QueryEvalResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "num_queries": self.num_queries,
            "mean_precision_at_k": self.mean_precision_at_k,
            "mean_recall_at_k": self.mean_recall_at_k,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "per_query": [dataclasses.asdict(q) for q in self.per_query],
        }


def run_retrieval_evaluation(
    k: int = DEFAULT_K,
    benchmark_path: str | Path | None = None,
    persist_dir: str | Path = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    use_drug_filter: bool = True,
) -> RetrievalEvalSummary:
    """Run Precision@k / Recall@k / MRR evaluation over the benchmark QA set.

    Args:
        k: Cutoff rank for precision/recall.
        benchmark_path: Optional path to an external benchmark JSON file;
            defaults to the built-in benchmark.
        persist_dir: Directory backing the persistent Chroma database.
        collection_name: Chroma collection to query.
        embedding_model: Embedding model id (must match index-time model).
        use_drug_filter: Whether to restrict retrieval to the query's
            labeled drug via metadata filtering (mirrors production
            retrieval behavior in ``retrieve_for_context``).

    Returns:
        A :class:`RetrievalEvalSummary` with per-query and aggregate
        metrics.

    Raises:
        ValueError: If ``k <= 0`` or the benchmark is empty.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    benchmark = load_benchmark(benchmark_path)
    if not benchmark:
        raise ValueError("Benchmark dataset is empty")

    generator = ClinicalReportGenerator(
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_model=embedding_model,
        top_k=max(k, DEFAULT_K),
        allow_missing_index=False,
    )

    per_query_results: list[QueryEvalResult] = []
    for query in benchmark:
        try:
            retrieved_chunks = generator.retrieve(
                query.question,
                top_k=max(k * 3, k),  # over-retrieve chunks before doc-level dedupe
                drug_filter=query.drug if use_drug_filter else None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Retrieval failed for query '%s' (%s): %s", query.qid, query.question, exc)
            retrieved_chunks = []

        retrieved_doc_ids = _dedupe_doc_ids_by_rank(retrieved_chunks)

        p_at_k = precision_at_k(retrieved_doc_ids, query.relevant_doc_ids, k)
        r_at_k = recall_at_k(retrieved_doc_ids, query.relevant_doc_ids, k)
        rr = reciprocal_rank(retrieved_doc_ids, query.relevant_doc_ids)

        per_query_results.append(
            QueryEvalResult(
                qid=query.qid,
                question=query.question,
                precision_at_k=p_at_k,
                recall_at_k=r_at_k,
                reciprocal_rank=rr,
                retrieved_doc_ids=retrieved_doc_ids[:k],
                relevant_doc_ids=sorted(query.relevant_doc_ids),
            )
        )

    n = len(per_query_results)
    mean_p = sum(r.precision_at_k for r in per_query_results) / n
    mean_r = sum(r.recall_at_k for r in per_query_results) / n
    mean_rr = sum(r.reciprocal_rank for r in per_query_results) / n

    summary = RetrievalEvalSummary(
        k=k,
        num_queries=n,
        mean_precision_at_k=mean_p,
        mean_recall_at_k=mean_r,
        mean_reciprocal_rank=mean_rr,
        per_query=per_query_results,
    )
    logger.info(
        "Retrieval eval (k=%d): P@k=%.3f R@k=%.3f MRR=%.3f over %d queries.",
        k, mean_p, mean_r, mean_rr, n,
    )
    return summary


# --------------------------------------------------------------------------
# Faithfulness / citation-coverage evaluation
# --------------------------------------------------------------------------

@dataclasses.dataclass
class FaithfulnessEvalSummary:
    """Aggregate citation-coverage / faithfulness evaluation results."""

    num_reports: int
    mean_coverage_rate: float
    total_claims: int
    total_grounded_claims: int
    faithfulness_rate: float  # total_grounded_claims / total_claims across all reports
    per_report_coverage: list[float]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def run_faithfulness_evaluation(
    generator: ClinicalReportGenerator,
    sample_reports: list[dict[str, Any]],
) -> FaithfulnessEvalSummary:
    """Evaluate citation-coverage faithfulness across a set of generated reports.

    Args:
        generator: A :class:`ClinicalReportGenerator` instance (used only
            for its :meth:`check_citation_coverage` method here, since
            reports are already generated).
        sample_reports: A list of dicts, each with keys
            ``"report_text"`` (str) and ``"retrieved_chunks"``
            (list of :class:`RetrievedChunk`), as returned by
            ``ClinicalReportGenerator.generate_report``.

    Returns:
        A :class:`FaithfulnessEvalSummary` aggregating per-report and
        overall (claim-weighted) faithfulness rates.

    Raises:
        ValueError: If ``sample_reports`` is empty.
    """
    if not sample_reports:
        raise ValueError("sample_reports must be non-empty")

    per_report_coverage: list[float] = []
    total_claims = 0
    total_grounded = 0

    for report in sample_reports:
        coverage = generator.check_citation_coverage(
            report["report_text"], report["retrieved_chunks"]
        )
        per_report_coverage.append(coverage.coverage_rate)
        total_claims += coverage.total_claims
        total_grounded += coverage.grounded_claims

    mean_coverage = sum(per_report_coverage) / len(per_report_coverage)
    faithfulness_rate = total_grounded / total_claims if total_claims else 0.0

    summary = FaithfulnessEvalSummary(
        num_reports=len(sample_reports),
        mean_coverage_rate=mean_coverage,
        total_claims=total_claims,
        total_grounded_claims=total_grounded,
        faithfulness_rate=faithfulness_rate,
        per_report_coverage=per_report_coverage,
    )
    logger.info(
        "Faithfulness eval: mean_coverage=%.3f faithfulness_rate=%.3f over %d report(s), "
        "%d/%d claims grounded.",
        mean_coverage, faithfulness_rate, len(sample_reports), total_grounded, total_claims,
    )
    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the AetherMed RAG pipeline.")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Cutoff rank for Precision/Recall@k")
    parser.add_argument("--benchmark-path", type=str, default=None)
    parser.add_argument("--persist-dir", type=str, default=str(DEFAULT_CHROMA_DIR))
    parser.add_argument("--collection-name", type=str, default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to write results as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: runs retrieval evaluation and prints/writes a summary."""
    args = _parse_args(argv)
    try:
        summary = run_retrieval_evaluation(
            k=args.k,
            benchmark_path=args.benchmark_path,
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            embedding_model=args.embedding_model,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Retrieval evaluation failed: %s", exc)
        return 1

    result_dict = summary.to_dict()
    print(json.dumps(result_dict, indent=2))

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
        logger.info("Wrote evaluation results to %s", out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())