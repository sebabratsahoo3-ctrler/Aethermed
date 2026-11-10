"""
src/rag/build_corpus.py

AetherMed 3.0 - RAG Corpus Builder
-----------------------------------
Builds a small, local, offline clinical literature corpus for the target
drugs used by the digital-twin PK/PD simulation engine (Warfarin, Digoxin,
Vancomycin), chunks the corpus into retrieval-sized passages, embeds those
passages with a local sentence-transformers model
(``BAAI/bge-small-en-v1.5``), and persists the resulting vectors into a
local Chroma database at ``data/processed/chroma_db``.

Notes on data provenance
~~~~~~~~~~~~~~~~~~~~~~~~
This module does NOT scrape or reproduce verbatim text from FDA labels,
DailyMed, or PubMed (doing so at build time would require live network
access and would risk redistributing copyrighted/verbatim regulatory
text). Instead, it ships a small set of **synthetic, clinically-informed
reference documents** written in the *style* of FDA drug labels, DailyMed
monographs, and PubMed abstracts, tagged with a ``source_type`` metadata
field (``fda_label`` | ``dailymed`` | ``pubmed_abstract``). This keeps the
pipeline fully offline/deterministic for local development and testing.

If real regulatory/literature text is available (e.g. downloaded
separately and placed under ``data/raw/``), set the ``CORPUS_DIR``
environment variable or pass ``corpus_dir=`` to ``build_corpus`` and this
module will ingest ``*.txt`` / ``*.json`` files from there instead of (or
in addition to) the synthetic seed set.

Usage
~~~~~
    python -m src.rag.build_corpus

    # or programmatically
    from src.rag.build_corpus import build_corpus
    build_corpus()
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

logger = logging.getLogger("aethermed.rag.build_corpus")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_CHROMA_DIR = Path("data/processed/chroma_db")
DEFAULT_COLLECTION_NAME = "aethermed_clinical_corpus"
DEFAULT_CHUNK_SIZE = 220          # target words per chunk
DEFAULT_CHUNK_OVERLAP = 40        # words of overlap between consecutive chunks
DEFAULT_EMBED_BATCH_SIZE = 32

SourceType = Literal["fda_label", "dailymed", "pubmed_abstract"]


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class RawDocument:
    """A single unchunked source document prior to ingestion."""

    doc_id: str
    drug: str
    source_type: SourceType
    title: str
    text: str
    citation: str


@dataclasses.dataclass(frozen=True)
class Chunk:
    """A retrieval-ready passage derived from a :class:`RawDocument`."""

    chunk_id: str
    doc_id: str
    drug: str
    source_type: SourceType
    title: str
    citation: str
    chunk_index: int
    text: str


# --------------------------------------------------------------------------
# Seed corpus (synthetic, offline, clinically-informed reference text)
# --------------------------------------------------------------------------

def _seed_documents() -> list[RawDocument]:
    """Return the built-in synthetic reference corpus.

    Each entry is an original, paraphrased summary written for this
    project (not a verbatim reproduction of any single copyrighted
    source) and is tagged with the regulatory/literature source type it
    emulates so downstream retrieval and citation logic can reason about
    provenance and evidence strength.
    """
    docs: list[RawDocument] = [
        # ---------------------------------------------------------- Warfarin
        RawDocument(
            doc_id="warfarin_label_01",
            drug="Warfarin",
            source_type="fda_label",
            title="Warfarin Sodium - Prescribing Information (Label Summary)",
            citation="FDA Prescribing Information, Warfarin Sodium Tablets (label summary)",
            text=(
                "Warfarin sodium is a vitamin K antagonist anticoagulant indicated for "
                "prophylaxis and treatment of venous thrombosis and its extension, "
                "pulmonary embolism, and for the prevention of thromboembolic "
                "complications associated with atrial fibrillation and mechanical heart "
                "valves. Dosing must be individualized according to the patient's INR "
                "response, with routine INR monitoring required throughout therapy. "
                "Warfarin has a narrow therapeutic index; the target INR range for most "
                "indications is 2.0 to 3.0, with a higher target of 2.5 to 3.5 for "
                "certain mechanical valve patients. Warfarin carries a boxed warning for "
                "bleeding risk, which can be fatal. Risk factors for bleeding include "
                "high intensity anticoagulation, advanced age, variability in INR, "
                "history of gastrointestinal bleeding, hypertension, cerebrovascular "
                "disease, serious heart disease, anemia, malignancy, trauma, renal "
                "insufficiency, and concomitant use of drugs that increase bleeding "
                "risk. Regular monitoring of INR should be performed on all treated "
                "patients. Warfarin metabolism is primarily mediated by cytochrome P450 "
                "isoenzyme CYP2C9, and genetic polymorphisms of CYP2C9 and VKORC1 can "
                "substantially affect dose requirements. Patients with CYP2C9 and/or "
                "VKORC1 variant genotypes may require lower warfarin doses and closer "
                "INR monitoring, especially during treatment initiation."
            ),
        ),
        RawDocument(
            doc_id="warfarin_label_02",
            drug="Warfarin",
            source_type="fda_label",
            title="Warfarin Sodium - Drug Interactions and Contraindications",
            citation="FDA Prescribing Information, Warfarin Sodium Tablets (label summary)",
            text=(
                "Warfarin is contraindicated in pregnancy, in patients with a "
                "hemorrhagic tendency, and in patients with recent or contemplated "
                "surgery of the central nervous system or eye, or traumatic surgery "
                "resulting in large open surfaces. Warfarin interacts with a large "
                "number of commonly used drugs and foods through pharmacokinetic and "
                "pharmacodynamic mechanisms. Drugs that inhibit CYP2C9 (for example "
                "amiodarone, fluconazole, metronidazole, and certain sulfonamides) can "
                "increase warfarin plasma concentration and INR, increasing bleeding "
                "risk. Drugs that induce CYP2C9 or related pathways (for example "
                "rifampin, carbamazepine) can decrease warfarin effect. Concomitant use "
                "of antiplatelet agents, NSAIDs, or other anticoagulants increases the "
                "risk of bleeding through additive pharmacodynamic effects even when INR "
                "is within the therapeutic range. Vitamin K intake, whether from diet or "
                "supplements, can reduce warfarin's anticoagulant effect and contribute "
                "to INR variability. Patients should be counseled to maintain consistent "
                "vitamin K intake rather than eliminating it entirely."
            ),
        ),
        RawDocument(
            doc_id="warfarin_dailymed_01",
            drug="Warfarin",
            source_type="dailymed",
            title="Warfarin - Monograph Excerpt (Monitoring Parameters)",
            citation="DailyMed-style monograph excerpt, Warfarin Sodium",
            text=(
                "Monitoring parameters for warfarin therapy include baseline and "
                "periodic INR (or prothrombin time), hemoglobin/hematocrit, and clinical "
                "assessment for signs of bleeding such as bruising, hematuria, "
                "melena, or mucosal bleeding. During initiation, INR should be checked "
                "frequently (for example every 1 to 2 days) until stable, then the "
                "interval can be extended to weekly and eventually to every 4 weeks in a "
                "stable outpatient on a consistent dose. Any change in concomitant "
                "medication, diet, alcohol intake, or hepatic/renal function warrants "
                "closer INR monitoring. In cases of supratherapeutic INR without "
                "bleeding, management may include withholding one or more doses; for "
                "INR values markedly above range or with clinically significant "
                "bleeding, vitamin K administration and/or reversal agents such as "
                "prothrombin complex concentrate may be indicated per institutional "
                "protocol."
            ),
        ),
        RawDocument(
            doc_id="warfarin_pubmed_01",
            drug="Warfarin",
            source_type="pubmed_abstract",
            title="Pharmacogenomic dosing algorithms for warfarin: a synthesis",
            citation="Synthetic literature abstract (pharmacogenomics of warfarin dosing)",
            text=(
                "Background: Warfarin dose requirements vary widely between patients, "
                "driven substantially by polymorphisms in CYP2C9 and VKORC1. Methods: "
                "This synthesis summarizes evidence from pharmacogenomic dosing trials "
                "comparing genotype-guided dosing algorithms with standard clinical "
                "dosing during warfarin initiation. Results: Genotype-informed dosing "
                "algorithms incorporating CYP2C9 and VKORC1 status, along with clinical "
                "covariates such as age, body surface area, and interacting "
                "medications, more accurately predict stable maintenance dose than "
                "clinical algorithms alone, particularly in patients carrying reduced "
                "function CYP2C9 alleles or VKORC1 low-dose haplotypes. These patients "
                "are at elevated risk of supratherapeutic INR and bleeding events during "
                "the initiation phase if dosed using standard fixed protocols. "
                "Conclusion: Incorporating pharmacogenomic and physiologic covariates "
                "into initial dose selection may reduce time to stable INR and lower "
                "early bleeding risk, supporting model-informed dosing approaches in "
                "digital twin simulation frameworks."
            ),
        ),
        # ----------------------------------------------------------- Digoxin
        RawDocument(
            doc_id="digoxin_label_01",
            drug="Digoxin",
            source_type="fda_label",
            title="Digoxin - Prescribing Information (Label Summary)",
            citation="FDA Prescribing Information, Digoxin Tablets (label summary)",
            text=(
                "Digoxin is a cardiac glycoside indicated for the treatment of mild to "
                "moderate heart failure and for control of ventricular response rate in "
                "patients with chronic atrial fibrillation. Digoxin has a narrow "
                "therapeutic index; the therapeutic serum concentration range is "
                "generally considered to be approximately 0.5 to 2.0 ng/mL, though "
                "toxicity can occur even within this range, particularly in the "
                "presence of hypokalemia, hypomagnesemia, hypercalcemia, hypothyroidism, "
                "or renal impairment. Digoxin is eliminated primarily by renal "
                "excretion, largely as unchanged drug, and dose adjustment is required "
                "in patients with reduced renal function. Signs and symptoms of digoxin "
                "toxicity include anorexia, nausea, vomiting, visual disturbances "
                "(including altered color perception), and cardiac arrhythmias such as "
                "ectopic beats, AV block, and ventricular arrhythmias. Serum digoxin "
                "concentration monitoring, along with serum electrolytes and renal "
                "function, is recommended, particularly after dose changes or in "
                "patients with fluctuating renal function."
            ),
        ),
        RawDocument(
            doc_id="digoxin_dailymed_01",
            drug="Digoxin",
            source_type="dailymed",
            title="Digoxin - Monograph Excerpt (Toxicity and Interactions)",
            citation="DailyMed-style monograph excerpt, Digoxin",
            text=(
                "Digoxin toxicity risk is increased by concomitant use of drugs that "
                "raise digoxin plasma concentration through P-glycoprotein inhibition, "
                "including amiodarone, verapamil, and certain macrolide antibiotics. "
                "Electrolyte disturbances, especially hypokalemia and hypomagnesemia, "
                "sensitize the myocardium to digoxin's arrhythmogenic effects even at "
                "therapeutic serum concentrations. Renal impairment reduces digoxin "
                "clearance and necessitates dose reduction and more frequent serum "
                "concentration monitoring. In cases of suspected toxicity with "
                "life-threatening arrhythmia, digoxin immune Fab (antibody fragments) "
                "may be used for reversal. Patients on chronic therapy should have "
                "periodic assessment of renal function, serum electrolytes, and, when "
                "clinically indicated, serum digoxin concentration drawn at least six "
                "hours after the last dose to avoid overestimating steady-state levels "
                "due to the distribution phase."
            ),
        ),
        RawDocument(
            doc_id="digoxin_pubmed_01",
            drug="Digoxin",
            source_type="pubmed_abstract",
            title="Renal function and digoxin exposure: a pharmacokinetic synthesis",
            citation="Synthetic literature abstract (digoxin renal clearance and exposure)",
            text=(
                "Background: Digoxin is predominantly cleared unchanged by the kidneys, "
                "making renal function a primary determinant of steady-state exposure. "
                "Methods: This synthesis reviews population pharmacokinetic evidence "
                "describing the relationship between creatinine clearance and digoxin "
                "clearance across adult populations, including older adults with "
                "age-related renal decline. Results: Digoxin clearance correlates "
                "closely with creatinine clearance, and models incorporating "
                "estimated glomerular filtration rate substantially improve prediction "
                "of steady-state trough concentration compared with fixed-dose "
                "approaches. Patients with reduced renal function who are not "
                "dose-adjusted are disproportionately represented among cases of "
                "digoxin toxicity, particularly when combined with diuretic-induced "
                "hypokalemia. Conclusion: Renal-function-aware dosing, ideally "
                "supported by therapeutic drug monitoring, meaningfully reduces the "
                "predicted incidence of supratherapeutic digoxin exposure in simulated "
                "patient cohorts."
            ),
        ),
        # --------------------------------------------------------- Vancomycin
        RawDocument(
            doc_id="vancomycin_label_01",
            drug="Vancomycin",
            source_type="fda_label",
            title="Vancomycin Hydrochloride - Prescribing Information (Label Summary)",
            citation="FDA Prescribing Information, Vancomycin Hydrochloride for Injection (label summary)",
            text=(
                "Vancomycin is a glycopeptide antibacterial indicated for the treatment "
                "of serious infections caused by susceptible strains of "
                "methicillin-resistant staphylococci and other gram-positive organisms, "
                "including complicated skin and skin structure infections, bacteremia, "
                "endocarditis, and infections of bone. Vancomycin is associated with "
                "dose-dependent nephrotoxicity, and the risk is increased with "
                "prolonged therapy, concurrent use of other nephrotoxic agents (such as "
                "aminoglycosides), pre-existing renal impairment, and higher trough or "
                "AUC exposure. Ototoxicity has also been reported. Vancomycin dosing "
                "should be individualized based on renal function, and therapeutic drug "
                "monitoring using AUC-guided dosing (targeting an AUC24/MIC ratio of "
                "approximately 400 to 600 for serious MRSA infections, assuming an MIC "
                "of 1 mcg/mL) is recommended over trough-only monitoring where "
                "feasible, per current consensus guidance. Rapid intravenous infusion "
                "can cause 'vancomycin flushing syndrome' (previously termed 'red man "
                "syndrome'), a histamine-mediated reaction distinct from true allergy; "
                "infusions should generally be administered over at least 60 minutes."
            ),
        ),
        RawDocument(
            doc_id="vancomycin_dailymed_01",
            drug="Vancomycin",
            source_type="dailymed",
            title="Vancomycin - Monograph Excerpt (Monitoring and Renal Dosing)",
            citation="DailyMed-style monograph excerpt, Vancomycin Hydrochloride",
            text=(
                "Vancomycin serum concentration monitoring is recommended for patients "
                "receiving therapy beyond a few doses, those with unstable or declining "
                "renal function, and those receiving concomitant nephrotoxic agents. "
                "AUC-guided monitoring is preferred where available; where trough-only "
                "monitoring is used, target troughs of 15 to 20 mcg/mL are commonly "
                "applied for serious infections, though this approach is associated "
                "with higher nephrotoxicity risk than AUC-guided dosing for a "
                "comparable efficacy target. Renal function should be reassessed "
                "periodically during therapy, with more frequent monitoring in "
                "patients with fluctuating creatinine, since vancomycin clearance is "
                "primarily renal and dose or interval adjustment is required as renal "
                "function changes."
            ),
        ),
        RawDocument(
            doc_id="vancomycin_pubmed_01",
            drug="Vancomycin",
            source_type="pubmed_abstract",
            title="AUC-guided versus trough-guided vancomycin dosing: a synthesis",
            citation="Synthetic literature abstract (AUC-guided vancomycin dosing and nephrotoxicity)",
            text=(
                "Background: Trough-only vancomycin monitoring has historically been "
                "used to guide dosing, but has been associated with higher rates of "
                "acute kidney injury than exposure metrics based on the area under the "
                "concentration-time curve (AUC). Methods: This synthesis summarizes "
                "comparative evidence on AUC-guided versus trough-guided vancomycin "
                "dosing strategies in adult inpatients treated for suspected or "
                "confirmed MRSA infection. Results: AUC-guided dosing achieving an "
                "AUC24/MIC target in the range of 400 to 600 was associated with lower "
                "incidence of nephrotoxicity relative to trough-guided strategies "
                "targeting 15 to 20 mcg/mL, without a demonstrated reduction in "
                "clinical efficacy. Higher cumulative AUC exposure and longer duration "
                "of therapy were independently associated with increased nephrotoxicity "
                "risk in multivariable models. Conclusion: Simulation frameworks "
                "estimating vancomycin-related organ toxicity risk should model AUC "
                "exposure directly rather than relying solely on trough concentration "
                "as a toxicity proxy."
            ),
        ),
    ]
    return docs


# --------------------------------------------------------------------------
# Corpus loading (seed + optional external files)
# --------------------------------------------------------------------------

def load_documents(corpus_dir: str | os.PathLike[str] | None = None) -> list[RawDocument]:
    """Load the raw document corpus.

    Always includes the built-in synthetic seed corpus. If ``corpus_dir``
    is provided (or the ``CORPUS_DIR`` environment variable is set) and
    the directory exists, any additional ``*.json`` files found there are
    also loaded. Each external JSON file must be either a single object
    or a list of objects with the keys: ``doc_id``, ``drug``,
    ``source_type``, ``title``, ``text``, ``citation``.

    Args:
        corpus_dir: Optional path to a directory of additional raw
            documents in JSON form.

    Returns:
        A list of :class:`RawDocument` instances, seed corpus first.

    Raises:
        ValueError: If an external JSON document is missing required
            fields or has an invalid ``source_type``.
    """
    docs = _seed_documents()

    resolved_dir = corpus_dir or os.environ.get("CORPUS_DIR")
    if not resolved_dir:
        return docs

    path = Path(resolved_dir)
    if not path.exists():
        logger.warning("corpus_dir '%s' does not exist; using seed corpus only.", path)
        return docs

    valid_sources = {"fda_label", "dailymed", "pubmed_abstract"}
    for json_file in sorted(path.glob("*.json")):
        try:
            raw = json.loads(json_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable corpus file %s: %s", json_file, exc)
            continue

        entries = raw if isinstance(raw, list) else [raw]
        for entry in entries:
            required = {"doc_id", "drug", "source_type", "title", "text", "citation"}
            missing = required - entry.keys()
            if missing:
                raise ValueError(
                    f"Document in {json_file} missing required fields: {sorted(missing)}"
                )
            if entry["source_type"] not in valid_sources:
                raise ValueError(
                    f"Document {entry['doc_id']} in {json_file} has invalid "
                    f"source_type '{entry['source_type']}'; must be one of {valid_sources}"
                )
            docs.append(RawDocument(**entry))

    logger.info("Loaded %d total raw documents (including seed corpus).", len(docs))
    return docs


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------

def _word_windows(
    words: list[str], chunk_size: int, overlap: int
) -> Iterator[tuple[int, int]]:
    """Yield (start, end) index windows over ``words`` with overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    step = chunk_size - overlap
    n = len(words)
    if n == 0:
        return
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        yield start, end
        if end == n:
            break
        start += step


def chunk_document(
    doc: RawDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split a single :class:`RawDocument` into overlapping word-window chunks.

    Args:
        doc: The source document to chunk.
        chunk_size: Target number of words per chunk.
        overlap: Number of words shared between consecutive chunks.

    Returns:
        A list of :class:`Chunk` objects derived from ``doc``.
    """
    words = doc.text.split()
    chunks: list[Chunk] = []
    for idx, (start, end) in enumerate(_word_windows(words, chunk_size, overlap)):
        chunk_text = " ".join(words[start:end]).strip()
        if not chunk_text:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::chunk{idx}",
                doc_id=doc.doc_id,
                drug=doc.drug,
                source_type=doc.source_type,
                title=doc.title,
                citation=doc.citation,
                chunk_index=idx,
                text=chunk_text,
            )
        )
    return chunks


def chunk_corpus(
    docs: Iterable[RawDocument],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk an entire corpus of documents.

    Args:
        docs: Iterable of raw documents.
        chunk_size: Target number of words per chunk.
        overlap: Number of overlapping words between chunks.

    Returns:
        A flat list of all chunks across all documents.
    """
    docs = list(docs)
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
    logger.info("Chunked %d documents into %d passages.", len(docs), len(all_chunks))
    return all_chunks


# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------

class Embedder:
    """Thin wrapper around a local sentence-transformers embedding model.

    Loading the model is deferred until first use so importing this
    module (e.g. from ``generate_report.py``) does not require the model
    weights to be present unless embedding/indexing is actually invoked.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for embedding. Install it "
                    "with `pip install sentence-transformers`."
                ) from exc
            logger.info("Loading embedding model '%s' ...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(
        self, texts: list[str], batch_size: int = DEFAULT_EMBED_BATCH_SIZE, is_query: bool = False
    ) -> list[list[float]]:
        """Embed a list of texts.

        For BGE-family models, retrieval quality is improved by prefixing
        *queries* (not passages) with an instruction string. This is
        handled automatically when ``is_query=True``.

        Args:
            texts: Texts to embed.
            batch_size: Batch size for encoding.
            is_query: Whether ``texts`` are search queries (as opposed to
                corpus passages), which affects BGE-style prefixing.

        Returns:
            A list of embedding vectors (as plain Python lists of floats),
            one per input text, in the same order.

        Raises:
            ValueError: If ``texts`` is empty.
        """
        if not texts:
            raise ValueError("texts must be a non-empty list")

        model = self._load()
        prefixed = texts
        if is_query and "bge" in self.model_name.lower():
            prefixed = [
                f"Represent this sentence for searching relevant passages: {t}" for t in texts
            ]

        embeddings = model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [vec.tolist() if hasattr(vec, "tolist") else list(vec) for vec in embeddings]


# --------------------------------------------------------------------------
# Chroma indexing
# --------------------------------------------------------------------------

def get_chroma_collection(
    persist_dir: str | os.PathLike[str] = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> Any:
    """Open (creating if necessary) a persistent Chroma collection.

    Args:
        persist_dir: Directory on disk backing the persistent Chroma
            client.
        collection_name: Name of the collection to open/create.

    Returns:
        A ``chromadb.api.models.Collection.Collection`` instance.

    Raises:
        ImportError: If ``chromadb`` is not installed.
    """
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError(
            "chromadb is required for vector storage. Install it with `pip install chromadb`."
        ) from exc

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_path))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def index_chunks(
    chunks: list[Chunk],
    embedder: Embedder,
    persist_dir: str | os.PathLike[str] = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embed_batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
    reset: bool = False,
) -> int:
    """Embed and upsert chunks into the persistent Chroma collection.

    Args:
        chunks: Chunks to index.
        embedder: An :class:`Embedder` used to compute passage vectors.
        persist_dir: Directory backing the persistent Chroma client.
        collection_name: Target collection name.
        embed_batch_size: Batch size used for embedding calls.
        reset: If True, delete and recreate the collection before
            indexing (useful for a clean rebuild).

    Returns:
        The number of chunks successfully indexed.

    Raises:
        ValueError: If ``chunks`` is empty.
    """
    if not chunks:
        raise ValueError("chunks must be a non-empty list")

    try:
        import chromadb
    except ImportError as exc:
        raise ImportError(
            "chromadb is required for vector storage. Install it with `pip install chromadb`."
        ) from exc

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(persist_path))

    if reset:
        try:
            client.delete_collection(collection_name)
            logger.info("Deleted existing collection '%s' for reset.", collection_name)
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass

    collection = client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    total_indexed = 0
    for i in range(0, len(chunks), embed_batch_size):
        batch = chunks[i : i + embed_batch_size]
        texts = [c.text for c in batch]
        try:
            vectors = embedder.embed(texts, batch_size=embed_batch_size, is_query=False)
        except Exception as exc:  # noqa: BLE001
            logger.error("Embedding failed for batch starting at index %d: %s", i, exc)
            raise

        ids = [c.chunk_id for c in batch]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "drug": c.drug,
                "source_type": c.source_type,
                "title": c.title,
                "citation": c.citation,
                "chunk_index": c.chunk_index,
            }
            for c in batch
        ]
        try:
            collection.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
        except Exception as exc:  # noqa: BLE001
            logger.error("Chroma upsert failed for batch starting at index %d: %s", i, exc)
            raise
        total_indexed += len(batch)
        logger.info("Indexed %d/%d chunks.", total_indexed, len(chunks))

    return total_indexed


# --------------------------------------------------------------------------
# Top-level orchestration
# --------------------------------------------------------------------------

def build_corpus(
    corpus_dir: str | os.PathLike[str] | None = None,
    persist_dir: str | os.PathLike[str] = DEFAULT_CHROMA_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    reset: bool = False,
) -> dict[str, Any]:
    """End-to-end corpus build: load -> chunk -> embed -> index.

    Args:
        corpus_dir: Optional directory of additional external documents.
        persist_dir: Directory backing the persistent Chroma database.
        collection_name: Name of the Chroma collection to populate.
        embedding_model: HuggingFace model id for sentence-transformers.
        chunk_size: Target words per chunk.
        overlap: Overlap words between adjacent chunks.
        reset: Whether to wipe the collection before indexing.

    Returns:
        A summary dict with document/chunk counts and index location.
    """
    logger.info("Building AetherMed RAG corpus...")
    docs = load_documents(corpus_dir=corpus_dir)
    logger.info("Loaded %d raw documents.", len(docs))

    chunks = []
    for doc in docs:
        chunks.extend(chunk_document(doc, chunk_size=chunk_size, overlap=overlap))
    logger.info("Produced %d chunks from %d documents.", len(chunks), len(docs))

    embedder = Embedder(model_name=embedding_model)
    n_indexed = index_chunks(
        chunks,
        embedder=embedder,
        persist_dir=persist_dir,
        collection_name=collection_name,
        reset=reset,
    )

    summary = {
        "num_documents": len(docs),
        "num_chunks": len(chunks),
        "num_indexed": n_indexed,
        "persist_dir": str(Path(persist_dir).resolve()),
        "collection_name": collection_name,
        "embedding_model": embedding_model,
    }
    logger.info("Corpus build complete: %s", json.dumps(summary, indent=2))
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and index the AetherMed clinical RAG corpus."
    )
    parser.add_argument("--corpus-dir", type=str, default=None, help="Optional external corpus dir")
    parser.add_argument("--persist-dir", type=str, default=str(DEFAULT_CHROMA_DIR))
    parser.add_argument("--collection-name", type=str, default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--reset", action="store_true", help="Wipe the collection before indexing")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code."""
    args = _parse_args(argv)
    try:
        build_corpus(
            corpus_dir=args.corpus_dir,
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            reset=args.reset,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Corpus build failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())