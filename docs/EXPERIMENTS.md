# Experiments Log

Running log of what was tried, what worked, and what didn't, per phase.
Update this file as you go — it's meant to make the git history and this
log tell the real story of the build, including dead ends.

## Phase 1 — Data Pipeline

**Date:**
**What was done:**
- Repo scaffold created.
- Download scripts written for MIMIC-IV Demo, Synthea, ChEMBL.
- Schema validation module (`src/data_pipeline/schema.py`) written and
  manually verified (see note below — pytest wasn't runnable in the
  authoring environment; verify with `pytest tests/ -v` in your own
  Claude Code / terminal session with the venv installed).
- Cohort builder (`src/data_pipeline/build_cohort.py`) joins MIMIC-IV Demo
  + Synthea patient tables into `data/processed/cohort.parquet`.

**Known gaps / to do before calling Phase 1 complete:**
- [ ] Actually run `download_mimic_demo.py`, `download_synthea.py`,
      `fetch_chembl.py` against the real network and confirm success.
- [ ] Run `pytest tests/ -v` for real and paste results here.
- [ ] Write the EDA notebook (`notebooks/01_eda.ipynb`) summarizing age,
      lab-value, and comorbidity distributions once real data is pulled.
- [ ] Confirm ChEMBL molecule IDs pinned in `config.py` are still valid
      (ChEMBL occasionally deprecates/merges IDs).

**Metrics / results:** none yet — schema validation logic confirmed
correct against synthetic in-test fixtures only, not real pulled data.

---

## Phase 2 — Synthetic Patient Generator

**What was done:** `synthetic_generator.py` written (CTGAN via sdv,
KS-statistic + correlation-similarity fidelity report to
`docs/SYNTHETIC_FIDELITY_REPORT.md`).
**Known gaps:** Not run against real data — needs Phase 1's real
`cohort.parquet` first. Only `age_years` is a numeric column right now,
so correlation-similarity is not yet meaningful; extend the cohort table
with lab values before trusting that metric.

## Phase 3 — Physics-Informed PK/PD Engine

**What was done:** Real 3-compartment ODE (`compartmental_ode.py`),
bounded learned correction term (`neural_correction.py`), validation
script against warfarin/digoxin/vancomycin (`train.py`), tests confirming
correction_scale=0 reduces exactly to pure physics and the correction is
bounded (`tests/test_pk_model.py`).
**Known gaps:** k12/k21/ke0 are placeholder values, not literature-sourced
— flagged explicitly in `train.py`. Don't cite half-life recovery numbers
as validated until these are replaced.

## Phase 4 — Multi-Agent Physiological Graph

**What was done:** 7 node types + physiologically-justified edges
(`hetero_graph.py`, each edge has a docstring justification),
GATConv-based HeteroConv model with attention-weight extraction
(`model.py`).
**Known gaps:** Untrained — no cohort has been run through it yet.
Attention-weight surfacing is implemented but unverified against real
data.

## Phase 5 — Offline RL Treatment Optimizer

**What was done:** Behavior-policy trajectory logging
(`dataset.py`), CQL training wrapper (`train_cql.py`), FQE + weighted
importance sampling OPE (`evaluate_ope.py`).
**Known gaps:** Blocked on trained Phase 3/4 outputs to generate real
trajectories — `train_cql.py`'s `main()` explains the dependency instead
of faking a run. Reward shape in `dataset.py` is a placeholder distance
function, not derived from real clinical utility.

## Phase 6 — RAG Clinical Reasoning Layer

**What was done:** Corpus chunking/embedding into Chroma
(`build_corpus.py`), MLX-backed generation with mechanical citation
validation (`generate_report.py`), retrieval precision@k + citation-
coverage eval harness (`eval_rag.py`).
**Known gaps:** `QA_EVAL_SET` has 3 illustrative items; spec calls for
20-30 hand-verified against the real corpus. No DailyMed/PubMed fetch
script exists yet (only ChEMBL, from Phase 1) — corpus dir currently
empty.

## Phase 7 — Uncertainty & Safety Layer

**What was done:** mapie-based conformal wrapper (`conformal.py`),
Mahalanobis OOD detector (`ood_detector.py`), empirical coverage
validator across multiple alphas (`calibration_eval.py`).
**Known gaps:** Not run against a trained regressor yet — no toxicity
model exists to wrap.

## Phase 8 — Dashboard & Integration

**What was done:** Streamlit app scaffold with all 6 tabs
(`src/dashboard/app.py`) — cohort tab is fully wired to real Phase 1
output; PK/graph/RL/RAG/uncertainty tabs show explicit "not wired yet"
messages rather than fabricated placeholder charts presented as results.
**Known gaps:** Everything past the cohort tab needs its upstream
phase's trained artifacts before it can render real output.

## Phase 9 — Writeup

**What was done:** `docs/ARCHITECTURE.md` (ASCII diagram, per-component
eval-status table, offline-RL rationale, limitations, next steps).
**Known gaps:** The eval-status table needs updating with real numbers
as each phase above actually gets run against real data.
