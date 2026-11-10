# Architecture

```
 ┌───────────────┐     ┌────────────────────┐     ┌──────────────────────┐
 │  Phase 1-2     │     │  Phase 3            │     │  Phase 4              │
 │  Data pipeline │────▶│  Physics-informed   │────▶│  Heterogeneous GNN    │
 │  + CTGAN        │     │  PK/PD Neural ODE    │     │  (7 node types,       │
 │  synthetic       │     │  (real compartmental │     │  HeteroConv,          │
 │  cohort           │     │  ODE + bounded        │     │  attention exposed)   │
 └───────────────┘     │  learned correction)   │     └──────────┬────────────┘
                          └────────────────────┘                │
                                                                   ▼
 ┌───────────────┐     ┌────────────────────┐     ┌──────────────────────┐
 │  Phase 6        │     │  Phase 7             │     │  Phase 5               │
 │  RAG (Chroma +   │◀────│  Conformal intervals  │◀────│  Offline RL (CQL)      │
 │  bge-small +      │     │  + Mahalanobis OOD     │     │  dosing policy,         │
 │  MLX local LLM,    │     │  detector               │     │  evaluated via FQE/WIS  │
 │  citation-forced)   │     └────────────────────┘     │  against behavior policy │
 └────────┬──────────┘                                    └──────────┬──────────────┘
            │                                                          │
            └──────────────────────────┬───────────────────────────────┘
                                          ▼
                             ┌───────────────────────┐
                             │  Phase 8                │
                             │  Streamlit dashboard      │
                             │  (cohort, PK curves,        │
                             │  organ risk, RL rec +         │
                             │  confidence, RAG citations,     │
                             │  conformal intervals + OOD)       │
                             └───────────────────────┘
```

## Component evaluation summary

| Component | Metric | Status |
|---|---|---|
| Data pipeline (P1) | Schema validation pass rate | Logic verified; real data pull pending network run |
| CTGAN synthetic generator (P2) | KS statistic, correlation similarity | Report generator written; not yet run on real cohort |
| PK/PD Neural ODE (P3) | Half-life/clearance RMSE vs. literature | k10=CL/V identity verified; secondary rate constants (k12/k21/ke0) are placeholders pending literature lookup |
| Heterogeneous GNN (P4) | Attention-weight explainability | Architecture + edge justifications written; untrained |
| Offline RL (P5) | FQE/WIS estimate vs. behavior policy | Framework written; blocked on trained P3/P4 outputs to generate real trajectories |
| RAG (P6) | Retrieval precision@k, citation-coverage rate | Eval harness written; QA set has 3/20-30 required items (template only) |
| Uncertainty (P7) | Empirical coverage vs. nominal (e.g. 90%) | Calibration check implemented; not yet run against a trained regressor |
| Dashboard (P8) | N/A (integration surface) | Scaffolded with explicit "not wired yet" placeholders per tab |

## Why offline RL, not online RL

See the docstring at the top of `src/rl_policy/train_cql.py` for the full
reasoning — in short: CQL trains only from logged behavior-policy
trajectories and explicitly penalizes out-of-distribution actions, so the
learned policy is never the one selecting doses and observing
consequences during training. That's the property worth defending in an
interview, not just the fact that "CQL" appears in the tech stack.

## Limitations (be ready to state these unprompted)

- MIMIC-IV Demo is 100 patients — not representative; Synthea supplements
  volume but is itself synthetic-generator-biased.
- PK secondary rate constants (k12, k21, ke0) are placeholders in this
  scaffold, not literature-sourced yet — Phase 3 isn't validated until
  they are.
- RAG QA eval set needs to grow from 3 illustrative items to the
  specified 20-30 hand-verified items before citation-coverage numbers
  mean anything.
- Offline RL reward shape (`compute_reward` in `dataset.py`) is a
  placeholder distance-to-target function, not derived from a clinician's
  actual utility function — flagged explicitly in that file's docstring.
- No component has been run end-to-end against real pulled data yet in
  this scaffold; see `docs/EXPERIMENTS.md` for the concrete per-phase
  checklist of what's left before each phase's numbers are trustworthy.

## Next steps

- Apply for credentialed full MIMIC-IV access to replace/supplement the
  100-patient demo once the pipeline is validated end-to-end on the demo.
- Replace PK placeholder rate constants with literature-sourced values
  per drug, with citations recorded in `EXPERIMENTS.md`.
- Expand the RAG QA eval set to the full 20-30 items.
- Run a formal bootstrap confidence interval for the OPE delta in
  `evaluate_ope.compare_to_behavior_policy`, rather than the current rough
  1-std heuristic.
