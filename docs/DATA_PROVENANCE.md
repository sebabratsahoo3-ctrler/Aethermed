# Data Provenance & Licensing

All data sources used in AetherMed are public and require no credentialing
beyond a free account (where noted). Full credentialed MIMIC-IV is
explicitly **not** used.

## MIMIC-IV Clinical Database Demo (v2.2)
- Source: PhysioNet, https://physionet.org/content/mimic-iv-demo/2.2/
- Access: Open access, no DUA/credentialing required (this is the express
  purpose of the demo release — 100 de-identified patients, a subset of
  full MIMIC-IV).
- License: PhysioNet Credentialed Health Data License is NOT required for
  the demo; it's released under the Open Data Commons Open Database
  License (ODbL) v1.0. Verify current terms on the PhysioNet page before
  redistribution.
- Fields used: `hosp/patients`, `hosp/admissions`, `hosp/labevents`,
  `hosp/diagnoses_icd`, `hosp/prescriptions`.
- Caveat: 100 patients is a small, non-representative sample — fine for
  pipeline development and schema validation, not for claims of
  generalizable clinical findings. This is the reason Phase 9's "next
  steps" section should discuss applying for full credentialed access.

## Synthea Synthetic Patient Data
- Source: https://github.com/synthetichealth/synthea (sample CSV release)
- Access: Fully open, Apache 2.0 license.
- Nature: 100% synthetic — no real patient data, no PHI, no consent
  concerns. Used to supplement MIMIC-IV Demo's small cohort size with a
  larger, longitudinal synthetic population for training the CTGAN
  (Phase 2) and graph model (Phase 4).
- Caveat: Synthea's generative model has its own biases/artifacts; treat
  it as a volume/diversity supplement, not a ground-truth clinical source.

## ChEMBL
- Source: https://www.ebi.ac.uk/chembl/ REST API
- Access: Free, no API key required.
- License: ChEMBL data is available under a Creative Commons
  Attribution-ShareAlike 3.0 Unported license.
- Used for: drug molecule metadata and mechanism-of-action data, feeding
  the Phase 3 PK/PD validation drugs (warfarin, digoxin, vancomycin).

## DrugBank Open Data
- Source: https://go.drugbank.com/releases/latest#open-data
- Access: Free open-data subset (structure/name/ID mappings), no
  credentialing for the open portion. The full commercial dataset is
  separate and NOT used here.
- Used for: cross-referencing drug identity between ChEMBL and label text.

## DailyMed / FDA Structured Product Labels
- Source: https://dailymed.nlm.nih.gov/, NLM/FDA, fully open.
- License: US Government work, public domain.
- Used for: Phase 6 RAG corpus (drug label text).

## PubMed Open-Access Abstracts (NCBI E-utilities)
- Source: https://www.ncbi.nlm.nih.gov/books/NBK25501/ (E-utilities API)
- Access: Free, rate-limited without an API key, higher rate limit with a
  free NCBI account API key.
- Used for: Phase 6 RAG corpus (literature abstracts). Only
  open-access-flagged abstracts are pulled; full-text is not scraped.

## What is explicitly NOT used
- Credentialed full MIMIC-IV (requires PhysioNet CITI training + DUA).
- Any DrugBank field outside the open-data license tier.
- Any dataset requiring institutional/IRB approval this project doesn't have.

If a download script in `src/data_pipeline/` fails due to an access
change, the correct response is to update this doc and stop — not to
substitute fabricated values.
