"""
Central config for Phase 1 data sources.

All sources here are public / no-credential. If PhysioNet ever gates the
MIMIC-IV Demo further, or an endpoint changes, update it here rather than
scattering URLs through the pipeline scripts.
"""
from pathlib import Path

# --- Paths -------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

MIMIC_DEMO_DIR = RAW_DIR / "mimic_iv_demo"
SYNTHEA_DIR = RAW_DIR / "synthea"
CHEMBL_DIR = RAW_DIR / "chembl"

# --- MIMIC-IV Demo (PhysioNet, open access, 100 patients) --------------
# Official open-access demo dataset. No credentialing required, unlike full MIMIC-IV.
MIMIC_DEMO_BASE_URL = "https://physionet.org/files/mimic-iv-demo/2.2/"
MIMIC_DEMO_FILES = [
    "hosp/patients.csv.gz",
    "hosp/admissions.csv.gz",
    "hosp/labevents.csv.gz",
    "hosp/d_labitems.csv.gz",
    "hosp/diagnoses_icd.csv.gz",
    "hosp/d_icd_diagnoses.csv.gz",
    "hosp/prescriptions.csv.gz",
]

# --- Synthea (synthetic longitudinal EHR, github release) --------------
SYNTHEA_RELEASE_URL = (
    "https://github.com/synthetichealth/synthea-sample-data/raw/master/"
    "downloads/synthea_sample_data_csv_apr2020.zip"
)

# --- ChEMBL REST API (free, no key) -------------------------------------
CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"

# Well-characterized drugs used later in Phase 3 PK validation.
# ChEMBL IDs looked up once and pinned here so re-runs are reproducible.
CHEMBL_TARGET_DRUGS = {
    "warfarin": "CHEMBL1464",
    "digoxin": "CHEMBL1160",
    "vancomycin": "CHEMBL262777",
}

# --- Sanity-check bounds used by schema validation (Phase 1 tests) -----
PLAUSIBLE_RANGES = {
    "age_years": (0, 110),
    "age": (0, 110),
    "creatinine_mg_dl": (0.1, 20.0),
    "serum_creatinine": (0.1, 20.0),
    "weight_kg": (20.0, 250.0),
    "weight": (20.0, 250.0),
    "gfr": (1.0, 200.0),
    "egfr": (1.0, 200.0),
    "egfr_ml_min": (1.0, 200.0),
    "alt": (2.0, 1000.0),
    "ast": (2.0, 1000.0),
    "potassium_mmol_l": (1.5, 9.0),
    "serum_potassium": (1.5, 9.0),
    "sodium_mmol_l": (100, 180),
    "albumin": (1.0, 7.0),
    "heart_rate_bpm": (20, 250),
    "systolic_bp_mmhg": (40, 260),
}
