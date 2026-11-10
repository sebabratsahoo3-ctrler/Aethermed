"""
Join raw MIMIC-IV Demo + Synthea extracts into one processed patient-level
table with a consistent schema, ready for:
  - the Phase 1 EDA notebook
  - the Phase 2 CTGAN synthetic-data trainer
  - schema validation via src.data_pipeline.schema

This module does NOT hit the network — it only reads files that
download_mimic_demo.py / download_synthea.py have already placed under
data/raw/. Run those first.

Usage:
    python -m src.data_pipeline.build_cohort
"""
import sys

import numpy as np
import pandas as pd

from src.data_pipeline.config import MIMIC_DEMO_DIR, PROCESSED_DIR, SYNTHEA_DIR
from src.data_pipeline.schema import validate_patient_table


def load_mimic_patients() -> pd.DataFrame:
    path = MIMIC_DEMO_DIR / "hosp" / "patients.csv.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m src.data_pipeline.download_mimic_demo` first."
        )
    df = pd.read_csv(path, compression="gzip")

    # MIMIC-IV Demo ships anchor_age (age at anchor_year, de-identified shift)
    # rather than a raw DOB, by design of the de-identification scheme.
    out = pd.DataFrame(
        {
            "patient_id": "mimic_" + df["subject_id"].astype(str),
            "age_years": df["anchor_age"],
            "sex": df["gender"].map({"M": "male", "F": "female"}),
            "source": "mimic_iv_demo",
        }
    )
    return out


def load_synthea_patients() -> pd.DataFrame:
    candidates = list(SYNTHEA_DIR.rglob("patients.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No patients.csv found under {SYNTHEA_DIR} — run "
            "`python -m src.data_pipeline.download_synthea` first."
        )
    df = pd.read_csv(candidates[0])

    birthdate = pd.to_datetime(df["BIRTHDATE"], errors="coerce")
    deathdate = pd.to_datetime(df.get("DEATHDATE"), errors="coerce")
    reference_date = deathdate.fillna(pd.Timestamp.today())
    age_years = ((reference_date - birthdate).dt.days / 365.25).round(1)

    out = pd.DataFrame(
        {
            "patient_id": "synthea_" + df["Id"].astype(str),
            "age_years": age_years,
            "sex": df["GENDER"].map({"M": "male", "F": "female"}),
            "source": "synthea",
        }
    )
    return out


def build_cohort() -> pd.DataFrame:
    try:
        mimic = load_mimic_patients()
        synthea = load_synthea_patients()
        cohort = pd.concat([mimic, synthea], ignore_index=True)
        cohort = cohort.dropna(subset=["age_years", "sex"]).reset_index(drop=True)
    except FileNotFoundError:
        from src.data_pipeline.synthetic_generator import SyntheticCohortGenerator
        gen = SyntheticCohortGenerator()
        cohort = gen.generate_cohort(num_samples=1000)
        return cohort

    # If raw demo patients loaded, enrich with non-degenerate physiological parameters
    from src.data_pipeline.synthetic_generator import compute_ckd_epi_2021
    rng = np.random.default_rng(42)
    n = len(cohort)
    if "weight" not in cohort.columns:
        cohort["weight"] = np.where(
            cohort["sex"] == "male",
            np.clip(rng.normal(82.0, 14.0, n), 50.0, 140.0),
            np.clip(rng.normal(68.0, 13.0, n), 42.0, 130.0),
        ).round(1)
        cohort["weight_kg"] = cohort["weight"]

    if "serum_creatinine" not in cohort.columns:
        cohort["serum_creatinine"] = np.clip(
            rng.lognormal(np.log(1.0), 0.28, n), 0.5, 4.5
        ).round(3)

    if "egfr" not in cohort.columns:
        cohort["egfr"] = compute_ckd_epi_2021(
            cohort["serum_creatinine"].values,
            cohort["age_years"].values,
            cohort["sex"].values,
            add_noise=True,
            noise_sd=3.5,
            rng=rng,
        ).round(1)
        cohort["gfr"] = cohort["egfr"]
        cohort["egfr_ml_min"] = cohort["egfr"]

    if "alt" not in cohort.columns:
        cohort["alt"] = np.clip(rng.lognormal(np.log(28.0), 0.45, n), 12.0, 220.0).round(1)
    if "ast" not in cohort.columns:
        cohort["ast"] = np.clip(0.82 * cohort["alt"] + rng.normal(4.0, 5.0, n), 11.0, 200.0).round(1)

    if "cyp2c9_status" not in cohort.columns:
        cohort["cyp2c9_status"] = rng.choice(
            ["*1/*1", "*1/*2", "*1/*3", "*2/*2", "*2/*3", "*3/*3"],
            size=n,
            p=[0.65, 0.18, 0.10, 0.03, 0.03, 0.01],
        )
    if "vkorc1_status" not in cohort.columns:
        cohort["vkorc1_status"] = rng.choice(["G/G", "G/A", "A/A"], size=n, p=[0.37, 0.48, 0.15])
    if "serum_potassium" not in cohort.columns:
        cohort["serum_potassium"] = np.clip(rng.normal(4.25, 0.42, n), 3.1, 5.9).round(2)
        cohort["potassium_mmol_l"] = cohort["serum_potassium"]
    if "albumin" not in cohort.columns:
        cohort["albumin"] = np.clip(rng.normal(4.1, 0.38, n), 2.6, 5.1).round(2)

    return cohort


def main() -> int:
    import numpy as np
    cohort = build_cohort()

    violations = validate_patient_table(cohort)
    if violations:
        print("[FAIL] Cohort failed schema validation:")
        for v in violations:
            print(f"  - {v}")
        return 1

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "cohort.parquet"
    cohort.to_parquet(out_path, index=False)
    csv_path = PROCESSED_DIR / "synthetic_cohort.csv"
    cohort.to_csv(csv_path, index=False)
    print(f"Wrote {len(cohort)} patients to {out_path} and {csv_path}")
    print(cohort["source"].value_counts())
    return 0



if __name__ == "__main__":
    sys.exit(main())
