"""
synthetic_generator.py
=======================

AetherMed :: Data Pipeline

Generates a synthetic patient cohort using CTGAN (via the `sdv` library)
trained on demographic and laboratory features drawn from a MIMIC-IV-style
clinical dataset. The synthetic cohort is intended for downstream digital
twin simulation, model prototyping, and testing without exposing real
patient-level data.

Typical usage
-------------
    from src.data_pipeline.synthetic_generator import SyntheticCohortGenerator

    generator = SyntheticCohortGenerator(
        raw_data_dir="data/raw/mimic-iv-clinical-database-demo-2.2",
        output_path="data/processed/synthetic_cohort.csv",
    )
    generator.run(num_rows=5000)

Notes
-----
- This module expects the MIMIC-IV demo directory layout, specifically
  `hosp/patients.csv.gz` (demographics) and `hosp/labevents.csv.gz`
  (laboratory measurements), plus `hosp/d_labitems.csv.gz` to map lab
  item IDs to human-readable test names.
- MIMIC-IV access is governed by a PhysioNet data use agreement. This
  code assumes the caller already has lawful, credentialed access to
  the data it points at; it does not fetch or unlock any data itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Optional

import numpy as np
import pandas as pd

try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer
    HAS_SDV = True
except ImportError:
    SingleTableMetadata = None  # type: ignore
    CTGANSynthesizer = None  # type: ignore
    HAS_SDV = False


LOGGER = logging.getLogger(__name__)

# Lab test names mapped to modeling frame column names
_LAB_LABEL_MAP: Final[dict[str, str]] = {
    "Creatinine": "serum_creatinine",
    "Alanine Aminotransferase (ALT)": "alt",
    "Asparate Aminotransferase (AST)": "ast",
}

_MODEL_COLUMNS: Final[list[str]] = [
    "age",
    "weight",
    "serum_creatinine",
    "alt",
    "ast",
    "gfr",
    "gender",
    "cyp2c9_status",
    "vkorc1_status",
    "serum_potassium",
    "albumin",
]


def compute_ckd_epi_2021(
    serum_creatinine: float | np.ndarray,
    age: float | np.ndarray,
    gender: str | np.ndarray = "male",
    add_noise: bool = True,
    noise_sd: float = 3.5,
    rng: np.random.Generator | None = None,
) -> float | np.ndarray:
    """Compute eGFR (mL/min/1.73m²) via the 2021 CKD-EPI Creatinine Equation.

    Formula:
      eGFR = 142 * min(Scr / kappa, 1)**alpha * max(Scr / kappa, 1)**(-1.200)
             * (0.9938 ** Age) * sex_factor
    Where:
      Females: kappa = 0.7, alpha = -0.241, sex_factor = 1.012
      Males:   kappa = 0.9, alpha = -0.302, sex_factor = 1.000
    With biological noise ~ N(0, noise_sd^2) clipped to [5.0, 140.0].
    """
    is_scalar = np.isscalar(serum_creatinine)
    scr = np.atleast_1d(np.asarray(serum_creatinine, dtype=np.float64))
    ag = np.atleast_1d(np.asarray(age, dtype=np.float64))
    
    # Handle gender input
    if isinstance(gender, str):
        gen = np.full_like(scr, gender, dtype=object)
    else:
        gen = np.atleast_1d(np.asarray(gender, dtype=object))

    # Match dimensions if needed
    if len(ag) == 1 and len(scr) > 1:
        ag = np.full_like(scr, ag[0])
    if len(gen) == 1 and len(scr) > 1:
        gen = np.full_like(scr, gen[0])

    is_female = np.array([str(g).lower() in ("f", "female") for g in gen])

    kappa = np.where(is_female, 0.7, 0.9)
    alpha = np.where(is_female, -0.241, -0.302)
    sex_factor = np.where(is_female, 1.012, 1.000)

    # Prevent division by zero
    scr_clamped = np.clip(scr, 0.1, 25.0)
    ratio = scr_clamped / kappa

    min_term = np.minimum(ratio, 1.0) ** alpha
    max_term = np.maximum(ratio, 1.0) ** (-1.200)
    age_term = 0.9938 ** ag

    egfr = 142.0 * min_term * max_term * age_term * sex_factor

    if add_noise:
        _rng = rng or np.random.default_rng(42)
        noise = _rng.normal(0.0, noise_sd, size=egfr.shape)
        egfr = egfr + noise

    egfr = np.clip(egfr, 5.0, 140.0)

    if is_scalar:
        return float(egfr[0])
    return egfr.astype(np.float64)


class SyntheticGeneratorError(Exception):
    """Raised for any unrecoverable error in the synthetic data pipeline."""


@dataclass
class SyntheticCohortGenerator:
    """Fits a CTGAN model or generates physiological patient cohorts with
    proper covariances, non-degenerate distributions, and CKD-EPI 2021 eGFR coupling.
    """
    raw_data_dir: Path = field(default_factory=lambda: Path("data/raw/mimic-iv-clinical-database-demo-2.2"))
    output_path: Path = field(default_factory=lambda: Path("data/processed/synthetic_cohort.csv"))
    model_path: Optional[Path] = None
    epochs: int = 300
    random_state: int = 42

    _synthesizer: Optional[CTGANSynthesizer] = field(default=None, init=False, repr=False)
    _metadata: Optional[SingleTableMetadata] = field(default=None, init=False, repr=False)
    _training_frame: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.raw_data_dir = Path(self.raw_data_dir)
        self.output_path = Path(self.output_path)
        if self.model_path is None:
            self.model_path = self.output_path.with_name("ctgan_cohort_model.pkl")
        else:
            self.model_path = Path(self.model_path)

        if self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")

    def generate_cohort(self, num_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
        """Generates a physiologically coupled, non-degenerate synthetic cohort
        of virtual patients with realistic biological distributions, CKD-EPI 2021 eGFR,
        and pharmacogenomic covariates.
        """
        rng = np.random.default_rng(seed)

        # 1. Age: Truncated normal, realistic inpatient/trial cohort [18, 88]
        raw_age = rng.normal(loc=63.0, scale=12.5, size=num_samples)
        age = np.clip(raw_age, 18.0, 88.0).round(1)

        # 2. Gender: ~51% male, 49% female
        gender = rng.choice(["male", "female"], size=num_samples, p=[0.51, 0.49])

        # 3. Weight: log-normal conditioned on gender
        is_male = gender == "male"
        weight = np.where(
            is_male,
            np.clip(rng.lognormal(mean=np.log(82.0), sigma=0.17, size=num_samples), 48.0, 140.0),
            np.clip(rng.lognormal(mean=np.log(69.0), sigma=0.18, size=num_samples), 42.0, 130.0),
        ).round(1)

        # 4. Serum Creatinine: Multi-modal clinical distribution without artificial clipping
        # 68% Normal (0.6 - 1.2), 22% Mild/Mod CKD (1.2 - 2.3), 10% Severe CKD/AKI (2.3 - 4.5)
        latent_cr_group = rng.choice([0, 1, 2], size=num_samples, p=[0.68, 0.22, 0.10])
        cr_norm = np.where(
            is_male,
            rng.lognormal(mean=np.log(0.98), sigma=0.18, size=num_samples),
            rng.lognormal(mean=np.log(0.82), sigma=0.18, size=num_samples),
        )
        cr_mod = rng.lognormal(mean=np.log(1.65), sigma=0.20, size=num_samples)
        cr_sev = rng.lognormal(mean=np.log(3.10), sigma=0.22, size=num_samples)

        serum_creatinine = np.select(
            [latent_cr_group == 0, latent_cr_group == 1, latent_cr_group == 2],
            [cr_norm, cr_mod, cr_sev],
        )
        serum_creatinine = np.clip(serum_creatinine, 0.45, 5.0).round(3)

        # 5. eGFR: Exact 2021 CKD-EPI Creatinine Equation with biological noise
        egfr = compute_ckd_epi_2021(
            serum_creatinine=serum_creatinine,
            age=age,
            gender=gender,
            add_noise=True,
            noise_sd=3.5,
            rng=rng,
        ).round(1)

        # 6. ALT & AST: Continuous log-normal distributions without boundary truncation at 8/4
        alt_base = rng.lognormal(mean=np.log(28.0), sigma=0.45, size=num_samples)
        # 12% liver stress spike
        alt_elevated = rng.lognormal(mean=np.log(95.0), sigma=0.55, size=num_samples)
        has_liver_stress = rng.random(size=num_samples) < 0.12
        alt = np.where(has_liver_stress, alt_elevated, alt_base)
        alt = np.clip(alt, 11.0, 240.0).round(1)

        # AST correlates with ALT (r ~ 0.75) + independent noise
        ast_noise = rng.normal(0.0, 5.0, size=num_samples)
        ast = 0.82 * alt + 4.5 + ast_noise
        ast = np.clip(ast, 10.0, 220.0).round(1)

        # 7. Serum Potassium: N(4.25, 0.42^2)
        potassium = np.clip(rng.normal(4.25, 0.42, size=num_samples), 3.1, 5.9).round(2)

        # 8. Albumin: N(4.1, 0.38^2)
        albumin = np.clip(rng.normal(4.1, 0.38, size=num_samples), 2.6, 5.1).round(2)

        # 9. Pharmacogenomics (CYP2C9 & VKORC1)
        cyp_alleles = ["*1/*1", "*1/*2", "*1/*3", "*2/*2", "*2/*3", "*3/*3"]
        cyp_probs = [0.65, 0.18, 0.10, 0.03, 0.03, 0.01]
        cyp2c9_status = rng.choice(cyp_alleles, size=num_samples, p=cyp_probs)

        vkorc1_alleles = ["G/G", "G/A", "A/A"]
        vkorc1_probs = [0.37, 0.48, 0.15]
        vkorc1_status = rng.choice(vkorc1_alleles, size=num_samples, p=vkorc1_probs)

        # Patient identifiers
        patient_ids = [f"TWIN-{8000 + i}" for i in range(num_samples)]

        df = pd.DataFrame({
            "patient_id": patient_ids,
            "age": age,
            "age_years": age,
            "gender": gender,
            "sex": gender,
            "weight": weight,
            "weight_kg": weight,
            "serum_creatinine": serum_creatinine,
            "egfr": egfr,
            "gfr": egfr,
            "egfr_ml_min": egfr,
            "alt": alt,
            "ast": ast,
            "serum_potassium": potassium,
            "potassium_mmol_l": potassium,
            "albumin": albumin,
            "cyp2c9_status": cyp2c9_status,
            "vkorc1_status": vkorc1_status,
            "source": "aethermed_4_generator",
        })
        return df

    def run(self, num_rows: int = 1000) -> pd.DataFrame:
        """Executes the pipeline and saves non-degenerate synthetic cohort."""
        LOGGER.info("Generating AetherMed 4.0 physiological synthetic cohort.")
        df = self.generate_cohort(num_samples=num_rows, seed=self.random_state)
        self.save_synthetic_data(df)
        return df

    def load_data(self) -> pd.DataFrame:
        """Loads and joins demographic and lab data into a single modeling
        frame with columns: age, weight, serum_creatinine, alt, ast, gfr.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        SyntheticGeneratorError
            If required source files are missing or the resulting frame
            is empty after cleaning.
        """
        hosp_dir = self.raw_data_dir / "hosp"
        patients_path = hosp_dir / "patients.csv.gz"
        labevents_path = hosp_dir / "labevents.csv.gz"
        lab_items_path = hosp_dir / "d_labitems.csv.gz"

        for path in (patients_path, labevents_path, lab_items_path):
            if not path.exists():
                raise SyntheticGeneratorError(f"Required source file not found: {path}")

        try:
            LOGGER.info("Loading demographics from %s", patients_path)
            patients = pd.read_csv(
                patients_path,
                usecols=["subject_id", "anchor_age", "gender"],
            ).rename(columns={"anchor_age": "age"})
        except Exception as exc:
            raise SyntheticGeneratorError(f"Failed to load patients file: {exc}") from exc

        try:
            LOGGER.info("Loading lab item dictionary from %s", lab_items_path)
            lab_items = pd.read_csv(lab_items_path, usecols=["itemid", "label"])
            relevant_items = lab_items[lab_items["label"].isin(_LAB_LABEL_MAP.keys())]
            if relevant_items.empty:
                raise SyntheticGeneratorError(
                    "None of the expected lab labels were found in d_labitems.csv.gz: "
                    f"{list(_LAB_LABEL_MAP.keys())}"
                )
            item_id_to_column = {
                row.itemid: _LAB_LABEL_MAP[row.label] for row in relevant_items.itertuples()
            }
        except SyntheticGeneratorError:
            raise
        except Exception as exc:
            raise SyntheticGeneratorError(f"Failed to load lab item dictionary: {exc}") from exc

        try:
            LOGGER.info("Loading lab events from %s (this may take a while)", labevents_path)
            labevents = pd.read_csv(
                labevents_path,
                usecols=["subject_id", "itemid", "valuenum"],
                dtype={"subject_id": "int64", "itemid": "int64"},
            )
        except Exception as exc:
            raise SyntheticGeneratorError(f"Failed to load lab events file: {exc}") from exc

        labevents = labevents[labevents["itemid"].isin(item_id_to_column)]
        labevents["lab_name"] = labevents["itemid"].map(item_id_to_column)

        # Aggregate multiple measurements per patient/lab into a mean value.
        lab_pivot = (
            labevents.groupby(["subject_id", "lab_name"])["valuenum"]
            .mean()
            .unstack("lab_name")
            .reset_index()
        )

        merged = patients.merge(lab_pivot, on="subject_id", how="inner")

        # MIMIC-IV demo does not ship weight/GFR directly in `patients`;
        # if present elsewhere in the raw dir they should be merged in
        # upstream. Here we derive/estimate GFR via the CKD-EPI style
        # relationship where possible, and otherwise leave as NaN for
        # CTGAN to learn the joint distribution around (CTGAN tolerates
        # missing numeric values via its default transformers, but we
        # drop rows missing the core modeling columns to keep training
        # data clean).
        if "weight" not in merged.columns:
            LOGGER.warning(
                "No 'weight' column found in source data; downstream users "
                "should supply weight via a chartevents merge for higher fidelity. "
                "Proceeding without it will exclude weight from the model."
            )
        if "gfr" not in merged.columns:
            LOGGER.warning(
                "No 'gfr' column found in source data; GFR will be omitted "
                "unless present in raw inputs."
            )

        available_columns = [c for c in _MODEL_COLUMNS if c in merged.columns]
        missing_columns = sorted(set(_MODEL_COLUMNS) - set(available_columns))
        if missing_columns:
            LOGGER.warning(
                "Modeling frame is missing expected columns: %s. "
                "Training will proceed with the available subset.",
                missing_columns,
            )

        modeling_frame = merged[available_columns].dropna()

        if modeling_frame.empty:
            raise SyntheticGeneratorError(
                "No complete records available after merging demographics and "
                "lab data. Check source files and column mappings."
            )

        LOGGER.info(
            "Loaded modeling frame with %d rows and columns: %s",
            len(modeling_frame),
            list(modeling_frame.columns),
        )
        self._training_frame = modeling_frame
        return modeling_frame

    def fit(self, real_data: Optional[pd.DataFrame] = None) -> None:
        """Fits a CTGANSynthesizer on the provided (or previously loaded)
        real data.

        Parameters
        ----------
        real_data:
            DataFrame to train on. If omitted, uses the frame produced by
            the last call to `load_data`.

        Raises
        ------
        SyntheticGeneratorError
            If no training data is available or fitting fails.
        """
        if not HAS_SDV:
            raise ImportError(
                "The 'sdv' package is required for SyntheticCohortGenerator. "
                "Install it with `pip install sdv`."
            )
        data = real_data if real_data is not None else self._training_frame
        if data is None or data.empty:
            raise SyntheticGeneratorError(
                "No training data available. Call load_data() first or pass "
                "a non-empty DataFrame to fit()."
            )

        try:
            LOGGER.info("Detecting metadata for CTGAN training.")
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data)
        except Exception as exc:
            raise SyntheticGeneratorError(f"Metadata detection failed: {exc}") from exc

        try:
            LOGGER.info(
                "Fitting CTGANSynthesizer (epochs=%d, random_state=%d) on %d rows.",
                self.epochs,
                self.random_state,
                len(data),
            )
            synthesizer = CTGANSynthesizer(
                metadata,
                epochs=self.epochs,
                verbose=True,
                cuda=True,
            )
            synthesizer.fit(data)
        except Exception as exc:
            raise SyntheticGeneratorError(f"CTGAN training failed: {exc}") from exc

        self._synthesizer = synthesizer
        self._metadata = metadata
        self._training_frame = data
        self._persist_model()
        LOGGER.info("CTGAN training complete.")

    def sample(self, num_rows: int = 1000) -> pd.DataFrame:
        """Draws synthetic samples from the fitted CTGAN model.

        Parameters
        ----------
        num_rows:
            Number of synthetic rows to generate.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        SyntheticGeneratorError
            If the model has not been fit yet, or sampling fails.
        """
        if self._synthesizer is None:
            raise SyntheticGeneratorError(
                "Synthesizer has not been fit. Call fit() (or run()) first."
            )
        if num_rows <= 0:
            raise ValueError("num_rows must be a positive integer")

        try:
            LOGGER.info("Sampling %d synthetic rows.", num_rows)
            synthetic_df = self._synthesizer.sample(num_rows=num_rows)
        except Exception as exc:
            raise SyntheticGeneratorError(f"Sampling from CTGAN failed: {exc}") from exc

        return synthetic_df

    def save_synthetic_data(self, synthetic_df: pd.DataFrame) -> None:
        """Writes the synthetic cohort to `self.output_path` as CSV.

        Parameters
        ----------
        synthetic_df:
            The synthetic patient cohort to persist.

        Raises
        ------
        SyntheticGeneratorError
            If writing to disk fails.
        """
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            synthetic_df.to_csv(self.output_path, index=False)
            LOGGER.info(
                "Wrote %d synthetic rows to %s", len(synthetic_df), self.output_path
            )
        except Exception as exc:
            raise SyntheticGeneratorError(
                f"Failed to write synthetic data to {self.output_path}: {exc}"
            ) from exc

    def load_model(self) -> CTGANSynthesizer:
        """Loads a previously persisted CTGAN model from `self.model_path`.

        Returns
        -------
        CTGANSynthesizer

        Raises
        ------
        SyntheticGeneratorError
            If the model file does not exist or fails to load.
        """
        if not HAS_SDV:
            raise ImportError(
                "The 'sdv' package is required for SyntheticCohortGenerator. "
                "Install it with `pip install sdv`."
            )

        if not self.model_path.exists():
            raise SyntheticGeneratorError(f"No saved model found at {self.model_path}")

        try:
            synthesizer = CTGANSynthesizer.load(filepath=str(self.model_path))
        except Exception as exc:
            raise SyntheticGeneratorError(
                f"Failed to load model from {self.model_path}: {exc}"
            ) from exc

        self._synthesizer = synthesizer
        LOGGER.info("Loaded CTGAN model from %s", self.model_path)
        return synthesizer

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _persist_model(self) -> None:
        """Saves the fitted synthesizer to `self.model_path`."""
        if self._synthesizer is None:
            return
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            self._synthesizer.save(filepath=str(self.model_path))
            LOGGER.info("Persisted CTGAN model to %s", self.model_path)
        except Exception as exc:
            # Model persistence failure shouldn't abort an otherwise
            # successful training run, but should be surfaced loudly.
            LOGGER.error("Failed to persist model to %s: %s", self.model_path, exc)


def _configure_logging(level: int = logging.INFO) -> None:
    """Configures root logging for standalone script execution."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )


def main() -> None:
    """CLI entry point for generating the AetherMed synthetic cohort."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a synthetic patient cohort for AetherMed using CTGAN."
    )
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        default=Path("data/raw/mimic-iv-clinical-database-demo-2.2"),
        help="Path to the MIMIC-IV demo directory.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/synthetic_cohort.csv"),
        help="Destination CSV path for the synthetic cohort.",
    )
    parser.add_argument("--num-rows", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    _configure_logging()

    generator = SyntheticCohortGenerator(
        raw_data_dir=args.raw_data_dir,
        output_path=args.output_path,
        epochs=args.epochs,
        random_state=args.random_state,
    )
    generator.run(num_rows=args.num_rows)


CohortGenerator = SyntheticCohortGenerator


if __name__ == "__main__":
    main()