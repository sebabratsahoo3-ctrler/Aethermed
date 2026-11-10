"""
Pull drug-property data from the free ChEMBL REST API (no API key needed).

Fetches molecule metadata and, where available, mechanism-of-action and
approved-indication data for the drugs pinned in config.CHEMBL_TARGET_DRUGS.
These are the drugs later used in Phase 3 to validate the PK/PD engine
against known half-life/clearance values.

Usage:
    python -m src.data_pipeline.fetch_chembl
"""
import json
import sys

import requests

from src.data_pipeline.config import CHEMBL_API_BASE, CHEMBL_DIR, CHEMBL_TARGET_DRUGS


def fetch_molecule(chembl_id: str) -> dict:
    url = f"{CHEMBL_API_BASE}/molecule/{chembl_id}.json"
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"ChEMBL request failed for {chembl_id}: HTTP {resp.status_code}")
    return resp.json()


def fetch_mechanism(chembl_id: str) -> dict:
    url = f"{CHEMBL_API_BASE}/mechanism.json"
    resp = requests.get(url, params={"molecule_chembl_id": chembl_id}, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"ChEMBL mechanism request failed for {chembl_id}: HTTP {resp.status_code}")
    return resp.json()


def main() -> int:
    CHEMBL_DIR.mkdir(parents=True, exist_ok=True)
    failures = []

    for name, chembl_id in CHEMBL_TARGET_DRUGS.items():
        print(f"Fetching {name} ({chembl_id}) from ChEMBL...")
        try:
            molecule = fetch_molecule(chembl_id)
            mechanism = fetch_mechanism(chembl_id)
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failures.append(name)
            continue

        out = {"molecule": molecule, "mechanism": mechanism}
        out_path = CHEMBL_DIR / f"{name}.json"
        out_path.write_text(json.dumps(out, indent=2))
        print(f"  saved -> {out_path}")

    if failures:
        print(f"\n{len(failures)} drug(s) failed: {failures}")
        return 1

    print("\nAll ChEMBL drug records fetched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
