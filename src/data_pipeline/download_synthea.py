"""
Download and unpack the public Synthea sample-data CSV release.

Synthea generates fully synthetic patient records — there is no PHI/consent
concern, which is why it's used here to supplement the small 100-patient
MIMIC-IV Demo cohort with a larger longitudinal population.

Usage:
    python -m src.data_pipeline.download_synthea
"""
import sys
import zipfile
from io import BytesIO

import requests

from src.data_pipeline.config import SYNTHEA_DIR, SYNTHEA_RELEASE_URL


def main() -> int:
    print(f"Downloading Synthea sample data to {SYNTHEA_DIR}")
    SYNTHEA_DIR.mkdir(parents=True, exist_ok=True)

    resp = requests.get(SYNTHEA_RELEASE_URL, timeout=120)
    if resp.status_code != 200:
        print(f"[FAIL] HTTP {resp.status_code} fetching {SYNTHEA_RELEASE_URL}")
        print(resp.text[:300])
        return 1

    try:
        with zipfile.ZipFile(BytesIO(resp.content)) as zf:
            zf.extractall(SYNTHEA_DIR)
    except zipfile.BadZipFile:
        print("[FAIL] Downloaded content was not a valid zip file.")
        return 1

    csvs = list(SYNTHEA_DIR.rglob("*.csv"))
    print(f"Extracted {len(csvs)} CSV files to {SYNTHEA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
