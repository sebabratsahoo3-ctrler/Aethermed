"""
Download the MIMIC-IV Clinical Database Demo (v2.2) from PhysioNet.

This is the OPEN ACCESS demo (100 de-identified patients), not the
credentialed full MIMIC-IV. No PhysioNet account or DUA required.

Usage:
    python -m src.data_pipeline.download_mimic_demo

If any file fails to download, this script prints the real HTTP error and
exits non-zero — it does NOT fall back to fabricating or skipping data
silently, per project working rules.
"""
import gzip
import shutil
import sys
from pathlib import Path

import requests
from tqdm import tqdm

from src.data_pipeline.config import MIMIC_DEMO_BASE_URL, MIMIC_DEMO_DIR, MIMIC_DEMO_FILES


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to download {url} — HTTP {resp.status_code}: {resp.text[:300]}"
        )
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name
    ) as bar:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def main() -> int:
    print(f"Downloading MIMIC-IV Demo v2.2 to {MIMIC_DEMO_DIR}")
    failures = []
    for rel_path in MIMIC_DEMO_FILES:
        url = MIMIC_DEMO_BASE_URL + rel_path
        dest = MIMIC_DEMO_DIR / rel_path
        if dest.exists():
            print(f"  [skip] {rel_path} already present")
            continue
        try:
            download_file(url, dest)
        except Exception as e:
            print(f"  [FAIL] {rel_path}: {e}")
            failures.append(rel_path)

    if failures:
        print(f"\n{len(failures)} file(s) failed to download: {failures}")
        print("Not substituting fabricated data. Re-run once network/access is fixed.")
        return 1

    print("\nAll MIMIC-IV Demo files downloaded successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
