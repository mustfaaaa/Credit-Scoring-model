"""Dataset acquisition and cleaning (STEP 1 - STEP 3 of the roadmap).

Downloads the UCI "Default of Credit Card Clients" dataset, caches the raw
file, and applies the cleaning decisions justified in report.md.
"""
from __future__ import annotations

import io
import logging
import urllib.request
import zipfile

import pandas as pd

from src import config

log = logging.getLogger(__name__)

# Codes documented in the UCI dataset description.  Anything outside the
# documented set is folded into the "others" bucket rather than silently kept
# as a meaningless integer.
EDUCATION_MAP = {1: "graduate_school", 2: "university", 3: "high_school"}
MARRIAGE_MAP = {1: "married", 2: "single", 3: "others"}
SEX_MAP = {1: "male", 2: "female"}

PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]


def download_raw(force: bool = False) -> "config.Path":
    """Download and cache the raw .xls file. Returns the local path."""
    if config.RAW_FILE.exists() and not force:
        log.info("Raw file already cached at %s", config.RAW_FILE)
        return config.RAW_FILE

    log.info("Downloading dataset from %s", config.DATA_URL)
    payload = urllib.request.urlopen(config.DATA_URL, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        # The archive holds exactly one .xls workbook.
        name = next(n for n in zf.namelist() if n.lower().endswith(".xls"))
        config.RAW_FILE.write_bytes(zf.read(name))
    log.info("Saved raw data to %s", config.RAW_FILE)
    return config.RAW_FILE


def load_raw() -> pd.DataFrame:
    """Load the raw workbook. Row 0 is a merged banner, so header=1."""
    download_raw()
    return pd.read_excel(config.RAW_FILE, header=1)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning steps decided during data understanding.

    1. Drop ``ID`` -- a row identifier carries no predictive signal and would
       act as a leakage-style shortcut for tree models.
    2. Rename the verbose target to ``default``.
    3. Drop exact duplicate applicants (35 rows) to avoid the same record
       landing in both train and test.
    4. Recode ``SEX`` / ``EDUCATION`` / ``MARRIAGE`` to readable labels and
       fold undocumented codes (EDUCATION 0/5/6, MARRIAGE 0) into "others".
    """
    df = df.copy()
    df = df.drop(columns=["ID"])
    df = df.rename(columns={"default payment next month": config.TARGET})

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log.info("Dropped %d duplicate rows", before - len(df))

    df["SEX"] = df["SEX"].map(SEX_MAP).fillna("unknown")
    df["EDUCATION"] = df["EDUCATION"].map(EDUCATION_MAP).fillna("others")
    df["MARRIAGE"] = df["MARRIAGE"].map(MARRIAGE_MAP).fillna("others")

    return df


def load_clean() -> pd.DataFrame:
    """Convenience wrapper: download -> read -> clean."""
    return clean(load_raw())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    data = load_clean()
    print(f"Clean shape: {data.shape}")
    print(data[config.TARGET].value_counts().to_string())
