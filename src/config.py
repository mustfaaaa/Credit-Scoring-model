"""Central configuration: paths, constants and reproducibility settings.

Every module imports paths from here so that nothing is hard-coded and the
project runs unchanged from any working directory.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths -----
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RESULTS_DIR = OUTPUTS_DIR / "results"

RAW_FILE = RAW_DIR / "default_of_credit_card_clients.xls"
PROCESSED_FILE = PROCESSED_DIR / "credit_clean_engineered.csv"
FINAL_MODEL_FILE = MODELS_DIR / "final_model.pkl"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------- dataset ------
# UCI Machine Learning Repository -- "Default of Credit Card Clients" (id 350).
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)
TARGET = "default"

# ------------------------------------------------------ reproducibility -----
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

# --------------------------------------------------------- cost model -------
# Business assumption used for threshold selection (documented in report.md):
# missing a defaulter (FN) loses the outstanding balance, while wrongly
# rejecting a good customer (FP) only loses the profit margin on that account.
COST_FN = 5.0   # cost of approving a borrower who defaults
COST_FP = 1.0   # cost of rejecting a borrower who would have repaid
