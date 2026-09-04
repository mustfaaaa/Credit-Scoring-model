"""Prediction interface for new applicants (PHASE 16).

Loads the saved bundle -- which contains the *whole* pipeline (imputation,
scaling, one-hot encoding and the trained classifier) plus the cost-optimal
decision threshold -- so raw applicant data can be scored directly.

Command line demo::

    python -m src.predict
"""
from __future__ import annotations

from typing import Any

import joblib
import pandas as pd

from src import config
from src.feature_engineering import add_financial_features

# Risk bands are reported alongside the raw probability so a non-technical
# reader gets an actionable label rather than a bare number.
RISK_BANDS = [
    (0.20, "LOW RISK"),
    (0.40, "MODERATE RISK"),
    (0.60, "HIGH RISK"),
    (1.01, "VERY HIGH RISK"),
]

_BUNDLE_CACHE: dict[str, Any] = {}


def load_model(path=None) -> dict:
    """Load (and cache) the trained bundle."""
    path = str(path or config.FINAL_MODEL_FILE)
    if path not in _BUNDLE_CACHE:
        if not config.FINAL_MODEL_FILE.exists() and path == str(config.FINAL_MODEL_FILE):
            raise FileNotFoundError(
                f"No trained model at {path}. Run `python -m src.train` first."
            )
        _BUNDLE_CACHE[path] = joblib.load(path)
    return _BUNDLE_CACHE[path]


def risk_category(prob_default: float) -> str:
    for upper, label in RISK_BANDS:
        if prob_default < upper:
            return label
    return RISK_BANDS[-1][1]


def _to_frame(applicant: dict | pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Validate the raw input and return a one-or-more-row DataFrame."""
    df = (applicant.copy() if isinstance(applicant, pd.DataFrame)
          else pd.DataFrame([applicant]))
    missing = [c for c in bundle["raw_input_columns"] if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required applicant fields: " + ", ".join(missing)
        )
    return df


def predict_applicant(applicant: dict | pd.DataFrame, bundle: dict | None = None) -> dict:
    """Score a single applicant supplied as a dict of raw financial fields.

    Returns the decision, both class probabilities and a risk band -- all
    derived from the trained model, none hard-coded.
    """
    bundle = bundle or load_model()
    df = _to_frame(applicant, bundle)

    # Identical feature engineering to training, then the fitted pipeline.
    features = add_financial_features(df)
    cols = bundle["numeric_features"] + bundle["categorical_features"]
    prob_default = float(bundle["pipeline"].predict_proba(features[cols])[0, 1])

    threshold = bundle["threshold"]
    is_default = prob_default >= threshold

    return {
        "prediction": "BAD CREDIT (likely to default)" if is_default
                      else "GOOD CREDIT (likely to repay)",
        "decision": "DECLINE / REVIEW" if is_default else "APPROVE",
        "probability_of_default": round(prob_default, 4),
        "probability_of_good_credit": round(1.0 - prob_default, 4),
        "risk_category": risk_category(prob_default),
        "threshold_used": round(threshold, 4),
        "model": bundle["model_name"],
    }


def predict_batch(df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """Score many applicants at once; returns a DataFrame of results."""
    bundle = bundle or load_model()
    frame = _to_frame(df, bundle)
    features = add_financial_features(frame)
    cols = bundle["numeric_features"] + bundle["categorical_features"]
    proba = bundle["pipeline"].predict_proba(features[cols])[:, 1]
    threshold = bundle["threshold"]

    return pd.DataFrame({
        "probability_of_default": proba.round(4),
        "probability_of_good_credit": (1 - proba).round(4),
        "risk_category": [risk_category(p) for p in proba],
        "decision": ["DECLINE / REVIEW" if p >= threshold else "APPROVE"
                     for p in proba],
    }, index=frame.index)


def format_report(result: dict, title: str = "CREDIT DECISION") -> str:
    """Human-readable block for the console / viva demo."""
    return "\n".join([
        "-" * 58,
        f"  {title}",
        "-" * 58,
        f"  Model used                 : {result['model']}",
        f"  Prediction                 : {result['prediction']}",
        f"  Decision                   : {result['decision']}",
        f"  Risk category              : {result['risk_category']}",
        f"  Probability of GOOD credit : "
        f"{result['probability_of_good_credit'] * 100:.1f}%",
        f"  Probability of BAD credit  : "
        f"{result['probability_of_default'] * 100:.1f}%",
        f"  Decision threshold         : {result['threshold_used']:.3f}",
        "-" * 58,
    ])


# ---------------------------------------------------------------- examples --
def example_applicants() -> dict[str, dict]:
    """Two realistic, hand-built raw applications used for the demo.

    The *inputs* are constructed by hand; every predicted number comes from
    the trained model at run time.
    """
    low_risk = {
        "LIMIT_BAL": 350_000, "SEX": "female", "EDUCATION": "graduate_school",
        "MARRIAGE": "married", "AGE": 41,
        # Never late: -1 = paid in full every month.
        "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
        # Modest balances relative to a 350k limit.
        "BILL_AMT1": 28_000, "BILL_AMT2": 26_500, "BILL_AMT3": 31_000,
        "BILL_AMT4": 24_000, "BILL_AMT5": 22_500, "BILL_AMT6": 20_000,
        # Pays the statement off in full each month.
        "PAY_AMT1": 26_500, "PAY_AMT2": 31_000, "PAY_AMT3": 24_000,
        "PAY_AMT4": 22_500, "PAY_AMT5": 20_000, "PAY_AMT6": 19_000,
    }

    high_risk = {
        "LIMIT_BAL": 20_000, "SEX": "male", "EDUCATION": "high_school",
        "MARRIAGE": "single", "AGE": 24,
        # Progressively later every month: 1 -> 3 months delayed.
        "PAY_0": 3, "PAY_2": 3, "PAY_3": 2, "PAY_4": 2, "PAY_5": 1, "PAY_6": 1,
        # Balances sit at or above the 20k limit.
        "BILL_AMT1": 19_800, "BILL_AMT2": 19_400, "BILL_AMT3": 19_100,
        "BILL_AMT4": 18_600, "BILL_AMT5": 18_000, "BILL_AMT6": 17_500,
        # Pays only a token minimum, and nothing at all in two months.
        "PAY_AMT1": 0, "PAY_AMT2": 700, "PAY_AMT3": 800,
        "PAY_AMT4": 0, "PAY_AMT5": 600, "PAY_AMT6": 700,
    }
    return {"Applicant A -- strong payer": low_risk,
            "Applicant B -- distressed borrower": high_risk}


def main() -> None:
    bundle = load_model()
    print(f"Loaded model : {bundle['model_name']}")
    print(f"Trained at   : {bundle['trained_at']}")
    print(f"Threshold    : {bundle['threshold']:.3f}")

    for title, applicant in example_applicants().items():
        print()
        print(format_report(predict_applicant(applicant, bundle), title))

    print("\nBatch scoring check:")
    batch = pd.DataFrame(list(example_applicants().values()))
    print(predict_batch(batch, bundle).to_string())


if __name__ == "__main__":
    main()
