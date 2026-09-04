"""Financial feature engineering (STEP 5 of the roadmap).

Every feature below is computed **row-wise** from a single applicant's own
record.  No statistic is pooled across rows, therefore this transformation is
leakage-free and may safely be applied before the train/test split.
(Statistics that *are* fitted from data -- imputation medians, scaler means,
one-hot categories -- live inside the sklearn Pipeline instead.)

Column semantics from the UCI documentation (April..September 2005):
  BILL_AMT1 = September bill      ... BILL_AMT6 = April bill
  PAY_AMT1  = paid during September ... PAY_AMT6 = paid during April
  PAY_0     = September repayment status ... PAY_6 = April repayment status
  repayment status: -2 = no consumption, -1 = paid in full,
                     0 = revolving credit, 1..9 = months of payment delay
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_loader import BILL_COLS, PAY_AMT_COLS, PAY_COLS
from src import config

# A payment made in month t settles the bill raised in month t+1.
# Ratios above 10x carry no extra meaning (the bill is simply cleared), so we
# cap them to stop a near-zero bill producing an extreme outlier.
MAX_PAY_RATIO = 10.0

BASE_CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
BASE_NUMERIC = ["LIMIT_BAL", "AGE"] + PAY_COLS + BILL_COLS + PAY_AMT_COLS


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, returning NaN where the denominator is non-positive.

    A non-positive bill means there was nothing to repay, so the ratio is
    undefined rather than zero.  NaNs are handled by the imputer in the
    preprocessing pipeline.
    """
    denom = denominator.where(denominator > 0)
    return (numerator / denom).replace([np.inf, -np.inf], np.nan)


def add_financial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered credit-risk features added."""
    out = df.copy()
    limit = out["LIMIT_BAL"]

    # -- 1. Credit utilisation -------------------------------------------
    # Share of the granted limit already consumed. The single strongest
    # concept in credit scoring: high sustained utilisation signals stress.
    util_cols = []
    for i, col in enumerate(BILL_COLS, start=1):
        name = f"util_{i}"
        out[name] = out[col] / limit
        util_cols.append(name)

    out["avg_utilization"] = out[util_cols].mean(axis=1)
    out["max_utilization"] = out[util_cols].max(axis=1)
    # Positive trend = debt growing relative to the limit over six months.
    out["utilization_trend"] = out["util_1"] - out["util_6"]
    out["over_limit_months"] = (out[BILL_COLS].gt(limit, axis=0)).sum(axis=1)

    # -- 2. Repayment effort ---------------------------------------------
    # How much of last month's bill the customer actually paid off.
    ratio_cols = []
    for t in range(1, 6):
        name = f"pay_ratio_{t}"
        out[name] = _safe_ratio(out[f"PAY_AMT{t}"], out[f"BILL_AMT{t + 1}"]).clip(
            upper=MAX_PAY_RATIO
        )
        ratio_cols.append(name)

    out["avg_pay_ratio"] = out[ratio_cols].mean(axis=1)
    out["min_pay_ratio"] = out[ratio_cols].min(axis=1)
    # Months where the customer paid literally nothing.
    out["zero_payment_months"] = (out[PAY_AMT_COLS] == 0).sum(axis=1)

    # -- 3. Delinquency history -------------------------------------------
    pay_status = out[PAY_COLS]
    out["n_delinquent_months"] = (pay_status >= 1).sum(axis=1)
    out["max_delinquency"] = pay_status.max(axis=1)
    out["recent_delinquency"] = (out["PAY_0"] >= 1).astype(int)
    # Positive = repayment behaviour deteriorating between April and September.
    out["delinquency_trend"] = out["PAY_0"] - out["PAY_6"]
    out["months_paid_full"] = (pay_status == -1).sum(axis=1)
    out["months_no_consumption"] = (pay_status == -2).sum(axis=1)

    # -- 4. Absolute financial burden --------------------------------------
    out["avg_bill_amt"] = out[BILL_COLS].mean(axis=1)
    out["avg_pay_amt"] = out[PAY_AMT_COLS].mean(axis=1)
    out["bill_volatility"] = out[BILL_COLS].std(axis=1)
    # Payment capacity relative to the credit granted.
    out["payment_to_limit"] = out["avg_pay_amt"] / limit
    # Headroom still available on the card.
    out["remaining_credit"] = limit - out["BILL_AMT1"]

    return out


def engineered_feature_names() -> list[str]:
    """Names of the columns created by :func:`add_financial_features`."""
    return (
        [f"util_{i}" for i in range(1, 7)]
        + ["avg_utilization", "max_utilization", "utilization_trend", "over_limit_months"]
        + [f"pay_ratio_{t}" for t in range(1, 6)]
        + ["avg_pay_ratio", "min_pay_ratio", "zero_payment_months"]
        + [
            "n_delinquent_months",
            "max_delinquency",
            "recent_delinquency",
            "delinquency_trend",
            "months_paid_full",
            "months_no_consumption",
        ]
        + [
            "avg_bill_amt",
            "avg_pay_amt",
            "bill_volatility",
            "payment_to_limit",
            "remaining_credit",
        ]
    )


def get_feature_lists(include_sex: bool = True) -> tuple[list[str], list[str]]:
    """Return ``(numeric_features, categorical_features)`` for the pipeline."""
    categorical = list(BASE_CATEGORICAL)
    if not include_sex:
        categorical.remove("SEX")
    return BASE_NUMERIC + engineered_feature_names(), categorical


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features and drop the target if present (X matrix)."""
    feat = add_financial_features(df)
    return feat.drop(columns=[config.TARGET], errors="ignore")


if __name__ == "__main__":
    from src.data_loader import load_clean

    data = add_financial_features(load_clean())
    num, cat = get_feature_lists()
    print(f"Shape after engineering: {data.shape}")
    print(f"{len(num)} numeric + {len(cat)} categorical features")
    missing = data[num].isna().sum()
    print("Engineered columns containing NaN (handled by the imputer):")
    print(missing[missing > 0].to_string() or "  none")
