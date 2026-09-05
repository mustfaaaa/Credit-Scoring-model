"""Flask web dashboard for the credit scoring model.

Serves a single-page dashboard that explains what the model does, shows every
artefact produced by ``python -m src.train``, and scores new applicants live
using the real saved pipeline.

Run with::

    python -m src.webapp          # then open http://127.0.0.1:5000

Nothing on the page is hard-coded: tables are read from ``outputs/results/``
and every prediction comes from ``models/final_model.pkl`` at request time.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from src import config
from src.data_loader import BILL_COLS, PAY_AMT_COLS, PAY_COLS
from src.feature_engineering import add_financial_features
from src.predict import load_model, predict_applicant, risk_category

app = Flask(
    __name__,
    template_folder=str(config.PROJECT_ROOT / "templates"),
    static_folder=str(config.PROJECT_ROOT / "static"),
)

# Human-readable labels for the six repayment-status codes.
PAY_STATUS_CHOICES = [
    (-2, "-2  No consumption"),
    (-1, "-1  Paid in full"),
    (0, " 0  Revolving credit"),
    (1, " 1  1 month late"),
    (2, " 2  2 months late"),
    (3, " 3  3 months late"),
    (4, " 4  4 months late"),
    (5, " 5  5 months late"),
    (6, " 6  6 months late"),
    (7, " 7  7 months late"),
    (8, " 8  8+ months late"),
]

MONTH_LABELS = ["September", "August", "July", "June", "May", "April"]

# Indicators shown back to the user to explain an individual score.
EXPLAIN_FIELDS = [
    ("n_delinquent_months", "Late months (of 6)", "{:.0f}", "{:.2f}"),
    ("max_delinquency", "Worst delay", "delay", "delay"),
    ("avg_utilization", "Average credit utilisation", "{:.1%}", "{:.1%}"),
    ("max_utilization", "Peak credit utilisation", "{:.1%}", "{:.1%}"),
    ("avg_pay_ratio", "Average repayment effort", "{:.2f}", "{:.2f}"),
    ("remaining_credit", "Remaining credit (NT$)", "{:,.0f}", "{:,.0f}"),
]


def _fmt(value, spec: str) -> str:
    """Format one indicator. The "delay" spec renders repayment-status codes,
    where anything below 1 means the customer was never actually late."""
    if spec == "delay":
        return "never late" if value < 1 else f"{value:.1f} month(s)"
    return spec.format(value)

_CACHE: dict = {}


# --------------------------------------------------------------- helpers ----
def _clean_nans(obj):
    """Recursively replace NaN/inf with None so the result is valid JSON."""
    if isinstance(obj, dict):
        return {k: _clean_nans(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nans(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _read_csv(name: str, index_col=0) -> pd.DataFrame | None:
    path = config.RESULTS_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=index_col)


def _table(df: pd.DataFrame | None, decimals: int = 4) -> dict | None:
    """Turn a DataFrame into a template-friendly {columns, rows} dict."""
    if df is None:
        return None
    out = df.copy()
    for col in out.select_dtypes("number").columns:
        out[col] = out[col].round(decimals)
    rows = []
    for idx, row in out.iterrows():
        cells = ["" if pd.isna(v) else (f"{v:,.4f}".rstrip("0").rstrip(".")
                                        if isinstance(v, float) else str(v))
                 for v in row]
        rows.append({"label": str(idx), "cells": cells})
    return {"columns": [str(c) for c in out.columns], "rows": rows}


def get_dataset():
    """Load the dataset once, preferring the processed CSV for speed."""
    if "data" not in _CACHE:
        if config.PROCESSED_FILE.exists():
            _CACHE["data"] = pd.read_csv(config.PROCESSED_FILE)
        else:  # fall back to rebuilding from the raw source
            from src.data_loader import load_clean

            _CACHE["data"] = add_financial_features(load_clean())
    return _CACHE["data"]


def get_context() -> dict:
    """Assemble everything the dashboard needs, cached after the first build."""
    if "context" in _CACHE:
        return _CACHE["context"]

    bundle = load_model()
    summary_path = config.RESULTS_DIR / "final_model_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) \
        if summary_path.exists() else {}

    # Feature importance (permutation is the reference measure).
    imp = _read_csv("feature_importance.csv", index_col=None)
    top_features = []
    if imp is not None:
        perm = imp[imp["type"] == "permutation_roc_auc_drop"].nlargest(12, "importance")
        peak = float(perm["importance"].max()) or 1.0
        top_features = [
            {
                "feature": r["feature"],
                "importance": float(r["importance"]),
                "pct": max(2.0, float(r["importance"]) / peak * 100.0),
                "engineered": r["feature"] not in (
                    ["LIMIT_BAL", "AGE", "SEX", "EDUCATION", "MARRIAGE"]
                    + PAY_COLS + BILL_COLS + PAY_AMT_COLS
                ),
            }
            for _, r in perm.iterrows()
        ]

    # Confusion matrix at the tuned threshold, taken from the error analysis.
    err = _read_csv("error_analysis.csv", index_col=None)
    confusion = None
    if err is not None:
        counts = dict(zip(err["group"], err["count"]))
        confusion = {
            "tn": int(counts.get("True Negative (correctly approved)", 0)),
            "fp": int(counts.get("False Positive (good customer rejected)", 0)),
            "fn": int(counts.get("False Negative (defaulter approved)", 0)),
            "tp": int(counts.get("True Positive (defaulter caught)", 0)),
        }
        confusion["defaulters"] = confusion["tp"] + confusion["fn"]
        confusion["caught_pct"] = (confusion["tp"] / confusion["defaulters"] * 100
                                   if confusion["defaulters"] else 0.0)

    metrics = summary.get("test_metrics_at_threshold", {})
    fairness = _read_csv("fairness_sensitivity.csv", index_col=None)

    ctx = {
        "model_name": bundle["model_name"],
        "threshold": bundle["threshold"],
        "trained_at": bundle["trained_at"],
        "best_params": bundle["best_params"],
        "cv_roc_auc": bundle.get("cv_roc_auc"),
        "n_train": bundle.get("n_train"),
        "n_test": bundle.get("n_test"),
        "sklearn_version": bundle.get("sklearn_version"),
        "cost_fn": bundle.get("cost_fn"),
        "cost_fp": bundle.get("cost_fp"),
        "n_features": len(bundle["numeric_features"]) + len(bundle["categorical_features"]),
        "metrics": metrics,
        "metrics_at_half": summary.get("test_metrics_at_0.5", {}),
        "confusion": confusion,
        "top_features": top_features,
        "cv_table": _table(_read_csv("cv_comparison.csv")),
        "test_table": _table(_read_csv("test_comparison.csv")),
        "imbalance_table": _table(_read_csv("imbalance_study.csv", index_col=None)),
        "tuning_table": _table(_read_csv("tuning_results.csv", index_col=None)),
        "error_table": _table(_read_csv("error_analysis.csv", index_col=None)),
        "fairness_auc": (float(fairness["test_roc_auc"].iloc[0])
                         if fairness is not None and len(fairness) else None),
        "pay_choices": PAY_STATUS_CHOICES,
        "months": MONTH_LABELS,
        "pay_cols": PAY_COLS,
        "bill_cols": BILL_COLS,
        "pay_amt_cols": PAY_AMT_COLS,
        "figures": sorted(p.name for p in config.FIGURES_DIR.glob("*.png")),
    }
    _CACHE["context"] = ctx
    return ctx


def explain(applicant: dict) -> list[dict]:
    """Compute the engineered risk indicators for one applicant.

    These are the actual feature values the model saw -- not a post-hoc
    approximation -- shown next to the population averages so the number has
    context.
    """
    row = add_financial_features(pd.DataFrame([applicant])).iloc[0]
    data = get_dataset()
    out = []
    for key, label, fmt, avg_fmt in EXPLAIN_FIELDS:
        if key not in row.index:
            continue
        value = row[key]
        if pd.isna(value):
            continue
        entry = {"label": label, "value": _fmt(value, fmt)}
        if key in data.columns and config.TARGET in data.columns:
            grouped = data.groupby(config.TARGET)[key].mean()
            entry["avg_repaid"] = _fmt(grouped.get(0, float("nan")), avg_fmt)
            entry["avg_defaulted"] = _fmt(grouped.get(1, float("nan")), avg_fmt)
            # Flag the indicator when the applicant is worse than the typical
            # defaulter on a "higher is riskier" measure.
            riskier_when_high = key not in ("avg_pay_ratio", "remaining_credit")
            defaulter_mean = float(grouped.get(1, float("nan")))
            if not math.isnan(defaulter_mean):
                entry["flagged"] = bool(
                    value >= defaulter_mean if riskier_when_high
                    else value <= defaulter_mean
                )
        out.append(entry)
    return out


# ---------------------------------------------------------------- routes ----
@app.route("/")
def index():
    return render_template("index.html", **get_context())


@app.route("/figures/<path:filename>")
def figure(filename: str):
    """Serve a generated figure. Restricted to PNGs inside outputs/figures."""
    if not filename.endswith(".png") or "/" in filename or "\\" in filename:
        abort(404)
    return send_from_directory(config.FIGURES_DIR, filename)


@app.route("/api/sample")
def api_sample():
    """Return a real customer from the dataset to pre-fill the form.

    ``kind=risky`` picks an actual defaulter, ``kind=safe`` a repayer, and
    ``kind=random`` anyone. Useful for demoing without typing 23 fields.
    """
    kind = request.args.get("kind", "random")
    data = get_dataset()
    pool = data
    if kind == "risky" and config.TARGET in data.columns:
        pool = data[data[config.TARGET] == 1]
    elif kind == "safe" and config.TARGET in data.columns:
        pool = data[data[config.TARGET] == 0]
    if pool.empty:
        pool = data

    bundle = load_model()
    row = pool.sample(1).iloc[0]
    applicant = {c: row[c] for c in bundle["raw_input_columns"]}
    payload = {
        "applicant": {
            k: (int(v) if isinstance(v, (int, np.integer)) or
                (isinstance(v, float) and float(v).is_integer()) else v)
            for k, v in applicant.items()
        },
        "actual_outcome": (
            "defaulted" if row.get(config.TARGET) == 1 else "repaid"
        ) if config.TARGET in data.columns else None,
    }
    return jsonify(_clean_nans(payload))


@app.route("/api/score", methods=["POST"])
def api_score():
    """Score one applicant with the real trained pipeline."""
    payload = request.get_json(silent=True) or {}
    bundle = load_model()

    applicant, errors = {}, []
    for col in bundle["raw_input_columns"]:
        if col not in payload or payload[col] in ("", None):
            errors.append({"field": col, "message": "This field is required."})
            continue
        value = payload[col]
        if col in ("SEX", "EDUCATION", "MARRIAGE"):
            applicant[col] = str(value)
            continue
        try:
            applicant[col] = float(value)
        except (TypeError, ValueError):
            errors.append({"field": col, "message": "Enter a number."})

    if not errors:
        if applicant.get("LIMIT_BAL", 0) <= 0:
            errors.append({"field": "LIMIT_BAL",
                           "message": "Credit limit must be greater than 0."})
        if not 18 <= applicant.get("AGE", 0) <= 120:
            errors.append({"field": "AGE", "message": "Age must be between 18 and 120."})

    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    result = predict_applicant(applicant, bundle)
    prob = result["probability_of_default"]
    result.update({
        "ok": True,
        "risk_category": risk_category(prob),
        "percent_default": round(prob * 100, 1),
        "percent_good": round((1 - prob) * 100, 1),
        "indicators": explain(applicant),
    })
    return jsonify(_clean_nans(result))


def main() -> None:
    host = "127.0.0.1"
    port = 5000
    print(f"Credit Scoring dashboard -> http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
