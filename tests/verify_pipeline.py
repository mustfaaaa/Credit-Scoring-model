"""End-to-end verification suite (STEP 21 / PHASE 20).

Runs the eleven checks required by the assignment against the *real* trained
artefacts.  Deliberately dependency-free (plain asserts, no pytest) so it runs
with nothing beyond requirements.txt::

    python -m tests.verify_pipeline

Exit code 0 = every check passed.
"""
from __future__ import annotations

import sys
import traceback

import numpy as np
import pandas as pd

from src import config

PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def check(name):
    """Decorator turning a function into a reported pass/fail check."""
    def wrapper(fn):
        try:
            detail = fn()
            PASSED.append(f"{name} -- {detail}")
            print(f"  [PASS] {name}: {detail}")
        except Exception as exc:  # noqa: BLE001 - we want every failure reported
            FAILED.append((name, traceback.format_exc()))
            print(f"  [FAIL] {name}: {exc}")
        return fn
    return wrapper


print("=" * 70)
print("CREDIT SCORING PROJECT -- VERIFICATION SUITE")
print("=" * 70)


# ------------------------------------------------------------ 1. loading ----
@check("1. Dataset loading")
def _test_load():
    from src.data_loader import load_clean

    df = load_clean()
    assert df.shape[0] > 25_000, f"unexpected row count {df.shape[0]}"
    assert config.TARGET in df.columns, "target column missing"
    assert "ID" not in df.columns, "identifier column was not dropped"
    assert df.duplicated().sum() == 0, "duplicates were not removed"
    globals()["_DF"] = df
    return f"{df.shape[0]:,} rows x {df.shape[1]} cols, no duplicates, ID dropped"


# ------------------------------------------------ 2. feature engineering ----
@check("2. Feature engineering")
def _test_features():
    from src.feature_engineering import (
        add_financial_features, engineered_feature_names, get_feature_lists,
    )

    feat = add_financial_features(_DF)
    new = engineered_feature_names()
    missing = [c for c in new if c not in feat.columns]
    assert not missing, f"features not created: {missing}"
    assert len(feat) == len(_DF), "row count changed during engineering"
    # No infinities may survive into the model matrix.
    num, _ = get_feature_lists()
    assert not np.isinf(feat[num].to_numpy(dtype="float64")).any(), \
        "infinite values present"
    globals()["_FEAT"] = feat
    return f"{len(new)} engineered features, no infinities, row count preserved"


# ---------------------------------------------------- 3. no data leakage ----
@check("3. Leakage guard (identifier absent, target excluded from X)")
def _test_leakage():
    from src.feature_engineering import get_feature_lists

    num, cat = get_feature_lists()
    cols = num + cat
    assert config.TARGET not in cols, "target leaked into the feature list"
    assert "ID" not in cols, "identifier leaked into the feature list"
    return f"{len(cols)} predictors, target and ID both excluded"


# --------------------------------------------------------- 4. split -------
@check("4. Stratified train/test split")
def _test_split():
    from sklearn.model_selection import train_test_split
    from src.feature_engineering import get_feature_lists

    num, cat = get_feature_lists()
    X, y = _FEAT[num + cat], _FEAT[config.TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y,
        random_state=config.RANDOM_STATE,
    )
    assert len(X_tr) + len(X_te) == len(X), "split lost rows"
    assert abs(y_tr.mean() - y_te.mean()) < 0.01, "stratification failed"
    # Train and test indices must not overlap.
    assert not set(X_tr.index) & set(X_te.index), "train/test overlap"
    globals().update(_X_TR=X_tr, _X_TE=X_te, _Y_TR=y_tr, _Y_TE=y_te)
    return (f"train {len(X_tr):,} / test {len(X_te):,}, "
            f"default rate {y_tr.mean():.4f} vs {y_te.mean():.4f}, no overlap")


# --------------------------------------- 5. preprocessing fits per fold ----
@check("5. Preprocessing pipeline (imputation + scaling + one-hot)")
def _test_preprocessing():
    from src.preprocessing import build_preprocessor
    from src.feature_engineering import get_feature_lists

    num, cat = get_feature_lists()
    pre = build_preprocessor(num, cat, scale=True)
    Xt = pre.fit_transform(_X_TR.head(2000))
    assert not np.isnan(Xt).any(), "NaNs survived imputation"
    names = list(pre.get_feature_names_out())
    assert len(names) == Xt.shape[1], "feature-name/column mismatch"
    # Unseen category must not crash the transformer.
    probe = _X_TE.head(5).copy()
    probe.loc[probe.index[0], "EDUCATION"] = "brand_new_category"
    assert pre.transform(probe).shape[1] == Xt.shape[1], \
        "unknown category changed the output width"
    return f"{Xt.shape[1]} output columns, no NaNs, unknown categories handled"


# ------------------------------------------- 6. training + 7. CV works ----
@check("6. Model training (all five pipelines fit)")
def _test_training():
    from src.models import build_models
    from src.feature_engineering import get_feature_lists

    num, cat = get_feature_lists()
    sub_X, sub_y = _X_TR.head(3000), _Y_TR.head(3000)
    fitted = []
    for name, pipe in build_models(num, cat).items():
        pipe.fit(sub_X, sub_y)
        preds = pipe.predict(_X_TE.head(500))
        assert set(np.unique(preds)) <= {0, 1}, f"{name} produced bad labels"
        fitted.append(name)
    return f"fitted and predicted with: {', '.join(fitted)}"


@check("7. Cross-validation")
def _test_cv():
    from sklearn.model_selection import cross_val_score
    from src.models import build_models
    from src.feature_engineering import get_feature_lists

    num, cat = get_feature_lists()
    lr = build_models(num, cat)["Logistic Regression"]
    scores = cross_val_score(lr, _X_TR.head(4000), _Y_TR.head(4000),
                             cv=3, scoring="roc_auc")
    assert len(scores) == 3 and (scores > 0.6).all(), f"weak CV scores {scores}"
    return f"3-fold ROC-AUC = {np.round(scores, 4).tolist()}"


# ------------------------------------------------- 8. tuning machinery ----
@check("8. Hyper-parameter tuning")
def _test_tuning():
    from sklearn.model_selection import GridSearchCV
    from src.models import build_models
    from src.feature_engineering import get_feature_lists

    num, cat = get_feature_lists()
    dt = build_models(num, cat)["Decision Tree"]
    gs = GridSearchCV(dt, {"model__max_depth": [3, 5]}, cv=3, scoring="roc_auc")
    gs.fit(_X_TR.head(4000), _Y_TR.head(4000))
    assert "model__max_depth" in gs.best_params_
    return f"best_params={gs.best_params_}, CV ROC-AUC={gs.best_score_:.4f}"


# ---------------------------------------------------------- 9. metrics ----
@check("9. Evaluation metrics")
def _test_metrics():
    from src.evaluate import compute_metrics, best_threshold_by_cost

    rng = np.random.default_rng(config.RANDOM_STATE)
    y = rng.integers(0, 2, 1000)
    p = np.clip(y * 0.3 + rng.random(1000) * 0.7, 0, 1)
    m = compute_metrics(y, (p >= 0.5).astype(int), p)
    for key in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"):
        assert key in m and 0 <= m[key] <= 1, f"{key} invalid: {m.get(key)}"
    thr, cost = best_threshold_by_cost(y, p)
    assert 0.05 <= thr <= 0.95 and cost >= 0
    return f"6 metrics computed; cost-optimal threshold {thr:.3f}"


# ------------------------------------- 10. saved model loads and scores ----
@check("10. Model saving / reloading")
def _test_reload():
    from src.predict import load_model

    assert config.FINAL_MODEL_FILE.exists(), \
        f"no model at {config.FINAL_MODEL_FILE} -- run `python -m src.train`"
    bundle = load_model()
    for key in ("pipeline", "threshold", "model_name", "numeric_features",
                "categorical_features", "raw_input_columns"):
        assert key in bundle, f"bundle missing '{key}'"
    # The reloaded pipeline must reproduce predictions on real held-out rows.
    cols = bundle["numeric_features"] + bundle["categorical_features"]
    proba = bundle["pipeline"].predict_proba(_X_TE[cols].head(100))[:, 1]
    assert proba.shape == (100,) and ((proba >= 0) & (proba <= 1)).all()
    globals()["_BUNDLE"] = bundle
    return (f"'{bundle['model_name']}' reloaded, threshold "
            f"{bundle['threshold']:.3f}, scored 100 held-out rows")


# ------------------------------------------- 11. prediction on new data ----
@check("11. Prediction on brand-new raw applicants")
def _test_predict():
    from src.predict import example_applicants, predict_applicant, predict_batch

    examples = example_applicants()
    results = {k: predict_applicant(v, _BUNDLE) for k, v in examples.items()}
    for name, r in results.items():
        p = r["probability_of_default"]
        assert 0.0 <= p <= 1.0, f"{name}: probability out of range"
        assert abs(p + r["probability_of_good_credit"] - 1.0) < 1e-6, \
            "probabilities do not sum to 1"

    low = results["Applicant A -- strong payer"]["probability_of_default"]
    high = results["Applicant B -- distressed borrower"]["probability_of_default"]
    assert high > low, (
        f"model ranks the distressed borrower ({high:.3f}) no riskier than "
        f"the strong payer ({low:.3f})"
    )

    batch = predict_batch(pd.DataFrame(list(examples.values())), _BUNDLE)
    assert len(batch) == 2 and "risk_category" in batch.columns

    # A missing required field must raise a clear error rather than silently
    # scoring a malformed application.
    broken = dict(list(examples.values())[0])
    broken.pop("LIMIT_BAL")
    try:
        predict_applicant(broken, _BUNDLE)
    except ValueError:
        pass
    else:
        raise AssertionError("missing-field validation did not trigger")

    return (f"P(default) = {low:.3f} (good payer) vs {high:.3f} (distressed); "
            "batch + input validation OK")


# ------------------------------------------------- 12. artefacts on disk ----
@check("12. Output artefacts exist")
def _test_artifacts():
    figures = sorted(p.name for p in config.FIGURES_DIR.glob("*.png"))
    results = sorted(p.name for p in config.RESULTS_DIR.glob("*.csv"))
    assert len(figures) >= 8, f"expected >=8 figures, found {len(figures)}"
    assert len(results) >= 5, f"expected >=5 result tables, found {len(results)}"
    return f"{len(figures)} figures, {len(results)} result tables"


# ------------------------------------------------- 13. web dashboard -------
@check("13. Web dashboard routes")
def _test_webapp():
    """Exercise the Flask app through its test client (no server needed)."""
    from src.webapp import app
    from src.predict import example_applicants

    client = app.test_client()

    page = client.get("/")
    assert page.status_code == 200, f"GET / returned {page.status_code}"
    html = page.get_data(as_text=True)
    for token in ("Score an Applicant", "Confusion matrix", "n_delinquent_months"):
        assert token in html, f"dashboard is missing {token!r}"

    png = client.get("/figures/roc_curves.png")
    assert png.status_code == 200, "figure route failed"
    # PNG magic number: 0x89 followed by "PNG".
    assert png.data[0] == 0x89 and png.data[1:4] == b"PNG", "not a PNG"
    assert client.get("/figures/../config.py").status_code == 404, (
        "path traversal was not blocked")

    sample = client.get("/api/sample?kind=risky").get_json()
    assert sample["actual_outcome"] == "defaulted", "risky sample is not a defaulter"

    good, bad = example_applicants().values()
    p_good = client.post("/api/score", json=good).get_json()
    p_bad = client.post("/api/score", json=bad).get_json()
    assert p_good["ok"] and p_bad["ok"], "scoring endpoint failed"
    assert p_bad["probability_of_default"] > p_good["probability_of_default"],         "dashboard ranks the distressed borrower no riskier than the strong payer"
    assert p_bad["indicators"], "no risk indicators returned"

    invalid = client.post("/api/score", json={})
    assert invalid.status_code == 400, "missing fields were not rejected"
    assert len(invalid.get_json()["errors"]) == 23, "expected one error per field"

    return (f"page + figures + samples OK; scored "
            f"{p_good['percent_default']}% vs {p_bad['percent_default']}%; "
            "validation and traversal guard OK")


# --------------------------------------------------------------- report ----
print("\n" + "=" * 70)
print(f"RESULT: {len(PASSED)} passed, {len(FAILED)} failed")
print("=" * 70)
if FAILED:
    for name, tb in FAILED:
        print(f"\n--- {name} ---\n{tb}")
    sys.exit(1)
print("All checks passed.")
