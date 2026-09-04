"""End-to-end training pipeline (STEPS 6, 8-19 of the roadmap).

Run with::

    python -m src.train

Everything written to ``outputs/`` is produced by this script from executed
code -- no metric in the report is typed by hand.

Guard-rails enforced here
-------------------------
* The test set is created once by a stratified split and is **never** used for
  model selection, hyper-parameter tuning or threshold selection.
* All preprocessing lives inside the Pipeline, so imputation/scaling/encoding
  statistics are fitted per CV fold, never on held-out data.
* The decision threshold is chosen from *out-of-fold* training predictions.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)

from src import config, eda, evaluate
from src.data_loader import BILL_COLS, PAY_AMT_COLS, PAY_COLS, load_clean
from src.feature_engineering import add_financial_features, get_feature_lists
from src.models import PARAM_DISTRIBUTIONS, TUNABLE, build_models

log = logging.getLogger("train")

# Raw fields a new applicant must supply (before feature engineering).
RAW_INPUT_COLS = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + PAY_COLS + BILL_COLS + PAY_AMT_COLS
)

# zero_division=0 mirrors evaluate.compute_metrics: the majority-class
# baseline predicts no positives, so precision is undefined and scored as 0
# rather than raising a warning.
_P = make_scorer(precision_score, zero_division=0)
_R = make_scorer(recall_score, zero_division=0)
_F1 = make_scorer(f1_score, zero_division=0)

SCORING = {
    "accuracy": "accuracy",
    "precision": _P,
    "recall": _R,
    "f1": _F1,
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}

# Number of random draws for the two ensemble searches (keeps runtime sane).
N_ITER = {"Random Forest": 12, "HistGradientBoosting": 15}

# --quick runs the identical code path on a subsample with smaller searches.
# It exists to smoke-test the pipeline end to end; it is NOT used for reported
# results, which always come from a full `python -m src.train` run.
QUICK = False


def _cv() -> StratifiedKFold:
    """Stratified folds preserve the 22% default rate inside every fold."""
    return StratifiedKFold(
        n_splits=3 if QUICK else config.CV_FOLDS,
        shuffle=True, random_state=config.RANDOM_STATE,
    )


def _banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


# --------------------------------------------------------------------------
# STEP 1-5: data
# --------------------------------------------------------------------------
def prepare_data():
    _banner("STEP 1-3 | LOAD AND CLEAN")
    df = load_clean()
    if QUICK:
        df = df.sample(n=4000, random_state=config.RANDOM_STATE).reset_index(drop=True)
        print("QUICK MODE: subsampled to 4,000 rows -- results are not reportable")
    print(f"Clean dataset            : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Missing values           : {int(df.isna().sum().sum())}")
    print(f"Duplicate rows remaining : {int(df.duplicated().sum())}")
    counts = df[config.TARGET].value_counts().sort_index()
    rate = counts[1] / counts.sum()
    print(f"Target distribution      : repaid={counts[0]:,}  "
          f"defaulted={counts[1]:,}  (default rate {rate:.2%})")
    print(f"Imbalance ratio          : {counts[0] / counts[1]:.2f} : 1")

    _banner("STEP 4 | EXPLORATORY DATA ANALYSIS")
    for path in eda.run_all(df):
        print("  figure ->", path)

    _banner("STEP 5 | FEATURE ENGINEERING")
    feat = add_financial_features(df)
    numeric, categorical = get_feature_lists()
    print(f"Raw predictors      : {len(RAW_INPUT_COLS)}")
    print(f"After engineering   : {len(numeric)} numeric + "
          f"{len(categorical)} categorical = {len(numeric) + len(categorical)}")
    feat.to_csv(config.PROCESSED_FILE, index=False)
    print("Processed dataset ->", config.PROCESSED_FILE)
    return feat, numeric, categorical


# --------------------------------------------------------------------------
# STEP 6: split
# --------------------------------------------------------------------------
def split_data(feat, numeric, categorical):
    _banner("STEP 6 | TRAIN / TEST SPLIT")
    X = feat[numeric + categorical]
    y = feat[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,                       # keeps the default rate identical
        random_state=config.RANDOM_STATE,
    )
    print(f"Train : {len(X_train):,} rows  (default rate {y_train.mean():.4f})")
    print(f"Test  : {len(X_test):,} rows  (default rate {y_test.mean():.4f})")
    print(f"Validation strategy : {config.CV_FOLDS}-fold stratified CV *inside* "
          "the training set")
    print("The test set is now sealed until STEP 14.")
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------
# STEP 8-11: cross-validated baseline comparison
# --------------------------------------------------------------------------
def cross_validate_models(models, X_train, y_train) -> pd.DataFrame:
    _banner("STEP 8-12 | CROSS-VALIDATED COMPARISON (training set only)")
    rows = {}
    for name, pipe in models.items():
        t0 = time.time()
        cv_res = cross_validate(
            pipe, X_train, y_train, cv=_cv(), scoring=SCORING, n_jobs=1,
            error_score="raise",
        )
        rows[name] = {
            "Accuracy": cv_res["test_accuracy"].mean(),
            "Precision": cv_res["test_precision"].mean(),
            "Recall": cv_res["test_recall"].mean(),
            "F1": cv_res["test_f1"].mean(),
            "ROC-AUC": cv_res["test_roc_auc"].mean(),
            "ROC-AUC_std": cv_res["test_roc_auc"].std(),
            "PR-AUC": cv_res["test_average_precision"].mean(),
            "fit_seconds": time.time() - t0,
        }
        r = rows[name]
        print(f"  {name:<24} ROC-AUC {r['ROC-AUC']:.4f} "
              f"(+/-{r['ROC-AUC_std']:.4f})  F1 {r['F1']:.4f}  "
              f"Recall {r['Recall']:.4f}  [{r['fit_seconds']:.0f}s]")

    table = pd.DataFrame(rows).T.sort_values("ROC-AUC", ascending=False)
    table.to_csv(config.RESULTS_DIR / "cv_comparison.csv")
    return table


# --------------------------------------------------------------------------
# STEP 11 (imbalance): class_weight on vs off
# --------------------------------------------------------------------------
def imbalance_study(numeric, categorical, X_train, y_train) -> pd.DataFrame:
    _banner("PHASE 11 | CLASS-IMBALANCE MITIGATION STUDY")
    print("Comparing class_weight='balanced' against no re-weighting, "
          "5-fold CV on the training set.\n")
    subset = ["Logistic Regression", "Random Forest", "HistGradientBoosting"]
    rows = []
    for balanced in (False, True):
        models = build_models(numeric, categorical, balanced=balanced)
        for name in subset:
            cv_res = cross_validate(
                models[name], X_train, y_train, cv=_cv(),
                scoring={"recall": _R, "precision": _P,
                         "f1": _F1, "roc_auc": "roc_auc"},
                n_jobs=1, error_score="raise",
            )
            rows.append({
                "Model": name,
                "class_weight": "balanced" if balanced else "none",
                "Precision": cv_res["test_precision"].mean(),
                "Recall": cv_res["test_recall"].mean(),
                "F1": cv_res["test_f1"].mean(),
                "ROC-AUC": cv_res["test_roc_auc"].mean(),
            })
            r = rows[-1]
            print(f"  {name:<24} {r['class_weight']:<9} "
                  f"Recall {r['Recall']:.4f}  Precision {r['Precision']:.4f}  "
                  f"F1 {r['F1']:.4f}  ROC-AUC {r['ROC-AUC']:.4f}")

    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS_DIR / "imbalance_study.csv", index=False)
    return table


# --------------------------------------------------------------------------
# STEP 13: hyper-parameter tuning
# --------------------------------------------------------------------------
def tune_models(models, X_train, y_train):
    _banner("STEP 13 | HYPER-PARAMETER TUNING (CV on the training set only)")
    tuned, records = {}, []

    for name in TUNABLE:
        grid = PARAM_DISTRIBUTIONS[name]
        n_combos = int(np.prod([len(v) for v in grid.values()]))
        t0 = time.time()

        if name in N_ITER:
            n_iter = 3 if QUICK else N_ITER[name]
            search = RandomizedSearchCV(
                models[name], grid, n_iter=n_iter, cv=_cv(),
                scoring="roc_auc", random_state=config.RANDOM_STATE,
                n_jobs=1, refit=True, error_score="raise",
            )
            tested = n_iter
            kind = f"RandomizedSearchCV ({tested}/{n_combos} combos)"
        else:
            search = GridSearchCV(
                models[name], grid, cv=_cv(), scoring="roc_auc",
                n_jobs=1, refit=True, error_score="raise",
            )
            tested = n_combos
            kind = f"GridSearchCV ({n_combos} combos)"

        search.fit(X_train, y_train)
        tuned[name] = search.best_estimator_
        elapsed = time.time() - t0
        records.append({
            "Model": name, "Search": kind, "Combinations_tested": tested,
            "Best_CV_ROC_AUC": search.best_score_,
            "Best_params": json.dumps(search.best_params_),
            "Seconds": round(elapsed, 1),
        })
        print(f"  {name}")
        print(f"    {kind} -> best CV ROC-AUC {search.best_score_:.4f} "
              f"[{elapsed:.0f}s]")
        print(f"    best params: {search.best_params_}")

    table = pd.DataFrame(records).sort_values("Best_CV_ROC_AUC", ascending=False)
    table.to_csv(config.RESULTS_DIR / "tuning_results.csv", index=False)
    return tuned, table


# --------------------------------------------------------------------------
# STEP 14: test-set evaluation
# --------------------------------------------------------------------------
def evaluate_on_test(estimators, X_test, y_test, threshold_map=None):
    _banner("STEP 14 | FINAL EVALUATION ON THE HELD-OUT TEST SET")
    results = {}
    for name, est in estimators.items():
        proba = (est.predict_proba(X_test)[:, 1]
                 if hasattr(est, "predict_proba") else None)
        thr = (threshold_map or {}).get(name, 0.5)
        pred = (proba >= thr).astype(int) if proba is not None else est.predict(X_test)
        results[name] = {
            "y_true": np.asarray(y_test), "y_pred": pred, "y_proba": proba,
            "threshold": thr,
            "metrics": evaluate.compute_metrics(y_test, pred, proba),
        }
        m = results[name]["metrics"]
        print(f"  {name:<24} Acc {m['Accuracy']:.4f}  Prec {m['Precision']:.4f}  "
              f"Rec {m['Recall']:.4f}  F1 {m['F1']:.4f}  "
              f"ROC-AUC {m['ROC-AUC']:.4f}")
    return results


def results_table(results) -> pd.DataFrame:
    return pd.DataFrame(
        {n: r["metrics"] for n, r in results.items()}
    ).T[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]]


# --------------------------------------------------------------------------
# STEP 16: interpretability
# --------------------------------------------------------------------------
def interpret(best_name, best_pipe, X_train, X_test, y_test):
    _banner("STEP 16 | FEATURE IMPORTANCE / INTERPRETABILITY")
    pre = best_pipe.named_steps["preprocess"]
    feature_names = list(pre.get_feature_names_out())
    model = best_pipe.named_steps["model"]
    frames = []

    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        frames.append(pd.DataFrame(
            {"feature": feature_names, "importance": imp, "type": "impurity"}
        ))
        evaluate.plot_feature_importance(
            feature_names, imp,
            f"{best_name}: impurity-based feature importance (top 20)",
            "feature_importance_native.png",
        )
    elif hasattr(model, "coef_"):
        coef = model.coef_.ravel()
        frames.append(pd.DataFrame(
            {"feature": feature_names, "importance": np.abs(coef),
             "coefficient": coef, "type": "abs_coefficient"}
        ))
        evaluate.plot_feature_importance(
            feature_names, np.abs(coef),
            f"{best_name}: |coefficient| (top 20)",
            "feature_importance_native.png",
        )
    else:
        # HistGradientBoosting exposes neither feature_importances_ nor coef_,
        # so permutation importance below is the only importance measure.
        print(f"  {best_name} has no built-in importance attribute; "
              "relying on permutation importance only.")

    # Permutation importance is model-agnostic and measures the real drop in
    # ROC-AUC when a column is shuffled. Computed on the TEST set for reporting
    # only -- it never feeds back into model selection.
    print("  computing permutation importance (this takes a minute) ...")
    perm = permutation_importance(
        best_pipe, X_test, y_test, n_repeats=2 if QUICK else 5,
        scoring="roc_auc",
        random_state=config.RANDOM_STATE, n_jobs=1,
    )
    perm_df = pd.DataFrame({
        "feature": list(X_test.columns),
        "importance": perm.importances_mean,
        "std": perm.importances_std,
        "type": "permutation_roc_auc_drop",
    }).sort_values("importance", ascending=False)

    evaluate.plot_feature_importance(
        perm_df["feature"].to_numpy(), perm_df["importance"].to_numpy(),
        f"{best_name}: permutation importance (drop in test ROC-AUC, top 20)",
        "feature_importance_permutation.png",
    )

    print("\n  Top 12 features by permutation importance:")
    for _, row in perm_df.head(12).iterrows():
        print(f"    {row['feature']:<26} {row['importance']:+.5f} "
              f"(+/-{row['std']:.5f})")

    out = pd.concat(frames + [perm_df], ignore_index=True)
    out.to_csv(config.RESULTS_DIR / "feature_importance.csv", index=False)
    return perm_df


# --------------------------------------------------------------------------
# STEP 17: error analysis
# --------------------------------------------------------------------------
def error_analysis(X_test, y_test, y_pred, y_proba):
    _banner("STEP 17 | ERROR ANALYSIS")
    y_true = np.asarray(y_test)
    groups = {
        "True Negative (correctly approved)": (y_true == 0) & (y_pred == 0),
        "False Positive (good customer rejected)": (y_true == 0) & (y_pred == 1),
        "False Negative (defaulter approved)": (y_true == 1) & (y_pred == 0),
        "True Positive (defaulter caught)": (y_true == 1) & (y_pred == 1),
    }
    cols = ["LIMIT_BAL", "AGE", "PAY_0", "n_delinquent_months",
            "avg_utilization", "avg_pay_ratio", "max_delinquency"]

    rows = []
    for label, mask in groups.items():
        profile = {"group": label, "count": int(mask.sum()),
                   "share": mask.mean(),
                   "mean_predicted_prob": float(np.mean(y_proba[mask]))}
        profile.update({c: float(X_test.loc[mask, c].mean()) for c in cols})
        rows.append(profile)

    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS_DIR / "error_analysis.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(3).to_string(index=False))

    fn = groups["False Negative (defaulter approved)"]
    fp = groups["False Positive (good customer rejected)"]
    print(f"\n  False negatives are {fn.sum():,} customers who defaulted but "
          f"were approved.")
    print(f"  Their mean predicted probability is "
          f"{np.mean(y_proba[fn]):.3f} -- close to the threshold, i.e. the "
          "model was uncertain rather than confidently wrong.")
    print(f"  Mean delinquent months: FN={X_test.loc[fn, 'n_delinquent_months'].mean():.2f} "
          f"vs FP={X_test.loc[fp, 'n_delinquent_months'].mean():.2f}")
    return table


# --------------------------------------------------------------------------
# Fairness sensitivity check
# --------------------------------------------------------------------------
def fairness_check(best_name, numeric, categorical, X_train, y_train,
                   X_test, y_test, tuned_params):
    _banner("FAIRNESS SENSITIVITY | does dropping SEX cost predictive power?")
    from sklearn.base import clone

    numeric_ns, categorical_ns = get_feature_lists(include_sex=False)
    base = build_models(numeric_ns, categorical_ns, balanced=True)[best_name]
    base.set_params(**{k: v for k, v in tuned_params.items()})

    cols = numeric_ns + categorical_ns
    est = clone(base).fit(X_train[cols], y_train)
    proba = est.predict_proba(X_test[cols])[:, 1]
    from sklearn.metrics import roc_auc_score

    auc_without = roc_auc_score(y_test, proba)
    print(f"  Test ROC-AUC without SEX : {auc_without:.4f}")

    table = pd.DataFrame([{"variant": "without_SEX", "test_roc_auc": auc_without}])
    table.to_csv(config.RESULTS_DIR / "fairness_sensitivity.csv", index=False)
    return auc_without


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    # matplotlib logs an INFO line for every categorical axis; keep the run log
    # focused on our own progress messages.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    np.random.seed(config.RANDOM_STATE)
    started = time.time()

    feat, numeric, categorical = prepare_data()
    X_train, X_test, y_train, y_test = split_data(feat, numeric, categorical)

    models = build_models(numeric, categorical, balanced=True)
    cv_table = cross_validate_models(models, X_train, y_train)
    imbalance_study(numeric, categorical, X_train, y_train)

    tuned, tuning_table = tune_models(models, X_train, y_train)

    # ---- STEP 18: pick the winner on CROSS-VALIDATED score, not on test ----
    _banner("STEP 18 | FINAL MODEL SELECTION (decided on CV, before touching test)")
    best_name = tuning_table.iloc[0]["Model"]
    best_pipe = tuned[best_name]
    best_params = json.loads(tuning_table.iloc[0]["Best_params"])
    print(f"Winner by CV ROC-AUC : {best_name} "
          f"({tuning_table.iloc[0]['Best_CV_ROC_AUC']:.4f})")

    # ---- Threshold from OUT-OF-FOLD training predictions -------------------
    _banner("DECISION THRESHOLD | chosen on out-of-fold TRAINING predictions")
    oof = cross_val_predict(
        best_pipe, X_train, y_train, cv=_cv(), method="predict_proba", n_jobs=1
    )[:, 1]
    threshold, oof_cost = evaluate.best_threshold_by_cost(y_train, oof)
    cost_at_half = evaluate.expected_cost(y_train, (oof >= 0.5).astype(int))
    print(f"Cost model            : FN costs {config.COST_FN:.0f}x, "
          f"FP costs {config.COST_FP:.0f}x")
    print(f"Cost at threshold 0.50: {cost_at_half:,.0f}")
    print(f"Optimal threshold     : {threshold:.3f}  (cost {oof_cost:,.0f}, "
          f"{(1 - oof_cost / cost_at_half) * 100:.1f}% lower)")
    evaluate.plot_threshold_analysis(y_train, oof, threshold)

    # ---- STEP 14/15: evaluate everything on the sealed test set ------------
    final_estimators = {"Baseline (Majority)": models["Baseline (Majority)"].fit(
        X_train, y_train)}
    final_estimators.update(tuned)
    results_default = evaluate_on_test(final_estimators, X_test, y_test)
    table_default = results_table(results_default)
    table_default.to_csv(config.RESULTS_DIR / "test_comparison.csv")

    _banner("STEP 15 | MODEL COMPARISON TABLE (test set, threshold = 0.50)")
    print(table_default.round(4).to_string())

    # Best model re-scored at the cost-optimal threshold.
    proba_best = best_pipe.predict_proba(X_test)[:, 1]
    pred_tuned = (proba_best >= threshold).astype(int)
    metrics_tuned = evaluate.compute_metrics(y_test, pred_tuned, proba_best)
    print(f"\n{best_name} at the cost-optimal threshold {threshold:.3f}:")
    for k, v in metrics_tuned.items():
        print(f"  {k:<10} {v:.4f}")

    combined = table_default.copy()
    combined.loc[f"{best_name} @ thr={threshold:.2f}"] = pd.Series(metrics_tuned)
    combined.to_csv(config.RESULTS_DIR / "test_comparison.csv")

    # ---- figures ----------------------------------------------------------
    evaluate.plot_confusion_matrices(results_default)
    evaluate.plot_roc_curves(results_default)
    evaluate.plot_pr_curves(results_default, y_test)
    evaluate.plot_model_comparison(table_default)

    # ---- STEP 16-17 -------------------------------------------------------
    perm_df = interpret(best_name, best_pipe, X_train, X_test, y_test)
    err_table = error_analysis(X_test, y_test, pred_tuned, proba_best)
    fairness_check(best_name, numeric, categorical, X_train, y_train,
                   X_test, y_test, best_params)

    # ---- STEP 19: persist -------------------------------------------------
    _banner("STEP 19 | SAVING THE FINAL MODEL")
    bundle = {
        "pipeline": best_pipe,              # preprocessing + estimator together
        "threshold": threshold,
        "model_name": best_name,
        "best_params": best_params,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "raw_input_columns": RAW_INPUT_COLS,
        "test_metrics_at_threshold": metrics_tuned,
        "test_metrics_at_0.5": results_default[best_name]["metrics"],
        "cv_roc_auc": float(tuning_table.iloc[0]["Best_CV_ROC_AUC"]),
        "cost_fn": config.COST_FN,
        "cost_fp": config.COST_FP,
        "random_state": config.RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
        "trained_at": dt.datetime.now().isoformat(timespec="seconds"),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    joblib.dump(bundle, config.FINAL_MODEL_FILE)
    print("Saved bundle (pipeline + threshold + metadata) ->",
          config.FINAL_MODEL_FILE)

    summary = {k: v for k, v in bundle.items() if k != "pipeline"}
    summary["top_features"] = perm_df.head(10)[["feature", "importance"]].to_dict(
        "records")
    (config.RESULTS_DIR / "final_model_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")

    _banner(f"PIPELINE COMPLETE in {time.time() - started:.0f}s")
    return bundle


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the credit scoring models.")
    parser.add_argument(
        "--quick", action="store_true",
        help="smoke-test the full pipeline on a 4k subsample with small searches",
    )
    args = parser.parse_args()
    QUICK = args.quick
    main()
