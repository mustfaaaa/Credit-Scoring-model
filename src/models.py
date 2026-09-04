"""Model zoo and hyper-parameter search spaces (STEPS 8-13).

Each entry is a complete ``Pipeline`` (preprocessing + estimator) so that a
model can never be fitted on pre-transformed data by accident.
"""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE
from src.preprocessing import build_preprocessor


def _pipe(numeric, categorical, estimator, scale):
    return Pipeline(
        [
            ("preprocess", build_preprocessor(numeric, categorical, scale=scale)),
            ("model", estimator),
        ]
    )


def build_models(
    numeric: list[str],
    categorical: list[str],
    balanced: bool = True,
) -> dict[str, Pipeline]:
    """Return the candidate models.

    ``balanced=True`` applies ``class_weight="balanced"`` to every model that
    supports it, which re-weights the 22% minority class during training.
    Running with ``balanced=False`` gives the before/after imbalance comparison.
    """
    cw = "balanced" if balanced else None

    return {
        # STEP 8 -- statistical floor: always predicts the majority class.
        "Baseline (Majority)": _pipe(
            numeric, categorical,
            DummyClassifier(strategy="most_frequent"), scale=False,
        ),
        # STEP 9 -- linear, fully interpretable, the industry standard for
        # scorecards because each coefficient is a defensible risk weight.
        "Logistic Regression": _pipe(
            numeric, categorical,
            LogisticRegression(
                max_iter=2000, class_weight=cw, random_state=RANDOM_STATE,
            ),
            scale=True,
        ),
        # STEP 10 -- captures non-linear rules; depth-capped to fight the
        # severe overfitting an unpruned tree shows on 30k rows.
        "Decision Tree": _pipe(
            numeric, categorical,
            DecisionTreeClassifier(
                max_depth=5, min_samples_leaf=50,
                class_weight=cw, random_state=RANDOM_STATE,
            ),
            scale=False,
        ),
        # STEP 11 -- bagged trees, strong default for tabular credit data.
        "Random Forest": _pipe(
            numeric, categorical,
            RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                class_weight=cw, random_state=RANDOM_STATE,
            ),
            scale=False,
        ),
        # STEP 12 -- boosting is the strongest known family for tabular data.
        # sklearn ships it, so this adds capability without a new dependency.
        "HistGradientBoosting": _pipe(
            numeric, categorical,
            HistGradientBoostingClassifier(
                class_weight=cw, random_state=RANDOM_STATE,
            ),
            scale=False,
        ),
    }


# ------------------------------------------------------- search spaces ------
# Deliberately compact grids: enough to matter, small enough to run in minutes.
PARAM_DISTRIBUTIONS: dict[str, dict] = {
    "Logistic Regression": {
        # `penalty` is deliberately NOT tuned: it is deprecated in
        # scikit-learn 1.8 and removed in 1.10.  L2 is the default, so tuning
        # C alone covers the same regularisation strength.
        "model__C": [0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
        "model__solver": ["lbfgs", "liblinear"],
    },
    "Decision Tree": {
        "model__max_depth": [3, 4, 5, 6, 8, 10],
        "model__min_samples_leaf": [20, 50, 100, 200],
        "model__criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "model__n_estimators": [200, 400, 600],
        "model__max_depth": [8, 12, 16, None],
        "model__min_samples_leaf": [1, 5, 10, 20],
        "model__max_features": ["sqrt", 0.3, 0.5],
    },
    "HistGradientBoosting": {
        "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [20, 50, 100],
        "model__l2_regularization": [0.0, 0.5, 1.0],
        "model__max_iter": [150, 300],
    },
}

TUNABLE = list(PARAM_DISTRIBUTIONS)
