"""Leakage-free preprocessing pipeline (STEP 7 of the roadmap).

The ColumnTransformer is *not* applied to the raw dataframe up-front.  It is
embedded inside every model Pipeline, so imputation medians, scaler statistics
and one-hot categories are learned **only from the training fold** and then
applied to validation/test data.  This is what prevents information from the
held-out data leaking into training.
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale: bool = True,
) -> ColumnTransformer:
    """Build the ColumnTransformer.

    Parameters
    ----------
    scale:
        ``True`` for distance/gradient based models (Logistic Regression),
        which need comparable feature magnitudes.  ``False`` for tree
        ensembles, which are invariant to monotone rescaling -- skipping the
        scaler there keeps the fitted trees interpretable in original units.
    """
    numeric_steps: list = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # handle_unknown="ignore" keeps the model usable if a new applicant
            # arrives with a category never seen during training.
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",          # anything not listed is deliberately excluded
        verbose_feature_names_out=False,
    )


def get_output_feature_names(fitted_preprocessor: ColumnTransformer) -> list[str]:
    """Feature names after transformation (needed for importance plots)."""
    return list(fitted_preprocessor.get_feature_names_out())
