"""Preprocessing pipelines for both tasks.

Design notes (these become §8 of the report):

* **Crop recommendation** has no categorical columns and no missing values, so the
  only preprocessing is standardisation - and only for the models that need it
  (Logistic Regression). Trees are scale-invariant, so scaling them would be
  noise dressed up as a step.
* **Yield forecasting** has two high-cardinality categorical columns (700+
  districts, 115 crops). One-hot encoding them would produce a very wide,
  mostly empty matrix that does not fit comfortably in memory. We use
  scikit-learn's ``TargetEncoder`` instead: it replaces each category with a
  cross-fitted mean of the target, which is compact and works for every model
  in the comparison.

  The ``cv=5`` argument is the important part. It means the encoding for a row is
  learned from *other* folds, so the target never sees itself. An encoder fitted
  on the full training set and then scored on the same data leaks badly - this is
  the most common mistake in tabular projects that use target encoding at all.

  Area spans 0.5 to 1e6 hectares, so it is log-transformed inside the pipeline
  (a one-to-one transformer keeps the column name "Area" for SHAP labels) and
  then standardised for the linear baseline. Trees are unaffected by the affine
  part; the log part genuinely helps every model.

Every transformer lives inside a ``Pipeline``, so it is fitted on training folds
only and serialised together with the model. There is no separate "prepared"
CSV that could drift out of sync with what the model actually saw.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, TargetEncoder

from src import config


# ------------------------------------------------------------------ task 1
def rec_preprocessor(*, scale: bool) -> ColumnTransformer:
    """Numeric-only preprocessing. Scaling is applied only where it matters."""
    return ColumnTransformer(
        [("scale", StandardScaler(), config.REC_FEATURES)],
        remainder="drop",
        verbose_feature_names_out=False,
    ) if scale else "passthrough"


def rec_pipeline(model, *, scale: bool) -> Pipeline:
    return Pipeline([("prep", rec_preprocessor(scale=scale)), ("model", model)])


def split_rec(df: pd.DataFrame):
    """Stratified 80:20 split, so all 22 classes appear in both halves."""
    X = df[config.REC_FEATURES]
    y = df[config.REC_TARGET]
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED, stratify=y
    )


# ------------------------------------------------------------------ task 2
def _log1p_float(X):
    """log1p that survives a DataFrame built from a single row.

    ``series.to_frame().T`` (the way the app builds a one-row input) produces
    object-dtype columns, and numpy ufuncs refuse object arrays. Coercing to
    float first keeps training and serving on identical code. Module-level so
    the fitted pipeline pickles cleanly.
    """
    return np.log1p(np.asarray(X, dtype=float))


def yield_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "cat",
                TargetEncoder(
                    # cross-fitted: each row is encoded from the other folds only.
                    # A shuffled KFold (rather than a plain int) keeps the result
                    # deterministic and silences the 1.9 deprecation warning.
                    cv=KFold(n_splits=5, shuffle=True, random_state=config.SEED),
                    target_type="continuous",
                ),
                config.PROD_CATEGORICAL,
            ),
            (
                "num",
                Pipeline(
                    [
                        ("log", FunctionTransformer(_log1p_float, feature_names_out="one-to-one")),
                        ("scale", StandardScaler()),
                    ]
                ),
                config.PROD_NUMERIC,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def yield_pipeline(model) -> Pipeline:
    return Pipeline([("prep", yield_preprocessor()), ("model", model)])


def split_yield(df: pd.DataFrame):
    """Split the yield task and return the *log-transformed* target.

    Raw yield spans 0-250 t/ha with a median near 1: a handful of sugarcane
    rows would otherwise dominate squared error and R². We model
    ``log1p(yield)`` and invert with ``expm1`` when reporting errors in t/ha.
    """
    X = df[config.PROD_CATEGORICAL + config.PROD_NUMERIC]
    y = np.log1p(df[config.YIELD_TARGET].to_numpy())
    return train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.SEED)


def split_yield_temporal(df: pd.DataFrame, cutoff_year: int | None = None):
    """Honest out-of-time split: train on years <= cutoff, test on later years.

    The random split scores a model that has already seen the test year's other
    districts. A temporal split answers the question the app actually asks:
    can the model forecast a year it has never seen?
    """
    if cutoff_year is None:
        cutoff_year = int(df["Crop_Year"].quantile(0.8))
    train_mask = df["Crop_Year"] <= cutoff_year
    X_tr = df.loc[train_mask, config.PROD_CATEGORICAL + config.PROD_NUMERIC]
    y_tr = np.log1p(df.loc[train_mask, config.YIELD_TARGET].to_numpy())
    X_te = df.loc[~train_mask, config.PROD_CATEGORICAL + config.PROD_NUMERIC]
    y_te = np.log1p(df.loc[~train_mask, config.YIELD_TARGET].to_numpy())
    return X_tr, X_te, y_tr, y_te, cutoff_year


def split_calibration(X: pd.DataFrame, y: np.ndarray):
    """Carve a calibration set for split-conformal intervals.

    The final model is scored on rows it never saw *and* whose residuals were
    never used to fit it; the calibration residuals then set the interval width.
    """
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X, y, test_size=config.CONFORMAL_CALIB_FRACTION, random_state=config.SEED
    )
    return X_fit, X_cal, y_fit, y_cal


def inverse_target(y_log):
    """Map predictions back to tonnes/hectare."""
    return np.expm1(y_log)
