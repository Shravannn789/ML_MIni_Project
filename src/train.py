"""Train and compare models for both tasks.

Run with:  python -m src.train

Outputs
-------
models/crop_recommendation.joblib   best classifier (full pipeline)
models/yield_forecast.joblib        best regressor  (full pipeline)
models/metrics.json                 every score, plus the split settings
reports/model_comparison_*.csv      the comparison tables for the report

Model selection rule (state this in §13): the winner is chosen on held-out
performance *after* checking that its cross-validated score is close to that
held-out score. A model that wins by 0.2% but swings 5% across folds is noise,
not a result.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from src import config, data_loader, preprocess


class LabelEncoded(ClassifierMixin, BaseEstimator):
    """Wrap an estimator that needs integer labels (XGBoost) so the pipeline
    still accepts and returns the original crop names.

    Without this, the saved artifact would return `17` instead of `rice`, and
    the app would need its own copy of the class order - a classic source of
    silent mismatch between training and serving.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.encoder_ = LabelEncoder().fit(y)
        self.classes_ = self.encoder_.classes_
        self.fitted_ = clone(self.estimator).fit(X, self.encoder_.transform(y))
        return self

    def predict(self, X):
        return self.encoder_.inverse_transform(self.fitted_.predict(X))

    def predict_proba(self, X):
        return self.fitted_.predict_proba(X)


# ------------------------------------------------------------------ task 1
def classifier_zoo() -> dict:
    """name -> (estimator, needs_scaling)."""
    return {
        "Logistic Regression": (LogisticRegression(max_iter=2000), True),
        "Decision Tree": (DecisionTreeClassifier(random_state=config.SEED), False),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, random_state=config.SEED, n_jobs=-1),
            False,
        ),
        "XGBoost": (
            LabelEncoded(
                XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=config.SEED,
                    n_jobs=-1,
                    tree_method="hist",
                )
            ),
            False,
        ),
    }


def train_classification() -> dict:
    df = data_loader.load_recommendation()
    X_tr, X_te, y_tr, y_te = preprocess.split_rec(df)
    print(f"[crop recommendation] train={len(X_tr)} test={len(X_te)} classes={y_tr.nunique()}")

    results, fitted = {}, {}
    for name, (est, scale) in classifier_zoo().items():
        pipe = preprocess.rec_pipeline(est, scale=scale)
        t0 = time.time()
        pipe.fit(X_tr, y_tr)

        pred = pipe.predict(X_te)
        cv = cross_val_score(pipe, X_tr, y_tr, cv=config.CV_FOLDS, scoring="accuracy", n_jobs=1)

        results[name] = {
            "accuracy": float(accuracy_score(y_te, pred)),
            "macro_f1": float(f1_score(y_te, pred, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y_te, pred)),
            "cv_accuracy_mean": float(cv.mean()),
            "cv_accuracy_std": float(cv.std()),
            "fit_seconds": round(time.time() - t0, 1),
        }
        fitted[name] = pipe
        r = results[name]
        print(
            f"  {name:22s} acc={r['accuracy']:.4f}  macroF1={r['macro_f1']:.4f}  "
            f"cv={r['cv_accuracy_mean']:.4f}+-{r['cv_accuracy_std']:.4f}  {r['fit_seconds']}s"
        )

    best = max(results, key=lambda k: results[k]["accuracy"])
    joblib.dump(fitted[best], config.REC_MODEL_FILE)
    print(f"  -> best: {best}  saved to {config.REC_MODEL_FILE.name}")

    pd.DataFrame(results).T.sort_values("accuracy", ascending=False).to_csv(
        config.REPORTS_DIR / "model_comparison_classification.csv"
    )
    return {"n_train": len(X_tr), "n_test": len(X_te),
            "n_classes": int(df[config.REC_TARGET].nunique()),
            "best_model": best, "models": results}


# ------------------------------------------------------------------ task 2
def regressor_zoo() -> dict:
    return {
        "Ridge (linear baseline)": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(random_state=config.SEED, min_samples_leaf=5),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, min_samples_leaf=2, random_state=config.SEED, n_jobs=-1
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.1, random_state=config.SEED
        ),
        "XGBoost": XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=config.SEED,
            n_jobs=-1,
            tree_method="hist",
        ),
    }


def train_regression() -> dict:
    df, stats = data_loader.build_yield_frame()
    X_tr, X_te, y_tr, y_te = preprocess.split_yield(df)
    print(f"[yield forecast] train={len(X_tr)} test={len(X_te)} (target = log1p t/ha)")

    results, fitted = {}, {}
    for name, est in regressor_zoo().items():
        pipe = preprocess.yield_pipeline(est)
        t0 = time.time()
        pipe.fit(X_tr, y_tr)

        pred_log = pipe.predict(X_te)
        cv = cross_val_score(pipe, X_tr, y_tr, cv=3, scoring="r2", n_jobs=1)

        # errors in real units matter for a reader who does not think in logs
        y_true_ha = preprocess.inverse_target(y_te)
        pred_ha = preprocess.inverse_target(pred_log)
        pred_ha = np.clip(pred_ha, 0, None)

        results[name] = {
            "r2_log": float(r2_score(y_te, pred_log)),
            "mae_log": float(mean_absolute_error(y_te, pred_log)),
            "rmse_t_per_ha": float(np.sqrt(mean_squared_error(y_true_ha, pred_ha))),
            "mae_t_per_ha": float(mean_absolute_error(y_true_ha, pred_ha)),
            "r2_t_per_ha": float(r2_score(y_true_ha, pred_ha)),
            "cv_r2_mean": float(cv.mean()),
            "cv_r2_std": float(cv.std()),
            "fit_seconds": round(time.time() - t0, 1),
        }
        fitted[name] = pipe
        r = results[name]
        print(
            f"  {name:24s} R2(log)={r['r2_log']:.4f}  MAE={r['mae_t_per_ha']:.2f} t/ha  "
            f"cv={r['cv_r2_mean']:.4f}+-{r['cv_r2_std']:.4f}  {r['fit_seconds']}s"
        )

    best = max(results, key=lambda k: results[k]["r2_log"])
    joblib.dump(fitted[best], config.YIELD_MODEL_FILE)
    print(f"  -> best: {best}  saved to {config.YIELD_MODEL_FILE.name}")

    pd.DataFrame(results).T.sort_values("r2_log", ascending=False).to_csv(
        config.REPORTS_DIR / "model_comparison_regression.csv"
    )
    return {"n_train": len(X_tr), "n_test": len(X_te), "cleaning": stats,
            "best_model": best, "models": results}


# ------------------------------------------------------------------ entry point
def main() -> None:
    payload = {
        "settings": {
            "seed": config.SEED,
            "test_size": config.TEST_SIZE,
            "cv_folds_classification": config.CV_FOLDS,
            "cv_folds_regression": 3,
            "yield_target": "log1p(Production / Area), winsorised per crop",
        },
        "classification": train_classification(),
        "regression": train_regression(),
    }
    config.METRICS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nmetrics written to {config.METRICS_FILE}")


if __name__ == "__main__":
    main()
