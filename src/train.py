"""Train and compare models for both tasks.

Run with:  python -m src.train

Outputs
-------
models/crop_recommendation.joblib   best classifier (full pipeline)
models/yield_forecast.joblib        best regressor  (full pipeline, conformal)
models/yield_conformal.json         interval half-width on the log target
models/metrics.json                 every score, plus the split settings
reports/model_comparison_*.csv      the comparison tables for the report

Three evaluation lenses for the yield model, in increasing harshness:

1. **Random 80:20 split** - standard, but the model has seen other rows from
   the test year, so it flatters performance.
2. **Temporal split** - train on years <= cutoff, test on later years. This is
   what the app does when a user forecasts a future season.
3. **Split-conformal calibration** - the shipped model is fit on 90% of the
   training rows; the residual quantile on the held-out 10% sets a prediction
   interval with guaranteed marginal coverage (under exchangeability).

Model selection rule (state this in §13): the winner is chosen on the *temporal*
split, because that is the regime the app deploys into - forecasting years the
model has never seen. The random-split score is reported next to it; a model
that wins the random split but collapses out-of-time is overfitting the calendar,
not learning agronomy.
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
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
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from scipy.stats import loguniform, randint, uniform
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
        "Extra Trees": (
            ExtraTreesClassifier(n_estimators=300, random_state=config.SEED, n_jobs=-1),
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


def tune_yield_model(X_tr: pd.DataFrame, y_tr: np.ndarray, name: str, est) -> tuple[object, dict]:
    """Small, honest random search for the two gradient-boosting baselines.

    ``n_iter`` and ``cv=3`` are deliberately modest: the search budget is a
    design choice you can defend, and a 6-draw search that beats the default
    everywhere is more convincing than a 200-draw search that overfits the
    CV split.
    """
    if name == "XGBoost":
        space = {
            "model__n_estimators": randint(300, 900),
            "model__max_depth": randint(4, 10),
            "model__learning_rate": loguniform(0.03, 0.25),
            "model__subsample": uniform(0.7, 0.3),
            "model__colsample_bytree": uniform(0.7, 0.3),
            "model__min_child_weight": randint(1, 8),
        }
    elif name == "Hist Gradient Boosting":
        space = {
            "model__max_iter": randint(200, 600),
            "model__learning_rate": loguniform(0.03, 0.25),
            "model__max_leaf_nodes": randint(15, 63),
            "model__min_samples_leaf": randint(10, 50),
            "model__l2_regularization": loguniform(1e-3, 1.0),
        }
    else:
        return est, {}

    pipe = preprocess.yield_pipeline(est)
    search = RandomizedSearchCV(
        pipe,
        space,
        n_iter=8,
        cv=3,
        scoring="r2",
        random_state=config.SEED,
        n_jobs=-1,
    )
    t0 = time.time()
    search.fit(X_tr, y_tr)
    elapsed = round(time.time() - t0, 1)
    print(f"  tuned {name}: best CV R2={search.best_score_:.4f} ({elapsed}s)")
    return search.best_estimator_, {
        "best_params": {k: (float(v) if isinstance(v, float) else int(v))
                        for k, v in search.best_params_.items()},
        "best_cv_r2": float(search.best_score_),
        "search_seconds": elapsed,
    }


def _regression_metrics(y_true_log: np.ndarray, pred_log: np.ndarray) -> dict:
    y_true_ha = preprocess.inverse_target(y_true_log)
    pred_ha = np.clip(preprocess.inverse_target(pred_log), 0, None)
    return {
        "r2_log": float(r2_score(y_true_log, pred_log)),
        "mae_log": float(mean_absolute_error(y_true_log, pred_log)),
        "rmse_t_per_ha": float(np.sqrt(mean_squared_error(y_true_ha, pred_ha))),
        "mae_t_per_ha": float(mean_absolute_error(y_true_ha, pred_ha)),
        "r2_t_per_ha": float(r2_score(y_true_ha, pred_ha)),
    }


def train_regression() -> dict:
    df, stats = data_loader.build_yield_frame()
    X_tr, X_te, y_tr, y_te = preprocess.split_yield(df)
    print(f"[yield forecast] train={len(X_tr)} test={len(X_te)} (target = log1p t/ha)")

    results, fitted, tuning = {}, {}, {}
    for name, est in regressor_zoo().items():
        pipe = preprocess.yield_pipeline(est)
        t0 = time.time()
        pipe.fit(X_tr, y_tr)

        pred_log = pipe.predict(X_te)
        cv = cross_val_score(pipe, X_tr, y_tr, cv=3, scoring="r2", n_jobs=1)

        # temporal lens: same model family, fit only on years the test never reaches
        Xt_tr, Xt_te, yt_tr, yt_te, cutoff = preprocess.split_yield_temporal(df)
        pipe_t = preprocess.yield_pipeline(est)
        pipe_t.fit(Xt_tr, yt_tr)
        temporal = _regression_metrics(yt_te, pipe_t.predict(Xt_te))

        results[name] = {
            **_regression_metrics(y_te, pred_log),
            "cv_r2_mean": float(cv.mean()),
            "cv_r2_std": float(cv.std()),
            "temporal_r2_log": temporal["r2_log"],
            "temporal_mae_t_per_ha": temporal["mae_t_per_ha"],
            "fit_seconds": round(time.time() - t0, 1),
        }
        fitted[name] = pipe
        r = results[name]
        print(
            f"  {name:24s} R2(log)={r['r2_log']:.4f}  MAE={r['mae_t_per_ha']:.2f} t/ha  "
            f"cv={r['cv_r2_mean']:.4f}+-{r['cv_r2_std']:.4f}  "
            f"temporal R2={r['temporal_r2_log']:.4f}  {r['fit_seconds']}s"
        )

    # ---- tuning the two boosting candidates, judged on the temporal split too
    print("  hyperparameter search (XGBoost, HistGB):")
    for name in ("XGBoost", "Hist Gradient Boosting"):
        tuned, info = tune_yield_model(X_tr, y_tr, name, regressor_zoo()[name])
        if not info:
            continue
        pred_log = tuned.predict(X_te)
        metrics = _regression_metrics(y_te, pred_log)
        Xt_tr, Xt_te, yt_tr, yt_te, _ = preprocess.split_yield_temporal(df)
        temporal = _regression_metrics(yt_te, tuned.predict(Xt_te))
        label = f"{name} (tuned)"
        results[label] = {
            **metrics,
            "cv_r2_mean": info["best_cv_r2"],
            "cv_r2_std": float("nan"),
            "temporal_r2_log": temporal["r2_log"],
            "temporal_mae_t_per_ha": temporal["mae_t_per_ha"],
            "fit_seconds": info["search_seconds"],
            "best_params": info["best_params"],
        }
        fitted[label] = tuned
        print(
            f"  {label:24s} R2(log)={metrics['r2_log']:.4f}  "
            f"temporal R2={temporal['r2_log']:.4f}"
        )

    # selection: the temporal split is the deployment regime (the app forecasts
    # years the model has not seen), so it decides; the random-split score is
    # reported alongside as the more flattering lens.
    best = max(results, key=lambda k: results[k]["temporal_r2_log"])
    print(f"  -> best by temporal R2(log): {best}")

    # ---- conformal: refit the winner on 90% of train, calibrate on 10%
    X_fit, X_cal, y_fit, y_cal = preprocess.split_calibration(X_tr, y_tr)
    final = clone(fitted[best])
    final.fit(X_fit, y_fit)
    resid = np.abs(y_cal - final.predict(X_cal))
    alpha = 1.0 - config.CONFORMAL_COVERAGE
    q = float(np.quantile(resid, min(1.0, (1 - alpha) * (1 + 1 / len(resid)))))
    conformal = {
        "coverage": config.CONFORMAL_COVERAGE,
        "q_log_scale": q,
        "n_calibration": int(len(y_cal)),
        "calib_fraction": config.CONFORMAL_CALIB_FRACTION,
        "note": "interval = prediction +- q on log1p(t/ha); invert with expm1",
    }
    config.YIELD_CONFORMAL_FILE.write_text(json.dumps(conformal, indent=2), encoding="utf-8")

    final_test = _regression_metrics(y_te, final.predict(X_te))
    results[best]["final_conformal_r2_log"] = final_test["r2_log"]
    results[best]["final_conformal_mae_t_per_ha"] = final_test["mae_t_per_ha"]
    joblib.dump(final, config.YIELD_MODEL_FILE)
    print(f"  -> saved conformal model {best} to {config.YIELD_MODEL_FILE.name} "
          f"(q={q:.4f} log, {config.CONFORMAL_COVERAGE:.0%} coverage)")

    pd.DataFrame(results).T.sort_values("r2_log", ascending=False).to_csv(
        config.REPORTS_DIR / "model_comparison_regression.csv"
    )
    return {"n_train": len(X_tr), "n_test": len(X_te), "cleaning": stats,
            "best_model": best, "conformal": conformal,
            "temporal_cutoff_year": int(cutoff), "models": results}


# ------------------------------------------------------------------ entry point
def main() -> None:
    payload = {
        "settings": {
            "seed": config.SEED,
            "test_size": config.TEST_SIZE,
            "cv_folds_classification": config.CV_FOLDS,
            "cv_folds_regression": 3,
            "yield_target": "log1p(Production / Area), winsorised per crop",
            "conformal_coverage": config.CONFORMAL_COVERAGE,
        },
        "classification": train_classification(),
        "regression": train_regression(),
    }
    config.METRICS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nmetrics written to {config.METRICS_FILE}")


if __name__ == "__main__":
    main()
