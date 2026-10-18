"""SHAP explanations for both models.

Run with:  python -m src.explain

Why SHAP here rather than built-in feature importance: impurity importance in a
tree ensemble is biased toward high-cardinality features (our district column has
646 values), and it says nothing about the direction of an effect. SHAP gives a
per-prediction contribution in the units of the model output, so "yield is high
because crop=rice and area is large" can be shown as numbers that add up.

Two shapes of explanation are produced:

* **Local** - why *this* row got *this* prediction. This is what the app shows.
* **Global** - mean absolute SHAP per feature, for the report's §16 figures.
"""
from __future__ import annotations

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import config, data_loader, preprocess

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9})


# ------------------------------------------------------------------ helpers
def _to_matrix(sv) -> np.ndarray:
    """Normalise SHAP output across versions.

    shap returns either a list of per-class arrays or a single (n, features,
    classes) array depending on version and model type.
    """
    if isinstance(sv, list):
        return np.stack(sv, axis=-1)
    return np.asarray(sv)


def crop_local(model, row: pd.Series, top_k: int = 6, explainer=None) -> pd.DataFrame:
    """Per-prediction contributions for the recommended crop.

    ``explainer`` may be passed in so a UI can cache the expensive
    ``TreeExplainer`` construction across interactions.
    """
    X = row[config.REC_FEATURES].to_frame().T
    if explainer is None:
        explainer = shap.TreeExplainer(model.named_steps["model"])
    sv = _to_matrix(explainer.shap_values(X))

    pred = model.predict(X)[0]
    classes = list(model.classes_)
    cls_idx = classes.index(pred)
    contributions = sv[0, :, cls_idx]

    out = pd.DataFrame({
        "feature": config.REC_FEATURES,
        "value": X.iloc[0].to_numpy(),
        "shap": contributions,
    })
    out["label"] = [f"{f} = {v:.1f}" for f, v in zip(out["feature"], out["value"])]
    out["abs"] = out["shap"].abs()
    return out.sort_values("abs", ascending=False).head(top_k).drop(columns="abs")


def yield_local(model, row: pd.Series, top_k: int = 6, explainer=None) -> pd.DataFrame:
    """Per-prediction contributions for the yield model.

    The model works on target-encoded categoricals, so the contribution is
    reported against the *original* column (e.g. ``Crop = rice``) rather than the
    encoded number, which would be meaningless to a reader.
    """
    X = row[config.PROD_CATEGORICAL + config.PROD_NUMERIC].to_frame().T
    prep = model.named_steps["prep"]
    Xt = prep.transform(X)
    names = list(prep.get_feature_names_out())

    if explainer is None:
        explainer = shap.TreeExplainer(model.named_steps["model"])
    sv = np.asarray(explainer.shap_values(Xt))
    contrib = sv[0]

    out = pd.DataFrame({"feature": names, "shap": contrib})
    # attach the human-readable value for each column
    out["value"] = [
        row[c] if c in row.index else Xt[0][names.index(c)] for c in names
    ]
    out["label"] = [f"{f} = {v}" for f, v in zip(out["feature"], out["value"])]
    out["abs"] = out["shap"].abs()
    return out.sort_values("abs", ascending=False).head(top_k).drop(columns="abs")


def _bar(df: pd.DataFrame, title: str, fname: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in df["shap"]]
    ax.barh(df["feature"], df["shap"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / fname, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {fname}")


def global_importance(sample_n: int = 400) -> None:
    """Mean |SHAP| across a sample of the test set, for both tasks."""
    # --- crop recommendation
    rec = data_loader.load_recommendation()
    X_tr, X_te, y_tr, y_te = preprocess.split_rec(rec)
    clf = joblib.load(config.REC_MODEL_FILE)
    Xs = X_te.head(sample_n)

    sv = _to_matrix(shap.TreeExplainer(clf.named_steps["model"]).shap_values(Xs))
    # average |contribution| over classes, then over rows
    mean_abs = np.abs(sv).mean(axis=(0, 2))
    imp = pd.DataFrame({"feature": config.REC_FEATURES, "mean_abs_shap": mean_abs})
    imp = imp.sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(config.REPORTS_DIR / "shap_importance_crop.csv", index=False)
    _bar(imp.rename(columns={"shap": "mean_abs_shap"}).assign(shap=imp["mean_abs_shap"]),
         "Figure 8 - Crop recommendation: mean |SHAP| across 22 classes",
         "fig_08_shap_crop.png", "mean |SHAP|")

    # --- yield
    prod, _ = data_loader.build_yield_frame()
    Xr_tr, Xr_te, yr_tr, yr_te = preprocess.split_yield(prod)
    reg = joblib.load(config.YIELD_MODEL_FILE)
    prep = reg.named_steps["prep"]
    Xs_r = Xr_te.head(sample_n)
    Xt = prep.transform(Xs_r)
    names = list(prep.get_feature_names_out())

    sv_r = np.asarray(shap.TreeExplainer(reg.named_steps["model"]).shap_values(Xt))
    imp_r = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(sv_r).mean(axis=0)})
    imp_r = imp_r.sort_values("mean_abs_shap", ascending=False)
    imp_r.to_csv(config.REPORTS_DIR / "shap_importance_yield.csv", index=False)
    _bar(imp_r.rename(columns={"mean_abs_shap": "shap"}),
         "Figure 9 - Yield forecast: mean |SHAP| (log yield)",
         "fig_09_shap_yield.png", "mean |SHAP|")


def main() -> None:
    global_importance()

    # one worked example per task, printed for the report
    rec = data_loader.load_recommendation()
    clf = joblib.load(config.REC_MODEL_FILE)
    row = rec.iloc[0]
    print("\nCrop recommendation example ->", clf.predict(row[config.REC_FEATURES].to_frame().T)[0])
    print(crop_local(clf, row).to_string(index=False))

    prod, _ = data_loader.build_yield_frame()
    reg = joblib.load(config.YIELD_MODEL_FILE)
    r = prod.iloc[0]
    pred_log = reg.predict(r[config.PROD_CATEGORICAL + config.PROD_NUMERIC].to_frame().T)[0]
    print(f"\nYield example -> {preprocess.inverse_target(np.array([pred_log]))[0]:.2f} t/ha "
          f"(actual {r[config.YIELD_TARGET]:.2f})")
    print(yield_local(reg, r).to_string(index=False))


if __name__ == "__main__":
    main()
