"""Evaluation artifacts: figures and human-readable reports.

Run with:  python -m src.evaluate

Writes numbered figures to reports/figures/ (the report refers to them as
Figure 1..N) plus a plain-text classification report. Nothing here re-trains a
model: it loads the saved pipelines and reproduces the same test split, so the
figures always describe the artifact that ships with the app.
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")  # headless: no window needed when run from a script
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src import config, data_loader, preprocess

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9})


def _save(fig, name: str) -> None:
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df[config.REC_TARGET].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(counts.index, counts.values, color="#4C72B0")
    ax.set_xlabel("rows")
    ax.set_title("Figure 1 - Crop recommendation: rows per class (perfectly balanced, 100 each)")
    _save(fig, "fig_01_class_distribution.png")


def plot_correlation(df: pd.DataFrame) -> None:
    corr = df[config.REC_FEATURES].corr()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Figure 2 - Feature correlation (no pair above 0.95)")
    _save(fig, "fig_02_correlation.png")


def plot_yield_distribution(y_clean: pd.Series, y_log: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].hist(y_clean, bins=80, color="#4C72B0")
    axes[0].set_title("Raw yield (t/ha), after cleaning")
    axes[0].set_xlabel("t/ha")
    axes[1].hist(y_log, bins=80, color="#55A868")
    axes[1].set_title("Modelled target: log1p(yield)")
    axes[1].set_xlabel("log1p(t/ha)")
    fig.suptitle("Figure 3 - Why the target is log-transformed", y=1.02)
    _save(fig, "fig_03_yield_distribution.png")


def plot_model_comparison() -> None:
    clf = pd.read_csv(config.REPORTS_DIR / "model_comparison_classification.csv", index_col=0)
    reg = pd.read_csv(config.REPORTS_DIR / "model_comparison_regression.csv", index_col=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    c = clf.sort_values("accuracy")
    axes[0].barh(c.index, c["accuracy"], color="#4C72B0")
    axes[0].set_xlim(max(0, c["accuracy"].min() - 0.08), 1.0)
    axes[0].set_title("Figure 4 - Classification accuracy (crop recommendation)")
    axes[0].set_xlabel("accuracy")

    r = reg.sort_values("r2_log")
    axes[1].barh(r.index, r["r2_log"], color="#55A868")
    axes[1].set_title("Figure 5 - Regression R2 on log yield")
    axes[1].set_xlabel("R2 (log scale)")
    _save(fig, "fig_04_05_model_comparison.png")


def plot_confusion(y_true, y_pred, labels) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labels)), labels, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    fig.colorbar(im, ax=ax, shrink=0.75)
    ax.set_title("Figure 6 - Best classifier: row-normalised confusion matrix")
    _save(fig, "fig_06_confusion_matrix.png")


def plot_pred_vs_actual(y_true_ha, y_pred_ha) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.scatter(y_true_ha, y_pred_ha, s=4, alpha=0.15, color="#4C72B0", edgecolors="none")
    lim = max(float(np.percentile(y_true_ha, 99.5)), float(np.percentile(y_pred_ha, 99.5)))
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="perfect prediction")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("actual yield (t/ha)")
    ax.set_ylabel("predicted yield (t/ha)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Figure 7 - Yield: predicted vs actual (test set)")
    _save(fig, "fig_07_pred_vs_actual.png")


def main() -> None:
    metrics = json.loads(config.METRICS_FILE.read_text(encoding="utf-8"))

    # ---------------------------------------------------------- classification
    rec = data_loader.load_recommendation()
    plot_class_distribution(rec)
    plot_correlation(rec)

    X_tr, X_te, y_tr, y_te = preprocess.split_rec(rec)
    clf = joblib.load(config.REC_MODEL_FILE)
    y_pred = clf.predict(X_te)

    labels = sorted(rec[config.REC_TARGET].unique())
    plot_confusion(y_te, y_pred, labels)

    report = classification_report(y_te, y_pred, digits=4)
    (config.REPORTS_DIR / "classification_report.txt").write_text(
        f"Model: {metrics['classification']['best_model']}\n\n{report}", encoding="utf-8"
    )
    print("  saved classification_report.txt")

    # -------------------------------------------------------------- regression
    prod, _ = data_loader.build_yield_frame()
    plot_yield_distribution(prod[config.YIELD_TARGET], np.log1p(prod[config.YIELD_TARGET]))

    Xr_tr, Xr_te, yr_tr, yr_te = preprocess.split_yield(prod)
    reg = joblib.load(config.YIELD_MODEL_FILE)
    pred_ha = np.clip(preprocess.inverse_target(reg.predict(Xr_te)), 0, None)
    plot_pred_vs_actual(preprocess.inverse_target(yr_te), pred_ha)

    plot_model_comparison()

    # ---------------------------------------------------------- summary table
    lines = ["# Results summary (generated)", ""]
    for task, key, metric in [
        ("Crop recommendation (classification)", "classification", "accuracy"),
        ("Yield forecast (regression)", "regression", "r2_log"),
    ]:
        lines += [f"## {task}", "", "| Model | " + " | ".join(
            k for k in metrics[key]["models"][metrics[key]["best_model"]]
        ) + " |"]
        header = lines[-1]
        lines.append("|" + "---|" * (header.count("|") - 1))
        for name, row in sorted(
            metrics[key]["models"].items(), key=lambda kv: kv[1][metric], reverse=True
        ):
            cells = " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.values())
            lines.append(f"| {name} | {cells} |")
        lines.append("")
    (config.REPORTS_DIR / "results_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("  saved results_summary.md")


if __name__ == "__main__":
    main()
