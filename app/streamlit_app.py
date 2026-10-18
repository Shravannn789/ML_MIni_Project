"""AgroStar - Explainable AI for crop recommendation and yield forecasting.

Run from the project root:

    streamlit run app/streamlit_app.py

Three tabs, matching the two modelling tasks plus the comparison evidence:

1. **Crop recommendation** - soil/climate sliders -> crop probability + SHAP reasons.
2. **Yield forecast** - state/district/crop/season/year/area -> expected t/ha + SHAP.
3. **Model comparison** - the held-out scores computed by ``src.train``.

Design rule: the app never re-implements preprocessing. It loads the fitted
pipelines from ``models/`` and calls ``predict`` on them, so what the user sees is
exactly the model that was evaluated.
"""
from __future__ import annotations

import base64
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config, data_loader, explain, preprocess  # noqa: E402

st.set_page_config(page_title="AgroStar | Explainable Crop AI", layout="wide")


def inject_background() -> None:
    """Blurred farmland photo behind the app, dimmed so text stays readable.

    The image is embedded as base64 rather than served as a static file: Streamlit
    does not expose an assets route, and this keeps the app to a single file plus
    one ~30 KB JPEG. The dark overlay is composited with the image in one
    ``background-image`` declaration, which avoids needing a positioned overlay
    element that could swallow clicks.
    """
    path = ROOT / "app" / "assets" / "farmland_blurred.jpg"
    if not path.exists():
        return
    b64 = base64.b64encode(path.read_bytes()).decode()
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background-image:
                linear-gradient(rgba(9, 15, 11, 0.58), rgba(9, 15, 11, 0.58)),
                url("data:image/jpeg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        /* Translucent rather than transparent: with a fully see-through header,
           content scrolls *under* it and labels clip against the toolbar. */
        [data-testid="stHeader"] {{
            background: rgba(9, 15, 11, 0.55);
            backdrop-filter: blur(6px);
        }}
        [data-testid="stToolbar"] {{ right: 1rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_background()


# --------------------------------------------------------------------- loading
@st.cache_resource
def load_models():
    if not config.REC_MODEL_FILE.exists() or not config.YIELD_MODEL_FILE.exists():
        st.error("Models not found. Run `python -m src.train` first.")
        st.stop()
    return joblib.load(config.REC_MODEL_FILE), joblib.load(config.YIELD_MODEL_FILE)


@st.cache_resource
def crop_explainer(_model):
    return shap.TreeExplainer(_model.named_steps["model"])


@st.cache_resource
def yield_explainer(_model):
    return shap.TreeExplainer(_model.named_steps["model"])


@st.cache_data
def load_data():
    rec = data_loader.load_recommendation()
    prod, stats = data_loader.build_yield_frame()
    return rec, prod, stats


def shap_bar(df: pd.DataFrame, xlabel: str):
    """Horizontal SHAP bar: red pushes the prediction up, blue pushes it down.

    Uses the ``label`` column when present, so a bar reads "Crop = Rice" rather
    than just "Crop" - the difference between a chart someone can act on and one
    they have to decode.
    """
    d = df.iloc[::-1]
    names = d["label"] if "label" in d.columns else d["feature"]
    fig, ax = plt.subplots(figsize=(6, max(2.2, 0.42 * len(d))))
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in d["shap"]]
    ax.barh(names, d["shap"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


clf, reg = load_models()
rec_df, prod_df, prod_stats = load_data()

st.title("AgroStar")
st.caption(
    "Explainable crop recommendation and yield forecasting · "
    "Random Forest · SHAP explanations per prediction"
)

tab1, tab2, tab3 = st.tabs(["Crop recommendation", "Yield forecast", "Model comparison"])


# ------------------------------------------------------------------ tab 1
with tab1:
    st.subheader("Given soil and climate, which crop?")
    st.caption(
        "Inputs are the seven measurements the dataset provides. "
        "Every prediction comes with the reasons behind it, not just a label."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        N = st.slider("Nitrogen (N)", 0, 140, 90)
        P = st.slider("Phosphorus (P)", 5, 145, 42)
        K = st.slider("Potassium (K)", 5, 205, 43)
    with c2:
        temp = st.slider("Temperature (°C)", 8.0, 44.0, 20.9, 0.1)
        humidity = st.slider("Humidity (%)", 14.0, 100.0, 82.0, 0.1)
    with c3:
        ph = st.slider("pH", 3.5, 10.0, 6.5, 0.1)
        rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 202.9, 0.1)

    if st.button("Recommend a crop", type="primary"):
        row = pd.Series({
            "N": N, "P": P, "K": K, "temperature": temp,
            "humidity": humidity, "ph": ph, "rainfall": rainfall,
        })
        X = row[config.REC_FEATURES].to_frame().T
        pred = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0]
        classes = list(clf.classes_)

        top = pd.Series(proba, index=classes).sort_values(ascending=False).head(5)

        left, right = st.columns([1, 1.4])
        with left:
            st.metric("Recommended crop", pred.title())
            st.write("**Most likely alternatives**")
            st.dataframe(
                top.rename("probability").to_frame().style.format("{:.1%}"),
                width="stretch",
            )
            st.caption(
                f"Confidence: {proba[classes.index(pred)]:.1%}. "
                "With 22 balanced classes, chance level is 4.5%."
            )
        with right:
            st.write("**Why this crop?** (SHAP contributions)")
            contributions = explain.crop_local(
                clf, row, explainer=crop_explainer(clf)
            )
            st.pyplot(shap_bar(contributions, "SHAP value for the predicted class (log-odds)"))
            st.caption(
                "Red pushes toward this crop, blue away from it. "
                "Values are in log-odds, so they add up to the model's decision."
            )


# ------------------------------------------------------------------ tab 2
with tab2:
    st.subheader("Expected yield for a district-level crop record")
    st.caption(
        "Trained on 237,849 cleaned district records (1997–2015). "
        "Target is tonnes per hectare, modelled on a log scale."
    )

    states = sorted(prod_df["State_Name"].unique())
    c1, c2, c3 = st.columns(3)
    with c1:
        state = st.selectbox("State", states, index=states.index("Punjab") if "Punjab" in states else 0)
        districts = sorted(prod_df.loc[prod_df["State_Name"] == state, "District_Name"].unique())
        district = st.selectbox("District", districts)
    with c2:
        crops = sorted(prod_df["Crop"].unique())
        crop = st.selectbox("Crop", crops, index=crops.index("Rice") if "Rice" in crops else 0)
        season = st.selectbox("Season", sorted(prod_df["Season"].unique()))
    with c3:
        year = st.number_input(
            "Crop year", int(prod_stats["year_min"]), int(prod_stats["year_max"]), 2012
        )
        area = st.number_input("Area (hectares)", 1.0, 1_000_000.0, 500.0, step=50.0)

    if st.button("Forecast yield", type="primary"):
        row = pd.Series({
            "State_Name": state,
            "District_Name": district,
            "Crop": crop,
            "Season": season,
            "Crop_Year": int(year),
            "Area": float(area),
        })
        X = row[config.PROD_CATEGORICAL + config.PROD_NUMERIC].to_frame().T
        pred_log = reg.predict(X)[0]
        pred_ha = float(preprocess.inverse_target(np.array([pred_log]))[0])

        # context: how does this compare with the same crop historically?
        hist = prod_df.loc[prod_df["Crop"] == crop, config.YIELD_TARGET]
        median_ha = float(hist.median())

        left, right = st.columns([1, 1.4])
        with left:
            st.metric("Predicted yield", f"{pred_ha:.2f} t/ha")
            st.metric(
                f"Typical {crop} yield",
                f"{median_ha:.2f} t/ha",
                delta=f"{pred_ha - median_ha:+.2f} t/ha",
            )
            st.caption(
                f"Total production estimate: **{pred_ha * area:,.0f} tonnes** "
                f"over {area:,.0f} ha."
            )
        with right:
            st.write("**Drivers of this forecast** (SHAP contributions)")
            contributions = explain.yield_local(reg, row, explainer=yield_explainer(reg))
            st.pyplot(shap_bar(contributions, "SHAP value (log t/ha)"))
            st.caption(
                "Categorical effects are target-encoded, so the bar shows the column "
                "and its value rather than a raw code."
            )


# ------------------------------------------------------------------ tab 3
with tab3:
    st.subheader("Held-out performance")
    st.caption(
        "Scores on the 20% test split that was never used for tuning. "
        "Every model shares the same split and seed."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Crop recommendation** (accuracy)")
        clf_tbl = pd.read_csv(
            config.REPORTS_DIR / "model_comparison_classification.csv", index_col=0
        ).sort_values("accuracy", ascending=False)
        st.dataframe(
            clf_tbl[["accuracy", "macro_f1", "cv_accuracy_mean", "cv_accuracy_std"]]
            .style.format("{:.4f}"),
            width="stretch",
        )
    with col_b:
        st.write("**Yield forecast** (R² on log yield)")
        reg_tbl = pd.read_csv(
            config.REPORTS_DIR / "model_comparison_regression.csv", index_col=0
        ).sort_values("r2_log", ascending=False)
        st.dataframe(
            reg_tbl[["r2_log", "mae_t_per_ha", "cv_r2_mean", "cv_r2_std"]]
            .style.format("{:.4f}"),
            width="stretch",
        )

    st.divider()
    st.write("**The finding worth reporting**")
    spread = clf_tbl["accuracy"].max() - clf_tbl["accuracy"].min()
    st.info(
        f"On crop recommendation, all four algorithms land within "
        f"{spread:.1%} of each other ({clf_tbl['accuracy'].min():.1%}–"
        f"{clf_tbl['accuracy'].max():.1%}), on a dataset with exactly 100 rows per "
        "class. The benchmark is too easily separable to discriminate between "
        "algorithms — so the interesting contribution is the *explanation*, and the "
        "harder yield task, not the headline accuracy."
    )

    with st.expander("Data cleaning decisions"):
        st.json(prod_stats)

    st.caption(f"Seed {config.SEED} · test size {config.TEST_SIZE} · "
               f"models in `{config.MODELS_DIR.name}/`")

st.divider()
st.caption(
    "Background: aerial view of agricultural fields in Punjab, India — "
    "Wikimedia Commons, CC BY-SA 4.0 (blurred and dimmed for use as a backdrop)."
)
