"""AgroStar - Explainable AI for crop recommendation, yield and earnings.

Run from the project root:

    streamlit run app/streamlit_app.py

Five views:

1. **Overview**       - headline numbers and today's market at a glance.
2. **Crop recommend** - soil/climate sliders -> crop probability + SHAP reasons.
3. **Yield & earnings** - district record -> t/ha with a conformal interval,
   live mandi price, and an estimated net return with scenarios.
4. **Live market**    - Agmarknet mandi prices, refreshed from data.gov.in.
5. **Models**         - the held-out, temporal and calibration evidence.

Design rules: the app never re-implements preprocessing - it loads the fitted
pipelines from ``models/``; and it never invents a price - every rupee figure
traces back to Agmarknet or to a user-edited field.
"""
from __future__ import annotations

import base64
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config, data_loader, economics, explain, market, preprocess  # noqa: E402

st.set_page_config(page_title="AgroStar | Explainable Crop AI", layout="wide")

PX_TEMPLATE = "plotly_dark"


# ------------------------------------------------------------------- helpers
def inject_background() -> None:
    """Blurred farmland photo behind the app, dimmed so text stays readable."""
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
        [data-testid="stHeader"] {{
            background: rgba(9, 15, 11, 0.55);
            backdrop-filter: blur(6px);
        }}
        [data-testid="stToolbar"] {{ right: 1rem; }}
        div[data-testid="stMetric"] {{
            background: rgba(13, 24, 17, 0.72);
            border: 1px solid rgba(95, 168, 69, 0.25);
            border-radius: 10px;
            padding: 12px 14px;
        }}
        div[data-testid="stMetricValue"] {{ font-size: 1.55rem; }}
        .agro-note {{
            background: rgba(13, 24, 17, 0.72);
            border-left: 3px solid #5FA845;
            border-radius: 6px;
            padding: 10px 14px;
            color: #e9f1ea;
            font-size: 0.92rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inr(x: float) -> str:
    """Short Indian-rupee formatting: 1.23 Cr / 45.6 L / 789."""
    x = float(x)
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e7:
        return f"{sign}Rs {x/1e7:,.2f} Cr"
    if x >= 1e5:
        return f"{sign}Rs {x/1e5:,.2f} L"
    return f"{sign}Rs {x:,.0f}"


def shap_bar_plotly(df: pd.DataFrame, xlabel: str, height: int | None = None) -> go.Figure:
    """Interactive SHAP bar: red pushes the prediction up, blue away."""
    d = df.iloc[::-1]
    names = d["label"] if "label" in d.columns else d["feature"]
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in d["shap"]]
    fig = go.Figure(go.Bar(x=d["shap"], y=names, orientation="h", marker_color=colors))
    fig.update_layout(
        template=PX_TEMPLATE,
        height=height or max(240, 42 * len(d)),
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis_title=xlabel,
        yaxis_title="",
        showlegend=False,
    )
    fig.add_vline(x=0, line_width=1, line_color="white")
    return fig


inject_background()


# --------------------------------------------------------------------- loading
@st.cache_resource
def load_models():
    if not config.REC_MODEL_FILE.exists() or not config.YIELD_MODEL_FILE.exists():
        st.error("Models not found. Run `python -m src.train` first.")
        st.stop()
    conformal = {}
    if config.YIELD_CONFORMAL_FILE.exists():
        conformal = json.loads(config.YIELD_CONFORMAL_FILE.read_text(encoding="utf-8"))
    return joblib.load(config.REC_MODEL_FILE), joblib.load(config.YIELD_MODEL_FILE), conformal


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


@st.cache_data(ttl=config.MANDI_CACHE_TTL_MINUTES * 60, show_spinner=False)
def load_prices(force: bool = False):
    return market.get_prices(force_refresh=force)


clf, reg, conformal = load_models()
rec_df, prod_df, prod_stats = load_data()

if "prices" not in st.session_state:
    st.session_state.prices, st.session_state.price_meta = load_prices()
prices, price_meta = st.session_state.prices, st.session_state.price_meta

YEAR_MIN = int(prod_stats["year_min"])
YEAR_MAX = int(prod_stats["year_max"])          # last year with training data
FORECAST_MAX = datetime.now().year + 1          # UI allows extrapolation with a warning


# ------------------------------------------------------------------- chrome
st.title("AgroStar")
st.caption(
    "Explainable crop recommendation and yield forecasting · "
    "DES production data 1997-2023 · live Agmarknet mandi prices · SHAP per prediction"
)

tab_overview, tab_rec, tab_yield, tab_market, tab_models = st.tabs(
    ["Overview", "Crop recommendation", "Yield & earnings", "Live market", "Models & methods"]
)


# ---------------------------------------------------------------- tab: overview
with tab_overview:
    st.subheader("The farm at a glance")
    m1, m2, m3, m4 = st.columns(4)
    metrics = json.loads(config.METRICS_FILE.read_text(encoding="utf-8"))
    m1.metric("Crop recommendation accuracy",
              f"{metrics['classification']['models'][metrics['classification']['best_model']]['accuracy']:.1%}")
    m2.metric("Yield R2 (unseen years)",
              f"{metrics['regression']['models'][metrics['regression']['best_model']]['temporal_r2_log']:.3f}")
    m3.metric("District records modelled", f"{prod_stats['rows_modelled']:,}")
    m4.metric("Live mandi listings", f"{price_meta['n_listings']:,}",
              help=f"{price_meta['source']}, fetched {price_meta['fetched_at']}")

    st.markdown(
        f"""
        <div class="agro-note">
        Pick a crop from your soil and climate, forecast its yield for a district,
        and see what today's mandi prices say that harvest is worth.
        Every prediction ships with a SHAP explanation and - for yield - a
        90% prediction interval from split-conformal calibration.
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary = market.commodity_summary(prices)
    if len(summary):
        top = summary.head(20)
        fig = px.bar(
            top, x="median_price", y="commodity", orientation="h",
            title=f"Today's median modal mandi price (Rs/quintal) - top {len(top)}",
            template=PX_TEMPLATE, height=520,
            hover_data=["listings", "states", "p25", "p75"],
            labels={"median_price": "Rs / quintal", "commodity": ""},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width='stretch')
        st.caption(
            f"Source: {price_meta['source']} · fetched {price_meta['fetched_at']} · "
            f"n = {price_meta['n_listings']:,} listings. Refresh from the Live market tab."
        )


# ------------------------------------------------------- tab: crop recommendation
with tab_rec:
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

    if st.button("Recommend a crop", type="primary", key="rec_btn"):
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
            st.metric("Recommended crop", pred.title(),
                      delta=f"{proba[classes.index(pred)]:.1%} confidence")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(proba[classes.index(pred)] * 100),
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#5FA845"},
                    "steps": [
                        {"range": [0, 50], "color": "#17221b"},
                        {"range": [50, 100], "color": "#24382a"},
                    ],
                },
            ))
            gauge.update_layout(template=PX_TEMPLATE, height=220, margin=dict(t=30, b=10))
            st.plotly_chart(gauge, width='stretch')

            fig_top = px.bar(
                x=top.values, y=[c.title() for c in top.index], orientation="h",
                title="Most likely alternatives", template=PX_TEMPLATE, height=240,
                labels={"x": "probability", "y": ""},
            )
            fig_top.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_top, width='stretch')
            st.caption(
                f"With 22 balanced classes, chance level is 4.5%. "
                f"Trained accuracy of this model: "
                f"{metrics['classification']['models'][metrics['classification']['best_model']]['accuracy']:.1%}."
            )
        with right:
            st.write("**Why this crop?** (SHAP contributions)")
            contributions = explain.crop_local(clf, row, explainer=crop_explainer(clf))
            st.plotly_chart(
                shap_bar_plotly(contributions, "SHAP value for the predicted class (log-odds)"),
                width='stretch',
            )
            st.caption(
                "Red pushes toward this crop, blue away from it. "
                "Values are in log-odds, so they add up to the model's decision."
            )


# ------------------------------------------------------- tab: yield & earnings
with tab_yield:
    st.subheader("Expected yield, market price and earnings")
    st.caption(
        f"Trained on {prod_stats['rows_modelled']:,} cleaned district records "
        f"({YEAR_MIN}-{YEAR_MAX}). Target is tonnes per hectare, modelled on a log "
        "scale; the interval is split-conformal at "
        f"{conformal.get('coverage', 0.9):.0%} nominal coverage."
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
        year = st.number_input("Crop year", YEAR_MIN, FORECAST_MAX, min(YEAR_MAX, 2021))
        area = st.number_input("Area (hectares)", 0.1, 1_000_000.0, 500.0, step=10.0)

    if year > YEAR_MAX:
        st.warning(
            f"The newest production data ends at {YEAR_MAX}. Years beyond that are "
            "extrapolation: the model has no observations of that season's weather, "
            "pests or policy."
        )

    # live price for the selected crop, editable before the forecast is computed
    quote = market.price_for_crop(prices, crop, state)
    if quote:
        default_price = quote["median_price"]
        price_help = (
            f"Live Agmarknet quote: {quote['commodity']}, median Rs {default_price:,.0f}/quintal "
            f"across {quote['n_listings']} listings ({quote['scope']}, arrivals {quote['latest_arrival']})."
        )
    else:
        default_price = 2000.0
        price_help = (
            "No Agmarknet listings matched this crop in the current data "
            "(some crops trade at mills, not mandis). Enter an estimated farm-gate price."
        )

    p1, p2 = st.columns(2)
    with p1:
        price = st.number_input("Price (Rs/quintal)", 1.0, 100_000.0, default_price, step=10.0, help=price_help)
    with p2:
        default_cost = economics.default_cost_per_ha(crop)
        cost = st.number_input(
            "Cost of cultivation (Rs/hectare)", 0.0, 1_000_000.0, float(default_cost), step=1000.0,
            help="Order-of-magnitude all-India default (A2+FL basis). Override with your own record.",
        )

    if st.button("Forecast yield and earnings", type="primary", key="yield_btn"):
        row = pd.Series({
            "State_Name": state,
            "District_Name": district,
            "Crop": crop,
            "Season": season,
            "Crop_Year": int(year),
            "Area": float(area),
        })
        X = row[config.PROD_CATEGORICAL + config.PROD_NUMERIC].to_frame().T
        pred_log = float(reg.predict(X)[0])
        pred_ha = float(preprocess.inverse_target(np.array([pred_log]))[0])
        q = float(conformal.get("q_log_scale", 0.0))
        lo_ha = float(max(0.0, preprocess.inverse_target(np.array([pred_log - q]))[0]))
        hi_ha = float(preprocess.inverse_target(np.array([pred_log + q]))[0])

        hist = prod_df.loc[prod_df["Crop"] == crop, config.YIELD_TARGET]
        median_ha = float(hist.median())

        earn = economics.estimate_earnings(
            yield_t_per_ha=pred_ha, area_ha=area,
            price_inr_per_quintal=price, cost_inr_per_ha=cost, crop=crop,
        )

        left, right = st.columns([1, 1.25])
        with left:
            st.metric("Predicted yield", f"{pred_ha:.2f} t/ha",
                      delta=f"{pred_ha - median_ha:+.2f} vs typical {crop.lower()}")
            st.metric("90% interval", f"{lo_ha:.2f} - {hi_ha:.2f} t/ha")
            st.metric("Expected production", f"{earn['production_t']:,.0f} t "
                                              f"({earn['production_q']:,.0f} quintals)")
            st.metric("Gross revenue", inr(earn["revenue_inr"]))
            c_net, c_roi = st.columns(2)
            c_net.metric("Net after cost", inr(earn["net_inr"]))
            c_roi.metric("Return on cost", f"{earn['roi_pct']:.0f}%")

            st.caption(
                f"Price used: Rs {price:,.0f}/quintal "
                f"({quote['scope'] if quote else 'manual estimate'}). "
                f"Cost: Rs {cost:,.0f}/ha."
            )

        with right:
            st.write("**Drivers of this forecast** (SHAP contributions)")
            contributions = explain.yield_local(reg, row, explainer=yield_explainer(reg))
            st.plotly_chart(
                shap_bar_plotly(contributions, "SHAP value (log t/ha)"),
                width='stretch',
            )

        st.divider()
        st.write("**Earnings scenarios** - a forecast is a range, not a number")
        scenarios = economics.scenario_table(
            yield_t_per_ha=pred_ha, area_ha=area,
            price_inr_per_quintal=price, cost_inr_per_ha=cost, crop=crop,
            low_yield=lo_ha, high_yield=hi_ha,
            # quartiles, not min/max: a single outlier mandi quote should not
            # define the optimistic case
            price_min=quote["p25"] if quote else price * 0.9,
            price_max=quote["p75"] if quote else price * 1.1,
        )
        scen_df = pd.DataFrame(scenarios)
        cols_show = ["scenario", "yield_t_per_ha", "price_inr_per_quintal",
                     "production_t", "revenue_inr", "total_cost_inr", "net_inr", "roi_pct"]
        st.dataframe(
            scen_df[cols_show].style.format({
                "yield_t_per_ha": "{:.2f}", "price_inr_per_quintal": "{:,.0f}",
                "production_t": "{:,.0f}", "revenue_inr": "{:,.0f}",
                "total_cost_inr": "{:,.0f}", "net_inr": "{:,.0f}", "roi_pct": "{:.0f}%",
            }),
            width="stretch", hide_index=True,
        )
        fig_scen = px.bar(
            scen_df, x="scenario", y="net_inr", color="scenario",
            title="Net return by scenario (Rs)", template=PX_TEMPLATE, height=300,
            labels={"net_inr": "net return (Rs)", "scenario": ""},
        )
        st.plotly_chart(fig_scen, width='stretch')

        # where the prediction sits in the crop's historical distribution
        st.write(f"**Where this forecast sits in historical {crop.lower()} yields**")
        hist_fig = px.histogram(
            prod_df[prod_df["Crop"] == crop], x=config.YIELD_TARGET, nbins=60,
            title=f"{crop} yields in the training data ({YEAR_MIN}-{YEAR_MAX})",
            template=PX_TEMPLATE, height=280,
            labels={config.YIELD_TARGET: "t/ha"},
        )
        hist_fig.add_vline(x=pred_ha, line_color="#5FA845", line_width=2,
                           annotation_text=f"forecast {pred_ha:.2f}")
        hist_fig.add_vrect(x0=lo_ha, x1=hi_ha, fillcolor="#5FA845", opacity=0.15,
                           line_width=0, annotation_text="90% interval")
        st.plotly_chart(hist_fig, width='stretch')


# ------------------------------------------------------------- tab: live market
with tab_market:
    st.subheader("Live mandi prices (Agmarknet)")
    badge = "LIVE" if price_meta["live"] else "OFFLINE SNAPSHOT"
    st.caption(
        f"{badge} · {price_meta['source']} · fetched {price_meta['fetched_at']} · "
        f"{price_meta['n_listings']:,} listings. Prices are modal wholesale quotes in "
        "Rs/quintal from regulated mandis across India."
    )

    if st.button("Refresh live prices now", key="refresh_prices"):
        with st.spinner("Fetching from api.data.gov.in ..."):
            fresh_prices, fresh_meta = market.get_prices(force_refresh=True)
        st.session_state.prices, st.session_state.price_meta = fresh_prices, fresh_meta
        if fresh_meta["live"]:
            st.success(f"Updated: {fresh_meta['n_listings']:,} listings at {fresh_meta['fetched_at']}.")
        else:
            st.warning("Live fetch failed - showing the bundled snapshot instead.")
        st.rerun()

    summary = market.commodity_summary(prices)
    if len(summary):
        left, right = st.columns([1.3, 1])
        with left:
            fig = px.bar(
                summary, x="median_price", y="commodity", orientation="h",
                title="Median modal price by commodity (Rs/quintal)",
                template=PX_TEMPLATE, height=max(420, 22 * len(summary)),
                hover_data=["listings", "states", "min_price", "max_price"],
                labels={"median_price": "Rs / quintal", "commodity": ""},
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, width='stretch')
        with right:
            st.dataframe(summary, width="stretch", hide_index=True, height=520)

        st.divider()
        commodities = sorted(summary["commodity"].tolist())
        sel = st.selectbox("State-level detail for commodity", commodities,
                           index=commodities.index("Rice") if "Rice" in commodities else 0)
        st_table = market.state_price_table(prices, sel)
        if len(st_table):
            left2, right2 = st.columns([1.2, 1])
            with left2:
                fig2 = px.bar(
                    st_table, x="median_price", y="state", orientation="h",
                    title=f"{sel}: median modal price by state (Rs/quintal)",
                    template=PX_TEMPLATE, height=max(360, 20 * len(st_table)),
                    hover_data=["listings", "min_price", "max_price"],
                    labels={"median_price": "Rs / quintal", "state": ""},
                )
                fig2.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig2, width='stretch')
            with right2:
                st.dataframe(st_table, width="stretch", hide_index=True, height=420)
    else:
        st.info("No market data available. Check the network or re-run `python -m src.market`.")


# ------------------------------------------------------------- tab: models
with tab_models:
    st.subheader("Held-out performance")
    st.caption(
        "The random split flatters every model (the test year's other districts "
        "were seen in training). The temporal split - train on early years, test "
        "on later ones - is the regime this app deploys into, and it decides "
        "which model ships."
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
        st.write("**Yield forecast** (R2 on log yield; temporal split decides)")
        reg_tbl = pd.read_csv(
            config.REPORTS_DIR / "model_comparison_regression.csv", index_col=0
        ).sort_values("temporal_r2_log", ascending=False)
        show_cols = [c for c in ["r2_log", "mae_t_per_ha", "temporal_r2_log",
                                 "temporal_mae_t_per_ha", "cv_r2_mean"] if c in reg_tbl.columns]
        st.dataframe(
            reg_tbl[show_cols].style.format("{:.4f}"),
            width="stretch",
        )

    st.divider()
    left, right = st.columns(2)
    with left:
        st.write("**The finding worth reporting**")
        spread = clf_tbl["accuracy"].max() - clf_tbl["accuracy"].min()
        st.info(
            f"On crop recommendation, all algorithms land within "
            f"{spread:.1%} of each other ({clf_tbl['accuracy'].min():.1%}-"
            f"{clf_tbl['accuracy'].max():.1%}), on a dataset with exactly 100 rows "
            "per class. The benchmark is too easily separable to discriminate "
            "between algorithms - so the interesting contribution is the "
            "explanation, the economics and the harder yield task."
        )
        st.write("**Why the temporal split matters**")
        best_row = reg_tbl.iloc[0]
        st.info(
            f"The shipped model ({reg_tbl.index[0]}) scores "
            f"{best_row['r2_log']:.3f} R2 on the random split but "
            f"{best_row['temporal_r2_log']:.3f} on unseen years - and models that "
            "look similar on the random split separate cleanly out-of-time. "
            "Selecting on the random split alone would pick the wrong model."
        )
    with right:
        with st.expander("Data cleaning decisions"):
            st.json(prod_stats)
        with st.expander("Conformal calibration"):
            st.json(conformal)
        with st.expander("Data provenance"):
            st.write("Production:", config.PROD_SOURCE)
            st.write("Recommendation:", config.REC_SOURCE)
            st.write("Prices:", price_meta["source"])

    st.caption(
        f"Seed {config.SEED} · test size {config.TEST_SIZE} · "
        f"models in `{config.MODELS_DIR.name}/` · "
        f"trained {datetime.fromtimestamp(config.METRICS_FILE.stat().st_mtime):%Y-%m-%d %H:%M}"
    )

st.divider()
st.caption(
    "\u201cThe farmer is the backbone of our nation.\u201d - Lal Bahadur Shastri · "
    "Prices: Agmarknet/data.gov.in (Govt. of India, open data) · "
    "Production: DES, Ministry of Agriculture & Farmers Welfare"
)
