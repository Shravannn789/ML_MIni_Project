# AgroStar

**Explainable AI for crop recommendation, yield forecasting and farm economics.**

An ML mini project in two modelling tasks — a multi-class classifier that
recommends a crop from soil and climate readings, and a regressor that
forecasts yield (t/ha) for a district-level crop record — plus the two things
that turn an agronomic number into a decision: **live mandi prices** from
Agmarknet and an **estimated-earnings calculator**. Both models are explained
per prediction with SHAP, so the app answers *why*, not just *what*.

Every number below was measured by running the code in this repository.

---

## Results

### Crop recommendation (22 classes, 2,200 rows)

| Model | Accuracy | Macro F1 | CV accuracy (5-fold) |
|---|---|---|---|
| Random Forest | **0.9705** | 0.9702 | 0.9676 ± 0.0069 |
| Extra Trees | 0.9682 | 0.9681 | 0.9659 ± 0.0074 |
| XGBoost | 0.9523 | 0.9522 | 0.9568 ± 0.0116 |
| Decision Tree | 0.9409 | 0.9404 | 0.9250 ± 0.0199 |
| Logistic Regression | 0.9318 | 0.9311 | 0.9330 ± 0.0161 |

### Yield forecast (395,097 cleaned district records, 1997-98 to 2022-23)

Two evaluation lenses. The **random split** is standard but flattering: the
model has already seen other districts in the test year. The **temporal split**
(train ≤ 2018, test 2019-2022) is the regime the app deploys into, and it
decides which model ships.

| Model | R² (log, random) | MAE (t/ha) | R² (temporal) |
|---|---|---|---|
| Random Forest | **0.9559** | **0.78** | 0.9136 |
| XGBoost (tuned) — **shipped** | 0.9395 | 1.00 | **0.9496** |
| Hist Gradient Boosting (tuned) | 0.9341 | 1.07 | 0.9379 |
| Decision Tree | 0.9352 | 0.92 | 0.8866 |
| XGBoost | 0.9161 | 1.20 | 0.9032 |
| Hist Gradient Boosting | 0.9152 | 1.23 | 0.9023 |
| Ridge (linear baseline) | 0.8132 | 2.09 | 0.8148 |

The shipped model is tuned XGBoost, wrapped in **split-conformal calibration**
(90% coverage, interval half-width q = 0.331 on log yield), so every forecast in
the app comes with an honest interval, not just a point.

---

## The findings worth reporting

1. **The crop-recommendation benchmark cannot discriminate between
   algorithms.** All five models land within 3.9% of each other (93.2%–97.0%)
   on a dataset with *exactly* 100 rows per class — a sign the data is curated
   rather than observed. The contribution is therefore reframed around the
   explanation and the harder tasks, not the headline accuracy.
2. **Model rankings reverse when the evaluation regime changes.** Random
   Forest wins the random split (0.956 vs 0.940) but loses the temporal split
   (0.914 vs 0.950) to tuned XGBoost. Selecting on the random split alone —
   the default in most tutorials — would ship the wrong model for forecasting.
3. **A yield number without a price is only half a decision.** The app joins
   the forecast to live Agmarknet mandi quotes and a cost-of-cultivation
   default to produce revenue, net return and ROI scenarios.

---

## Quickstart

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash
pip install -r requirements.txt

python -m src.data_loader          # verifies the raw data loads and cleans
python -m src.train                # trains both task families -> models/ + metrics.json
python -m src.evaluate             # figures -> reports/figures/, summary tables
python -m src.explain              # SHAP global importance + worked examples
python -m src.market               # optional: refresh the live mandi cache
streamlit run app/streamlit_app.py # the application
```

Training takes about 3 minutes on a laptop CPU (including the hyperparameter
search and the temporal evaluation). No GPU is required. The app starts
instantly on the bundled mandi snapshot and upgrades to live prices when the
network allows.

---

## Layout

```
data/raw/                 crop_recommendation.csv, crop_production_1997_2023.csv,
                          mandi_prices_snapshot.csv (offline fallback, dated)
data/cache/               refreshed Agmarknet cache (git-ignored)
src/config.py             paths, seed, feature lists, provenance, price keys
src/data_loader.py        loading + all cleaning decisions
src/preprocess.py         pipelines (target encoding, log-area, temporal split)
src/train.py              comparison, tuning, temporal eval, conformal calibration
src/evaluate.py           numbered figures + text reports
src/explain.py            SHAP, local and global
src/market.py             live Agmarknet prices with cache + snapshot fallback
src/economics.py          revenue / cost / net / ROI from yield and price
app/streamlit_app.py      the five-tab application
app/assets/               blurred farmland backdrop + image credit
models/                   saved pipelines, conformal width, metrics.json
reports/figures/          Figure 1 .. Figure 9, referenced by the report
```

---

## Data

| File | Rows | Coverage | Source |
|---|---|---|---|
| `crop_recommendation.csv` | 2,200 | — | HuggingFace mirror of the Kaggle *Crop Recommendation Dataset* |
| `crop_production_1997_2023.csv` | 402,342 | 1997-98 to 2022-23 | DES / India Data Portal: *Crop Wise Area Production Yield*, district level (ODbL) |
| `mandi_prices_snapshot.csv` | ~5,300 | fetched 2026-10-18 | Agmarknet via `api.data.gov.in` (Govt. of India open data) |
| `crop_production.csv` (legacy) | 246,091 | 1997-2015 | old Kaggle mirror, kept for provenance |

The production data was upgraded from the Kaggle mirror (which stops at 2015)
to the Directorate of Economics & Statistics release that runs through the
2022-23 crop year — the newest district-level statistics published by the
Government of India. State-level advance estimates exist for 2023-26, but they
are not district-level, so they cannot be joined to this model's features.

### Cleaning (documented, not silent)

402,342 rows → **395,097** modelled:

- 3,969 dropped for zero/negative `Production` or `Area`
- 3,276 dropped for yields above 250 t/ha — unit-entry errors (acres recorded
  as hectares, coconut historically reported in nuts)
- 1,994 winsorised per crop at the 99.5th percentile, because legitimate
  yields differ by two orders of magnitude between crops (sugarcane ~80 t/ha,
  rice ~2 t/ha)
- Season `Total` rows are annual aggregates of Kharif + Rabi and are removed
  to prevent double counting

### Live market data

`src/market.py` pulls current daily mandi prices (modal, min, max per listing
in Rs/quintal) from the Agmarknet mirror on `api.data.gov.in`, caches them for
six hours in `data/cache/`, and falls back to the dated snapshot when the
network is unavailable. The UI always shows which source and fetch time the
prices came from; an old snapshot can never masquerade as live data.

### Scope note

Rainfall forecasting remains excluded: the production archive has no
district-level weather station join, and a credible rainfall→yield relationship
requires geospatial climate data beyond this project's scope.

---

## Design decisions worth defending in a viva

- **Selection on the temporal split.** The app forecasts years the model has
  not seen; the metric used to pick the model must match that regime. The
  random split is reported for comparison, not for selection.
- **Target encoding with `cv=5`**, inside the pipeline. Encoding on the full
  training set and scoring on the same rows leaks the target; cross-fitting is
  the difference between an honest score and a fiction.
- **Log-transformed target.** Raw yield has a median near 1 and a long right
  tail; squared error on the raw scale would be dominated by sugarcane.
- **Split-conformal intervals.** The shipped model is fit on 90% of the
  training rows; the 90th percentile of absolute residuals on the held-out 10%
  sets the interval width. Model-agnostic, one number, honest coverage.
- **Pipelines serialised whole.** The app loads the fitted pipeline, so there
  is no second copy of the preprocessing that can drift away from training —
  including the in-pipeline `log1p(Area)`.
- **Prices are never invented.** Every rupee figure traces to an Agmarknet
  listing or a field the user edited; crops that do not trade at mandis
  (e.g. sugarcane, which moves through mills) ask for a manual price.

## UI

Five tabs: Overview, Crop recommendation, Yield & earnings, Live market, and
Models & methods. Charts are Plotly (interactive, dark-themed to match the
backdrop); no emoji in the interface, and the theme is pinned to dark in
`.streamlit/config.toml`. The background is a blurred aerial photo of Punjab
farmlands (Wikimedia Commons, CC BY-SA 4.0), credited in
`app/assets/CREDITS.md` and in the app footer.

## Limitations

- Both modelling datasets are single-country (India) snapshots, so results do
  not transfer to other agricultural contexts.
- The crop-recommendation features are soil/climate readings of unclear
  measurement provenance; the dataset's perfect class balance suggests
  synthesis.
- Yield R² is flattered by crop identity — the model partly learns "rice
  yields what rice yields", which SHAP shows explicitly.
- Mandi modal prices are wholesale quotes, not the price a farmer receives
  after transport, commissions and grading; the earnings calculator is an
  order-of-magnitude tool, not an accounting system.
- Cost-of-cultivation defaults are all-India A2+FL approximations; real costs
  vary by state, irrigation and season, which is why the field is editable.
