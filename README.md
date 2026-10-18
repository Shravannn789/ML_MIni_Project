# AgroStar

**Explainable AI for crop recommendation and yield forecasting.**

An ML mini project in two parts: a multi-class classifier that recommends a crop
from soil and climate readings, and a regressor that forecasts yield (t/ha) for a
district-level crop record. Both are explained per prediction with SHAP, so the
app answers *why*, not just *what*.

Every number below was measured by running the code in this repository.

---

## Results

### Crop recommendation (22 classes, 2,200 rows)

| Model | Accuracy | Macro F1 | CV accuracy (5-fold) |
|---|---|---|---|
| Random Forest | **0.9705** | 0.9702 | 0.9676 ± 0.0069 |
| XGBoost | 0.9523 | 0.9522 | 0.9568 ± 0.0116 |
| Decision Tree | 0.9409 | 0.9404 | 0.9250 ± 0.0199 |
| Logistic Regression | 0.9318 | 0.9311 | 0.9330 ± 0.0161 |

### Yield forecast (237,849 rows, target = log1p t/ha)

| Model | R² (log) | MAE (t/ha) | CV R² (3-fold) |
|---|---|---|---|
| Random Forest | **0.9485** | **0.88** | 0.9329 ± 0.0119 |
| Decision Tree | 0.9224 | 1.05 | 0.9016 ± 0.0136 |
| Hist Gradient Boosting | 0.9187 | 1.21 | 0.8972 ± 0.0127 |
| XGBoost | 0.9141 | 1.26 | 0.9021 ± 0.0146 |
| Ridge (baseline) | 0.8324 | 2.00 | 0.8272 ± 0.0027 |

---

## The finding worth reporting

**The crop-recommendation benchmark cannot discriminate between algorithms.**
All four models land within 3.9% of each other (93.2%–97.0%) on a dataset with
*exactly* 100 rows per class — a sign the data is curated rather than observed.
A report that presents "97% accuracy" as its contribution has nothing to say when
a logistic regression gets 93%. So the contribution is reframed:

1. **Report the saturation honestly** — the benchmark is easy, and that is itself
   a finding about the dataset, not a failure of the modelling.
2. **Put the difficulty in the yield task**, where the spread between models is
   wide (R² 0.83 → 0.95) and the preprocessing decisions actually matter.
3. **Explain both**, because an agronomist needs reasons, not probabilities.

---

## Quickstart

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash
pip install -r requirements.txt

python -m src.data_loader          # downloads nothing; verifies the raw data loads
python -m src.train                # trains both task families, writes models/ + metrics.json
python -m src.evaluate             # figures -> reports/figures/, summary tables
python -m src.explain              # SHAP global importance + worked examples
streamlit run app/streamlit_app.py # the application
```

Training takes about 40 seconds on a laptop CPU. No GPU is required.

---

## Layout

```
data/raw/                 crop_recommendation.csv, crop_production.csv (unmodified)
src/config.py             paths, seed, feature lists, provenance
src/data_loader.py        loading + all cleaning decisions
src/preprocess.py         pipelines (target encoding lives here)
src/train.py              model comparison for both tasks
src/evaluate.py           numbered figures + text reports
src/explain.py            SHAP, local and global
app/streamlit_app.py      the three-tab application
app/assets/               blurred farmland backdrop + image credit
models/                   saved pipelines + metrics.json
reports/figures/          Figure 1 .. Figure 9, referenced by the report
```

---

## Data

| File | Rows | Source |
|---|---|---|
| `crop_recommendation.csv` | 2,200 | HuggingFace mirror of the Kaggle *Crop Recommendation Dataset* |
| `crop_production.csv` | 246,091 | HuggingFace mirror of the Kaggle *Crop Production in India* dataset (1997–2015) |

**Note:** the mirrors declare no licence (`none` / `unknown`). For submission, cite the
original Kaggle datasets in the References section; treat the copies here as
mirrors, not authorities.

### Cleaning (documented, not silent)

246,091 rows → **237,849** modelled:

- 3,730 dropped for missing `Production`
- 3,523 dropped for zero/negative `Production` or `Area`
- 989 dropped for yields above 250 t/ha — unit-entry errors, e.g. Punjab sugarcane
  recorded with `Area = 3` and `Production = 264,000` (acres entered as hectares)
- 1,235 winsorised per crop at the 99.5th percentile, because legitimate yields
  differ by two orders of magnitude between crops (sugarcane ~53 t/ha, rice ~2 t/ha)

### Scope note

Rainfall forecasting was **excluded**: the IMD archive available to us covers
2015–2025 while the production archive covers 1997–2015, so they overlap in a
single year. No credible rainfall→yield relationship can be learned from that.

---

## Design decisions worth defending in a viva

- **Target encoding with `cv=5`**, inside the pipeline. Encoding on the full
  training set and scoring on the same rows leaks the target; cross-fitting is the
  difference between an honest 0.95 and a fiction.
- **Log-transformed target.** Raw yield has a median of 1.0 and a mean of 4.5 with
  a long right tail; squared error on the raw scale would be dominated by sugarcane.
- **Pipelines serialised whole.** The app loads the fitted pipeline, so there is no
  second copy of the preprocessing that can drift away from training.
- **Both models chosen on held-out data**, after checking the cross-validated score
  agrees. A model that wins by 0.2% while swinging 5% across folds is noise.

## UI

No emoji in the interface, and the theme is pinned to dark in
`.streamlit/config.toml` so the backdrop stays legible. The background is a
blurred aerial photo of Punjab farmlands (Wikimedia Commons, CC BY-SA 4.0),
credited in `app/assets/CREDITS.md` and in the app footer.

## Limitations

- Both datasets are single-country (India) snapshots, so results do not transfer
  to other agricultural contexts.
- The crop-recommendation features are soil/climate readings of unclear measurement
  provenance; the dataset's perfect class balance suggests synthesis.
- Yield R² is flattered by crop identity — the model is partly learning "rice yields
  what rice yields", which SHAP shows explicitly.
- No price or market data, so this is agronomy, not farm economics.
