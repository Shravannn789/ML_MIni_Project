"""Central configuration: paths, seeds, feature lists and dataset provenance.

Everything that a report needs to record for reproducibility (§ "settings")
lives here rather than being scattered through the scripts.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"

for _p in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- experiment settings
SEED = 42                 # every split / model uses this, never the default None
TEST_SIZE = 0.20          # stratified 80:20
CV_FOLDS = 5              # fold count for cross-validation on the training split

# ---------------------------------------------------------------- task 1: crop recommendation
REC_FILE = DATA_RAW / "crop_recommendation.csv"
REC_SOURCE = (
    "HuggingFace mirror 'randalakab/Crop-recommendation' of the Kaggle "
    "'Crop Recommendation Dataset'; original attribution required in the report"
)
REC_FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
REC_TARGET = "label"      # 22 balanced classes, 100 rows each

# ---------------------------------------------------------------- task 2: yield forecasting
PROD_FILE = DATA_RAW / "crop_production.csv"
PROD_SOURCE = (
    "HuggingFace mirror 'jason1966/abhinand05_crop-production-in-india' of the "
    "Kaggle 'Crop Production in India' dataset (1997-2015, district level)"
)
PROD_CATEGORICAL = ["State_Name", "District_Name", "Crop", "Season"]
PROD_NUMERIC = ["Crop_Year", "Area"]
YIELD_TARGET = "yield_t_per_ha"   # derived: Production / Area

# A global cap fails here: legitimate crops differ by two orders of magnitude
# (sugarcane ~53 t/ha, rice ~2 t/ha), so a single threshold either keeps the
# unit-entry errors or throws away real harvests. Winsorise *within each crop*
# at this quantile instead. 969 rows show yields above 1,000 t/ha, which are
# unit-entry errors (e.g. Punjab sugarcane with area recorded in acres).
YIELD_CLIP_QUANTILE = 0.995

# Absolute ceiling in tonnes/hectare. No field crop exceeds roughly 200 t/ha
# (sugarcane is the highest, ~80-150). Anything above this is a unit-entry error
# and is removed before the per-crop winsorising step.
YIELD_CEILING = 250.0

# ---------------------------------------------------------------- artifact names
REC_MODEL_FILE = MODELS_DIR / "crop_recommendation.joblib"
YIELD_MODEL_FILE = MODELS_DIR / "yield_forecast.joblib"
METRICS_FILE = MODELS_DIR / "metrics.json"
