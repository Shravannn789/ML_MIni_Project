"""Central configuration: paths, seeds, feature lists and dataset provenance.

Everything that a report needs to record for reproducibility (§ "settings")
lives here rather than being scattered through the scripts.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_CACHE = ROOT / "data" / "cache"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
REPORTS_DIR = ROOT / "reports"

for _p in (DATA_RAW, DATA_PROCESSED, DATA_CACHE, MODELS_DIR, FIGURES_DIR):
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
# Directorate of Economics & Statistics (DES), Ministry of Agriculture, via the
# India Data Portal: district-wise, season-wise crop production 1997-98 to
# 2022-23. Supersedes the Kaggle mirror that stopped at 2015.
PROD_FILE = DATA_RAW / "crop_production_1997_2023.csv"
PROD_LEGACY_FILE = DATA_RAW / "crop_production.csv"  # old Kaggle copy, kept for provenance
PROD_SOURCE = (
    "India Data Portal / DES, Ministry of Agriculture & Farmers Welfare: "
    "'Crop Wise Area Production Yield', district level, 1997-98 to 2022-23 "
    "(ODbL). Legacy Kaggle mirror (1997-2015) kept in data/raw."
)
PROD_CATEGORICAL = ["State_Name", "District_Name", "Crop", "Season"]
PROD_NUMERIC = ["Crop_Year", "Area"]
YIELD_TARGET = "yield_t_per_ha"   # derived: Production / Area

# A global cap fails here: legitimate crops differ by two orders of magnitude
# (sugarcane ~80 t/ha, rice ~2 t/ha), so a single threshold either keeps the
# unit-entry errors or throws away real harvests. Winsorise *within each crop*
# at this quantile instead.
YIELD_CLIP_QUANTILE = 0.995

# Absolute ceiling in tonnes/hectare. No field crop exceeds roughly 200 t/ha
# (sugarcane is the highest, ~80-150). Anything above this is a unit-entry error
# and is removed before the per-crop winsorising step.
YIELD_CEILING = 250.0

# Split-conformal prediction interval for yield (applied on the log target).
CONFORMAL_COVERAGE = 0.90      # nominal coverage of the reported interval
CONFORMAL_CALIB_FRACTION = 0.10  # share of training rows held out for calibration

# ---------------------------------------------------------------- live market data (Agmarknet)
MANDI_RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"
MANDI_API_BASE = f"https://api.data.gov.in/resource/{MANDI_RESOURCE_ID}"
# Public "sample" keys published in data.gov.in documentation; rotated if one
# is rate-limited. Never a secret - the endpoint is an open government API.
MANDI_API_KEYS = [
    "579b464db66ec23bdd0000012041aca96faa47a4602429dc165cd658",
    "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b",
    "579b464db66ec23bdd000001e86e1a57509c44215c7df6245ef774df",
    "579b464db66ec23bdd00000179e2c6795a5648977db569a70e54be63",
]
MANDI_SNAPSHOT_FILE = DATA_RAW / "mandi_prices_snapshot.csv"  # bundled offline fallback
MANDI_CACHE_FILE = DATA_CACHE / "mandi_prices.csv"            # refreshed live cache
MANDI_CACHE_TTL_MINUTES = 360
MANDI_TIMEOUT_SECONDS = 25

# ---------------------------------------------------------------- economics
# Approximate all-India cost of cultivation, A2+FL basis (DES/CCE norms,
# order-of-magnitude). Used only as a default; the UI lets the user override.
COST_OF_CULTIVATION_INR_PER_HA = {
    "Rice": 40_000, "Wheat": 32_000, "Maize": 26_000, "Jowar": 22_000,
    "Bajra": 20_000, "Ragi": 20_000, "Small Millets": 18_000,
    "Arhar/Tur": 26_000, "Moong(Green Gram)": 24_000, "Urad": 24_000,
    "Gram": 24_000, "Masoor": 24_000, "Horse-Gram": 18_000,
    "Groundnut": 28_000, "Soyabean": 26_000, "Sunflower": 26_000,
    "Sesamum": 24_000, "Rapeseed &Mustard": 24_000, "Linseed": 22_000,
    "Niger Seed": 18_000, "Castor Seed": 24_000,
    "Cotton(Lint)": 52_000, "Sugarcane": 130_000, "Jute": 38_000,
    "Mesta": 32_000, "Tobacco": 60_000, "Tea": 180_000, "Coffee": 150_000,
    "Onion": 55_000, "Potato": 60_000, "Sweet Potato": 40_000,
    "Tapioca": 35_000, "Turmeric": 45_000, "Ginger": 55_000,
    "Dry Chillies": 50_000, "Garlic": 60_000, "Peas": 30_000,
    "Banana": 90_000, "Mango": 100_000, "Coconut": 60_000,
    "Arecanut": 70_000, "Cashewnut": 80_000, "Papaya": 80_000,
    "Guava": 80_000, "Apple": 150_000, "Grapes": 120_000,
    "Orange": 100_000, "Lemon": 90_000, "Pomegranate": 110_000,
}
DEFAULT_COST_INR_PER_HA = 30_000
QUINTALS_PER_TONNE = 10.0

# ---------------------------------------------------------------- artifact names
REC_MODEL_FILE = MODELS_DIR / "crop_recommendation.joblib"
YIELD_MODEL_FILE = MODELS_DIR / "yield_forecast.joblib"
YIELD_CONFORMAL_FILE = MODELS_DIR / "yield_conformal.json"
METRICS_FILE = MODELS_DIR / "metrics.json"
