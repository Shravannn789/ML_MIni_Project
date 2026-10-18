"""Dataset loading and cleaning.

Two tasks, two sources:

1. ``crop_recommendation.csv`` - 2,200 rows, 7 soil/climate features, 22 crop classes.
2. ``crop_production.csv``     - 246,091 rows of Indian district-level production,
   which we turn into a *yield* regression problem (tonnes per hectare).

Every cleaning decision is explained where it happens, because §8 of the report
has to justify each one.
"""
from __future__ import annotations

import pandas as pd

from src import config


# --------------------------------------------------------------------- task 1
def load_recommendation() -> pd.DataFrame:
    """Load the crop-recommendation table (already numeric, no missing values)."""
    if not config.REC_FILE.exists():
        raise FileNotFoundError(
            f"{config.REC_FILE} missing. Re-download from {config.REC_SOURCE}."
        )
    df = pd.read_csv(config.REC_FILE)
    missing = [c for c in config.REC_FEATURES + [config.REC_TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"crop_recommendation.csv is missing columns: {missing}")
    return df


# --------------------------------------------------------------------- task 2
def load_production() -> pd.DataFrame:
    """Load the raw district-level production table, unmodified."""
    if not config.PROD_FILE.exists():
        raise FileNotFoundError(
            f"{config.PROD_FILE} missing. Re-download from {config.PROD_SOURCE}."
        )
    return pd.read_csv(config.PROD_FILE)


def build_yield_frame(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Clean production data and derive the regression target.

    Cleaning rules, in order:

    * Drop rows where ``Production`` or ``Area`` is missing (3,730 rows).
    * Drop rows where either is <= 0. A zero production row is not a real harvest
      observation, and a zero/negative area makes the ratio undefined (3,523 rows).
    * ``yield = Production / Area`` (tonnes per hectare).
    * Remove yields above ``YIELD_CEILING`` (250 t/ha) as unit errors.
    * Winsorise what remains **within each crop** at the 99.5th percentile.
      A global threshold cannot work: real yields differ by two orders of
      magnitude between crops (sugarcane ~53 t/ha, rice ~2 t/ha). The source also
      contains unit-entry errors - 969 rows exceed 1,000 t/ha, e.g. Punjab
      sugarcane with Area=3 and Production=264,000, i.e. acres recorded as hectares.
      Clipping per crop removes those without discarding legitimate high yields.

    Returns the cleaned frame plus a dict of counts for the report.
    """
    raw = load_production() if df is None else df.copy()
    n_raw = len(raw)

    work = raw.dropna(subset=["Production", "Area"]).copy()
    n_dropped_missing = n_raw - len(work)

    work = work[(work["Production"] > 0) & (work["Area"] > 0)].copy()

    # counted explicitly so the figures quoted in the report cannot drift
    after_missing = n_raw - n_dropped_missing
    n_dropped_nonpositive = after_missing - len(work)

    work[config.YIELD_TARGET] = work["Production"] / work["Area"]

    # Step 1: drop physiologically impossible yields. A per-crop quantile alone
    # is not enough - coconut's own 99.5th percentile is still ~19,912 t/ha,
    # because more than 0.5% of its rows are unit errors.
    before_ceiling = len(work)
    work = work[work[config.YIELD_TARGET] <= config.YIELD_CEILING].copy()
    n_dropped_implausible = before_ceiling - len(work)

    # Step 2: per-crop cap, applied with a transform so the index lines up
    caps = work.groupby("Crop")[config.YIELD_TARGET].transform(
        lambda s: s.quantile(config.YIELD_CLIP_QUANTILE)
    )
    n_clipped = int((work[config.YIELD_TARGET] > caps).sum())
    work[config.YIELD_TARGET] = work[config.YIELD_TARGET].clip(upper=caps)

    # Guard: the year range matters because the report claims rainfall cannot be
    # joined to this table (the rainfall archive starts where this one ends).
    stats = {
        "rows_raw": n_raw,
        "dropped_missing": int(n_dropped_missing),
        "dropped_nonpositive": int(n_dropped_nonpositive),
        "rows_modelled": len(work),
        "n_dropped_implausible": int(n_dropped_implausible),
        "n_clipped": n_clipped,
        "clip_quantile": config.YIELD_CLIP_QUANTILE,
        "ceiling": config.YIELD_CEILING,
        "year_min": int(work["Crop_Year"].min()),
        "year_max": int(work["Crop_Year"].max()),
        "n_crops": int(work["Crop"].nunique()),
        "n_states": int(work["State_Name"].nunique()),
        "n_districts": int(work["District_Name"].nunique()),
        "target_median": float(work[config.YIELD_TARGET].median()),
        "target_mean": float(work[config.YIELD_TARGET].mean()),
    }
    return work, stats


if __name__ == "__main__":  # quick manual check: python -m src.data_loader
    rec = load_recommendation()
    print(f"recommendation: {rec.shape}, classes={rec[config.REC_TARGET].nunique()}")

    prod, stats = build_yield_frame()
    print(f"production modelled: {prod.shape}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
