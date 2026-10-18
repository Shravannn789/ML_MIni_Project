"""Live mandi (wholesale market) prices from Agmarknet via data.gov.in.

Run with:  python -m src.market          # refreshes the local cache and prints

Data path:

* **Live** - ``api.data.gov.in`` "Current Daily Price of Various Commodities
  from Various Markets (Mandi)", the open-data mirror of Agmarknet. Several
  public documentation keys are rotated when one is rate-limited.
* **Cache** - the last successful fetch is written to ``data/cache/`` with a
  timestamp; refreshes are attempted at most once per ``MANDI_CACHE_TTL_MINUTES``.
* **Snapshot** - a bundled copy (``data/raw/mandi_prices_snapshot.csv``) keeps
  the app usable offline; its fetch date is shown in the UI so an old snapshot
  can never masquerade as live data.

Prices are in Rs per quintal (100 kg), as Agmarknet reports them.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from src import config

# Dataset crop name -> Agmarknet commodity names to try, in order. Agmarknet's
# naming is not standardised ("Gram" vs "Bengal Gram(Gram)(Whole)"), so each
# crop lists the exact names observed in the live feed first, then looser
# fallbacks. Matching is exact-then-contains, longest alias first.
CROP_TO_COMMODITY: dict[str, list[str]] = {
    "Rice": ["Rice"],
    "Wheat": ["Wheat"],
    "Maize": ["Maize"],
    "Jowar": ["Jowar(Sorghum)", "Jowar"],
    "Bajra": ["Bajra(Pearl Millet/Cumbu)", "Bajra"],
    "Ragi": ["Ragi(Finger Millet)", "Ragi"],
    "Small Millets": ["Small Millets"],
    "Arhar/Tur": ["Red gram/Arhar/Tur(whole)", "Tur(Arhar)(Whole)", "Red Gram"],
    "Moong(Green Gram)": ["Green Gram(Moong)(Whole)", "Moong", "Green Gram"],
    "Urad": ["Black Gram(Urd Beans)(Whole)", "Urad", "Black Gram"],
    "Gram": ["Bengal Gram(Gram)(Whole)", "Bengal Gram Dal(Chana Dal)", "Gram"],
    "Masoor": ["Lentil(Masur)(Whole)", "Masoor", "Lentil"],
    "Horse-Gram": ["Horse Gram"],
    "Groundnut": ["Groundnut"],
    "Soyabean": ["Soyabean"],
    "Sunflower": ["Sunflower/Sunflower Seed", "Sunflower"],
    "Sesamum": ["Sesamum(Sesame,Gingelly,Til)", "Sesame", "Sesamum"],
    "Rapeseed &Mustard": ["Mustard", "Rapeseed &Mustard"],
    "Linseed": ["Linseed"],
    "Niger Seed": ["Niger Seed"],
    "Castor Seed": ["Castor Seed"],
    "Cotton(Lint)": ["Cotton"],
    "Sugarcane": ["Sugarcane"],
    "Jute": ["Jute"],
    "Mesta": ["Mesta"],
    "Tobacco": ["Tobacco"],
    "Onion": ["Onion"],
    "Potato": ["Potato"],
    "Sweet Potato": ["Sweet Potato"],
    "Tapioca": ["Tapioca"],
    "Turmeric": ["Turmeric"],
    "Ginger": ["Ginger(Green)", "Ginger"],
    "Dry Chillies": ["Dry Chillies"],
    "Garlic": ["Garlic"],
    "Peas": ["Green Peas", "Peas Wet", "Peas"],
    "Banana": ["Banana"],
    "Mango": ["Mango", "Mango(Raw-Ripe)"],
    "Coconut": ["Coconut"],
    "Arecanut": ["Arecanut(Betelnut/Supari)", "Arecanut"],
    "Cashewnut": ["Cashew"],
    "Papaya": ["Papaya"],
    "Guava": ["Guava"],
    "Apple": ["Apple"],
    "Grapes": ["Grapes"],
    "Orange": ["Orange"],
    "Lemon": ["Lemon"],
    "Pomegranate": ["Pomegranate"],
}

_PRICE_COLUMNS = [
    "state", "district", "market", "commodity", "variety", "grade",
    "arrival_date", "min_price", "max_price", "modal_price",
]


# --------------------------------------------------------------------- fetch
def _fetch_one(commodity: str, key: str) -> list[dict]:
    r = requests.get(
        config.MANDI_API_BASE,
        params={
            "api-key": key,
            "format": "json",
            "limit": 250,
            "filters[commodity]": commodity,
        },
        timeout=config.MANDI_TIMEOUT_SECONDS,
        headers={"User-Agent": "AgroStar-ML-project/1.0"},
    )
    r.raise_for_status()
    return r.json().get("records", [])


def fetch_live(crops: list[str] | None = None, *, pause: float = 0.35) -> pd.DataFrame:
    """Fetch current mandi listings for the given dataset crop names.

    Falls through the public API keys on rate-limit errors. Returns an empty
    frame if the network is unreachable - callers decide what to do next.
    """
    crops = list(CROP_TO_COMMODITY) if crops is None else crops
    keys = list(config.MANDI_API_KEYS)
    rows: list[dict] = []

    for crop in crops:
        aliases = CROP_TO_COMMODITY.get(crop, [crop])
        fetched = False
        for alias in aliases:
            for key in keys:
                try:
                    records = _fetch_one(alias, key)
                    rows.extend(records)
                    fetched = True
                    break
                except requests.RequestException:
                    keys.append(keys.pop(0))  # rotate: maybe another key works
                    time.sleep(0.5)
                except ValueError:
                    break  # this alias is unknown; try the next one
            if fetched:
                break
            time.sleep(0.2)
        time.sleep(pause)

    if not rows:
        return pd.DataFrame(columns=_PRICE_COLUMNS + ["fetched_at"])
    df = pd.DataFrame(rows)
    df = df[[c for c in _PRICE_COLUMNS if c in df.columns]].copy()
    for col in ("min_price", "max_price", "modal_price"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["modal_price"])
    df = df[df["modal_price"] >= 10]  # Agmarknet contains 0.01 Rs junk rows
    df["fetched_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    return df.drop_duplicates()


def load_snapshot() -> pd.DataFrame:
    """The bundled offline copy, whatever date it was taken on."""
    if not config.MANDI_SNAPSHOT_FILE.exists():
        return pd.DataFrame(columns=_PRICE_COLUMNS + ["fetched_at"])
    df = pd.read_csv(config.MANDI_SNAPSHOT_FILE)
    df["fetched_at"] = df["fetched_at"].astype(str)
    return df


# ------------------------------------------------------------------ caching
def get_prices(*, force_refresh: bool = False, ttl_minutes: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (prices, meta). Live if possible, cache, then snapshot.

    ``meta`` always says where the numbers came from and when they were fetched;
    the UI must surface that rather than implying every number is real-time.
    """
    ttl = config.MANDI_CACHE_TTL_MINUTES if ttl_minutes is None else ttl_minutes
    cache = config.MANDI_CACHE_FILE

    if not force_refresh and cache.exists():
        cached = pd.read_csv(cache)
        age_min = (datetime.now() - datetime.fromtimestamp(cache.stat().st_mtime)).total_seconds() / 60
        if age_min < ttl and len(cached):
            return cached, {
                "source": "Agmarknet (cached live data)",
                "live": True,
                "fetched_at": str(cached["fetched_at"].iloc[0]),
                "n_listings": len(cached),
            }

    live = fetch_live()
    if len(live):
        live.to_csv(cache, index=False)
        return live, {
            "source": "Agmarknet via data.gov.in (live)",
            "live": True,
            "fetched_at": str(live["fetched_at"].iloc[0]),
            "n_listings": len(live),
        }

    snap = load_snapshot()
    return snap, {
        "source": "bundled snapshot (offline fallback)",
        "live": False,
        "fetched_at": str(snap["fetched_at"].iloc[0]) if len(snap) else "n/a",
        "n_listings": len(snap),
    }


# ---------------------------------------------------------------- summaries
def commodity_summary(prices: pd.DataFrame) -> pd.DataFrame:
    """One row per commodity: spread of today's modal prices across mandis."""
    if not len(prices):
        return pd.DataFrame()
    prices = prices[prices["modal_price"] >= 10]
    g = prices.groupby("commodity")
    out = pd.DataFrame({
        "listings": g.size(),
        "median_price": g["modal_price"].median().round(0),
        "p25": g["modal_price"].quantile(0.25).round(0),
        "p75": g["modal_price"].quantile(0.75).round(0),
        "min_price": g["modal_price"].min(),
        "max_price": g["modal_price"].max(),
        "states": g["state"].nunique(),
    }).sort_values("median_price", ascending=False)
    out.index.name = "commodity"
    return out.reset_index()


def state_price_table(prices: pd.DataFrame, commodity: str) -> pd.DataFrame:
    """Per-state median modal price for one commodity."""
    sub = prices[prices["commodity"].str.lower() == commodity.lower()]
    if not len(sub):
        return pd.DataFrame()
    out = sub.groupby("state").agg(
        listings=("modal_price", "size"),
        median_price=("modal_price", "median"),
        min_price=("modal_price", "min"),
        max_price=("modal_price", "max"),
    ).round(0).sort_values("median_price", ascending=False)
    return out.reset_index()


def _match_rows(prices: pd.DataFrame, crop: str) -> pd.DataFrame:
    """Rows whose commodity name matches a dataset crop, best alias first.

    Exact alias matches win outright; otherwise the longest alias that occurs
    as a substring decides, so "Gram" prefers "Bengal Gram(Gram)(Whole)" over
    unrelated dals. A floor on modal_price drops the 0.01 Rs data-entry junk
    that Agmarknet regularly contains.
    """
    aliases = sorted(
        (a.lower() for a in CROP_TO_COMMODITY.get(crop, [crop])),
        key=len,
        reverse=True,
    )
    names = prices["commodity"].str.lower()
    for alias in aliases:
        exact = names[names == alias]
        if len(exact) >= 3:
            return prices.loc[exact.index]
    for alias in aliases:
        contains = prices[names.str.contains(alias, regex=False, na=False)]
        if len(contains):
            return contains
    return prices.iloc[0:0]


def price_for_crop(
    prices: pd.DataFrame, crop: str, state: str | None = None
) -> dict | None:
    """Best available Rs/quintal quote for a dataset crop.

    Tries the state first (local mandis beat a national median); falls back to
    the national aggregate; returns None when the crop is not traded in the
    data at all (e.g. the snapshot is stale or the commodity is rare).
    """
    sub = _match_rows(prices, crop)
    sub = sub[sub["modal_price"] >= 10]
    if not len(sub):
        return None

    scope, label = sub, "national"
    if state:
        state_l = state.lower()
        local = sub[sub["state"].str.lower() == state_l]
        # Agmarknet spells states differently ("Keralam" vs "Kerala"), so also
        # accept a loose contains match before giving up on locality.
        if not len(local):
            local = sub[sub["state"].str.lower().str.contains(state_l, na=False)]
        if len(local) >= 3:
            scope, label = local, f"state ({state})"

    return {
        "commodity": scope["commodity"].mode().iloc[0],
        "median_price": float(scope["modal_price"].median()),
        "p25": float(scope["modal_price"].quantile(0.25)),
        "p75": float(scope["modal_price"].quantile(0.75)),
        "min_price": float(scope["modal_price"].min()),
        "max_price": float(scope["modal_price"].max()),
        "n_listings": int(len(scope)),
        "scope": label,
        "latest_arrival": str(scope["arrival_date"].max()) if "arrival_date" in scope else "n/a",
    }


if __name__ == "__main__":  # python -m src.market
    print("fetching live mandi prices from Agmarknet ...")
    prices, meta = get_prices(force_refresh=True)
    print(f"  {meta['source']} - {meta['n_listings']} listings, fetched {meta['fetched_at']}")
    summary = commodity_summary(prices)
    if len(summary):
        print(summary.to_string(index=False))
