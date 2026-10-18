"""Farm economics: turning a yield forecast and a mandi price into money.

All monetary figures are Indian rupees. Prices are Rs/quintal (Agmarknet
convention); yields are tonnes/hectare, so:

    production (t) = yield (t/ha) * area (ha)
    revenue        = production * 10 (quintals/t) * price (Rs/quintal)

Cost of cultivation defaults come from ``config.COST_OF_CULTIVATION_INR_PER_HA``,
which are order-of-magnitude all-India A2+FL norms. They are a convenience, not
an estimate of anyone's actual farm: the UI exposes the number as an editable
field precisely because real costs vary by state, irrigation and season.
"""
from __future__ import annotations

from src import config


def default_cost_per_ha(crop: str) -> float:
    """Approximate cost of cultivation (Rs/ha) for a dataset crop name."""
    if crop in config.COST_OF_CULTIVATION_INR_PER_HA:
        return float(config.COST_OF_CULTIVATION_INR_PER_HA[crop])
    # loose match: "Cotton(Lint)" vs dict key "Cotton(Lint)" is exact, but
    # tolerate case/parenthesis variants from newer datasets
    for key, val in config.COST_OF_CULTIVATION_INR_PER_HA.items():
        if key.lower().startswith(crop.lower()[:6]) and crop.lower()[:6]:
            return float(val)
    return float(config.DEFAULT_COST_INR_PER_HA)


def estimate_earnings(
    *,
    yield_t_per_ha: float,
    area_ha: float,
    price_inr_per_quintal: float,
    cost_inr_per_ha: float | None = None,
    crop: str = "",
) -> dict:
    """Expected economics for one harvest.

    Returns production, gross revenue, total cost, net return and ROI. Negative
    net returns are returned as-is rather than clipped: a forecast that loses
    money is information, not an error.
    """
    cost_per_ha = default_cost_per_ha(crop) if cost_inr_per_ha is None else float(cost_inr_per_ha)
    production_t = float(yield_t_per_ha) * float(area_ha)
    production_q = production_t * config.QUINTALS_PER_TONNE
    revenue = production_q * float(price_inr_per_quintal)
    total_cost = cost_per_ha * float(area_ha)
    net = revenue - total_cost
    roi = (net / total_cost * 100.0) if total_cost > 0 else float("nan")

    return {
        "yield_t_per_ha": float(yield_t_per_ha),
        "area_ha": float(area_ha),
        "production_t": production_t,
        "production_q": production_q,
        "price_inr_per_quintal": float(price_inr_per_quintal),
        "cost_inr_per_ha": cost_per_ha,
        "revenue_inr": revenue,
        "total_cost_inr": total_cost,
        "net_inr": net,
        "roi_pct": roi,
    }


def scenario_table(
    *,
    yield_t_per_ha: float,
    area_ha: float,
    price_inr_per_quintal: float,
    cost_inr_per_ha: float,
    crop: str = "",
    low_yield: float | None = None,
    high_yield: float | None = None,
    price_min: float | None = None,
    price_max: float | None = None,
) -> list[dict]:
    """Net-return scenarios across low/expected/high yield and price corners.

    Used by the app to show that an earnings estimate is a range, not a number.
    """
    yl = low_yield if low_yield is not None else yield_t_per_ha * 0.85
    yh = high_yield if high_yield is not None else yield_t_per_ha * 1.15
    pl = price_min if price_min is not None else price_inr_per_quintal * 0.9
    ph = price_max if price_max is not None else price_inr_per_quintal * 1.1

    cases = [
        ("Pessimistic", yl, pl),
        ("Expected", yield_t_per_ha, price_inr_per_quintal),
        ("Optimistic", yh, ph),
    ]
    return [
        {
            "scenario": name,
            **estimate_earnings(
                yield_t_per_ha=y,
                area_ha=area_ha,
                price_inr_per_quintal=p,
                cost_inr_per_ha=cost_inr_per_ha,
                crop=crop,
            ),
        }
        for name, y, p in cases
    ]
