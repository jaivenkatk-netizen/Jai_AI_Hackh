"""Derivation rules shared by the pipeline and the API.

Bucketing and confidence scoring live here rather than in either consumer so
the number on a screen can never disagree with the number in DuckDB. Pure
functions, no I/O, no model imports - safe to use anywhere.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm

from contracts.schemas import (
    PLANNED_MAX_DAYS,
    SERVICE_LEVEL,
    URGENT_MAX_DAYS,
    URGENT_THRESHOLD,
    Bucket,
    Confidence,
    ConfidenceBadge,
    Criticality,
    TTMBand,
)

# --------------------------------------------------------------------------
# AFT survival algebra
# --------------------------------------------------------------------------
# The XGBoost AFT model is trained on *remaining* life with
# aft_loss_distribution="normal", so for a component with feature vector x:
#
#     log(TTM) ~ Normal(mu(x), sigma)
#
# where mu = booster.predict(x) and sigma is the fitted
# aft_loss_distribution_scale, widened by the calibration multiplier `s`
# from models/conformal.py. Because the label is remaining life, the
# conditioning on "survived until now" is already carried by the features
# (age, days since PM, repair count) - there is no S(t+h)/S(t) renormalising
# to do, which is the main reason this parameterisation was chosen.


def ttm_band(mu: float, sigma: float, calibration: float = 1.0) -> TTMBand:
    """Read P10/P50/P90 off the fitted log-normal TTM distribution."""
    s = sigma * calibration
    return TTMBand(
        p10_days=float(math.exp(mu + s * norm.ppf(0.10))),
        p50_days=float(math.exp(mu)),
        p90_days=float(math.exp(mu + s * norm.ppf(0.90))),
        sigma_log=float(s),
    )


def p_fail_within(mu: float, sigma: float, horizon_days: float, calibration: float = 1.0) -> float:
    """P(TTM <= horizon) = Phi((ln h - mu) / sigma). Drives parts demand."""
    if horizon_days <= 0:
        return 0.0
    s = sigma * calibration
    if s <= 0:
        return 1.0 if math.log(horizon_days) >= mu else 0.0
    return float(norm.cdf((math.log(horizon_days) - mu) / s))


def ttm_quantile(mu: float, sigma: float, q: float, calibration: float = 1.0) -> float:
    """Inverse: the TTM at quantile q, in days."""
    return float(math.exp(mu + sigma * calibration * norm.ppf(q)))


def sample_ttm(
    mu: np.ndarray,
    sigma: np.ndarray,
    draws: int,
    rng: np.random.Generator,
    calibration: float = 1.0,
) -> np.ndarray:
    """Vectorised TTM draws, shape (len(mu), draws). The Monte Carlo core.

    Kept here so the parts-demand simulation and the inventory policy sample
    from provably the same distribution the UI displays.
    """
    mu = np.asarray(mu, dtype=float).reshape(-1, 1)
    s = np.asarray(sigma, dtype=float).reshape(-1, 1) * calibration
    return np.exp(rng.normal(loc=mu, scale=np.maximum(s, 1e-9), size=(mu.shape[0], draws)))


# --------------------------------------------------------------------------
# Bucketing
# --------------------------------------------------------------------------


def assign_bucket(
    p_fail_14d: float,
    p50_days: float,
    criticality: Criticality,
    has_active_critical_fault: bool = False,
) -> Bucket:
    """Urgent / Planned / Forecast, per the brief's rule.

    Urgent is driven by *probability*, not the median - a component with a
    P50 of 40 days but a fat left tail can still be urgent, and for a
    class-A asset it should be. Planned and Forecast then split on the median.
    """
    if has_active_critical_fault:
        return Bucket.URGENT
    if p_fail_14d >= URGENT_THRESHOLD[criticality]:
        return Bucket.URGENT
    if p50_days <= URGENT_MAX_DAYS:
        return Bucket.URGENT
    if p50_days <= PLANNED_MAX_DAYS:
        return Bucket.PLANNED
    return Bucket.FORECAST


def bucket_of_days(days: float) -> Bucket:
    """Pure horizon bucketing for a sampled failure time (used inside the MC)."""
    if days <= URGENT_MAX_DAYS:
        return Bucket.URGENT
    if days <= PLANNED_MAX_DAYS:
        return Bucket.PLANNED
    return Bucket.FORECAST


def bucket_edges() -> list[tuple[Bucket, float, float]]:
    return [
        (Bucket.URGENT, 0.0, float(URGENT_MAX_DAYS)),
        (Bucket.PLANNED, float(URGENT_MAX_DAYS), float(PLANNED_MAX_DAYS)),
        (Bucket.FORECAST, float(PLANNED_MAX_DAYS), float("inf")),
    ]


# --------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------

# Default "wide band" threshold, as a multiple of the band's own median - a
# relative measure, because 20 days of uncertainty means very different things
# on a 30-day and a 600-day prediction.
#
# Callers should pass the fleet's own upper-tercile relative width instead of
# relying on this default. A fixed threshold is a trap: set at 1.5 against a
# fleet whose median relative width was 2.1, it fired for almost every
# component, so the badge stopped discriminating. Judging each band against the
# fleet distribution keeps "wide" meaning "wide compared to what this model
# usually achieves".
_WIDE_BAND_RATIO = 2.5


def confidence_badge(
    band: TTMBand,
    sensor_coverage: float,
    hours_since_last_reading: float,
    imputed_share: float,
    ood_score: float,
    wide_band_ratio: float = _WIDE_BAND_RATIO,
) -> ConfidenceBadge:
    """Combine interval width, data quality and OOD distance into a badge.

    Deliberately a transparent rule rather than a learned score: a judge can
    ask "why Medium?" and get an answer, which is the point of the badge.
    """
    freshness = math.exp(-max(hours_since_last_reading, 0.0) / 72.0)  # 3-day half-life-ish
    quality = float(
        np.clip(sensor_coverage, 0, 1) * freshness * (1.0 - float(np.clip(imputed_share, 0, 1)))
    )

    band_ratio = band.width_days / max(band.p50_days, 1e-6)
    reasons: list[str] = []

    demerits = 0
    if band_ratio > wide_band_ratio:
        demerits += 1
        reasons.append(f"Wide TTM band ({band.p10_days:.0f}-{band.p90_days:.0f} d)")
    if quality < 0.6:
        demerits += 1
        if hours_since_last_reading > 48:
            reasons.append(f"{hours_since_last_reading / 24:.0f}-day telemetry gap")
        if imputed_share > 0.05:
            reasons.append(f"{imputed_share:.0%} of readings imputed")
        if sensor_coverage < 0.8:
            reasons.append(f"Only {sensor_coverage:.0%} of expected sensors reporting")
    if ood_score > 1.0:
        demerits += 1
        reasons.append("Asset sits outside the training distribution")

    level = Confidence.HIGH if demerits == 0 else (Confidence.MEDIUM if demerits == 1 else Confidence.LOW)
    if not reasons:
        reasons.append("Narrow band, fresh and complete telemetry")

    return ConfidenceBadge(
        level=level,
        interval_width_days=band.width_days,
        data_quality=quality,
        ood_score=float(ood_score),
        reasons=reasons,
    )


# --------------------------------------------------------------------------
# Inventory
# --------------------------------------------------------------------------


def service_level(criticality: Criticality) -> float:
    return SERVICE_LEVEL[criticality]


def safety_stock_closed_form(
    mean_demand_per_day: float,
    sd_demand_per_day: float,
    lead_time_mean_d: float,
    lead_time_sd_d: float,
    criticality: Criticality,
) -> float:
    """SS = z * sqrt(L * sd_D^2 + D^2 * sd_L^2).

    The *explanatory* form, for the slide and the recommendation card text.
    Real numbers come from the Monte Carlo in planning/inventory_policy.py,
    which needs no normality assumption; this is here so the card can say
    "a wider band raises safety stock" and show the algebra behind it.
    """
    z = float(norm.ppf(service_level(criticality)))
    var = lead_time_mean_d * sd_demand_per_day**2 + (mean_demand_per_day**2) * lead_time_sd_d**2
    return float(z * math.sqrt(max(var, 0.0)))


__all__ = [
    "ttm_band",
    "p_fail_within",
    "ttm_quantile",
    "sample_ttm",
    "assign_bucket",
    "bucket_of_days",
    "bucket_edges",
    "confidence_badge",
    "service_level",
    "safety_stock_closed_form",
]
