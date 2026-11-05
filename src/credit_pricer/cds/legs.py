"""Protection and premium leg integrals for a CDS.

Two integration schemes are provided:

``isda``
    Exact integration on every sub-interval where both the hazard rate and the
    risk-free forward rate are constant (the ISDA standard model approach, without the
    historical half-day accrual bias).

``isda_halfday``
    As ``isda`` but with the half-day accrual-on-default bias of the original ISDA
    Standard Model (the accrued premium at default is overstated by half a day). This is
    QuantLib's ``IsdaCdsEngine`` default and is what its ``SpreadCdsHelper`` uses.

``midpoint``
    Default is assumed to happen in the middle of each premium period (calendar-date
    windows, coupons conditional on survival to the payment date, accrued premium paid
    at the mid-point). This follows the conventions of QuantLib's ``MidPointCdsEngine``.

Timing convention for ``isda`` (ISDA standard model): protection is effective from the start of the
step-in day ``T+1``, which in end-of-day terms is the trade date ``T``. A premium period
with accrual dates ``[s, e)`` is therefore exposed to default over ``[s-1, e-1]`` in
end-of-day terms; its coupon is paid only if the name survives to ``e-1``. Accrual on
default is linear in time over that window.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from credit_pricer.curves.discount import DiscountCurve
from credit_pricer.curves.survival import SurvivalCurve
from credit_pricer.dates import AccrualPeriod

FloatArray = NDArray[np.float64]
Method = Literal["isda", "isda_halfday", "midpoint"]

_ONE_DAY = dt.timedelta(days=1)
_TAYLOR = 1e-4


def _e1(k: FloatArray, length: FloatArray) -> FloatArray:
    """(1 - exp(-k L)) / k, stable for small k L."""
    x = k * length
    small = np.abs(x) < _TAYLOR
    safe_k = np.where(small, 1.0, k)
    exact = -np.expm1(-x) / safe_k
    series = length * (1.0 - x / 2.0 + x * x / 6.0)
    return np.asarray(np.where(small, series, exact), dtype=float)


def _e2(k: FloatArray, length: FloatArray) -> FloatArray:
    """(1 - exp(-k L) (1 + k L)) / k^2, stable for small k L."""
    x = k * length
    small = np.abs(x) < _TAYLOR
    safe_k = np.where(small, 1.0, k)
    exact = (-np.expm1(-x) - x * np.exp(-x)) / (safe_k * safe_k)
    series = length * length * (0.5 - x / 3.0 + x * x / 8.0)
    return np.asarray(np.where(small, series, exact), dtype=float)


@dataclass(frozen=True)
class PeriodValues:
    """Leg components for one premium period (per unit notional)."""

    coupon_annuity: float  # accrual * DF(pay) * Q(end), per unit of spread
    aod_annuity: float  # accrual-on-default value per unit of spread
    protection: float  # (1 - R) weighted default leg over the period window


def _exact_periods(
    periods: Sequence[AccrualPeriod],
    t_start: float,
    discount: DiscountCurve,
    survival: SurvivalCurve,
    lgd: float,
    half_day_bias: bool = False,
) -> list[PeriodValues]:
    """Exact integration, vectorised over every (period, curve-node) sub-interval.

    On a sub-interval ``[a, a + L]`` with constant hazard ``lam`` and forward ``f``
    (``k = lam + f``), writing ``B = P(a) Q(a)``:

    * protection:        ``lam B (1 - e^{-kL}) / k``
    * accrual on default ``rate lam B [d0 (1 - e^{-kL}) / k + (1 - e^{-kL}(1 + kL)) / k^2]``

    where ``d0`` is the time already accrued at ``a`` and ``rate`` converts curve time
    (ACT/365F) into ACT/360 accrual. ``half_day_bias`` adds half a day to ``d0``.
    """
    bias = 0.5 / 365.0 if half_day_bias else 0.0
    acc = np.array([p.accrual for p in periods])
    t_acc0 = np.array([discount.t(p.start - _ONE_DAY) for p in periods])  # end-of-day frame
    t_end = np.array([discount.t(p.end - _ONE_DAY) for p in periods])
    t_pay = np.array([discount.t(p.pay) for p in periods])
    coupon = acc * discount.df(t_pay) * survival.survival(np.maximum(t_end, 0.0))

    lo_win = np.maximum(t_acc0, t_start)
    active = t_end > lo_win
    n = len(periods)
    prot: NDArray[np.float64] = np.zeros(n)
    aod: NDArray[np.float64] = np.zeros(n)
    if np.any(active):
        nodes = np.concatenate((discount.node_times, survival.node_times))
        first, last = lo_win[active].min(), t_end[active].max()
        inner = nodes[(nodes > first) & (nodes < last)]
        grid = np.unique(np.concatenate((lo_win[active], t_end[active], inner)))
        lo, hi = grid[:-1], grid[1:]
        idx = np.searchsorted(t_end, lo, side="right")
        ok = idx < n
        ok[ok] &= (lo_win[idx[ok]] <= lo[ok]) & active[idx[ok]]
        lo, hi, idx = lo[ok], hi[ok], idx[ok]
        length = hi - lo
        q0, q1 = survival.survival(lo), survival.survival(hi)
        p0, p1 = discount.df(lo), discount.df(hi)
        lam = np.log(q0 / q1) / length
        k = lam + np.log(p0 / p1) / length
        base = lam * p0 * q0
        e1 = _e1(k, length)
        rate = acc[idx] / (t_end[idx] - t_acc0[idx])
        # asarray: older numpy stubs type weighted bincount as integer
        prot = np.asarray(np.bincount(idx, base * e1, minlength=n), dtype=np.float64)
        aod_w = rate * base * ((lo - t_acc0[idx] + bias) * e1 + _e2(k, length))
        aod = np.asarray(np.bincount(idx, aod_w, minlength=n), dtype=np.float64)
    return [
        PeriodValues(float(c), float(a), float(lgd * pr)) for c, a, pr in zip(coupon, aod, prot)
    ]


def _midpoint_period(
    p: AccrualPeriod,
    protection_start: dt.date,
    is_first: bool,
    is_last: bool,
    discount: DiscountCurve,
    survival: SurvivalCurve,
    lgd: float,
) -> PeriodValues:
    # Calendar-date windows (no end-of-day shift), as in QuantLib's MidPointCdsEngine.
    end = p.end - _ONE_DAY if is_last else p.end
    coupon = p.accrual * discount.df_date(p.pay) * float(survival.survival(survival.t(p.pay)))
    # first period: the default window opens at the protection start, even when accrual
    # starts later (trade dated the day before a coupon date)
    start = protection_start if is_first else max(p.start, protection_start)
    if end <= start:
        return PeriodValues(coupon, 0.0, 0.0)
    default_date = start + dt.timedelta(days=(end - start).days // 2)
    dq = float(survival.survival(survival.t(start)) - survival.survival(survival.t(end)))
    df_mid = discount.df_date(default_date)
    accrued_days = max((default_date - p.start).days + (1 if is_last else 0), 0)
    return PeriodValues(coupon, accrued_days / 360.0 * dq * df_mid, lgd * dq * df_mid)


def integrate_periods(
    periods: Sequence[AccrualPeriod],
    protection_start: dt.date,
    discount: DiscountCurve,
    survival: SurvivalCurve,
    recovery: float,
    method: Method,
) -> list[PeriodValues]:
    """Leg values per premium period with the chosen integration ``method``.

    ``protection_start`` is the trade date: protection runs from the start of the
    step-in day ``T+1``, i.e. from the end of ``T``.
    """
    lgd = 1.0 - recovery
    if method in ("isda", "isda_halfday"):
        return _exact_periods(
            periods, discount.t(protection_start), discount, survival, lgd,
            half_day_bias=method == "isda_halfday",
        )
    if method == "midpoint":
        n = len(periods)
        return [
            _midpoint_period(p, protection_start, i == 0, i == n - 1, discount, survival, lgd)
            for i, p in enumerate(periods)
        ]
    raise ValueError(f"unknown method {method!r}")
