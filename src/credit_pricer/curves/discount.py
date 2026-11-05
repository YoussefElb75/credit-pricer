"""Risk-free discount curve: log-linear discount factors, bootstrapped from deposits and swaps.

Curve time is measured in ACT/365F years from the valuation date. Between nodes the
log discount factor is linear, i.e. the instantaneous forward rate is piecewise flat;
beyond the last node the last forward rate is extended.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from credit_pricer.dates import add_months, adjust_modified_following, year_fraction

FloatArray = NDArray[np.float64]


def time_from(valuation_date: dt.date, d: dt.date) -> float:
    """Curve time (ACT/365F years) of ``d`` seen from ``valuation_date``."""
    return (d - valuation_date).days / 365.0


class DiscountCurve:
    """Piecewise-flat-forward discount curve.

    Parameters
    ----------
    valuation_date:
        Curve reference date (time 0, discount factor 1).
    times:
        Strictly increasing node times (> 0) in ACT/365F years.
    discount_factors:
        Discount factors at the node times.
    spread:
        Optional constant continuously compounded spread added to every forward rate
        (used e.g. for an issuer funding spread).
    """

    def __init__(
        self,
        valuation_date: dt.date,
        times: ArrayLike,
        discount_factors: ArrayLike,
        spread: float = 0.0,
    ) -> None:
        t = np.asarray(times, dtype=float)
        p = np.asarray(discount_factors, dtype=float)
        if t.ndim != 1 or t.shape != p.shape or t.size == 0:
            raise ValueError("times and discount_factors must be 1-d arrays of equal length")
        if np.any(t <= 0) or np.any(np.diff(t) <= 0):
            raise ValueError("node times must be positive and strictly increasing")
        if np.any(p <= 0):
            raise ValueError("discount factors must be positive")
        self.valuation_date = valuation_date
        self.times: FloatArray = t
        self.discount_factors: FloatArray = p
        self.spread = float(spread)
        self._t = np.concatenate(([0.0], t))
        self._logp = np.concatenate(([0.0], np.log(p)))
        if t.size >= 1:
            dt_last = self._t[-1] - self._t[-2]
            self._last_fwd = -(self._logp[-1] - self._logp[-2]) / dt_last
        else:  # pragma: no cover
            self._last_fwd = 0.0

    # ------------------------------------------------------------------ constructors
    @classmethod
    def flat(cls, valuation_date: dt.date, rate: float, horizon: float = 50.0) -> DiscountCurve:
        """Flat continuously compounded curve."""
        return cls(valuation_date, [horizon], [np.exp(-rate * horizon)])

    # ------------------------------------------------------------------ evaluation
    def t(self, d: dt.date) -> float:
        """Curve time of a date."""
        return time_from(self.valuation_date, d)

    def log_df(self, t: ArrayLike) -> FloatArray:
        tt = np.asarray(t, dtype=float)
        if np.any(tt < 0):
            raise ValueError("negative curve time")
        inside = np.interp(tt, self._t, self._logp)
        beyond = self._logp[-1] - self._last_fwd * (tt - self._t[-1])
        out = np.where(tt > self._t[-1], beyond, inside)
        return np.asarray(out - self.spread * tt, dtype=float)

    def df(self, t: ArrayLike) -> FloatArray:
        """Discount factor(s) at curve time(s) ``t``."""
        return np.exp(self.log_df(t))

    def df_date(self, d: dt.date) -> float:
        """Discount factor at a date."""
        return float(self.df(self.t(d)))

    def forward_rate(self, t1: float, t2: float) -> float:
        """Continuously compounded forward rate between two curve times."""
        if t2 <= t1:
            raise ValueError("t2 must exceed t1")
        return float((self.log_df(t1) - self.log_df(t2)) / (t2 - t1))

    def zero_rate(self, t: ArrayLike) -> FloatArray:
        """Continuously compounded zero rate(s)."""
        tt = np.asarray(t, dtype=float)
        return np.asarray(-self.log_df(tt) / np.where(tt > 0, tt, np.nan), dtype=float)

    def with_spread(self, spread: float) -> DiscountCurve:
        """Same curve with an extra constant spread on every forward rate."""
        return DiscountCurve(
            self.valuation_date, self.times, self.discount_factors, self.spread + spread
        )

    @property
    def node_times(self) -> FloatArray:
        """Times where the forward rate may jump."""
        return self.times


# ---------------------------------------------------------------------- bootstrapping
@dataclass(frozen=True)
class DepositQuote:
    """Simple-rate deposit starting on the valuation date (ACT/360)."""

    months: int
    rate: float


@dataclass(frozen=True)
class SwapQuote:
    """Single-curve par swap (OIS-style): annual fixed ACT/360 coupons vs the curve."""

    years: int
    rate: float


def _swap_schedule(valuation_date: dt.date, years: int) -> list[dt.date]:
    return [
        adjust_modified_following(add_months(valuation_date, 12 * k))
        for k in range(1, years + 1)
    ]


def swap_par_rate(curve: DiscountCurve, years: int) -> float:
    """Par rate of an annual ACT/360 single-curve swap on ``curve``."""
    dates = _swap_schedule(curve.valuation_date, years)
    prev = curve.valuation_date
    annuity = 0.0
    for d in dates:
        annuity += year_fraction(prev, d, "ACT/360") * curve.df_date(d)
        prev = d
    return (1.0 - curve.df_date(dates[-1])) / annuity


def deposit_rate(curve: DiscountCurve, months: int) -> float:
    """Simple ACT/360 deposit rate implied by ``curve``."""
    end = adjust_modified_following(add_months(curve.valuation_date, months))
    tau = year_fraction(curve.valuation_date, end, "ACT/360")
    return (1.0 / curve.df_date(end) - 1.0) / tau


def bootstrap_discount_curve(
    valuation_date: dt.date,
    deposits: Sequence[DepositQuote],
    swaps: Sequence[SwapQuote],
) -> DiscountCurve:
    """Bootstrap a discount curve that reprices deposits and par swaps exactly.

    Deposits fix their end-date discount factor directly. Each swap adds one node at its
    maturity, solved with Brent's method because intermediate coupon dates are
    interpolated on the curve being built.
    """
    times: list[float] = []
    dfs: list[float] = []
    for dep in sorted(deposits, key=lambda q: q.months):
        end = adjust_modified_following(add_months(valuation_date, dep.months))
        tau = year_fraction(valuation_date, end, "ACT/360")
        times.append(time_from(valuation_date, end))
        dfs.append(1.0 / (1.0 + dep.rate * tau))
    for sw in sorted(swaps, key=lambda q: q.years):
        t_mat = time_from(valuation_date, _swap_schedule(valuation_date, sw.years)[-1])
        if times and t_mat <= times[-1]:
            raise ValueError(f"swap {sw.years}y does not extend the curve")

        def err(log_df: float, t_mat: float = t_mat, rate: float = sw.rate,
                years: int = sw.years) -> float:
            c = DiscountCurve(valuation_date, [*times, t_mat], [*dfs, np.exp(log_df)])
            return swap_par_rate(c, years) - rate

        sol = brentq(err, -10.0, 1.0, xtol=1e-15)
        times.append(t_mat)
        dfs.append(float(np.exp(sol)))
    if not times:
        raise ValueError("no instruments")
    return DiscountCurve(valuation_date, times, dfs)
