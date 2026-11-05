"""Survival curve with piecewise-constant hazard rates, bootstrapped from CDS par spreads."""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from credit_pricer.curves.discount import DiscountCurve, time_from

if TYPE_CHECKING:
    from credit_pricer.cds.legs import Method

FloatArray = NDArray[np.float64]


class SurvivalCurve:
    """Survival probabilities from a piecewise-constant hazard rate.

    ``hazards[k]`` applies on ``(times[k-1], times[k]]`` (with ``times[-1] = 0``); the
    last hazard rate is extended flat beyond the last node.

    .. math::  Q(t) = \\exp\\left(-\\int_0^t h(u)\\,du\\right)
    """

    def __init__(self, valuation_date: dt.date, times: ArrayLike, hazards: ArrayLike) -> None:
        t = np.asarray(times, dtype=float)
        h = np.asarray(hazards, dtype=float)
        if t.ndim != 1 or t.shape != h.shape or t.size == 0:
            raise ValueError("times and hazards must be 1-d arrays of equal length")
        if np.any(t <= 0) or np.any(np.diff(t) <= 0):
            raise ValueError("node times must be positive and strictly increasing")
        if np.any(h < 0):
            raise ValueError("hazard rates must be non-negative")
        self.valuation_date = valuation_date
        self.times: FloatArray = t
        self.hazards: FloatArray = h
        self._t = np.concatenate(([0.0], t))
        self._cum = np.concatenate(([0.0], np.cumsum(h * np.diff(self._t))))

    @classmethod
    def flat(cls, valuation_date: dt.date, hazard: float, horizon: float = 50.0) -> SurvivalCurve:
        """Constant hazard rate."""
        return cls(valuation_date, [horizon], [hazard])

    def t(self, d: dt.date) -> float:
        """Curve time (ACT/365F) of a date."""
        return time_from(self.valuation_date, d)

    def cumulative_hazard(self, t: ArrayLike) -> FloatArray:
        tt = np.asarray(t, dtype=float)
        if np.any(tt < 0):
            raise ValueError("negative curve time")
        inside = np.interp(tt, self._t, self._cum)
        beyond = self._cum[-1] + self.hazards[-1] * (tt - self._t[-1])
        return np.asarray(np.where(tt > self._t[-1], beyond, inside), dtype=float)

    def survival(self, t: ArrayLike) -> FloatArray:
        """Survival probability Q(t)."""
        return np.exp(-self.cumulative_hazard(t))

    def default_probability(self, t: ArrayLike) -> FloatArray:
        """Cumulative default probability 1 - Q(t)."""
        return -np.expm1(-self.cumulative_hazard(t))

    def hazard(self, t: ArrayLike) -> FloatArray:
        """Instantaneous hazard rate; at a node time the value of the segment ending there."""
        tt = np.asarray(t, dtype=float)
        idx = np.minimum(np.searchsorted(self.times, tt, side="left"), self.times.size - 1)
        return np.asarray(self.hazards[idx], dtype=float)

    def scaled(self, factor: float) -> SurvivalCurve:
        """Curve with every hazard rate multiplied by ``factor``."""
        if factor < 0:
            raise ValueError("factor must be non-negative")
        return SurvivalCurve(self.valuation_date, self.times, self.hazards * factor)

    def shifted(self, shift: float) -> SurvivalCurve:
        """Curve with ``shift`` added to every hazard rate (floored at zero)."""
        return SurvivalCurve(
            self.valuation_date, self.times, np.maximum(self.hazards + shift, 0.0)
        )

    def with_nodes(self, extra_times: ArrayLike) -> SurvivalCurve:
        """Identical curve with additional node times (hazard rates unchanged)."""
        extra = np.asarray(extra_times, dtype=float)
        times = np.unique(np.concatenate((self.times, extra[extra > 0])))
        return SurvivalCurve(self.valuation_date, times, self.hazard(times))

    def scaled_segments(self, boundaries: ArrayLike, factors: ArrayLike) -> SurvivalCurve:
        """Multiply hazard rates segment by segment.

        ``factors[j]`` applies on ``(boundaries[j-1], boundaries[j]]`` (``boundaries[-1]``
        being 0); the last factor also applies beyond the last boundary.
        """
        b = np.asarray(boundaries, dtype=float)
        f = np.asarray(factors, dtype=float)
        if b.shape != f.shape or np.any(f < 0):
            raise ValueError("need one non-negative factor per boundary")
        curve = self.with_nodes(b)
        seg = np.minimum(np.searchsorted(b, curve.times, side="left"), b.size - 1)
        return SurvivalCurve(self.valuation_date, curve.times, curve.hazards * f[seg])

    @property
    def node_times(self) -> FloatArray:
        """Times where the hazard rate may jump."""
        return self.times




def bootstrap_survival_curve(
    valuation_date: dt.date,
    discount: DiscountCurve,
    tenors: Sequence[int],
    par_spreads: Sequence[float],
    recovery: float = 0.4,
    method: Method = "isda",
    max_hazard: float = 50.0,
) -> SurvivalCurve:
    """Bootstrap piecewise-constant hazard rates from standard CDS par spreads.

    One node is placed at each standard maturity (``tenors`` in years, IMM roll; for the
    ``midpoint`` scheme at the last payment date, which it also observes). The
    hazard rate of each new segment is solved so that a CDS with coupon equal to the
    quoted par spread has zero value (protection leg = spread x clean RPV01).

    Parameters
    ----------
    par_spreads:
        Par spreads as decimals (0.01 = 100 bp).
    """
    from credit_pricer.cds.pricer import CDS, cds_legs  # local import: cds depends on curves

    if len(tenors) != len(par_spreads) or not tenors:
        raise ValueError("tenors and par_spreads must be non-empty and of equal length")
    if not 0.0 <= recovery < 1.0:
        raise ValueError("recovery must lie in [0, 1) to bootstrap from spreads")
    order = np.argsort(np.asarray(tenors))
    times: list[float] = []
    hazards: list[float] = []
    for i in order:
        cds = CDS.standard(valuation_date, int(tenors[i]), float(par_spreads[i]), recovery)
        # Pillar at the last date the instrument depends on: the maturity for the exact
        # schemes, the (adjusted) last payment date for the mid-point scheme.
        last = cds.maturity if method != "midpoint" else max(cds.maturity, cds.periods[-1].pay)
        t_mat = time_from(valuation_date, last)
        if times and t_mat <= times[-1]:
            raise ValueError("tenors must map to distinct maturities")

        def value(h: float, t_mat: float = t_mat, cds: CDS = cds) -> float:
            curve = SurvivalCurve(valuation_date, [*times, t_mat], [*hazards, h])
            legs = cds_legs(cds, discount, curve, method)
            return legs.protection - cds.coupon * legs.rpv01_clean

        if par_spreads[i] <= 0:
            raise ValueError("par spreads must be positive")
        lo, hi = value(0.0), value(max_hazard)
        if lo > 0.0 or hi < 0.0:
            raise ValueError(
                f"no non-negative hazard rate reprices the {tenors[i]}y spread "
                f"{par_spreads[i]:.6f} (up to max_hazard={max_hazard}); the spread curve is "
                "not consistent with this model (too inverted or too steep)"
            )
        h_sol = float(brentq(value, 0.0, max_hazard, xtol=1e-14, rtol=1e-14))
        times.append(t_mat)
        hazards.append(h_sol)
    return SurvivalCurve(valuation_date, times, hazards)
