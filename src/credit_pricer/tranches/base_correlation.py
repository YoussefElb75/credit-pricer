"""Base correlation: calibration from tranche quotes and the base correlation curve.

A tranche ``[A, D]`` is priced as the difference of two *base* (equity) tranches
``[0, D]`` and ``[0, A]``, each at its own flat correlation. Given contiguous quotes
``[0, K1], [K1, K2], ...`` the base correlations are bootstrapped one detachment at a
time: :math:`\\rho(K_1)` from the equity quote, then :math:`\\rho(K_{i})` so that

.. math:: \\mathrm{PV}_{[0,K_i]}(\\rho_{K_i}) - \\mathrm{PV}_{[0,K_{i-1}]}(\\rho_{K_{i-1}})
          = \\text{quoted value of } [K_{i-1}, K_i].

The value of a base tranche to the protection buyer is decreasing in correlation
(expected loss of an equity tranche falls as correlation rises), so each step has at
most one root.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from scipy.optimize import brentq

from credit_pricer.tranches.tranche import TrancheLegs, TranchePricer


@dataclass(frozen=True)
class TrancheQuote:
    """Market convention: protection buyer pays ``upfront`` (fraction of tranche notional)
    at inception plus ``coupon`` running on the outstanding tranche notional."""

    attach: float
    detach: float
    upfront: float
    coupon: float


@dataclass(frozen=True)
class BaseCorrelationPoint:
    detach: float
    correlation: float


def _value(legs: TrancheLegs, q: TrancheQuote) -> float:
    """Protection buyer's value of the quoted trade, per unit pool notional."""
    return legs.protection - q.coupon * legs.rpv01 - q.upfront * legs.width


def quote_from_base_correlations(
    pricer: TranchePricer,
    attach: float,
    detach: float,
    rho_attach: float,
    rho_detach: float,
    coupon: float,
) -> TrancheQuote:
    """Upfront quote (for a fixed running ``coupon``) implied by two base correlations."""
    legs = pricer.base_correlation_legs(attach, detach, rho_attach, rho_detach)
    return TrancheQuote(attach, detach, legs.upfront(coupon), coupon)


def calibrate_base_correlation(
    pricer: TranchePricer,
    quotes: Sequence[TrancheQuote],
    rho_bounds: tuple[float, float] = (1e-4, 0.995),
    tol: float = 1e-10,
) -> list[BaseCorrelationPoint]:
    """Bootstrap base correlations from contiguous tranche quotes starting at 0."""
    qs = sorted(quotes, key=lambda q: q.attach)
    if not qs or qs[0].attach != 0.0:
        raise ValueError("quotes must start with an equity tranche [0, K]")
    for a, b in zip(qs[:-1], qs[1:]):
        if a.detach != b.attach:
            raise ValueError("tranche quotes must be contiguous")
    lo, hi = rho_bounds
    curve: list[BaseCorrelationPoint] = []
    rho_prev = 0.0
    for q in qs:
        lower = pricer.legs(0.0, q.attach, rho_prev) if q.attach > 0.0 else None

        def f(rho: float, q: TrancheQuote = q, lower: TrancheLegs | None = lower) -> float:
            upper = pricer.legs(0.0, q.detach, rho)
            legs = upper if lower is None else upper - lower
            return _value(legs, q)

        f_lo, f_hi = f(lo), f(hi)
        if f_lo * f_hi > 0.0:
            raise ValueError(
                f"no base correlation in [{lo}, {hi}] reprices tranche "
                f"[{q.attach:.0%}, {q.detach:.0%}]"
            )
        rho = float(brentq(f, lo, hi, xtol=tol))
        curve.append(BaseCorrelationPoint(q.detach, rho))
        rho_prev = rho
    return curve
