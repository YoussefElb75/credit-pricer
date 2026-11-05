"""CDS index: intrinsic spread from constituents, index basis and hazard adjustment.

An index CDS with coupon ``C`` is economically a portfolio of single-name CDS with the
same coupon and maturity, weighted by ``w_i``. Its *intrinsic* value is

.. math:: V^{intr} = \\sum_i w_i \\left(\\mathrm{Prot}_i - C\\,\\mathrm{RPV01}_i\\right)

and the *intrinsic spread* is the RPV01-weighted average of constituent par spreads.
The market quotes the index as a single spread, converted to an upfront with a flat
hazard curve (ISDA convention). The difference between quoted and intrinsic spreads is
the *index basis*; the hazard adjustment rescales constituent curves so that the
intrinsic value matches the quote.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import brentq

from credit_pricer.cds.legs import Method
from credit_pricer.cds.pricer import CDS, cds_legs
from credit_pricer.curves.discount import DiscountCurve
from credit_pricer.curves.survival import SurvivalCurve, bootstrap_survival_curve


@dataclass(frozen=True)
class Constituent:
    """Index member: survival curve, recovery and weight (weights sum to one)."""

    name: str
    curve: SurvivalCurve
    recovery: float = 0.4
    weight: float = 1.0


def equal_weights(curves: Sequence[SurvivalCurve], recovery: float = 0.4) -> list[Constituent]:
    """Equally weighted constituents named ``N001``, ``N002``, ..."""
    n = len(curves)
    return [Constituent(f"N{i + 1:03d}", c, recovery, 1.0 / n) for i, c in enumerate(curves)]


@dataclass(frozen=True)
class IndexIntrinsic:
    """Intrinsic (bottom-up) valuation of an index CDS, per unit index notional."""

    protection: float
    rpv01: float  # clean risky annuity per unit spread
    coupon: float
    spread: float  # intrinsic spread = protection / rpv01
    pv: float  # protection - coupon * rpv01 at the trade date


def index_intrinsic(
    trade_date: dt.date,
    tenor_years: int,
    coupon: float,
    constituents: Sequence[Constituent],
    discount: DiscountCurve,
    method: Method = "isda",
) -> IndexIntrinsic:
    """Bottom-up valuation of an index CDS from its constituents' curves."""
    total_w = sum(c.weight for c in constituents)
    if not np.isclose(total_w, 1.0):
        raise ValueError("constituent weights must sum to one")
    prot = 0.0
    rpv01 = 0.0
    for c in constituents:
        cds = CDS.standard(trade_date, tenor_years, coupon, c.recovery)
        legs = cds_legs(cds, discount, c.curve, method)
        prot += c.weight * legs.protection
        rpv01 += c.weight * legs.rpv01_clean
    return IndexIntrinsic(prot, rpv01, coupon, prot / rpv01, prot - coupon * rpv01)


def quoted_spread_pv(
    trade_date: dt.date,
    tenor_years: int,
    coupon: float,
    quoted_spread: float,
    discount: DiscountCurve,
    recovery: float = 0.4,
    method: Method = "isda",
) -> float:
    """Value at the trade date of an index quoted as a spread (flat hazard convention)."""
    flat = bootstrap_survival_curve(
        trade_date, discount, [tenor_years], [quoted_spread], recovery, method
    )
    cds = CDS.standard(trade_date, tenor_years, coupon, recovery)
    legs = cds_legs(cds, discount, flat, method)
    return legs.protection - coupon * legs.rpv01_clean


@dataclass(frozen=True)
class IndexAdjustment:
    """Result of matching constituent curves to quoted index spreads."""

    tenors: list[int]
    factors: list[float]  # hazard multiplier per maturity segment
    constituents: list[Constituent]  # adjusted constituents


def adjust_to_index_quotes(
    trade_date: dt.date,
    tenors: Sequence[int],
    quoted_spreads: Sequence[float],
    coupon: float,
    constituents: Sequence[Constituent],
    discount: DiscountCurve,
    method: Method = "isda",
    index_recovery: float = 0.4,
) -> IndexAdjustment:
    """Scale every constituent's hazard rates so intrinsic values match index quotes.

    Maturity segments are processed in increasing order: on the segment ending at the
    ``k``-th index maturity all constituent hazard rates are multiplied by a common
    factor ``alpha_k``, solved so that the intrinsic value equals the value implied by
    the quoted spread (converted with a flat curve at ``index_recovery``). Earlier
    segments are left unchanged, so the result reprices every index quote while
    preserving the relative risk of the constituents. Repricing is exact for the
    ``isda`` schemes; with ``midpoint`` a payment date falling after a segment boundary
    (maturity on a weekend) leaves a residual of order 1e-9.
    """
    order = np.argsort(np.asarray(tenors))
    ten = [int(tenors[i]) for i in order]
    quotes = [float(quoted_spreads[i]) for i in order]
    bounds = [discount.t(CDS.standard(trade_date, k, coupon).maturity) for k in ten]
    factors: list[float] = []

    def adjusted(fs: Sequence[float]) -> list[Constituent]:
        b = bounds[: len(fs)]
        return [
            Constituent(c.name, c.curve.scaled_segments(b, fs), c.recovery, c.weight)
            for c in constituents
        ]

    for tenor, q in zip(ten, quotes):
        target = quoted_spread_pv(
            trade_date, tenor, coupon, q, discount, index_recovery, method
        )

        def gap(a: float, tenor: int = tenor, target: float = target) -> float:
            res = index_intrinsic(
                trade_date, tenor, coupon, adjusted([*factors, a]), discount, method
            )
            return res.pv - target

        hi = 2.0
        while gap(hi) < 0.0:
            hi *= 2.0
            if hi > 1e4:
                raise ValueError(f"cannot match the {tenor}y index quote")
        if gap(0.0) > 0.0:
            raise ValueError(f"cannot match the {tenor}y index quote with non-negative hazards")
        factors.append(float(brentq(gap, 0.0, hi, xtol=1e-14, rtol=1e-14)))
    return IndexAdjustment(ten, factors, adjusted(factors))


def constituents_from_spreads(
    trade_date: dt.date,
    discount: DiscountCurve,
    tenors: Sequence[int],
    spreads: ArrayLike,
    recovery: float = 0.4,
    names: Sequence[str] | None = None,
    method: Method = "isda",
) -> list[Constituent]:
    """Bootstrap one survival curve per constituent and weight them equally.

    ``spreads`` has one row of par spreads (decimals, matching ``tenors``) per name.
    """
    matrix = np.atleast_2d(np.asarray(spreads, dtype=float))
    n = matrix.shape[0]
    labels = list(names) if names is not None else [f"N{i + 1:03d}" for i in range(n)]
    return [
        Constituent(
            label,
            bootstrap_survival_curve(trade_date, discount, tenors, row.tolist(), recovery, method),
            recovery,
            1.0 / n,
        )
        for label, row in zip(labels, matrix)
    ]
