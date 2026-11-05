"""Synthetic CDO tranche pricing under the one-factor Gaussian copula.

A tranche ``[A, D]`` absorbs pool losses between the attachment ``A`` and detachment
``D`` (fractions of pool notional). With expected tranche loss
:math:`EL_j = E[\\min((L(t_j) - A)^+, D - A)]` on the premium dates :math:`t_j`:

.. math::

    \\mathrm{Prot} = \\sum_j P(\\bar t_j)\\,(EL_j - EL_{j-1}),\\qquad
    \\mathrm{RPV01} = \\sum_j \\alpha_j P(t_j)\\left(D - A - \\tfrac{EL_{j-1} + EL_j}{2}\\right)
    - \\alpha_{acc} P(t_{cs}) (D - A)

where :math:`\\bar t_j` is the mid-point of period ``j`` (losses settle on average in the
middle of the period) and the last term rebates the premium accrued before step-in at
cash settlement (clean convention, as for single-name CDS). Values are per unit of
*pool* notional; divide by ``D - A`` for per-unit-tranche quantities.
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
from credit_pricer.dates import add_business_days, cds_maturity, cds_schedule, year_fraction
from credit_pricer.tranches.loss import (
    finite_pool_default_pmf,
    lhp_excess_loss,
)

FloatArray = NDArray[np.float64]
Model = Literal["finite", "lhp"]


@dataclass(frozen=True)
class TrancheLegs:
    """Tranche legs per unit pool notional: protection and risky annuity (per unit spread)."""

    attach: float
    detach: float
    protection: float
    rpv01: float

    @property
    def width(self) -> float:
        return self.detach - self.attach

    @property
    def fair_spread(self) -> float:
        """Running spread that makes the tranche worth zero with no upfront."""
        return self.protection / self.rpv01

    def upfront(self, coupon: float) -> float:
        """Upfront (fraction of tranche notional) paid by the protection buyer for ``coupon``."""
        return (self.protection - coupon * self.rpv01) / self.width

    def __sub__(self, other: TrancheLegs) -> TrancheLegs:
        """``[0, D] - [0, A] = [A, D]`` (base-correlation decomposition)."""
        if self.attach != 0.0 or other.attach != 0.0 or other.detach >= self.detach:
            raise ValueError("subtract base tranches [0, A] from [0, D] with A < D")
        return TrancheLegs(
            other.detach,
            self.detach,
            self.protection - other.protection,
            self.rpv01 - other.rpv01,
        )


class TranchePricer:
    """Prices tranches on a pool of equally weighted names with a common recovery.

    Parameters
    ----------
    trade_date, tenor_years:
        Standard quarterly premium schedule (IMM dates, ACT/360) to the CDS maturity.
    curves:
        Survival curve of each name (equal notional weights).
    recovery:
        Common recovery rate.
    model:
        ``"finite"`` (exact recursion over the names) or ``"lhp"`` (large homogeneous
        pool using the average default probability).
    n_quad:
        Quadrature nodes for the factor integral (finite pool), a multiple of 8.
    """

    def __init__(
        self,
        trade_date: dt.date,
        tenor_years: int,
        curves: Sequence[SurvivalCurve],
        discount: DiscountCurve,
        recovery: float = 0.4,
        model: Model = "finite",
        n_quad: int = 192,
    ) -> None:
        if not curves:
            raise ValueError("need at least one name")
        if not 0.0 <= recovery <= 1.0:
            raise ValueError("recovery must lie in [0, 1]")
        if model not in ("finite", "lhp"):
            raise ValueError(f"unknown model {model!r}")
        self.trade_date = trade_date
        self.maturity = cds_maturity(trade_date, tenor_years)
        self.recovery = recovery
        self.model = model
        self.n_quad = n_quad
        self.n_names = len(curves)
        periods = cds_schedule(trade_date, self.maturity)
        self.accruals: FloatArray = np.array([p.accrual for p in periods])
        # loss observed at period end; first period starts at the trade date (EL = 0)
        self.times: FloatArray = np.array(
            [min(discount.t(p.end), discount.t(self.maturity)) for p in periods]
        )
        starts = np.concatenate(([0.0], self.times[:-1]))
        self.df_pay: FloatArray = discount.df([discount.t(p.pay) for p in periods])
        # clean convention: the premium accrued before step-in is rebated at cash settlement
        step_in = trade_date + dt.timedelta(days=1)
        self.accrued_fraction = year_fraction(periods[0].start, step_in, "ACT/360")
        self.df_cash_settle = discount.df_date(add_business_days(trade_date, 3))
        self.df_mid: FloatArray = discount.df(0.5 * (starts + self.times))
        self.default_probs: FloatArray = np.array(
            [c.default_probability(self.times) for c in curves]
        ).T  # shape (n_dates, n_names)
        self._pmf_cache: dict[float, FloatArray] = {}

    # ----------------------------------------------------------------- loss model
    def _pmf(self, rho: float) -> FloatArray:
        if rho not in self._pmf_cache:
            self._pmf_cache[rho] = np.array(
                [finite_pool_default_pmf(p, rho, self.n_quad) for p in self.default_probs]
            )
            if len(self._pmf_cache) > 64:
                self._pmf_cache.pop(next(iter(self._pmf_cache)))
        return self._pmf_cache[rho]

    def excess_loss(self, strike: float, rho: float) -> FloatArray:
        """E[(L(t_j) - K)^+] on every premium date."""
        if not 0.0 <= rho <= 1.0:
            raise ValueError("rho must lie in [0, 1]")
        if self.model == "lhp":
            pbar = self.default_probs.mean(axis=1)
            return np.array([lhp_excess_loss(strike, p, self.recovery, rho) for p in pbar])
        pmf = self._pmf(rho)
        losses = np.arange(self.n_names + 1) * (1.0 - self.recovery) / self.n_names
        return np.asarray(pmf @ np.maximum(losses - strike, 0.0), dtype=float)

    def expected_tranche_loss(self, attach: float, detach: float, rho: float) -> FloatArray:
        """Expected tranche loss on every premium date (fraction of pool notional)."""
        if not 0.0 <= attach < detach <= 1.0:
            raise ValueError("need 0 <= attach < detach <= 1")
        return self.excess_loss(attach, rho) - self.excess_loss(detach, rho)

    # ----------------------------------------------------------------- pricing
    def legs(self, attach: float, detach: float, rho: float) -> TrancheLegs:
        """Protection leg and risky annuity of ``[attach, detach]`` at correlation ``rho``."""
        el = self.expected_tranche_loss(attach, detach, rho)
        el_prev = np.concatenate(([0.0], el[:-1]))
        prot = float(self.df_mid @ (el - el_prev))
        outstanding = (detach - attach) - 0.5 * (el + el_prev)
        rpv01 = float((self.accruals * self.df_pay) @ outstanding)
        rpv01 -= self.accrued_fraction * self.df_cash_settle * (detach - attach)
        return TrancheLegs(attach, detach, prot, rpv01)

    def base_correlation_legs(
        self, attach: float, detach: float, rho_attach: float, rho_detach: float
    ) -> TrancheLegs:
        """Tranche legs as a difference of base tranches priced at their own correlations."""
        upper = self.legs(0.0, detach, rho_detach)
        if attach == 0.0:
            return upper
        return upper - self.legs(0.0, attach, rho_attach)

    def pool_expected_loss(self) -> FloatArray:
        """Expected pool loss on every premium date (independent of correlation)."""
        return np.asarray((1.0 - self.recovery) * self.default_probs.mean(axis=1), dtype=float)
