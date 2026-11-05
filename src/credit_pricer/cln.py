"""Credit-linked note: funded exposure to a reference entity plus issuer funding risk.

The investor pays the price up front and receives, while the reference entity survives,
a floating coupon (risk-free forward rate) plus a fixed spread ``s`` and par at
maturity. On a reference default the note terminates and the investor receives the
recovery ``R`` at the default time; unpaid coupons are lost.

Issuer credit risk is captured by discounting every cash flow at the risk-free rate plus
the issuer's funding spread ``f`` (equivalently, an issuer default intensity ``f`` with
zero recovery, independent of the reference entity). With :math:`D_f(t) = P(t) e^{-ft}`:

.. math::

    \\text{Price} = \\sum_j \\left(F_j \\alpha_j + s\\,\\alpha_j\\right) D_f(t_j) Q(t_j)
                  + D_f(T) Q(T) + R \\int_0^T D_f(u)\\,(-dQ(u))

where :math:`F_j\\alpha_j = P(t_{j-1})/P(t_j) - 1` is the risk-free forward coupon.
The fair spread (price = par) is close to the CDS par spread plus the funding spread.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from credit_pricer.cds.legs import integrate_periods
from credit_pricer.curves.discount import DiscountCurve
from credit_pricer.curves.survival import SurvivalCurve
from credit_pricer.dates import AccrualPeriod, cds_maturity, cds_schedule


@dataclass(frozen=True)
class CreditLinkedNote:
    """Floating-rate CLN issued on ``issue_date`` with quarterly IMM coupon dates."""

    issue_date: dt.date
    maturity: dt.date
    spread: float
    recovery: float = 0.4

    @classmethod
    def standard(
        cls, issue_date: dt.date, tenor_years: int, spread: float, recovery: float = 0.4
    ) -> CreditLinkedNote:
        return cls(issue_date, cds_maturity(issue_date, tenor_years), spread, recovery)

    @property
    def periods(self) -> list[AccrualPeriod]:
        """CDS-style schedule with the first period starting on the issue date."""
        periods = cds_schedule(self.issue_date, self.maturity)
        first = periods[0]
        return [AccrualPeriod(self.issue_date, first.end, first.pay), *periods[1:]]


@dataclass(frozen=True)
class CLNValuation:
    """CLN price decomposition per unit face value."""

    floating_coupons: float
    spread_annuity: float  # value of 1 unit of spread
    principal: float
    recovery: float

    def price(self, spread: float) -> float:
        return self.floating_coupons + spread * self.spread_annuity + self.principal + self.recovery

    @property
    def fair_spread(self) -> float:
        """Spread that prices the note at par."""
        return (1.0 - self.floating_coupons - self.principal - self.recovery) / self.spread_annuity


def value_cln(
    note: CreditLinkedNote,
    discount: DiscountCurve,
    survival: SurvivalCurve,
    funding_spread: float = 0.0,
) -> CLNValuation:
    """Decompose the CLN value into floating coupons, spread annuity, principal and recovery."""
    if discount.valuation_date != note.issue_date or survival.valuation_date != note.issue_date:
        raise ValueError("curves must be referenced at the issue date")
    risky_disc = discount.with_spread(funding_spread)
    floating = 0.0
    annuity = 0.0
    prev_pay = note.issue_date
    for p in note.periods:
        # same end-of-day observation convention as the CDS legs
        observed = p.end - dt.timedelta(days=1)
        weight = risky_disc.df_date(p.pay) * float(survival.survival(survival.t(observed)))
        # floating coupon fixes on the payment dates, so a riskless floater is worth par
        fwd = discount.df_date(prev_pay) / discount.df_date(p.pay) - 1.0
        floating += fwd * weight
        annuity += p.accrual * weight
        prev_pay = p.pay
    t_mat = survival.t(note.maturity)
    principal = risky_disc.df_date(prev_pay) * float(survival.survival(t_mat))
    # Recovery leg = R * integral of D_f dPD: the protection-leg integral with LGD := R.
    rec = sum(
        v.protection
        for v in integrate_periods(
            note.periods, note.issue_date, risky_disc, survival, 1.0 - note.recovery, "isda"
        )
    )
    return CLNValuation(floating, annuity, principal, rec)


def price_cln(
    note: CreditLinkedNote,
    discount: DiscountCurve,
    survival: SurvivalCurve,
    funding_spread: float = 0.0,
) -> float:
    """Dirty price per unit face value."""
    return value_cln(note, discount, survival, funding_spread).price(note.spread)
