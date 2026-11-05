"""Single-name CDS valuation: legs, par spread, upfront and risky PV01."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import cached_property

from credit_pricer.cds.legs import Method, integrate_periods
from credit_pricer.curves.discount import DiscountCurve
from credit_pricer.curves.survival import SurvivalCurve
from credit_pricer.dates import (
    AccrualPeriod,
    add_business_days,
    cds_maturity,
    cds_schedule,
    year_fraction,
)


@dataclass(frozen=True)
class CDS:
    """Standard single-name CDS, from the protection buyer's point of view.

    All values are per unit notional; scale the results by the trade notional.

    Attributes
    ----------
    trade_date:
        Trade (and valuation) date ``T``. Step-in is ``T+1``; cash settlement ``T+3``
        business days.
    maturity:
        Unadjusted scheduled termination date. Use :meth:`standard` for the standard IMM
        maturity; other dates are accepted and produce a schedule rolled back from the
        maturity (V1 is validated on standard maturities only).
    coupon:
        Running coupon as a decimal (0.01 = 100 bp), ACT/360, paid quarterly.
    recovery:
        Recovery rate used for the protection leg.
    """

    trade_date: dt.date
    maturity: dt.date
    coupon: float
    recovery: float = 0.4

    def __post_init__(self) -> None:
        if not 0.0 <= self.recovery <= 1.0:
            raise ValueError("recovery must lie in [0, 1]")
        if self.maturity <= self.trade_date:
            raise ValueError("maturity must be after the trade date")

    @classmethod
    def standard(
        cls, trade_date: dt.date, tenor_years: int, coupon: float, recovery: float = 0.4
    ) -> CDS:
        """CDS with the standard IMM maturity for ``tenor_years``."""
        return cls(trade_date, cds_maturity(trade_date, tenor_years), coupon, recovery)

    @cached_property
    def periods(self) -> list[AccrualPeriod]:
        return cds_schedule(self.trade_date, self.maturity)

    @property
    def step_in_date(self) -> dt.date:
        return self.trade_date + dt.timedelta(days=1)

    @property
    def cash_settle_date(self) -> dt.date:
        return add_business_days(self.trade_date, 3)

    @property
    def accrued_fraction(self) -> float:
        """ACT/360 accrual from the accrual start date to the step-in date."""
        return year_fraction(self.periods[0].start, self.step_in_date, "ACT/360")


@dataclass(frozen=True)
class CDSLegs:
    """Leg values per unit notional, discounted to the valuation date.

    ``rpv01_dirty`` is the risky annuity (coupons plus accrual on default) per unit of
    spread. ``rpv01_clean`` deducts the accrued premium rebated at cash settlement.
    """

    protection: float
    coupon_annuity: float
    aod_annuity: float
    accrued_rebate: float

    @property
    def rpv01_dirty(self) -> float:
        return self.coupon_annuity + self.aod_annuity

    @property
    def rpv01_clean(self) -> float:
        return self.rpv01_dirty - self.accrued_rebate

    @property
    def par_spread(self) -> float:
        """Spread that sets the clean value of the contract to zero."""
        return self.protection / self.rpv01_clean


def cds_legs(
    cds: CDS, discount: DiscountCurve, survival: SurvivalCurve, method: Method = "isda"
) -> CDSLegs:
    """Integrate both legs of ``cds`` (per unit notional, valued at the trade date)."""
    if discount.valuation_date != cds.trade_date or survival.valuation_date != cds.trade_date:
        raise ValueError("curves must be referenced at the CDS trade date")
    vals = integrate_periods(cds.periods, cds.trade_date, discount, survival, cds.recovery, method)
    rebate = cds.accrued_fraction * discount.df_date(cds.cash_settle_date)
    return CDSLegs(
        protection=sum(v.protection for v in vals),
        coupon_annuity=sum(v.coupon_annuity for v in vals),
        aod_annuity=sum(v.aod_annuity for v in vals),
        accrued_rebate=rebate,
    )


@dataclass(frozen=True)
class CDSValuation:
    """Valuation summary for a protection buyer (per unit notional unless stated)."""

    protection_leg: float
    premium_leg: float
    rpv01: float
    par_spread: float
    pv: float
    upfront: float
    accrued: float

    @property
    def rpv01_bp(self) -> float:
        """Risky PV01: value of 1 bp of running spread."""
        return self.rpv01 * 1e-4


def price_cds(
    cds: CDS, discount: DiscountCurve, survival: SurvivalCurve, method: Method = "isda"
) -> CDSValuation:
    """Price a CDS for the protection buyer.

    * ``pv`` = protection leg - coupon x clean RPV01, at the valuation date. This
      includes the accrued premium rebated by the buyer at cash settlement.
    * ``upfront`` = clean upfront paid by the buyer at cash settlement (``pv`` forwarded
      to the cash settlement date), the quantity quoted by the market.
    * ``accrued`` = coupon x accrual fraction to the step-in date.
    """
    legs = cds_legs(cds, discount, survival, method)
    premium = cds.coupon * legs.rpv01_clean
    pv = legs.protection - premium
    return CDSValuation(
        protection_leg=legs.protection,
        premium_leg=premium,
        rpv01=legs.rpv01_clean,
        par_spread=legs.par_spread,
        pv=pv,
        upfront=pv / discount.df_date(cds.cash_settle_date),
        accrued=cds.coupon * cds.accrued_fraction,
    )
