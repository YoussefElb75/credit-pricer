"""Limit cases: zero hazard, recovery 0% and 100%, closed forms on flat curves."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from credit_pricer.cds import CDS, cds_legs, price_cds
from credit_pricer.curves import DiscountCurve, SurvivalCurve


@pytest.mark.parametrize("method", ["isda", "midpoint"])
def test_zero_hazard(trade_date: dt.date, discount: DiscountCurve, method: str) -> None:
    cds = CDS.standard(trade_date, 5, 0.01)
    curve = SurvivalCurve.flat(trade_date, 0.0)
    legs = cds_legs(cds, discount, curve, method)  # type: ignore[arg-type]
    assert legs.protection == 0.0
    assert legs.aod_annuity == 0.0
    riskless = sum(p.accrual * discount.df_date(p.pay) for p in cds.periods)
    assert legs.coupon_annuity == pytest.approx(riskless, rel=1e-14)
    assert legs.par_spread == 0.0


@pytest.mark.parametrize("method", ["isda", "midpoint"])
def test_full_recovery(
    trade_date: dt.date, discount: DiscountCurve, survival: SurvivalCurve, method: str
) -> None:
    v = price_cds(CDS.standard(trade_date, 5, 0.01, recovery=1.0), discount, survival, method)  # type: ignore[arg-type]
    assert v.protection_leg == 0.0
    assert v.par_spread == 0.0


@pytest.mark.parametrize("method", ["isda", "midpoint"])
def test_protection_scales_with_lgd(
    trade_date: dt.date, discount: DiscountCurve, survival: SurvivalCurve, method: str
) -> None:
    p0 = price_cds(CDS.standard(trade_date, 5, 0.01, 0.0), discount, survival, method)  # type: ignore[arg-type]
    p4 = price_cds(CDS.standard(trade_date, 5, 0.01, 0.4), discount, survival, method)  # type: ignore[arg-type]
    assert p4.protection_leg == pytest.approx(0.6 * p0.protection_leg, rel=1e-14)
    assert p0.rpv01 == pytest.approx(p4.rpv01, rel=1e-14)


def test_zero_rates_zero_recovery_closed_form(trade_date: dt.date) -> None:
    """With r = 0 and R = 0 the protection leg is the default probability over the window."""
    disc = DiscountCurve.flat(trade_date, 0.0)
    h = 0.03
    curve = SurvivalCurve.flat(trade_date, h)
    cds = CDS.standard(trade_date, 5, 0.01, recovery=0.0)
    legs = cds_legs(cds, disc, curve)
    t_mat = curve.t(cds.maturity)
    assert legs.protection == pytest.approx(1 - np.exp(-h * t_mat), rel=1e-13)


def test_flat_curves_credit_triangle(trade_date: dt.date) -> None:
    """Flat hazard h and recovery R give a par spread close to h (1 - R) * 365/360."""
    disc = DiscountCurve.flat(trade_date, 0.03)
    for h in (0.001, 0.02, 0.2):
        curve = SurvivalCurve.flat(trade_date, h)
        s = price_cds(CDS.standard(trade_date, 5, 0.01), disc, curve).par_spread
        assert s == pytest.approx(h * 0.6 * 360 / 365, rel=0.01)
