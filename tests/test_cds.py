from __future__ import annotations

import datetime as dt
from dataclasses import replace

import numpy as np
import pytest

from credit_pricer.cds import (
    CDS,
    bucketed_cs01,
    cds_legs,
    cs01,
    price_cds,
    recovery_sensitivity,
)
from credit_pricer.curves import DiscountCurve, SurvivalCurve

from .conftest import SPREADS, TENORS


def test_value_is_linear_in_coupon(
    trade_date: dt.date, discount: DiscountCurve, survival: SurvivalCurve
) -> None:
    v1 = price_cds(CDS.standard(trade_date, 5, 0.01), discount, survival)
    v5 = price_cds(CDS.standard(trade_date, 5, 0.05), discount, survival)
    assert v1.protection_leg == pytest.approx(v5.protection_leg)
    assert v1.rpv01 == pytest.approx(v5.rpv01)
    assert v1.pv - v5.pv == pytest.approx(0.04 * v1.rpv01)
    assert v1.par_spread == pytest.approx(v5.par_spread)


def test_par_spread_zeroes_value(
    trade_date: dt.date, discount: DiscountCurve, survival: SurvivalCurve
) -> None:
    s = price_cds(CDS.standard(trade_date, 5, 0.01), discount, survival).par_spread
    v = price_cds(CDS.standard(trade_date, 5, s), discount, survival)
    assert abs(v.pv) < 1e-15
    assert abs(v.upfront) < 1e-15


def test_upfront_sign_and_settlement(
    trade_date: dt.date, discount: DiscountCurve, survival: SurvivalCurve
) -> None:
    cds = CDS.standard(trade_date, 5, 0.05)  # coupon above par spread: buyer receives
    v = price_cds(cds, discount, survival)
    assert v.pv < 0 and v.upfront < 0
    assert v.upfront == pytest.approx(v.pv / discount.df_date(cds.cash_settle_date))
    assert v.accrued == pytest.approx(0.05 * cds.accrued_fraction)
    legs = cds_legs(cds, discount, survival)
    assert legs.rpv01_dirty > legs.rpv01_clean > 0
    assert legs.aod_annuity > 0


def test_monotonic_in_maturity(
    trade_date: dt.date, discount: DiscountCurve, survival: SurvivalCurve
) -> None:
    vals = [price_cds(CDS.standard(trade_date, t, 0.01), discount, survival) for t in TENORS]
    assert all(a.protection_leg < b.protection_leg for a, b in zip(vals, vals[1:]))
    assert all(a.rpv01 < b.rpv01 for a, b in zip(vals, vals[1:]))


def test_monotonic_in_hazard(
    trade_date: dt.date, discount: DiscountCurve, survival: SurvivalCurve
) -> None:
    cds = CDS.standard(trade_date, 5, 0.01)
    vals = [price_cds(cds, discount, survival.scaled(f)) for f in (0.5, 1.0, 2.0)]
    assert vals[0].protection_leg < vals[1].protection_leg < vals[2].protection_leg
    assert vals[0].rpv01 > vals[1].rpv01 > vals[2].rpv01
    assert vals[0].par_spread < vals[1].par_spread < vals[2].par_spread


def test_cs01_close_to_rpv01(trade_date: dt.date, discount: DiscountCurve) -> None:
    cds = CDS.standard(trade_date, 5, 0.01)
    par = cs01(cds, discount, TENORS, SPREADS)
    buckets = bucketed_cs01(cds, discount, TENORS, SPREADS)
    assert par > 0
    assert buckets.sum() == pytest.approx(par, rel=1e-3)
    assert np.argmax(buckets) == TENORS.index(5)
    assert np.allclose(buckets[TENORS.index(7):], 0.0, atol=1e-15)  # beyond maturity
    from credit_pricer.curves import bootstrap_survival_curve

    curve = bootstrap_survival_curve(trade_date, discount, TENORS, SPREADS)
    rpv01_bp = price_cds(cds, discount, curve).rpv01_bp
    assert par == pytest.approx(rpv01_bp, rel=0.03)


def test_recovery_sensitivity(trade_date: dt.date, discount: DiscountCurve) -> None:
    cds = CDS.standard(trade_date, 5, 0.01)
    rs = recovery_sensitivity(cds, discount, TENORS, SPREADS)
    from credit_pricer.curves import bootstrap_survival_curve

    curve = bootstrap_survival_curve(trade_date, discount, TENORS, SPREADS)
    prot = price_cds(cds, discount, curve).protection_leg
    assert rs.fixed_hazard == pytest.approx(-0.01 * prot / 0.6, rel=1e-10)
    assert abs(rs.rebuilt) < 0.05 * abs(rs.fixed_hazard)
    with pytest.raises(ValueError):
        recovery_sensitivity(replace(cds, recovery=0.995), discount, TENORS, SPREADS)


def test_methods_agree_closely(
    trade_date: dt.date, discount: DiscountCurve, survival: SurvivalCurve
) -> None:
    for tenor in TENORS:
        cds = CDS.standard(trade_date, tenor, 0.01)
        a = price_cds(cds, discount, survival, "isda").par_spread
        b = price_cds(cds, discount, survival, "midpoint").par_spread
        assert abs(a - b) < 0.2e-4  # within 0.2 bp


def test_no_accrued_the_day_before_a_coupon_date(discount: DiscountCurve) -> None:
    import datetime as _dt

    from credit_pricer.curves import DiscountCurve as DC
    from credit_pricer.curves import SurvivalCurve as SC

    trade = _dt.date(2026, 3, 19)
    cds = CDS.standard(trade, 5, 0.01)
    assert cds.accrued_fraction == 0.0
    v = price_cds(cds, DC.flat(trade, 0.03), SC.flat(trade, 0.02))
    assert v.accrued == 0.0


def test_invalid(trade_date: dt.date, discount: DiscountCurve, survival: SurvivalCurve) -> None:
    with pytest.raises(ValueError):
        CDS(trade_date, trade_date, 0.01)
    with pytest.raises(ValueError):
        CDS.standard(trade_date, 5, 0.01, recovery=1.5)
    other = DiscountCurve.flat(trade_date + dt.timedelta(days=1), 0.03)
    with pytest.raises(ValueError):
        price_cds(CDS.standard(trade_date, 5, 0.01), other, survival)
    with pytest.raises(ValueError):
        price_cds(CDS.standard(trade_date, 5, 0.01), discount, survival, "trapezoid")  # type: ignore[arg-type]
