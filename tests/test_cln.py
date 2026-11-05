from __future__ import annotations

import datetime as dt

import pytest

from credit_pricer.cds import CDS, price_cds
from credit_pricer.cln import CreditLinkedNote, price_cln, value_cln
from credit_pricer.curves import DiscountCurve, SurvivalCurve, bootstrap_survival_curve


def test_riskless_floater_prices_at_par(trade_date: dt.date, discount: DiscountCurve) -> None:
    note = CreditLinkedNote.standard(trade_date, 5, 0.0)
    v = value_cln(note, discount, SurvivalCurve.flat(trade_date, 0.0))
    assert v.price(0.0) == pytest.approx(1.0, abs=1e-12)
    assert v.fair_spread == pytest.approx(0.0, abs=1e-12)


def test_funding_only(trade_date: dt.date, discount: DiscountCurve) -> None:
    note = CreditLinkedNote.standard(trade_date, 5, 0.0)
    fair = value_cln(note, discount, SurvivalCurve.flat(trade_date, 0.0), 0.01).fair_spread
    assert fair == pytest.approx(0.01 * 360 / 365, rel=0.02)


@pytest.mark.parametrize("spread", [0.002, 0.01, 0.04])
@pytest.mark.parametrize("funding", [0.0, 0.005, 0.015])
def test_fair_spread_close_to_cds_plus_funding(
    trade_date: dt.date, discount: DiscountCurve, spread: float, funding: float
) -> None:
    tenors = [1, 3, 5, 7]
    curve = bootstrap_survival_curve(trade_date, discount, tenors, [spread] * 4)
    cds = price_cds(CDS.standard(trade_date, 5, 0.01), discount, curve).par_spread
    fair = value_cln(CreditLinkedNote.standard(trade_date, 5, 0.0), discount, curve, funding)
    # equal up to ACT/360-vs-continuous and coupon-loss effects: within 3% of the total
    assert fair.fair_spread == pytest.approx(cds + funding * 360 / 365, rel=0.03)


def test_price_monotonicity(
    trade_date: dt.date, discount: DiscountCurve, survival: SurvivalCurve
) -> None:
    note = CreditLinkedNote.standard(trade_date, 5, 0.01)
    base = price_cln(note, discount, survival)
    assert price_cln(note, discount, survival.scaled(2.0)) < base
    assert price_cln(note, discount, survival, 0.01) < base
    rich = CreditLinkedNote.standard(trade_date, 5, 0.02)
    assert price_cln(rich, discount, survival) > base


def test_recovery_limits(
    trade_date: dt.date, discount: DiscountCurve, survival: SurvivalCurve
) -> None:
    # full recovery: default only costs the accrued coupon -> fair spread below 1 bp
    note = CreditLinkedNote.standard(trade_date, 5, 0.0, recovery=1.0)
    assert 0.0 < value_cln(note, discount, survival).fair_spread < 1e-4
    zero = CreditLinkedNote.standard(trade_date, 5, 0.0, recovery=0.0)
    forty = CreditLinkedNote.standard(trade_date, 5, 0.0, recovery=0.4)
    assert (
        value_cln(zero, discount, survival).fair_spread
        > value_cln(forty, discount, survival).fair_spread
    )
