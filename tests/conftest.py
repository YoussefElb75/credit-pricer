from __future__ import annotations

import datetime as dt

import pytest

from credit_pricer.curves import DiscountCurve, SurvivalCurve, bootstrap_survival_curve
from credit_pricer.data import VALUATION_DATE, load_discount_curve, load_index
from credit_pricer.index import Constituent, adjust_to_index_quotes, constituents_from_spreads

TENORS = [1, 3, 5, 7, 10]
SPREADS = [0.0040, 0.0060, 0.0080, 0.0095, 0.0105]


@pytest.fixture(scope="session")
def trade_date() -> dt.date:
    return VALUATION_DATE


@pytest.fixture(scope="session")
def flat_discount(trade_date: dt.date) -> DiscountCurve:
    return DiscountCurve.flat(trade_date, 0.03)


@pytest.fixture(scope="session")
def discount(trade_date: dt.date) -> DiscountCurve:
    return load_discount_curve()


@pytest.fixture(scope="session")
def survival(trade_date: dt.date, discount: DiscountCurve) -> SurvivalCurve:
    return bootstrap_survival_curve(trade_date, discount, TENORS, SPREADS, 0.4)


@pytest.fixture(scope="session")
def index_members(trade_date: dt.date, discount: DiscountCurve) -> list[Constituent]:
    data = load_index()
    return constituents_from_spreads(
        trade_date, discount, data.tenors, data.spreads, data.recovery, data.names
    )


@pytest.fixture(scope="session")
def adjusted_members(
    trade_date: dt.date, discount: DiscountCurve, index_members: list[Constituent]
) -> list[Constituent]:
    data = load_index()
    adj = adjust_to_index_quotes(
        trade_date, data.quoted_tenors, data.quoted_spreads, data.coupon, index_members, discount
    )
    return adj.constituents
