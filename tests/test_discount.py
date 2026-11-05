from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from credit_pricer.curves import (
    DiscountCurve,
    bootstrap_discount_curve,
    deposit_rate,
    swap_par_rate,
)
from credit_pricer.data import load_rate_quotes


def test_bootstrap_reprices_inputs(discount: DiscountCurve) -> None:
    deposits, swaps = load_rate_quotes()
    for d in deposits:
        assert deposit_rate(discount, d.months) == pytest.approx(d.rate, abs=1e-13)
    for s in swaps:
        assert swap_par_rate(discount, s.years) == pytest.approx(s.rate, abs=1e-12)


def test_curve_shape(discount: DiscountCurve) -> None:
    t = np.linspace(0.0, 40.0, 401)
    df = discount.df(t)
    assert df[0] == pytest.approx(1.0)
    assert np.all(np.diff(df) < 0)  # positive forward rates on the synthetic curve
    assert np.all(df > 0)


def test_flat_curve_and_spread(trade_date: dt.date) -> None:
    c = DiscountCurve.flat(trade_date, 0.03)
    assert float(c.df(2.0)) == pytest.approx(np.exp(-0.06))
    assert float(c.df(80.0)) == pytest.approx(np.exp(-2.4))  # flat extrapolation
    s = c.with_spread(0.01)
    assert float(s.df(2.0)) == pytest.approx(np.exp(-0.08))
    assert s.forward_rate(1.0, 3.0) == pytest.approx(0.04)
    assert float(c.zero_rate(5.0)) == pytest.approx(0.03)


def test_invalid_inputs(trade_date: dt.date) -> None:
    with pytest.raises(ValueError):
        DiscountCurve(trade_date, [1.0, 0.5], [0.9, 0.95])
    with pytest.raises(ValueError):
        DiscountCurve(trade_date, [1.0], [-0.9])
    with pytest.raises(ValueError):
        bootstrap_discount_curve(trade_date, [], [])
    with pytest.raises(ValueError):
        DiscountCurve.flat(trade_date, 0.03).df(-1.0)
