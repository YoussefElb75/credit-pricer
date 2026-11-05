from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from credit_pricer.cds import CDS, price_cds
from credit_pricer.curves import DiscountCurve, SurvivalCurve
from credit_pricer.data import load_index
from credit_pricer.index import (
    Constituent,
    adjust_to_index_quotes,
    equal_weights,
    index_intrinsic,
    quoted_spread_pv,
)


def test_intrinsic_is_rpv01_weighted_average(
    trade_date: dt.date, discount: DiscountCurve, index_members: list[Constituent]
) -> None:
    res = index_intrinsic(trade_date, 5, 0.01, index_members, discount)
    pars, rpvs = [], []
    for c in index_members:
        v = price_cds(CDS.standard(trade_date, 5, 0.01, c.recovery), discount, c.curve)
        pars.append(v.par_spread)
        rpvs.append(v.rpv01)
    w = np.array(rpvs) / np.sum(rpvs)
    assert res.spread == pytest.approx(float(w @ np.array(pars)), rel=1e-12)
    assert min(pars) < res.spread < max(pars)
    # high-spread names have smaller RPV01, so the intrinsic spread is below the average
    assert res.spread < float(np.mean(pars))
    assert res.pv == pytest.approx(res.protection - 0.01 * res.rpv01)


def test_identical_names(
    trade_date: dt.date, discount: DiscountCurve, survival: SurvivalCurve
) -> None:
    members = equal_weights([survival] * 10)
    res = index_intrinsic(trade_date, 5, 0.01, members, discount)
    single = price_cds(CDS.standard(trade_date, 5, 0.01), discount, survival)
    assert res.spread == pytest.approx(single.par_spread, rel=1e-13)


def test_quoted_spread_flat_convention(trade_date: dt.date, discount: DiscountCurve) -> None:
    assert abs(quoted_spread_pv(trade_date, 5, 0.01, 0.01, discount)) < 1e-14
    assert quoted_spread_pv(trade_date, 5, 0.01, 0.02, discount) > 0
    assert quoted_spread_pv(trade_date, 5, 0.01, 0.005, discount) < 0


def test_adjustment_matches_quotes(
    trade_date: dt.date, discount: DiscountCurve, index_members: list[Constituent]
) -> None:
    data = load_index()
    adj = adjust_to_index_quotes(
        trade_date, data.quoted_tenors, data.quoted_spreads, data.coupon, index_members, discount
    )
    for tenor, q in zip(data.quoted_tenors, data.quoted_spreads):
        target = quoted_spread_pv(trade_date, tenor, data.coupon, q, discount)
        got = index_intrinsic(trade_date, tenor, data.coupon, adj.constituents, discount).pv
        assert got == pytest.approx(target, abs=1e-12)
        raw = index_intrinsic(trade_date, tenor, data.coupon, index_members, discount)
        assert q < raw.spread  # the synthetic basis is negative
    assert all(0.8 < f < 1.0 for f in adj.factors)  # negative basis -> hazards scaled down


def test_zero_basis_gives_unit_factor(
    trade_date: dt.date, discount: DiscountCurve, survival: SurvivalCurve
) -> None:
    members = equal_weights([survival, survival.scaled(2.0), survival.scaled(0.5)])
    intr = index_intrinsic(trade_date, 5, 0.01, members, discount)
    pv = intr.pv
    # quoted spread whose flat-curve value equals the intrinsic value
    from scipy.optimize import brentq

    q = brentq(lambda s: quoted_spread_pv(trade_date, 5, 0.01, s, discount) - pv, 1e-4, 0.1)
    adj = adjust_to_index_quotes(trade_date, [5], [q], 0.01, members, discount)
    assert adj.factors[0] == pytest.approx(1.0, abs=1e-10)


def test_weights_validated(
    trade_date: dt.date, discount: DiscountCurve, survival: SurvivalCurve
) -> None:
    bad = [Constituent("A", survival, 0.4, 0.7)]
    with pytest.raises(ValueError):
        index_intrinsic(trade_date, 5, 0.01, bad, discount)
