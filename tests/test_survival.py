from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from credit_pricer.cds import CDS, price_cds
from credit_pricer.curves import DiscountCurve, SurvivalCurve, bootstrap_survival_curve

from .conftest import SPREADS, TENORS


@pytest.mark.parametrize("method", ["isda", "midpoint"])
def test_bootstrap_round_trip(
    trade_date: dt.date, discount: DiscountCurve, method: str
) -> None:
    curve = bootstrap_survival_curve(trade_date, discount, TENORS, SPREADS, 0.4, method)  # type: ignore[arg-type]
    for tenor, s in zip(TENORS, SPREADS):
        v = price_cds(CDS.standard(trade_date, tenor, s), discount, curve, method)  # type: ignore[arg-type]
        assert v.par_spread == pytest.approx(s, abs=1e-12)
        assert abs(v.pv) < 1e-13


def test_survival_properties(survival: SurvivalCurve) -> None:
    t = np.linspace(0.0, 15.0, 301)
    q = survival.survival(t)
    assert q[0] == pytest.approx(1.0)
    assert np.all(np.diff(q) <= 0)
    assert np.all(survival.hazards >= 0)
    assert np.allclose(survival.default_probability(t), 1 - q)


def test_credit_triangle(trade_date: dt.date, flat_discount: DiscountCurve) -> None:
    """Flat spread curve -> roughly flat hazard close to s / (1 - R)."""
    s, r = 0.01, 0.4
    curve = bootstrap_survival_curve(trade_date, flat_discount, TENORS, [s] * 5, r)
    assert np.allclose(curve.hazards, s / (1 - r), rtol=0.02)


def test_higher_spreads_mean_lower_survival(
    trade_date: dt.date, discount: DiscountCurve
) -> None:
    low = bootstrap_survival_curve(trade_date, discount, TENORS, SPREADS)
    high = bootstrap_survival_curve(trade_date, discount, TENORS, [2 * s for s in SPREADS])
    t = np.linspace(0.5, 10, 20)
    assert np.all(high.survival(t) < low.survival(t))


def test_recovery_increases_hazard(trade_date: dt.date, discount: DiscountCurve) -> None:
    """Same spreads, higher recovery -> more default probability needed."""
    lo = bootstrap_survival_curve(trade_date, discount, TENORS, SPREADS, 0.2)
    hi = bootstrap_survival_curve(trade_date, discount, TENORS, SPREADS, 0.6)
    assert np.all(hi.hazards > lo.hazards)


def test_segment_scaling(survival: SurvivalCurve) -> None:
    b = [survival.times[1], survival.times[3]]
    sc = survival.scaled_segments(b, [1.0, 2.0])
    t_in, t_out = 0.5 * (survival.times[0] + survival.times[1]), survival.times[2]
    assert float(sc.hazard(t_in)) == pytest.approx(float(survival.hazard(t_in)))
    assert float(sc.hazard(t_out)) == pytest.approx(2 * float(survival.hazard(t_out)))
    t = [1.0, 2.5, 7.0]
    assert np.allclose(survival.with_nodes([2.5]).survival(t), survival.survival(t))
    assert np.allclose(survival.scaled(0.0).survival([1, 5]), 1.0)


def test_inverted_curve_rejected(trade_date: dt.date, discount: DiscountCurve) -> None:
    with pytest.raises(ValueError, match="not consistent"):
        bootstrap_survival_curve(trade_date, discount, [1, 2], [0.05, 0.001])
    with pytest.raises(ValueError):
        bootstrap_survival_curve(trade_date, discount, [1], [0.01], recovery=1.0)
