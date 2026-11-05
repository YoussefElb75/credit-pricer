from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from credit_pricer.curves import DiscountCurve, SurvivalCurve
from credit_pricer.index import Constituent
from credit_pricer.tranches import TranchePricer

STRUCTURE = [(0.0, 0.03), (0.03, 0.07), (0.07, 0.10), (0.10, 0.15), (0.15, 0.30), (0.30, 1.0)]


@pytest.fixture(scope="module")
def pricer(
    trade_date: dt.date, discount: DiscountCurve, adjusted_members: list[Constituent]
) -> TranchePricer:
    return TranchePricer(trade_date, 5, [c.curve for c in adjusted_members], discount)


@pytest.mark.parametrize("rho", [0.0, 0.25, 0.6, 0.9, 1.0])
def test_capital_structure_adds_up(pricer: TranchePricer, rho: float) -> None:
    """Sum of tranche expected losses = pool expected loss, for any correlation."""
    total = sum(pricer.expected_tranche_loss(a, d, rho) for a, d in STRUCTURE)
    assert np.allclose(total, pricer.pool_expected_loss(), rtol=1e-9, atol=1e-15)
    prot = sum(pricer.legs(a, d, rho).protection for a, d in STRUCTURE)
    pool = pricer.legs(0.0, 1.0, rho).protection
    assert prot == pytest.approx(pool, rel=1e-9)


def test_correlation_monotonicity(pricer: TranchePricer) -> None:
    rhos = np.linspace(0.0, 0.95, 8)
    eq = [pricer.expected_tranche_loss(0.0, 0.03, r)[-1] for r in rhos]
    sen = [pricer.expected_tranche_loss(0.15, 0.30, r)[-1] for r in rhos]
    assert np.all(np.diff(eq) < 0)  # equity benefits from correlation
    assert np.all(np.diff(sen) > 0)  # senior suffers from correlation


def test_expected_loss_bounds_and_time(pricer: TranchePricer) -> None:
    for a, d in STRUCTURE:
        el = pricer.expected_tranche_loss(a, d, 0.3)
        assert np.all(el >= -1e-15) and np.all(el <= d - a + 1e-15)
        assert np.all(np.diff(el) >= -1e-15)


def test_spreads_decrease_with_seniority(pricer: TranchePricer) -> None:
    spreads = [pricer.legs(a, d, 0.3).fair_spread for a, d in STRUCTURE]
    assert all(x > y for x, y in zip(spreads, spreads[1:]))


def test_upfront_at_fair_spread_is_zero(pricer: TranchePricer) -> None:
    legs = pricer.legs(0.0, 0.03, 0.3)
    assert legs.upfront(legs.fair_spread) == pytest.approx(0.0, abs=1e-15)
    assert legs.upfront(0.05) > 0  # equity fair spread far above 500 bp


def test_lhp_close_to_finite_pool(
    trade_date: dt.date, discount: DiscountCurve, survival: SurvivalCurve
) -> None:
    curves = [survival] * 125
    fin = TranchePricer(trade_date, 5, curves, discount, model="finite")
    lhp = TranchePricer(trade_date, 5, curves, discount, model="lhp")
    for a, d in [(0.0, 0.03), (0.03, 0.07), (0.07, 0.15)]:
        assert fin.legs(a, d, 0.3).fair_spread == pytest.approx(
            lhp.legs(a, d, 0.3).fair_spread, rel=0.05
        )


def test_full_tranche_equals_index_without_recovery(
    trade_date: dt.date, discount: DiscountCurve
) -> None:
    """With zero recovery the [0, 100%] tranche is the index (clean premium leg)."""
    from credit_pricer.index import equal_weights, index_intrinsic

    curves = [SurvivalCurve.flat(trade_date, h) for h in (0.005, 0.01, 0.02, 0.04)] * 10
    tranche = TranchePricer(trade_date, 5, curves, discount, recovery=0.0)
    index = index_intrinsic(trade_date, 5, 0.01, equal_weights(curves, 0.0), discount)
    for rho in (0.0, 0.3, 0.8):
        assert tranche.legs(0.0, 1.0, rho).fair_spread == pytest.approx(index.spread, abs=0.05e-4)


def test_zero_hazard_and_full_recovery(trade_date: dt.date, discount: DiscountCurve) -> None:
    curves = [SurvivalCurve.flat(trade_date, 0.0)] * 10
    p = TranchePricer(trade_date, 5, curves, discount)
    assert p.legs(0.0, 0.03, 0.3).protection == 0.0
    curves = [SurvivalCurve.flat(trade_date, 0.05)] * 10
    p = TranchePricer(trade_date, 5, curves, discount, recovery=1.0)
    assert p.legs(0.0, 0.03, 0.3).protection == pytest.approx(0.0, abs=1e-15)


def test_invalid(trade_date: dt.date, discount: DiscountCurve, survival: SurvivalCurve) -> None:
    with pytest.raises(ValueError):
        TranchePricer(trade_date, 5, [], discount)
    p = TranchePricer(trade_date, 5, [survival], discount)
    with pytest.raises(ValueError):
        p.legs(0.1, 0.05, 0.3)
    with pytest.raises(ValueError):
        p.legs(0.0, 0.05, 1.5)
