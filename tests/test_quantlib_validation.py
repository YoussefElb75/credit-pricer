"""Independent check of the CDS engines and the bootstrap against QuantLib."""

from __future__ import annotations

import datetime as dt

import pytest

from credit_pricer.curves import DiscountCurve

pytest.importorskip("QuantLib")

from credit_pricer.curves import bootstrap_survival_curve  # noqa: E402
from credit_pricer.validation import compare_engines, run_validation  # noqa: E402

# Tolerances in basis points of notional (upfront) or of spread.
TOL = {"isda": 1e-3, "isda_halfday": 1e-3, "midpoint": 1e-8}


@pytest.fixture(scope="module")
def results(discount: DiscountCurve) -> tuple[list, list]:  # type: ignore[type-arg]
    return run_validation(discount)


def test_engine_level(results: tuple[list, list]) -> None:  # type: ignore[type-arg]
    rows, _ = results
    assert {r.method for r in rows} == set(TOL)
    for r in rows:
        assert abs(r.d_par_spread_bp) < TOL[r.method], r
        assert abs(r.d_upfront_bp) < TOL[r.method], r
        assert abs(r.d_rpv01) < 1e-5, r


def test_bootstrap_level(results: tuple[list, list]) -> None:  # type: ignore[type-arg]
    _, rows = results
    for r in rows:
        assert abs(r.d_survival) < 1e-6, r
        assert abs(r.d_upfront_bp) < 2e-3, r


ROLL_DATES = [
    dt.date(2026, 3, 18), dt.date(2026, 3, 19), dt.date(2026, 3, 20), dt.date(2026, 6, 19),
    dt.date(2026, 6, 21), dt.date(2026, 9, 19), dt.date(2026, 12, 18), dt.date(2026, 12, 20),
    dt.date(2027, 3, 19), dt.date(2027, 3, 21),
]


@pytest.mark.parametrize("trade", ROLL_DATES)
def test_engines_around_coupon_dates(trade: dt.date) -> None:
    """Trade dates around IMM dates and weekends, where step-in conventions matter."""
    disc = DiscountCurve.flat(trade, 0.03)
    for method in TOL:
        curve = bootstrap_survival_curve(
            trade, disc, [1, 3, 5, 7, 10], [0.03] * 5, 0.4, method  # type: ignore[arg-type]
        )
        rows = compare_engines(disc, curve, [1, 5, 10], 0.01, 0.4, method, 300.0)  # type: ignore[arg-type]
        for r in rows:
            assert abs(r.d_par_spread_bp) < TOL[r.method], r
            assert abs(r.d_upfront_bp) < TOL[r.method], r
