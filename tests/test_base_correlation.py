from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from credit_pricer.curves import DiscountCurve
from credit_pricer.data import load_tranche_quotes
from credit_pricer.index import Constituent
from credit_pricer.tranches import (
    TranchePricer,
    TrancheQuote,
    calibrate_base_correlation,
    quote_from_base_correlations,
)


@pytest.fixture(scope="module")
def pricer(
    trade_date: dt.date, discount: DiscountCurve, adjusted_members: list[Constituent]
) -> TranchePricer:
    return TranchePricer(trade_date, 5, [c.curve for c in adjusted_members], discount)


def test_round_trip_recovers_generating_skew(pricer: TranchePricer) -> None:
    data = load_tranche_quotes()
    curve = calibrate_base_correlation(pricer, data.quotes)
    got = np.array([pt.correlation for pt in curve])
    assert np.allclose(got, data.generating_correlations, atol=1e-5)
    assert [pt.detach for pt in curve] == [q.detach for q in data.quotes]
    assert np.all(np.diff(got) > 0)  # upward-sloping skew


def test_flat_correlation_gives_flat_curve(pricer: TranchePricer) -> None:
    rho = 0.35
    bounds = [0.0, 0.03, 0.07, 0.15]
    quotes = [
        quote_from_base_correlations(pricer, a, d, rho, rho, 0.01)
        for a, d in zip(bounds[:-1], bounds[1:])
    ]
    curve = calibrate_base_correlation(pricer, quotes)
    assert np.allclose([pt.correlation for pt in curve], rho, atol=1e-8)


def test_quote_validation(pricer: TranchePricer) -> None:
    with pytest.raises(ValueError, match="equity"):
        calibrate_base_correlation(pricer, [TrancheQuote(0.03, 0.07, 0.0, 0.05)])
    with pytest.raises(ValueError, match="contiguous"):
        calibrate_base_correlation(
            pricer, [TrancheQuote(0.0, 0.03, 0.3, 0.05), TrancheQuote(0.05, 0.07, 0.0, 0.05)]
        )
    with pytest.raises(ValueError, match="no base correlation"):
        calibrate_base_correlation(pricer, [TrancheQuote(0.0, 0.03, 0.99, 0.05)])
