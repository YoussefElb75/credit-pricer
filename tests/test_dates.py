from __future__ import annotations

import datetime as dt

import pytest

from credit_pricer.dates import (
    add_business_days,
    add_months,
    adjust_following,
    adjust_modified_following,
    cds_maturity,
    cds_schedule,
    is_business_day,
    previous_cds_date,
    year_fraction,
)

D = dt.date


def test_day_counts() -> None:
    assert year_fraction(D(2026, 1, 1), D(2026, 7, 1), "ACT/360") == pytest.approx(181 / 360)
    assert year_fraction(D(2026, 1, 1), D(2027, 1, 1), "ACT/365F") == pytest.approx(1.0)
    assert year_fraction(D(2026, 1, 31), D(2026, 3, 31), "30/360") == pytest.approx(60 / 360)
    with pytest.raises(ValueError):
        year_fraction(D(2026, 1, 1), D(2026, 2, 1), "ACT/ACT")  # type: ignore[arg-type]


def test_business_day_adjustments() -> None:
    sat = D(2026, 6, 20)
    assert not is_business_day(sat)
    assert adjust_following(sat) == D(2026, 6, 22)
    # Saturday 31 Oct 2026: following leaves the month, modified following goes back
    assert adjust_modified_following(D(2026, 10, 31)) == D(2026, 10, 30)
    assert add_business_days(D(2026, 11, 5), 3) == D(2026, 11, 10)
    assert add_months(D(2026, 1, 31), 1) == D(2026, 2, 28)


@pytest.mark.parametrize(
    ("trade", "tenor", "maturity"),
    [
        (D(2026, 11, 5), 5, D(2031, 12, 20)),
        (D(2026, 3, 19), 5, D(2030, 12, 20)),
        (D(2026, 3, 20), 5, D(2031, 6, 20)),
        (D(2026, 9, 19), 1, D(2027, 6, 20)),
        (D(2026, 9, 20), 1, D(2027, 12, 20)),
    ],
)
def test_semi_annual_roll(trade: dt.date, tenor: int, maturity: dt.date) -> None:
    assert cds_maturity(trade, tenor) == maturity


def test_previous_cds_date() -> None:
    assert previous_cds_date(D(2026, 11, 5)) == D(2026, 9, 20)
    assert previous_cds_date(D(2026, 12, 20)) == D(2026, 12, 20)
    assert previous_cds_date(D(2027, 1, 3)) == D(2026, 12, 20)


def test_accrual_starts_from_step_in() -> None:
    """Trade the day before a coupon date: accrual starts on that coupon date."""
    trade = D(2026, 3, 19)  # T+1 = Friday 20 March 2026
    periods = cds_schedule(trade, cds_maturity(trade, 5))
    assert periods[0].start == D(2026, 3, 20)
    trade = D(2026, 3, 20)
    assert cds_schedule(trade, cds_maturity(trade, 5))[0].start == D(2026, 3, 20)


def test_schedule_structure() -> None:
    trade = D(2026, 11, 5)
    mat = cds_maturity(trade, 5)
    periods = cds_schedule(trade, mat)
    assert len(periods) == 21
    assert periods[0].start == D(2026, 9, 21)  # 20 Sep 2026 is a Sunday
    assert periods[0].start <= trade < periods[0].end
    for a, b in zip(periods[:-1], periods[1:]):
        assert a.end == b.start
        assert is_business_day(a.pay)
    assert periods[-1].end == mat + dt.timedelta(days=1)
    assert periods[-1].pay == adjust_following(mat)
    with pytest.raises(ValueError):
        cds_schedule(trade, trade)


def test_schedule_matches_quantlib() -> None:
    ql = pytest.importorskip("QuantLib")

    def q(d: dt.date) -> object:
        return ql.Date(d.day, d.month, d.year)

    start = D(2025, 1, 1)
    for k in range(0, 800, 3):
        trade = start + dt.timedelta(days=k)
        for tenor in (1, 5, 10):
            mat = cds_maturity(trade, tenor)
            qm = ql.cdsMaturity(q(trade), ql.Period(tenor, ql.Years), ql.DateGeneration.CDS2015)
            assert (qm.year(), qm.month(), qm.dayOfMonth()) == (mat.year, mat.month, mat.day)
            # ISDA: the schedule is generated from the step-in date T+1
            sched = ql.Schedule(
                q(trade + dt.timedelta(days=1)), qm, ql.Period(3, ql.Months), ql.WeekendsOnly(),
                ql.Following, ql.Unadjusted, ql.DateGeneration.CDS2015, False,
            )
            theirs = [(x.year(), x.month(), x.dayOfMonth()) for x in sched]
            mine = [p.start for p in cds_schedule(trade, mat)] + [mat]
            assert theirs == [(x.year, x.month, x.day) for x in mine]
