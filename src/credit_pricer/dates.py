"""Date utilities: day counts, business-day adjustment and the standard CDS schedule.

Conventions (V1):

* Business days are Monday to Friday; no holiday calendar is applied.
* Standard CDS coupon dates are the 20th of March, June, September and December,
  adjusted *following*. Maturities are unadjusted.
* Maturities follow the semi-annual roll introduced in December 2015: a trade dated
  on or after 20 March and before 20 September rolls to a 20 June maturity, otherwise
  to a 20 December maturity.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Literal

DayCount = Literal["ACT/360", "ACT/365F", "30/360"]

_CDS_MONTHS = (3, 6, 9, 12)


def is_business_day(d: dt.date) -> bool:
    """Return True on Monday to Friday (no holidays in V1)."""
    return d.weekday() < 5


def adjust_following(d: dt.date) -> dt.date:
    """Roll a date forward to the next business day."""
    while not is_business_day(d):
        d += dt.timedelta(days=1)
    return d


def adjust_modified_following(d: dt.date) -> dt.date:
    """Following, unless that crosses into the next month, then preceding."""
    adj = adjust_following(d)
    if adj.month != d.month:
        adj = d
        while not is_business_day(adj):
            adj -= dt.timedelta(days=1)
    return adj


def add_business_days(d: dt.date, n: int) -> dt.date:
    """Add ``n >= 0`` business days."""
    if n < 0:
        raise ValueError("n must be non-negative")
    out = d
    while n > 0:
        out += dt.timedelta(days=1)
        if is_business_day(out):
            n -= 1
    return out


def add_months(d: dt.date, months: int) -> dt.date:
    """Add calendar months, clipping the day to the end of the target month."""
    m0 = d.month - 1 + months
    year, month = d.year + m0 // 12, m0 % 12 + 1
    for day in (d.day, 30, 29, 28):
        try:
            return dt.date(year, month, day)
        except ValueError:
            continue
    raise AssertionError("unreachable")  # pragma: no cover


def year_fraction(d1: dt.date, d2: dt.date, basis: DayCount) -> float:
    """Year fraction between two dates under the given day-count convention."""
    if basis == "ACT/360":
        return (d2 - d1).days / 360.0
    if basis == "ACT/365F":
        return (d2 - d1).days / 365.0
    if basis == "30/360":
        day1, day2 = min(d1.day, 30), d2.day
        if day1 == 30 and day2 == 31:
            day2 = 30
        return (
            360 * (d2.year - d1.year) + 30 * (d2.month - d1.month) + (day2 - day1)
        ) / 360.0
    raise ValueError(f"unknown day count {basis!r}")


def previous_cds_date(d: dt.date) -> dt.date:
    """Latest unadjusted CDS date (20 Mar/Jun/Sep/Dec) on or before ``d``."""
    year, month = d.year, d.month
    while True:
        if month in _CDS_MONTHS and dt.date(year, month, 20) <= d:
            return dt.date(year, month, 20)
        month -= 1
        if month == 0:
            year, month = year - 1, 12


def cds_maturity(trade_date: dt.date, tenor_years: int) -> dt.date:
    """Unadjusted standard maturity for a CDS traded on ``trade_date``.

    Uses the post-2015 semi-annual roll on 20 March and 20 September.
    """
    year = trade_date.year
    if trade_date < dt.date(year, 3, 20):
        base = dt.date(year - 1, 12, 20)
    elif trade_date < dt.date(year, 9, 20):
        base = dt.date(year, 6, 20)
    else:
        base = dt.date(year, 12, 20)
    return add_months(base, 12 * tenor_years)


@dataclass(frozen=True)
class AccrualPeriod:
    """One premium period: accrual from ``start`` to ``end`` (exclusive), paid on ``pay``.

    For the last period of a standard CDS ``end`` is the maturity *plus one day*, so
    that the maturity date itself accrues (ISDA convention).
    """

    start: dt.date
    end: dt.date
    pay: dt.date

    @property
    def accrual(self) -> float:
        """ACT/360 accrual fraction of the full period."""
        return year_fraction(self.start, self.end, "ACT/360")


def cds_schedule(trade_date: dt.date, maturity: dt.date) -> list[AccrualPeriod]:
    """Standard quarterly CDS premium schedule.

    Accrual starts on the latest adjusted CDS date on or before the step-in date ``T+1``
    (ISDA): a trade dated the day before a coupon date starts accruing on that date.
    Coupon dates are the following-adjusted quarterly CDS dates; the final payment is on
    the adjusted maturity and the final period accrues through the (unadjusted) maturity
    inclusive.
    """
    if maturity <= trade_date:
        raise ValueError("maturity must be after the trade date")
    unadjusted = [maturity]
    d = maturity
    step_in = trade_date + dt.timedelta(days=1)
    first = previous_cds_date(step_in)
    if adjust_following(first) > step_in:  # 20th on a weekend, step-in before the roll
        first = previous_cds_date(first - dt.timedelta(days=1))
    while d > first:
        d = add_months(d, -3)
        unadjusted.append(max(d, first))
    unadjusted.reverse()
    periods: list[AccrualPeriod] = []
    last = len(unadjusted) - 2
    for i, (u0, u1) in enumerate(zip(unadjusted[:-1], unadjusted[1:])):
        start = adjust_following(u0)
        if i == last:
            periods.append(AccrualPeriod(start, u1 + dt.timedelta(days=1), adjust_following(u1)))
        else:
            end = adjust_following(u1)
            periods.append(AccrualPeriod(start, end, end))
    return periods
