"""Independent validation against QuantLib (optional dependency: ``pip install QuantLib``).

Two comparisons are made on the same inputs:

* **Engine level** - our survival and discount curves are rebuilt node for node in
  QuantLib (``DiscountCurve`` with log-linear discount factors, ``HazardRateCurve`` with
  piecewise-flat hazard rates) and the same standard CDS is priced with QuantLib's
  ``IsdaCdsEngine`` (Taylor fix, piecewise forwards; with and without the half-day
  accrual bias) and ``MidPointCdsEngine``.
* **Bootstrap level** - QuantLib bootstraps its own hazard curve from the same par
  spreads with ``SpreadCdsHelper`` (ISDA model, which uses the half-day accrual bias)
  and ``PiecewiseFlatHazardRate``; we bootstrap with ``isda_halfday`` to match.
  QuantLib places pillars one day after each maturity; the hazard rates differ only on
  that unobserved day, so survival probabilities at the maturities are comparable.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from credit_pricer.cds.legs import Method
from credit_pricer.cds.pricer import CDS, price_cds
from credit_pricer.curves.discount import DiscountCurve
from credit_pricer.curves.survival import SurvivalCurve, bootstrap_survival_curve


def _ql() -> Any:
    try:
        import QuantLib
    except ImportError as exc:  # pragma: no cover - exercised only without QuantLib
        raise ImportError("validation requires QuantLib: pip install QuantLib") from exc
    return QuantLib


def _qdate(ql: Any, d: dt.date) -> Any:
    return ql.Date(d.day, d.month, d.year)


def _node_dates(ql: Any, ref: dt.date, times: Sequence[float]) -> list[Any]:
    return [_qdate(ql, ref + dt.timedelta(days=round(t * 365.0))) for t in times]


def ql_discount_curve(curve: DiscountCurve) -> Any:
    """QuantLib handle replicating ``curve`` (nodes must fall on whole days)."""
    ql = _ql()
    if curve.spread != 0.0:
        raise ValueError("spread-shifted curves are not supported")
    ref = curve.valuation_date
    dates = [_qdate(ql, ref), *_node_dates(ql, ref, curve.times.tolist())]
    ts = ql.DiscountCurve(dates, [1.0, *curve.discount_factors.tolist()], ql.Actual365Fixed())
    ts.enableExtrapolation()
    return ql.YieldTermStructureHandle(ts)


def ql_survival_curve(curve: SurvivalCurve) -> Any:
    """QuantLib handle replicating a piecewise-flat hazard curve."""
    ql = _ql()
    ref = curve.valuation_date
    dates = [_qdate(ql, ref), *_node_dates(ql, ref, curve.times.tolist())]
    hazards = [float(curve.hazards[0]), *curve.hazards.tolist()]
    ts = ql.HazardRateCurve(dates, hazards, ql.Actual365Fixed())
    ts.enableExtrapolation()
    return ql.DefaultProbabilityTermStructureHandle(ts)


def ql_cds(cds: CDS) -> Any:
    """QuantLib ``CreditDefaultSwap`` with the same standard (CDS2015) conventions."""
    ql = _ql()
    trade = _qdate(ql, cds.trade_date)
    # Schedule generated from the step-in date T+1, as in the ISDA standard model.
    schedule = ql.Schedule(
        trade + 1,
        _qdate(ql, cds.maturity),
        ql.Period(3, ql.Months),
        ql.WeekendsOnly(),
        ql.Following,
        ql.Unadjusted,
        ql.DateGeneration.CDS2015,
        False,
    )
    return ql.CreditDefaultSwap(
        ql.Protection.Buyer,
        1.0,
        cds.coupon,
        schedule,
        ql.Following,
        ql.Actual360(),
        True,
        True,
        trade,
        ql.FaceValueClaim(),
        ql.Actual360(True),
        True,
        trade,
        3,
    )


def ql_engine(method: Method, survival: Any, recovery: float, discount: Any) -> Any:
    ql = _ql()
    if method == "midpoint":
        return ql.MidPointCdsEngine(survival, recovery, discount)
    bias = ql.IsdaCdsEngine.HalfDayBias if method == "isda_halfday" else ql.IsdaCdsEngine.NoBias
    return ql.IsdaCdsEngine(
        survival, recovery, discount, False, ql.IsdaCdsEngine.Taylor, bias,
        ql.IsdaCdsEngine.Piecewise,
    )


@dataclass(frozen=True)
class ComparisonRow:
    """One validation case; differences are ours minus QuantLib's."""

    method: str
    tenor: int
    spread_bp: float
    par_spread_bp: float
    d_par_spread_bp: float
    upfront_bp: float
    d_upfront_bp: float
    rpv01: float
    d_rpv01: float


def compare_engines(
    discount: DiscountCurve,
    survival: SurvivalCurve,
    tenors: Sequence[int],
    coupon: float,
    recovery: float,
    method: Method,
    spread_label_bp: float,
) -> list[ComparisonRow]:
    """Price standard CDS with both libraries on identical curves."""
    ql = _ql()
    trade = discount.valuation_date
    ql.Settings.instance().evaluationDate = _qdate(ql, trade)
    qdisc = ql_discount_curve(discount)
    qsurv = ql_survival_curve(survival)
    engine = ql_engine(method, qsurv, recovery, qdisc)
    rows = []
    for tenor in tenors:
        cds = CDS.standard(trade, tenor, coupon, recovery)
        ours = price_cds(cds, discount, survival, method)
        q = ql_cds(cds)
        q.setPricingEngine(engine)
        q_rpv01 = -(q.couponLegNPV() + q.accrualRebateNPV()) / coupon
        rows.append(
            ComparisonRow(
                method=method,
                tenor=tenor,
                spread_bp=spread_label_bp,
                par_spread_bp=ours.par_spread * 1e4,
                d_par_spread_bp=(ours.par_spread - q.fairSpread()) * 1e4,
                upfront_bp=ours.pv * 1e4,
                d_upfront_bp=(ours.pv - q.NPV()) * 1e4,
                rpv01=ours.rpv01,
                d_rpv01=ours.rpv01 - q_rpv01,
            )
        )
    return rows


@dataclass(frozen=True)
class BootstrapRow:
    """Bootstrap comparison at one tenor (ours minus QuantLib)."""

    tenor: int
    spread_bp: float
    survival: float
    d_survival: float
    d_upfront_bp: float


def compare_bootstrap(
    discount: DiscountCurve,
    tenors: Sequence[int],
    spreads: Sequence[float],
    recovery: float,
    coupon: float,
) -> list[BootstrapRow]:
    """Bootstrap the same par spreads in both libraries and compare the curves."""
    ql = _ql()
    trade = discount.valuation_date
    qtrade = _qdate(ql, trade)
    ql.Settings.instance().evaluationDate = qtrade
    qdisc = ql_discount_curve(discount)
    helpers = [
        ql.SpreadCdsHelper(
            ql.QuoteHandle(ql.SimpleQuote(s)),
            ql.Period(t, ql.Years),
            0,
            ql.WeekendsOnly(),
            ql.Quarterly,
            ql.Following,
            ql.DateGeneration.CDS2015,
            ql.Actual360(),
            recovery,
            qdisc,
            True,
            True,
            ql.Date(),
            ql.Actual360(True),
            True,
            ql.CreditDefaultSwap.ISDA,
        )
        for t, s in zip(tenors, spreads)
    ]
    qcurve = ql.PiecewiseFlatHazardRate(qtrade, helpers, ql.Actual365Fixed())
    qcurve.enableExtrapolation()
    qhandle = ql.DefaultProbabilityTermStructureHandle(qcurve)
    ours = bootstrap_survival_curve(trade, discount, tenors, spreads, recovery, "isda_halfday")
    engine = ql_engine("isda_halfday", qhandle, recovery, qdisc)
    rows = []
    for t, s in zip(tenors, spreads):
        cds = CDS.standard(trade, t, coupon, recovery)
        q_surv = qcurve.survivalProbability(_qdate(ql, cds.maturity))
        o_surv = float(ours.survival(ours.t(cds.maturity)))
        q = ql_cds(cds)
        q.setPricingEngine(engine)
        o_pv = price_cds(cds, discount, ours, "isda_halfday").pv
        rows.append(BootstrapRow(t, s * 1e4, o_surv, o_surv - q_surv, (o_pv - q.NPV()) * 1e4))
    return rows


VALIDATION_TENORS = [1, 3, 5, 7, 10]
VALIDATION_SPREADS_BP = [25.0, 100.0, 300.0, 1000.0]
VALIDATION_METHODS: tuple[Method, ...] = ("isda", "isda_halfday", "midpoint")


def run_validation(
    discount: DiscountCurve, coupon: float = 0.01, recovery: float = 0.4
) -> tuple[list[ComparisonRow], list[BootstrapRow]]:
    """Full grid: flat par spread curves at several levels, both engines, all tenors."""
    engine_rows: list[ComparisonRow] = []
    boot_rows: list[BootstrapRow] = []
    trade = discount.valuation_date
    for level in VALIDATION_SPREADS_BP:
        spreads = [level * 1e-4] * len(VALIDATION_TENORS)
        for method in VALIDATION_METHODS:
            curve = bootstrap_survival_curve(
                trade, discount, VALIDATION_TENORS, spreads, recovery, method
            )
            engine_rows += compare_engines(
                discount, curve, VALIDATION_TENORS, coupon, recovery, method, level
            )
        boot_rows += compare_bootstrap(discount, VALIDATION_TENORS, spreads, recovery, coupon)
    return engine_rows, boot_rows
