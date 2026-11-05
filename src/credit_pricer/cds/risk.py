"""CDS risk measures: CS01 (parallel and bucketed) and recovery sensitivity.

All sensitivities are finite differences with full curve rebuilds, per unit notional,
for the protection buyer (long protection gains when spreads widen).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from credit_pricer.cds.legs import Method
from credit_pricer.cds.pricer import CDS, price_cds
from credit_pricer.curves.discount import DiscountCurve
from credit_pricer.curves.survival import SurvivalCurve, bootstrap_survival_curve

ONE_BP = 1e-4


def _pv(
    cds: CDS,
    discount: DiscountCurve,
    tenors: Sequence[int],
    spreads: Sequence[float],
    curve_recovery: float,
    method: Method,
) -> float:
    curve = bootstrap_survival_curve(
        cds.trade_date, discount, tenors, spreads, curve_recovery, method
    )
    return price_cds(cds, discount, curve, method).pv


def cs01(
    cds: CDS,
    discount: DiscountCurve,
    tenors: Sequence[int],
    par_spreads: Sequence[float],
    curve_recovery: float = 0.4,
    method: Method = "isda",
    bump: float = ONE_BP,
) -> float:
    """PV change for a parallel +1 bp shift of every quoted par spread (curve rebuilt)."""
    base = _pv(cds, discount, tenors, par_spreads, curve_recovery, method)
    bumped = [s + bump for s in par_spreads]
    return (_pv(cds, discount, tenors, bumped, curve_recovery, method) - base) * ONE_BP / bump


def bucketed_cs01(
    cds: CDS,
    discount: DiscountCurve,
    tenors: Sequence[int],
    par_spreads: Sequence[float],
    curve_recovery: float = 0.4,
    method: Method = "isda",
    bump: float = ONE_BP,
) -> NDArray[np.float64]:
    """PV change for a +1 bp shift of each quoted par spread separately."""
    base = _pv(cds, discount, tenors, par_spreads, curve_recovery, method)
    out = np.empty(len(par_spreads))
    for i in range(len(par_spreads)):
        bumped = list(par_spreads)
        bumped[i] += bump
        out[i] = (_pv(cds, discount, tenors, bumped, curve_recovery, method) - base)
    return np.asarray(out * ONE_BP / bump, dtype=float)


@dataclass(frozen=True)
class RecoverySensitivity:
    """PV change for a +1% (absolute) recovery bump.

    ``rebuilt``: curve and trade recovery both bumped, curve re-bootstrapped from the
    same spreads (the usual quoted-spread view; small for near-par trades).
    ``fixed_hazard``: hazard rates held fixed, only the trade recovery is bumped
    (pure loss-given-default exposure, equal to ``-protection / (1 - R)`` per unit R).
    """

    rebuilt: float
    fixed_hazard: float


def recovery_sensitivity(
    cds: CDS,
    discount: DiscountCurve,
    tenors: Sequence[int],
    par_spreads: Sequence[float],
    method: Method = "isda",
    bump: float = 0.01,
) -> RecoverySensitivity:
    """Recovery sensitivity of ``cds`` using the trade recovery for the curve as well."""
    r = cds.recovery
    if not 0.0 <= r + bump < 1.0:
        raise ValueError("bumped recovery must stay in [0, 1)")
    curve: SurvivalCurve = bootstrap_survival_curve(
        cds.trade_date, discount, tenors, par_spreads, r, method
    )
    base = price_cds(cds, discount, curve, method).pv
    bumped_trade = replace(cds, recovery=r + bump)
    fixed = price_cds(bumped_trade, discount, curve, method).pv - base
    rebuilt = _pv(bumped_trade, discount, tenors, par_spreads, r + bump, method) - base
    return RecoverySensitivity(rebuilt=rebuilt, fixed_hazard=fixed)
