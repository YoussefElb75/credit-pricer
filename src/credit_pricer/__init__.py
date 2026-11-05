"""credit-pricer: credit derivatives pricing validated against QuantLib.

Sub-packages
------------
``curves``    discount curve (deposits/swaps) and survival curve (CDS bootstrap)
``cds``       single-name CDS legs, par spread, upfront, RPV01, CS01, recovery risk
``index``     CDS index intrinsic spread, basis and hazard adjustment
``tranches``  one-factor Gaussian copula: LHP and finite pool, base correlation
``cln``       credit-linked note
"""

from credit_pricer.cds import CDS, cds_legs, cs01, price_cds
from credit_pricer.cln import CreditLinkedNote, price_cln, value_cln
from credit_pricer.curves import (
    DiscountCurve,
    SurvivalCurve,
    bootstrap_discount_curve,
    bootstrap_survival_curve,
)
from credit_pricer.tranches import TranchePricer, calibrate_base_correlation

__version__ = "0.1.0"

__all__ = [
    "CDS",
    "CreditLinkedNote",
    "DiscountCurve",
    "SurvivalCurve",
    "TranchePricer",
    "__version__",
    "bootstrap_discount_curve",
    "bootstrap_survival_curve",
    "calibrate_base_correlation",
    "cds_legs",
    "cs01",
    "price_cds",
    "price_cln",
    "value_cln",
]
