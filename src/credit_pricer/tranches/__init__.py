"""Synthetic CDO tranches under the one-factor Gaussian copula."""

from credit_pricer.tranches.base_correlation import (
    BaseCorrelationPoint,
    TrancheQuote,
    calibrate_base_correlation,
    quote_from_base_correlations,
)
from credit_pricer.tranches.copula import (
    conditional_default_probability,
    factor_quadrature,
    gauss_hermite,
)
from credit_pricer.tranches.loss import (
    finite_pool_default_pmf,
    lhp_excess_loss,
    lhp_loss_cdf,
    lhp_tranche_loss,
    tranche_loss_from_pmf,
)
from credit_pricer.tranches.tranche import TrancheLegs, TranchePricer

__all__ = [
    "BaseCorrelationPoint",
    "TrancheLegs",
    "TranchePricer",
    "TrancheQuote",
    "calibrate_base_correlation",
    "conditional_default_probability",
    "factor_quadrature",
    "finite_pool_default_pmf",
    "gauss_hermite",
    "lhp_excess_loss",
    "lhp_loss_cdf",
    "lhp_tranche_loss",
    "quote_from_base_correlations",
    "tranche_loss_from_pmf",
]
