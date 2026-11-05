from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ndtr, ndtri
from scipy.stats import binom, multivariate_normal

from credit_pricer.tranches import (
    conditional_default_probability,
    factor_quadrature,
    finite_pool_default_pmf,
    gauss_hermite,
    lhp_excess_loss,
    lhp_loss_cdf,
    lhp_tranche_loss,
    tranche_loss_from_pmf,
)
from credit_pricer.tranches.loss import bivariate_normal_cdf


def test_gauss_hermite_moments() -> None:
    z, w = gauss_hermite(32)
    assert w.sum() == pytest.approx(1.0)
    assert w @ z == pytest.approx(0.0, abs=1e-14)
    assert w @ z**2 == pytest.approx(1.0)
    assert w @ z**4 == pytest.approx(3.0)


@pytest.mark.parametrize("rho", [0.0, 0.2, 0.7, 0.95, 0.99])
def test_conditional_probability_averages_to_p(rho: float) -> None:
    z, w = factor_quadrature()
    assert w.sum() == pytest.approx(1.0)
    p = np.array([0.001, 0.03, 0.2])
    pz = conditional_default_probability(p[:, None], rho, z[None, :])
    assert np.allclose(pz @ w, p, rtol=1e-4 if rho > 0.95 else 1e-8)


def test_conditional_probability_limits() -> None:
    z = np.array([-1.0, 0.0, 1.0])
    assert np.allclose(conditional_default_probability(0.1, 0.0, z), 0.1)
    assert np.array_equal(
        conditional_default_probability(0.5, 1.0, z), np.array([1.0, 0.0, 0.0])
    )
    with pytest.raises(ValueError):
        conditional_default_probability(0.1, 1.2, z)


@pytest.mark.parametrize(
    ("h", "k", "r"), [(0.3, -1.2, 0.5), (-2.0, -1.0, 0.9), (1.0, 2.0, 0.1), (0.0, 0.4, 0.3)]
)
def test_bivariate_normal(h: float, k: float, r: float) -> None:
    ref = multivariate_normal([0, 0], [[1, r], [r, 1]]).cdf([h, k])
    assert bivariate_normal_cdf(h, k, r) == pytest.approx(ref, abs=1e-7)


@pytest.mark.parametrize("rho", [0.05, 0.3, 0.6, 0.9])
@pytest.mark.parametrize("strike", [0.0, 0.01, 0.03, 0.1, 0.5])
def test_lhp_excess_loss_against_quadrature(rho: float, strike: float) -> None:
    p, r = 0.05, 0.4

    def integrand(z: float) -> float:
        pz = float(ndtr((ndtri(p) - np.sqrt(rho) * z) / np.sqrt(1 - rho)))
        return float(max((1 - r) * pz - strike, 0.0) * np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi))

    z_star = (ndtri(p) - np.sqrt(1 - rho) * ndtri(strike / (1 - r))) / np.sqrt(rho)
    ref, _ = quad(integrand, -12, min(float(z_star), 12.0), limit=400, epsabs=1e-14)
    assert lhp_excess_loss(strike, p, r, rho) == pytest.approx(ref, abs=1e-10)


def test_lhp_limits() -> None:
    p, r = 0.05, 0.4
    # rho = 0: loss is deterministic (1-R)p
    assert lhp_tranche_loss(0.0, 0.03, p, r, 0.0) == pytest.approx(0.03)
    assert lhp_tranche_loss(0.03, 0.07, p, r, 0.0) == pytest.approx(0.0)
    # rho = 1: all-or-nothing
    assert lhp_tranche_loss(0.0, 0.03, p, r, 1.0) == pytest.approx(p * 0.03)
    assert lhp_tranche_loss(0.1, 0.3, p, r, 1.0) == pytest.approx(p * 0.2)
    # recovery 100%: no loss; recovery 0%: losses up to 100%
    assert lhp_tranche_loss(0.0, 0.03, p, 1.0, 0.3) == 0.0
    assert lhp_tranche_loss(0.0, 1.0, p, 0.0, 0.3) == pytest.approx(p)
    assert lhp_tranche_loss(0.0, 1.0, p, r, 0.3) == pytest.approx((1 - r) * p)
    assert lhp_excess_loss(0.5, 0.0, r, 0.3) == 0.0


def test_lhp_cdf_consistency() -> None:
    p, r, rho = 0.05, 0.4, 0.3
    x = np.linspace(0, 0.6, 3001)
    cdf = lhp_loss_cdf(x, p, r, rho)
    assert np.all(np.diff(cdf) >= 0)
    mean = np.trapezoid(1 - cdf, x)
    assert mean == pytest.approx((1 - r) * p, rel=1e-3)


def test_finite_pool_pmf_properties() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0.005, 0.1, 50)
    for rho in (0.0, 0.3, 0.8, 1.0):
        pmf = finite_pool_default_pmf(p, rho)
        assert pmf.sum() == pytest.approx(1.0)
        assert np.all(pmf >= -1e-15)
        assert np.arange(51) @ pmf == pytest.approx(p.sum(), rel=1e-6)


def test_finite_pool_rho_zero_is_binomial() -> None:
    pmf = finite_pool_default_pmf(np.full(40, 0.07), 0.0)
    assert np.allclose(pmf, binom.pmf(np.arange(41), 40, 0.07), atol=1e-15)


def test_finite_pool_rho_one_is_comonotone() -> None:
    p = np.array([0.1, 0.02, 0.05])
    pmf = finite_pool_default_pmf(p, 1.0)
    assert np.allclose(pmf, [0.9, 0.05, 0.03, 0.02])
    # correlation pushes probability mass to the extremes: variance increases with rho
    q = np.full(20, 0.05)
    k = np.arange(21)
    var = [finite_pool_default_pmf(q, r) @ (k - 1.0) ** 2 for r in (0.0, 0.3, 0.6, 0.9, 1.0)]
    assert np.all(np.diff(var) > 0)
    assert var[-1] == pytest.approx(0.05 * 19**2 + 0.95 * 1.0)


def test_large_finite_pool_approaches_lhp() -> None:
    n, p, r, rho = 1000, 0.05, 0.4, 0.3
    pmf = finite_pool_default_pmf(np.full(n, p), rho, 128)
    for a, d in [(0.0, 0.03), (0.03, 0.07), (0.07, 0.15)]:
        fin = tranche_loss_from_pmf(pmf, (1 - r) / n, a, d)
        assert fin == pytest.approx(lhp_tranche_loss(a, d, p, r, rho), rel=0.02)


def test_invalid_tranche() -> None:
    with pytest.raises(ValueError):
        lhp_tranche_loss(0.07, 0.03, 0.05, 0.4, 0.3)
    with pytest.raises(ValueError):
        tranche_loss_from_pmf([1.0], 0.01, 0.0, 1.5)
