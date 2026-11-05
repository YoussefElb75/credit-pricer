"""Portfolio loss distributions under the one-factor Gaussian copula.

Losses are fractions of the pool notional. Two models:

* **Large homogeneous pool (LHP)** - Vasicek limit of infinitely many identical names:
  the loss is a deterministic function of the factor, ``L = (1 - R) p(Z)``.
* **Finite pool** - exact distribution of the number of defaults among ``N`` equally
  weighted names with heterogeneous default probabilities and a common recovery, by the
  recursion of Andersen, Sidenius and Basu (2003) conditional on the factor, integrated
  over the factor with composite Gauss-Legendre quadrature.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri, owens_t

from credit_pricer.tranches.copula import conditional_default_probability, factor_quadrature

FloatArray = NDArray[np.float64]


def _check(p: float, recovery: float, rho: float) -> None:
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1]")
    if not 0.0 <= recovery <= 1.0:
        raise ValueError("recovery must lie in [0, 1]")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")


def bivariate_normal_cdf(h: float, k: float, r: float) -> float:
    """P(X <= h, Y <= k) for standard normals with correlation ``r`` (Owen's T form)."""
    if np.isinf(h) or np.isinf(k):
        if h == -np.inf or k == -np.inf:
            return 0.0
        return float(ndtr(k if np.isinf(h) else h))
    if abs(r) >= 1.0:
        return float(ndtr(min(h, k))) if r > 0 else max(float(ndtr(h) + ndtr(k) - 1.0), 0.0)
    h = h if h != 0.0 else 1e-15
    k = k if k != 0.0 else 1e-15
    s = np.sqrt(1.0 - r * r)
    beta = 0.0 if h * k > 0 else 0.5
    val = (
        0.5 * (ndtr(h) + ndtr(k))
        - owens_t(h, (k - r * h) / (h * s))
        - owens_t(k, (h - r * k) / (k * s))
        - beta
    )
    return float(min(max(val, 0.0), 1.0))


def lhp_loss_cdf(x: ArrayLike, p: float, recovery: float, rho: float) -> FloatArray:
    """Cumulative distribution P(L <= x) of the LHP portfolio loss."""
    _check(p, recovery, rho)
    xx = np.asarray(x, dtype=float)
    lgd = 1.0 - recovery
    if lgd == 0.0 or p == 0.0:
        return np.asarray((xx >= 0.0).astype(float), dtype=float)
    if rho == 0.0:
        return np.asarray((xx >= lgd * p).astype(float), dtype=float)
    if rho == 1.0:
        return np.asarray(np.where(xx < 0, 0.0, np.where(xx < lgd, 1.0 - p, 1.0)), dtype=float)
    frac = np.clip(xx / lgd, 0.0, 1.0)
    with np.errstate(divide="ignore"):
        arg = (np.sqrt(1.0 - rho) * ndtri(frac) - ndtri(p)) / np.sqrt(rho)
    out = ndtr(arg)
    return np.asarray(np.where(xx < 0, 0.0, np.where(xx >= lgd, 1.0, out)), dtype=float)


def lhp_excess_loss(strike: float, p: float, recovery: float, rho: float) -> float:
    """E[(L - K)^+] for the LHP loss, in closed form.

    With :math:`z^* = (\\Phi^{-1}(p) - \\sqrt{1-\\rho}\\,\\Phi^{-1}(K/(1-R)))/\\sqrt{\\rho}`:

    .. math:: E[(L-K)^+] = (1-R)\\,\\Phi_2(\\Phi^{-1}(p), z^*; \\sqrt{\\rho}) - K\\,\\Phi(z^*)
    """
    _check(p, recovery, rho)
    lgd = 1.0 - recovery
    if strike <= 0.0:
        return lgd * p - strike
    if strike >= lgd or p == 0.0:
        return 0.0
    if p == 1.0:
        return lgd - strike
    if rho == 0.0:
        return max(lgd * p - strike, 0.0)
    if rho == 1.0:
        return p * (lgd - strike)
    c = float(ndtri(p))
    z_star = (c - np.sqrt(1.0 - rho) * float(ndtri(strike / lgd))) / np.sqrt(rho)
    val = lgd * bivariate_normal_cdf(c, z_star, np.sqrt(rho)) - strike * float(ndtr(z_star))
    return max(val, 0.0)


def lhp_tranche_loss(
    attach: float, detach: float, p: float, recovery: float, rho: float
) -> float:
    """Expected tranche loss E[min((L - A)^+, D - A)] (fraction of pool notional)."""
    if not 0.0 <= attach < detach <= 1.0:
        raise ValueError("need 0 <= attach < detach <= 1")
    return lhp_excess_loss(attach, p, recovery, rho) - lhp_excess_loss(detach, p, recovery, rho)


def _comonotone_pmf(p: FloatArray) -> FloatArray:
    """Number-of-defaults distribution when all latent variables coincide (rho = 1)."""
    ps = np.sort(p)[::-1]
    upper = np.concatenate(([1.0], ps))
    lower = np.concatenate((ps, [0.0]))
    return np.asarray(upper - lower, dtype=float)


def conditional_default_count_pmf(pz: FloatArray) -> FloatArray:
    """Distribution of the number of defaults for independent names.

    ``pz`` has shape ``(n_names, n_scenarios)``; the result has shape
    ``(n_scenarios, n_names + 1)``.
    """
    n, m = pz.shape
    dist = np.zeros((m, n + 1))
    dist[:, 0] = 1.0
    for i in range(n):
        q = pz[i][:, None]
        new = dist[:, : i + 2] * (1.0 - q)
        new[:, 1:] += dist[:, : i + 1] * q
        dist[:, : i + 2] = new
    return dist


def finite_pool_default_pmf(p: ArrayLike, rho: float, n_quad: int = 192) -> FloatArray:
    """Unconditional distribution of the number of defaults among ``len(p)`` names."""
    pp = np.asarray(p, dtype=float)
    if pp.ndim != 1:
        raise ValueError("p must be one-dimensional")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    if rho == 1.0:
        return _comonotone_pmf(pp)
    if rho == 0.0:
        return np.asarray(conditional_default_count_pmf(pp[:, None])[0], dtype=float)
    z, w = factor_quadrature(n_quad)
    pz = conditional_default_probability(pp[:, None], rho, z[None, :])
    pmf = w @ conditional_default_count_pmf(pz)
    return np.asarray(pmf / pmf.sum(), dtype=float)


def tranche_loss_from_pmf(
    pmf: ArrayLike, loss_per_default: float, attach: float, detach: float
) -> float:
    """E[min((L - A)^+, D - A)] for a loss ``L = k * loss_per_default`` with pmf over k."""
    if not 0.0 <= attach < detach <= 1.0:
        raise ValueError("need 0 <= attach < detach <= 1")
    q = np.asarray(pmf, dtype=float)
    losses = np.arange(q.size) * loss_per_default
    return float(q @ np.clip(losses - attach, 0.0, detach - attach))
