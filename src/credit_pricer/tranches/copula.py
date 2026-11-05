"""One-factor Gaussian copula: conditional default probabilities and factor quadrature.

Name ``i`` defaults before ``t`` when its latent variable
:math:`X_i = \\sqrt{\\rho}\\,Z + \\sqrt{1-\\rho}\\,\\varepsilon_i` falls below
:math:`\\Phi^{-1}(p_i(t))`. Conditional on the common factor ``Z = z``, defaults are
independent with probability

.. math::

    p_i(t \\mid z) =
    \\Phi\\left(\\frac{\\Phi^{-1}(p_i(t)) - \\sqrt{\\rho}\\,z}{\\sqrt{1-\\rho}}\\right).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri, roots_hermitenorm, roots_legendre

FloatArray = NDArray[np.float64]


@lru_cache(maxsize=16)
def _gh(n: int) -> tuple[FloatArray, FloatArray]:
    z, w = roots_hermitenorm(n)
    return np.asarray(z, dtype=float), np.asarray(w / np.sqrt(2.0 * np.pi), dtype=float)


def gauss_hermite(n: int = 64) -> tuple[FloatArray, FloatArray]:
    """Nodes and weights integrating against the standard normal density."""
    if n < 1:
        raise ValueError("n must be positive")
    z, w = _gh(n)
    return z.copy(), w.copy()


@lru_cache(maxsize=16)
def _composite_gl(panels: int, order: int, half_width: float) -> tuple[FloatArray, FloatArray]:
    x, w = roots_legendre(order)
    edges = np.linspace(-half_width, half_width, panels + 1)
    mid, half = 0.5 * (edges[:-1] + edges[1:]), 0.5 * np.diff(edges)
    z = (mid[:, None] + half[:, None] * x[None, :]).ravel()
    wz = (half[:, None] * w[None, :]).ravel() * np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return np.asarray(z, dtype=float), np.asarray(wz / wz.sum(), dtype=float)


def factor_quadrature(n: int = 192) -> tuple[FloatArray, FloatArray]:
    """Nodes and weights for E[f(Z)], Z standard normal, used by the tranche models.

    Composite Gauss-Legendre (8 nodes per panel) on [-8.5, 8.5], weights normalised to
    one. Unlike Gauss-Hermite it resolves the steep conditional default probabilities
    met at high correlation: with 192 nodes E[p(Z)] matches p to ~1e-10 (relative) at
    rho = 0.95 and ~1e-5 at rho = 0.99, where 64-node Gauss-Hermite is off by ~10%.
    """
    if n < 8 or n % 8:
        raise ValueError("n must be a positive multiple of 8")
    z, w = _composite_gl(n // 8, 8, 8.5)
    return z.copy(), w.copy()


def conditional_default_probability(p: ArrayLike, rho: float, z: ArrayLike) -> FloatArray:
    """Conditional default probabilities ``p(z)``, broadcasting ``p`` against ``z``.

    ``rho = 0`` returns ``p`` unchanged; ``rho = 1`` gives the comonotone indicator
    ``1{z < Phi^{-1}(p)}``.
    """
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must lie in [0, 1]")
    pp = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    zz = np.asarray(z, dtype=float)
    if rho == 0.0:
        return np.asarray(np.broadcast_to(pp, np.broadcast(pp, zz).shape).copy(), dtype=float)
    with np.errstate(divide="ignore"):
        threshold = ndtri(pp)
    if rho == 1.0:
        return np.asarray((zz < threshold).astype(float), dtype=float)
    return np.asarray(ndtr((threshold - np.sqrt(rho) * zz) / np.sqrt(1.0 - rho)), dtype=float)
