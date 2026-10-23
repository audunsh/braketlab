"""
Integration engines for braketlab_core.

Provides:
- Analytical GTO overlap/kinetic integrals via Obara-Saika recurrence
- 1D quadrature via scipy.integrate.quad
- Multi-dimensional Monte Carlo with importance sampling
- Content-addressable expression cache for deduplication
"""

import numpy as np
from scipy import integrate
from scipy.stats import multivariate_normal
from scipy.special import hyp1f1
import sympy as sp


# ---------------------------------------------------------------------------
# Expression cache: avoids recomputing identical integrals
# ---------------------------------------------------------------------------

_integral_cache: dict[tuple, complex] = {}


def _cache_key(b1, b2) -> tuple:
    """Content-addressable key for a pair of basis functions."""
    return (
        str(b1.ket_sympy_expression),
        tuple(b1.position),
        str(b2.ket_sympy_expression),
        tuple(b2.position),
    )


def clear_cache():
    """Clear the integral cache."""
    _integral_cache.clear()


# ---------------------------------------------------------------------------
# Analytical GTO integrals (Obara-Saika)
# ---------------------------------------------------------------------------

def boys_function(n, T):
    """
    Boys function F_n(T) = int_0^1 t^(2n) exp(-T*t^2) dt

    Used for nuclear attraction and electron repulsion integrals.
    """
    if T < 1e-15:
        return 1.0 / (2 * n + 1)
    return hyp1f1(n + 0.5, n + 1.5, -T) / (2.0 * n + 1.0)


def gaussian_product_center(a, A, b, B):
    """Gaussian product theorem: P = (a*A + b*B) / (a + b)."""
    return (a * np.asarray(A, dtype=float) + b * np.asarray(B, dtype=float)) / (a + b)


def overlap_1d(l1, l2, PA, PB, gamma):
    """
    Obara-Saika recurrence for 1D overlap integrals S(l1, l2).

    PA, PB = components of (P - A) and (P - B) along one Cartesian axis.
    gamma = a + b (sum of exponents).
    """
    sz = max(l1, l2) + 3
    S = np.zeros((sz, sz))
    S[0, 0] = 1.0

    for i in range(1, l1 + 1):
        prev2 = S[i - 2, 0] if i >= 2 else 0.0
        S[i, 0] = PA * S[i - 1, 0] + (i - 1) / (2 * gamma) * prev2

    for j in range(1, l2 + 1):
        prev2 = S[0, j - 2] if j >= 2 else 0.0
        S[0, j] = PB * S[0, j - 1] + (j - 1) / (2 * gamma) * prev2

    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            prev2 = S[i, j - 2] if j >= 2 else 0.0
            S[i, j] = (
                PB * S[i, j - 1]
                + i / (2 * gamma) * S[i - 1, j - 1]
                + (j - 1) / (2 * gamma) * prev2
            )

    return S[l1, l2]


def overlap_3d_cartesian(a, lmn_a, A, b, lmn_b, B):
    """
    Analytical overlap integral between two primitive Cartesian GTOs.

    Parameters:
        a, b: exponents
        lmn_a, lmn_b: tuples of (lx, ly, lz) angular momentum
        A, B: center positions (3-vectors)

    Returns: overlap integral value
    """
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    gamma = a + b
    P = gaussian_product_center(a, A, b, B)
    PA = P - A
    PB = P - B
    AB2 = np.sum((A - B) ** 2)

    prefactor = np.exp(-a * b / gamma * AB2) * (np.pi / gamma) ** 1.5

    sx = overlap_1d(lmn_a[0], lmn_b[0], PA[0], PB[0], gamma)
    sy = overlap_1d(lmn_a[1], lmn_b[1], PA[1], PB[1], gamma)
    sz = overlap_1d(lmn_a[2], lmn_b[2], PA[2], PB[2], gamma)

    return prefactor * sx * sy * sz


def kinetic_1d(l1, l2, PA, PB, gamma, exp_b):
    """
    1D kinetic energy integral component (Szabo & Ostlund, eq. A.11).

    T^(x)_ij = exp_b*(2*l2+1)*S(l1,l2) - 2*exp_b^2*S(l1,l2+2) - l2*(l2-1)/2*S(l1,l2-2)

    Parameters:
        l1, l2: angular momentum along this axis
        PA, PB: Gaussian product center offsets
        gamma: sum of exponents (a + b)
        exp_b: exponent of the second Gaussian
    """
    S = overlap_1d
    s_ij = S(l1, l2, PA, PB, gamma)
    s_ij2 = S(l1, l2 + 2, PA, PB, gamma)
    s_ij_m2 = S(l1, max(l2 - 2, 0), PA, PB, gamma) if l2 >= 2 else 0.0

    return (
        exp_b * (2 * l2 + 1) * s_ij
        - 2 * exp_b ** 2 * s_ij2
        - 0.5 * l2 * (l2 - 1) * s_ij_m2
    )


def kinetic_3d_cartesian(a, lmn_a, A, b, lmn_b, B):
    """
    Analytical kinetic energy integral between two primitive Cartesian GTOs.

    Returns: <a|T|b> where T = -1/2 nabla^2
    """
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    gamma = a + b
    P = gaussian_product_center(a, A, b, B)
    PA, PB = P - A, P - B
    AB2 = np.sum((A - B) ** 2)
    prefactor = np.exp(-a * b / gamma * AB2) * (np.pi / gamma) ** 1.5

    la, lb = lmn_a, lmn_b
    S = overlap_1d
    K = kinetic_1d

    sx = S(la[0], lb[0], PA[0], PB[0], gamma)
    sy = S(la[1], lb[1], PA[1], PB[1], gamma)
    sz = S(la[2], lb[2], PA[2], PB[2], gamma)

    tx = K(la[0], lb[0], PA[0], PB[0], gamma, b)
    ty = K(la[1], lb[1], PA[1], PB[1], gamma, b)
    tz = K(la[2], lb[2], PA[2], PB[2], gamma, b)

    return prefactor * (tx * sy * sz + sx * ty * sz + sx * sy * tz)


def nuclear_attraction_1d(l1, l2, PA, PB, CP, gamma, n_max):
    """
    Obara-Saika recurrence for 1D nuclear attraction auxiliary integrals.
    Returns A[l1, l2, n] for n = 0..n_max.
    """
    sz = max(l1, l2) + 2
    A_arr = np.zeros((sz, sz, n_max + 1))

    for n in range(n_max + 1):
        A_arr[0, 0, n] = boys_function(n, gamma * np.sum(CP ** 2)) * (
            (-2 * gamma) ** n
        )

    # Full OS recurrence would go here for l1 > 0, l2 > 0
    # (Kept simple for the alpha; full recurrence is a future enhancement)

    return A_arr[l1, l2, 0] if l1 == 0 and l2 == 0 else None


# ---------------------------------------------------------------------------
# Monte Carlo integration
# ---------------------------------------------------------------------------

def onebody_mc(integrand, sigma, loc, n_samples):
    """
    Monte Carlo estimate with importance sampling (multivariate normal proposal).
    """
    sig = np.diag(np.asarray(sigma, dtype=float))
    rng = np.random.default_rng()
    R = rng.multivariate_normal(loc, sig, n_samples)
    P = multivariate_normal(mean=loc, cov=sig).pdf(R)
    return np.mean(integrand(*R.T) * P ** -1)


def eri_mc(phi_p, phi_q, phi_r, phi_s, pp=None, pq=None, pr=None, ps=None,
           n_samples=1000000, sigma=0.5, Pr=None, Qr=None, zeta=1, eta=1):
    """
    Electron repulsion integral via Monte Carlo.

    (pq|rs) = integral phi_p(r1) phi_q(r1) 1/|r1-r2| phi_r(r2) phi_s(r2) dr1 dr2
    """
    zero3 = np.array([0.0, 0.0, 0.0])
    pp = pp if pp is not None else zero3
    pq = pq if pq is not None else zero3
    pr = pr if pr is not None else zero3
    ps = ps if ps is not None else zero3
    Pr = Pr if Pr is not None else zero3
    Qr = Qr if Qr is not None else zero3

    rng = np.random.default_rng()
    x = rng.multivariate_normal(np.zeros(6), np.eye(6) * sigma, n_samples)
    P_pdf = multivariate_normal(mean=np.zeros(6), cov=np.eye(6) * sigma).pdf

    u1 = x[:, :3] * zeta ** -0.5 + Pr
    u2 = x[:, 3:] * eta ** -0.5 + Qr
    r12 = np.sqrt(np.sum((u1 - u2) ** 2, axis=1))

    val = (
        phi_p(u1[:, 0] - pp[0], u1[:, 1] - pp[1], u1[:, 2] - pp[2])
        * phi_q(u1[:, 0] - pq[0], u1[:, 1] - pq[1], u1[:, 2] - pq[2])
        * phi_r(u2[:, 0] - pr[0], u2[:, 1] - pr[1], u2[:, 2] - pr[2])
        * phi_s(u2[:, 0] - ps[0], u2[:, 1] - ps[1], u2[:, 2] - ps[2])
    )
    return np.mean(val / (P_pdf(x) * r12)) * (zeta * eta) ** -1.5


# ---------------------------------------------------------------------------
# Inner product (public API)
# ---------------------------------------------------------------------------

def inner_product(b1, b2, n_samples=int(1e6), sigma=None):
    """
    Compute the inner product <b1|b2> between two basis functions.

    Uses a content-addressable cache to avoid recomputing identical integrals.
    Dispatches to scipy.integrate.quad (1D) or importance-sampled Monte Carlo (nD).
    """
    key = _cache_key(b1, b2)
    if key in _integral_cache:
        return _integral_cache[key]

    ri = b1.position * 0
    rj = b2.position * 0

    def integrand(*R):
        coords_1 = np.array([R[k] - ri[k] for k in range(len(ri))])
        coords_2 = np.array([R[k] - rj[k] for k in range(len(rj))])
        return b1.bra_numeric_expression(*coords_1) * b2.ket_numeric_expression(*coords_2)

    ndim_b1 = len(b1.bra_sympy_expression.free_symbols)
    ndim_b2 = len(b2.ket_sympy_expression.free_symbols)

    if ndim_b1 == 1 and ndim_b2 == 1:
        result = integrate.quad(integrand, -np.inf, np.inf)[0]
    else:
        ai, aj = b1.decay, b2.decay
        ri_pos, rj_pos = b1.position, b2.position
        R_center = (ai * ri_pos + aj * rj_pos) / (ai + aj)
        if sigma is None:
            sigma = 0.5 * (ai + aj)
        result = onebody_mc(integrand, np.ones(len(R_center)) * sigma, R_center, n_samples)

    _integral_cache[key] = result
    return result
