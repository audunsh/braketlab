"""
Basis function factory for common quantum mechanical systems.

Provides GTOs, STOs, hydrogen eigenfunctions, and harmonic oscillator
eigenfunctions. Cleaned from original braketlab/basisbank.py + supporting modules.
"""

import math
import numpy as np
import sympy as sp

from .algebra import ket, basisfunction, get_ordered_symbols, get_default_variables


# ---------------------------------------------------------------------------
# Real solid harmonics (Helgaker Table 6.3 + eq. 6.4.47)
# ---------------------------------------------------------------------------

def solid_harmonic_table(l, m):
    """
    Real solid harmonic S_lm(x,y,z) from Helgaker Table 6.3.
    Supports l = 0..4.
    """
    x, y, z = sp.symbols("x y z")
    r = sp.sqrt(x**2 + y**2 + z**2)

    assert 0 <= l <= 4, f"Only l=0..4 supported, got l={l}"
    assert abs(m) <= l, f"|m|={abs(m)} > l={l}"

    table = {
        (0, 0): x**0,
        (1, 1): x, (1, 0): z, (1, -1): y,
        (2, 2): sp.sqrt(3) / 2 * (x**2 - y**2),
        (2, 1): sp.sqrt(3) * x * z,
        (2, 0): (3 * z**2 - r**2) / 2,
        (2, -1): sp.sqrt(3) * y * z,
        (2, -2): sp.sqrt(3) * x * y,
        (3, 3): x * (x**2 - 3*y**2) * sp.sqrt(sp.Rational(5, 2)) / 2,
        (3, 2): z * (x**2 - y**2) * sp.sqrt(15) / 2,
        (3, 1): x * (5*z**2 - r**2) * sp.sqrt(sp.Rational(3, 2)) / 2,
        (3, 0): z * (2*z**2 - 3*x**2 - 3*y**2) / 2,
        (3, -1): y * (5*z**2 - r**2) * sp.sqrt(sp.Rational(3, 2)) / 2,
        (3, -2): sp.sqrt(15) * x * y * z,
        (3, -3): sp.sqrt(sp.Rational(5, 2)) / 2 * (3*x**2 - y**2) * y,
        (4, 4): sp.sqrt(35) / 8 * (x**4 - 6*x**2*y**2 + y**4),
        (4, 3): sp.sqrt(sp.Rational(35, 2)) / 2 * (x**2 - 3*y**2) * x * z,
        (4, 2): sp.sqrt(5) / 4 * (7*z**2 - r**2) * (x**2 - y**2),
        (4, 1): sp.sqrt(sp.Rational(5, 2)) / 2 * (7*z**2 - 3*r**2) * x * z,
        (4, 0): (35*z**4 - 30*z**2*r**2 + 3*r**4) / 8,
        (4, -1): sp.sqrt(sp.Rational(5, 2)) / 2 * (7*z**2 - 3*r**2) * y * z,
        (4, -2): sp.sqrt(5) / 2 * (7*z**2 - r**2) * x * y,
        (4, -3): sp.sqrt(sp.Rational(35, 2)) / 2 * (3*x**2 - y**2) * y * z,
        (4, -4): sp.sqrt(35) / 2 * (x**2 - y**2) * x * y,
    }
    return table[(l, m)]


def _double_factorial(n):
    """Double factorial n!! (eq. 6.5.10)."""
    if n <= 0:
        return 1
    if n % 2 == 0:
        return int(np.prod(np.arange(0, n, 2) + 2))
    return int(np.prod(np.arange(0, n, 2) + 1))


def _binomial(a, b):
    return math.factorial(int(a)) // (math.factorial(int(b)) * math.factorial(int(a - b)))


def _V(m):
    return 0.5 if m < 0 else 0.0


def _c_factor(l, m, t, u, v):
    return (
        (-1) ** (t + v - _V(m))
        * 0.25 ** t
        * _binomial(l, t)
        * _binomial(l - t, abs(m) + t)
        * _binomial(t, u)
        * _binomial(abs(m), 2 * v)
    )


def _N_factor(l, m):
    return (
        1 / (2 ** abs(m) * math.factorial(l))
        * np.sqrt(
            2 * math.factorial(l + abs(m))
            * math.factorial(l - abs(m))
            / (2 if m == 0 else 1)
        )
    )


def _get_Slm_rsh(l, m):
    """Real solid harmonic via eq. 6.4.47 (Pink Bible)."""
    x, y, z = sp.symbols("x y z")
    slm = 0
    for t in range(int(np.floor((l - abs(m)) / 2)) + 1):
        for u in range(t + 1):
            vm = _V(m)
            for v_val in np.arange(vm, np.floor(abs(m) / 2 - vm) + vm + 1):
                slm += (
                    _c_factor(l, m, t, u, int(v_val))
                    * x ** int(2 * t + abs(m) - 2 * (u + v_val))
                    * y ** int(2 * (u + v_val))
                    * z ** int(l - 2 * t - abs(m))
                )
    return slm


def _gto_normalization(a, l):
    """GTO normalization prefactor (eq. 6.6.15)."""
    return (
        (2.0 * np.pi) ** (-0.75)
        * (4.0 * a) ** (0.75 + l / 2.0)
        * float(_double_factorial(2 * l - 1)) ** -0.5
    )


# ---------------------------------------------------------------------------
# Hermite polynomials for HO
# ---------------------------------------------------------------------------

def _hermite(f, z, n):
    """Physicist's Hermite polynomial H_n(z)."""
    return (-1) ** n * sp.exp(f ** 2) * sp.diff(sp.exp(-f ** 2), z, n)


# ---------------------------------------------------------------------------
# Laguerre polynomials for hydrogen
# ---------------------------------------------------------------------------

def _associated_laguerre(x, i, alpha):
    rr = sp.symbols("r_1")
    return (
        sp.exp(x) * x ** (-alpha) / math.factorial(i)
        * sp.diff(sp.exp(-rr) * rr ** (i + alpha), rr, i).subs(rr, x)
    )


def _spherical_harmonic(l, m):
    theta, phi = sp.symbols("theta phi")
    am = abs(m)
    x_var = sp.Symbol("_x")
    legendre = sp.diff((x_var ** 2 - 1) ** l, x_var, l) / (2 ** l * math.factorial(l))
    assoc = sp.sqrt((1 - x_var ** 2) ** am) * sp.diff(legendre, x_var, am)
    assoc = assoc.subs(x_var, sp.cos(theta))
    eps = (-1) ** m if m >= 0 else 1
    norm = sp.sqrt((2 * l + 1) * math.factorial(l - am) / (4 * sp.pi * math.factorial(l + am)))
    return eps * norm * assoc * sp.exp(sp.I * m * phi)


def _solid_harmonic_from_spherical(l, m):
    ylm = _spherical_harmonic(l, m)
    theta_, phi_ = sp.symbols("theta phi")
    x, y, z = sp.symbols("x y z")
    phi_val = sp.atan2(y, x)
    theta_val = sp.acos(z / sp.sqrt(x ** 2 + y ** 2 + z ** 2))
    return ylm.subs(theta_, theta_val).subs(phi_, phi_val)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_gto(a, l, m, position=None):
    """
    Normalized Gaussian-type orbital (primitive).

    Parameters:
        a: exponent
        l: angular momentum quantum number
        m: magnetic quantum number
        position: center (default: origin)

    Returns: ket
    """
    if position is None:
        position = np.array([0.0, 0.0, 0.0])
    position = np.asarray(position, dtype=float)

    x, y, z = sp.symbols("x y z")
    psi = (
        _gto_normalization(a, l)
        * _N_factor(l, m)
        * _get_Slm_rsh(l, m)
        * sp.exp(-sp.UnevaluatedExpr(a) * (x ** 2 + y ** 2 + z ** 2))
    )

    syms = get_ordered_symbols(psi)
    default_vars = get_default_variables(0, len(syms))
    for i, sym in enumerate(syms):
        psi = psi.subs(sym, default_vars[i])

    return ket(psi, name=f"\\chi_{{{l},{m}}}^{{{a:.2f}}}", position=position)


def get_sto(a, w, n, l, m, position=None):
    """
    Slater-type orbital.

    Parameters:
        a: exponent
        w: weight
        n: principal quantum number
        l, m: angular momentum quantum numbers
        position: center

    Returns: ket
    """
    if position is None:
        position = np.array([0.0, 0.0, 0.0])
    position = np.asarray(position, dtype=float)

    x, y, z = sp.symbols("x y z")
    r = sp.sqrt(x ** 2 + y ** 2 + z ** 2)
    S = (2.0 * a) ** n * np.sqrt(2.0 * a / math.factorial(2 * int(n)))
    psi = S * solid_harmonic_table(l, m) * r ** (n - 1) * sp.exp(-sp.UnevaluatedExpr(a) * r)

    syms = get_ordered_symbols(psi)
    default_vars = get_default_variables(0, len(syms))
    for i, sym in enumerate(syms):
        psi = psi.subs(sym, default_vars[i])

    return ket(psi, name=f"\\chi_{{{l},{m}}}^{{{a:.2f}}}", position=position)


def get_hydrogen_function(n, l, m, position=None):
    """
    Hydrogen atom eigenfunction |n,l,m>.

    Parameters:
        n: principal quantum number
        l: orbital angular momentum
        m: magnetic quantum number
        position: nucleus position

    Returns: ket
    """
    if position is None:
        position = np.array([0.0, 0.0, 0.0])
    position = np.asarray(position, dtype=float)

    x, y, z = sp.symbols("x y z")
    r = sp.sqrt(x ** 2 + y ** 2 + z ** 2)

    a_bohr = 1
    norm = sp.sqrt(
        (2.0 / (n * a_bohr)) ** 3
        * math.factorial(n - l - 1)
        / (2.0 * n * math.factorial(n + l) ** 3)
    )
    radial = (
        norm
        * (2 / (n * a_bohr) * r) ** l
        * _associated_laguerre(2 / (n * a_bohr) * r, n - l - 1, 2 * l + 1)
        * sp.exp(-r / (n * a_bohr))
    )
    psi = radial * _solid_harmonic_from_spherical(l, m)

    syms = get_ordered_symbols(psi)
    default_vars = get_default_variables(0, len(syms))
    for i, sym in enumerate(syms):
        psi = psi.subs(sym, default_vars[i])

    return ket(psi, name=f"{n},{l},{m}", position=position)


def get_harmonic_oscillator_function(n, omega=1, position=0):
    """
    1D harmonic oscillator eigenfunction |n>.

    Parameters:
        n: quantum number
        omega: angular frequency
        position: center

    Returns: ket with energy = omega*(n + 0.5)
    """
    x = sp.Symbol("x")
    norm = 1 / np.sqrt(math.factorial(n) * 2 ** n)
    prefactor = (omega / (np.pi)) ** 0.25
    gaussian = sp.exp(-omega * x ** 2 / 2)
    hermite = _hermite(sp.sqrt(omega) * x, x, n)
    psi = norm * prefactor * gaussian * hermite

    syms = get_ordered_symbols(psi)
    default_vars = get_default_variables(0, len(syms))
    for i, sym in enumerate(syms):
        psi = psi.subs(sym, default_vars[i])

    return ket(
        psi,
        name=str(n),
        energy=[omega * (0.5 + n)],
        position=np.array([position], dtype=float),
    )
