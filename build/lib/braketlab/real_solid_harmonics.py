"""
Real solid harmonic Gaussian basis function,
as presented in chapter 6 of the "Pink Bible" (*) Helgaker, T., Jorgensen, P., & Olsen, J. (2014). Molecular electronic-structure theory. John Wiley & Sons.
Author: Audun Skau Hansen
"""
import sympy as sp
import numpy as np


def get_default_variables(p: int, n: int = 3) -> list:
    """
    Generates sympy variables with indices $\{p,n\}$ of the kind
    ```[x_{p; 0}, x_{p; 1}, ..., x_{p; n-1}]```

    Example
    ===
    ```get_default_variables(3, 3)```
    will return a list
    ```[x_{3; 0}, x_{3; 1}, x_{3; 2}]```
    """
    variables = []
    for i in range(n):
        variables.append(sp.Symbol("x_{%i; %i}" % (p, i)))
    return variables


def compute_binomial_coefficient(a: int, b: int) -> int:
    """
    Compute the compute_binomial_coefficiential coefficient ( a , b )
    [<a href="https://en.wikipedia.org/wiki/compute_binomial_coefficiential_coefficient">Wikipedia</a>]
    """
    return np.math.factorial(int(a)) // (
        np.math.factorial(int(b)) * np.math.factorial(int(a - b))
    )


def V_factor(m: int) -> float:
    """
    eq. 6.4.50, pink bible (*)
    """
    vm = 0.0
    if m < 0:
        vm = 0.5
    return vm


def c_factor(l: int, m: int, t: int, u: int, v: int) -> float:
    """
    eq. 6.4.48, pink bible (*)
    """
    return (
        (-1) ** (t + v - V_factor(m))
        * (0.25) ** t
        * compute_binomial_coefficient(l, t)
        * compute_binomial_coefficient(l - t, abs(m) + t)
        * compute_binomial_coefficient(t, u)
        * compute_binomial_coefficient(abs(m), 2 * v)
    )


def N_factor(l: int, m: int) -> float:
    """
    eq. 6.4.49, pink bible (*)
    """
    return (
        1
        / (2 ** abs(m) * np.math.factorial(l))
        * np.sqrt(
            2
            * np.math.factorial(l + abs(m))
            * np.math.factorial(l - abs(m))
            * (2 ** (m == 0)) ** -1
        )
    )


def get_Slm(l: int, m: int) -> float:
    """
    eq. 6.4.47, pink bible (*)
    """
    x, y, z = sp.symbols("x y z")  # get_default_variables(0)
    slm = 0
    for t in range(int(np.floor((l - abs(m)) / 2)) + 1):
        for u in range(t + 1):
            vm = V_factor(m)
            for v in np.arange(vm, np.floor(abs(m) / 2 - vm) + vm + 1):
                slm += (
                    c_factor(l, m, t, u, v)
                    * x ** int(2 * t + abs(m) - 2 * (u + v))
                    * y ** int(2 * (u + v))
                    * z ** int(l - 2 * t - abs(m))
                )
    return slm


def get_gto(a: float, l: int, m: int):
    """
    eq. 6.6.15, pink bible (*)
    """
    # x,y,z = get_default_variables(0)
    x, y, z = sp.symbols("x y z")
    return (
        get_Npi(a, l)
        * N_factor(l, m)
        * get_Slm(l, m)
        * sp.exp(-sp.UnevaluatedExpr(a) * (x**2.0 + y**2.0 + z**2.0))
    )


def get_Npi(a_i: float, l: int):
    """
    Returns the normalization prefactor for S_lm(a_i, r)
    a_i = exponent
    l = angular quantum number
    """
    return (
        (2.0 * np.pi) ** (-0.75)
        * (4.0 * a_i) ** (0.75 + l / 2.0)
        * float(double_factorial(2 * l - 1)) ** -0.5
    )


def double_factorial(n: int) -> int:
    """
    'double' factorial function
    eq. 6.5.10 in pink bible (*)
    """
    if n >= 0 and n % 2 == 0:
        return np.prod(np.arange(0, n, 2) + 2)
    else:
        if n % 2 == 1:
            return np.prod(np.arange(0, n, 2) + 1)
