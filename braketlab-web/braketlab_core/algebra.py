"""
Core algebra engine for Dirac bra-ket notation.

Provides basisfunction, ket, and operator classes with full operator
overloading for quantum mechanics computations. Cleaned up from the
original braketlab/core.py for Pyodide compatibility:
  - No numba dependency
  - No matplotlib import at module level
  - No evince import (visualization handled by viewer.py)
  - Expression hashing for efficient flatten()
"""

import copy
import warnings
from functools import lru_cache

import numpy as np
import sympy as sp
from scipy import integrate
from scipy.stats import multivariate_normal


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_ordered_symbols(expr):
    """Return free symbols of a SymPy expression, sorted by name."""
    symbols = np.array(list(expr.free_symbols))
    order = np.argsort([s.name for s in symbols])
    return symbols[order]


def get_default_variables(p, n=3):
    """Generate standard indexed variables x_{p;0}, x_{p;1}, ..."""
    return [sp.Symbol("x_{%i; %i}" % (p, i)) for i in range(n)]


def translate_sympy_expression(sympy_expression, translation_vector):
    """Shift a SymPy expression by a translation vector."""
    symbols = get_ordered_symbols(sympy_expression)
    assert len(symbols) == len(translation_vector), (
        f"Translation vector length {len(translation_vector)} != "
        f"symbol count {len(symbols)}"
    )
    result = sympy_expression
    for i, sym in enumerate(symbols):
        if np.abs(translation_vector[i]) >= 20.0:
            result = result.subs(sym, sym - sp.UnevaluatedExpr(translation_vector[i]))
        else:
            result = result.subs(sym, sym - translation_vector[i])
    return result


def _expr_hash(expr):
    """Content-addressable hash for SymPy expressions (for efficient dedup)."""
    return hash(str(expr))


# ---------------------------------------------------------------------------
# Basis function
# ---------------------------------------------------------------------------

class basisfunction:
    """
    A basis function in R^n wrapping a SymPy expression.

    Supports pointwise arithmetic (+, -, *) and numerical evaluation via __call__.
    """

    def __init__(self, sympy_expression, position=None, domain=None, name="\\chi"):
        self.__name__ = name
        self.dimension = len(sympy_expression.free_symbols)
        self.domain = domain

        if position is not None:
            self.position = np.array(position, dtype=float)
        else:
            self.position = np.zeros(self.dimension, dtype=float)

        assert len(self.position) == self.dimension, (
            f"Position has {len(self.position)} dims but expression has {self.dimension}"
        )

        self.ket_sympy_expression = translate_sympy_expression(
            sympy_expression, self.position
        )
        self.bra_sympy_expression = translate_sympy_expression(
            sp.conjugate(sympy_expression), self.position
        )

        symbols = get_ordered_symbols(sympy_expression)
        self.ket_numeric_expression = sp.lambdify(
            symbols, self.ket_sympy_expression, "numpy"
        )
        self.bra_numeric_expression = sp.lambdify(
            symbols, self.bra_sympy_expression, "numpy"
        )

        self.normalization = 1.0
        self.decay = 1.0

    def normalize(self):
        """Set normalization so that <self|self> = 1."""
        from .integrals import inner_product
        s = inner_product(self, self)
        self.normalization = s ** -0.5

    def get_domain(self, other=None):
        if other is None:
            return self.domain
        if self.domain is None:
            return None
        domain = []
        for i in range(len(self.domain)):
            domain.append([
                max(self.domain[i][0], other.domain[i][0]),
                min(self.domain[i][1], other.domain[i][1]),
            ])
        return domain

    def __call__(self, *r):
        return self.normalization * self.ket_numeric_expression(*r)

    def __mul__(self, other):
        return basisfunction(
            self.ket_sympy_expression * other.ket_sympy_expression,
            position=0.5 * (self.position + other.position),
            domain=self.get_domain(other),
            name=self.__name__ + other.__name__,
        )

    def __rmul__(self, other):
        if isinstance(other, basisfunction):
            return basisfunction(
                self.ket_sympy_expression * other.ket_sympy_expression,
                position=0.5 * (self.position + other.position),
                domain=self.get_domain(other),
                name=self.__name__ + other.__name__,
            )
        return self.__mul__(other)

    def __add__(self, other):
        return basisfunction(
            self.ket_sympy_expression + other.ket_sympy_expression,
            position=0.5 * (self.position + other.position),
            domain=self.get_domain(other),
        )

    def __sub__(self, other):
        return basisfunction(
            self.ket_sympy_expression - other.ket_sympy_expression,
            position=0.5 * (self.position + other.position),
            domain=self.get_domain(other),
        )

    def _repr_html_(self):
        return "$ %s $" % sp.latex(self.ket_sympy_expression)


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class operator:
    """Parent class for quantum mechanical operators."""
    eigenstates = None
    eigenvalues = None


class operator_expression:
    """
    Algebraic combination of operator products with coefficients.

    Represents sum_i c_i * (Op_i1 * Op_i2 * ... )
    """

    def __init__(self, ops, coefficients=None):
        self.ops = ops
        if issubclass(type(ops), operator):
            self.ops = [[ops]]
        self.coefficients = coefficients
        if coefficients is None:
            self.coefficients = np.ones(len(self.ops))

    def __mul__(self, other):
        if isinstance(other, operator):
            new_ops = [i + j for i in self.ops for j in other.ops]
            return operator_expression(new_ops).flatten()
        return self.apply(other)

    def __add__(self, other):
        new_ops = self.ops + other.ops
        new_coeffs = list(self.coefficients) + list(other.coefficients)
        return operator_expression(new_ops, new_coeffs).flatten()

    def __sub__(self, other):
        new_ops = self.ops + other.ops
        new_coeffs = list(self.coefficients) + [-c for c in other.coefficients]
        return operator_expression(new_ops, new_coeffs).flatten()

    def flatten(self):
        new_ops, new_coeffs, found = [], [], set()
        for i in range(len(self.ops)):
            if i not in found:
                new_ops.append(self.ops[i])
                coeff = self.coefficients[i]
                for j in range(i + 1, len(self.ops)):
                    if self.ops[i] == self.ops[j]:
                        coeff += self.coefficients[j]
                        found.add(j)
                new_coeffs.append(coeff)
        return operator_expression(new_ops, new_coeffs)

    def apply(self, other_ket):
        ret = None
        for i in range(len(self.ops)):
            ret_term = other_ket * 1
            for j in range(len(self.ops[i])):
                ret_term = self.ops[i][-j] * ret_term
            ret = ret_term if ret is None else ret + ret_term
        return ret

    def _repr_html_(self):
        parts = []
        for i in range(len(self.ops)):
            c = self.coefficients[i]
            sign = "+" if c > 0 else ""
            if abs(c) == 1:
                sign = "+" if c > 0 else "-"
                coeff_str = sign
            else:
                coeff_str = f"{sign}{c:.2f}"
            op_strs = "".join(
                f"$\\big{{(}}${self.ops[i][j]._repr_html_()}$\\big{{)}}$"
                for j in range(len(self.ops[i]))
            )
            parts.append(coeff_str + op_strs)
        return "".join(parts)


class sympy_operator(operator):
    def __init__(self, sympy_expression):
        self.sympy_expression = sympy_expression

    def __mul__(self, other):
        return self.sympy_expression * other


class translation(operator):
    def __init__(self, translation_vector):
        self.translation_vector = np.array(translation_vector)

    def __mul__(self, other):
        new_expr = translate_sympy_expression(
            other.get_ket_sympy_expression(), self.translation_vector
        )
        return ket(new_expr)


class differential(operator):
    def __init__(self, order, variables=None):
        self.order = order
        self.variables = variables

    def __mul__(self, other):
        new_expression = 0
        if self.variables is None:
            symbols = get_ordered_symbols(other.get_ket_sympy_expression())
            for i, sym in enumerate(symbols):
                new_expression += sp.diff(
                    other.get_ket_sympy_expression(), sym, self.order[i]
                )
        else:
            for i, var in enumerate(self.variables):
                new_expression += sp.diff(
                    other.get_ket_sympy_expression(), var, self.order[i]
                )
        return ket(new_expression)


class kinetic_operator(operator):
    def __init__(self, p=None):
        self.p = p
        self.variables = get_default_variables(p) if p is not None else None

    def __mul__(self, other):
        variables = self.variables
        if variables is None:
            variables = other.basis[0].ket_sympy_expression.free_symbols

        new_basis = []
        for basis_fn in other.basis:
            laplacian = sum(
                sp.diff(basis_fn.ket_sympy_expression, var, 2)
                for var in variables
            )
            bf = basisfunction(laplacian)
            bf.position = basis_fn.position
            new_basis.append(bf)
        return ket([-0.5 * c for c in other.coefficients], basis=new_basis)

    def _repr_html_(self):
        return "$ -\\frac{1}{2} \\nabla^2 $"


class onebody_coulomb_operator(operator):
    def __init__(self, position=None, Z=1.0, p=None, variables=None):
        if position is None:
            position = np.array([0.0, 0.0, 0.0])
        self.position = np.array(position)
        self.Z = Z
        self.p = p
        self.variables = variables
        if p is not None:
            self.variables = get_default_variables(self.p, len(position))
            r2 = sum((v - self.position[j]) ** 2
                      for j, v in enumerate(self.variables))
            self.r_inv = r2 ** -0.5

    def __mul__(self, other):
        variables = self.variables
        if variables is None:
            symbols = get_ordered_symbols(other.basis[0].ket_sympy_expression)
            variables = symbols
            r2 = sum((variables[j] - self.position[j]) ** 2
                      for j in range(len(variables)))
            self.r_inv = r2 ** -0.5

        new_basis = [
            basisfunction(self.r_inv * bf.ket_sympy_expression)
            for bf in other.basis
        ]
        return ket([-self.Z * c for c in other.coefficients], basis=new_basis)

    def _repr_html_(self):
        return "$ -\\frac{Z}{|\\mathbf{r}|} $"


def _twobody_denominator(p1, p2, ndim):
    v1 = get_default_variables(p1, ndim)
    v2 = get_default_variables(p2, ndim)
    return sum((v1[i] - v2[i]) ** 2 for i in range(ndim)) ** 0.5


class twobody_coulomb_operator(operator):
    def __init__(self, p1=0, p2=1, ndim=3):
        self.p1, self.p2, self.ndim = p1, p2, ndim

    def __mul__(self, other):
        expr = sum(
            other.coefficients[i] * other.basis[i].ket_sympy_expression
            / _twobody_denominator(self.p1, self.p2, self.ndim)
            for i in range(len(other.basis))
        )
        return ket(expr)

    def _repr_html_(self):
        return "$ \\frac{1}{|\\mathbf{r}_1 - \\mathbf{r}_2|} $"


# ---------------------------------------------------------------------------
# Operator factory functions
# ---------------------------------------------------------------------------

def get_kinetic_operator(p=None):
    return operator_expression(kinetic_operator(p=p))

def get_translation_operator(pos):
    return operator_expression(translation(pos))

def get_sympy_operator(expr):
    return operator_expression(expr)

def get_differential_operator(order):
    return operator_expression(differential(order))

def get_onebody_coulomb_operator(position=None, Z=1.0, p=None, variables=None):
    if position is None:
        position = np.array([0, 0, 0.0])
    return operator_expression(onebody_coulomb_operator(position, Z=Z, p=p, variables=variables))

def get_twobody_coulomb_operator(p1=0, p2=1):
    return operator_expression(twobody_coulomb_operator(p1, p2))


# ---------------------------------------------------------------------------
# Variable relabelling for direct products
# ---------------------------------------------------------------------------

def relabel_direct(expr_a, expr_b):
    """Relabel variables for a direct (tensor) product of two expressions."""
    symbols_a = get_ordered_symbols(expr_a)
    symbols_b = get_ordered_symbols(expr_b)

    nd_a = len(symbols_a)
    nd_b = len(symbols_b)

    new_vars_a = get_default_variables(0, nd_a)
    new_vars_b = get_default_variables(1, nd_b)

    result = expr_a
    for i, sym in enumerate(symbols_a):
        result = result.subs(sym, new_vars_a[i])

    expr_b_relabeled = expr_b
    for i, sym in enumerate(symbols_b):
        expr_b_relabeled = expr_b_relabeled.subs(sym, new_vars_b[i])

    return result * expr_b_relabeled, (set(new_vars_a), set(new_vars_b))


# ---------------------------------------------------------------------------
# Ket
# ---------------------------------------------------------------------------

class ket:
    """
    A vector in a Hilbert space, supporting Dirac bra-ket notation.

    Can be constructed from:
    - A SymPy expression (becomes a single-term expansion)
    - A list of coefficients + basis functions
    - A numpy array (Cartesian vector)

    Supports: +, -, *, /, @, .bra, __call__, view()
    """

    def __init__(self, generic_input, name="", basis=None, position=None,
                 energy=None, autoflatten=True):
        self.autoflatten = autoflatten
        self.position = position
        self.variable_identities = None

        if isinstance(generic_input, (np.ndarray, list)):
            self.coefficients = list(generic_input)
            self.basis = list(np.eye(len(self.coefficients))) if basis is None else basis
        else:
            if position is None:
                position = np.zeros(len(generic_input.free_symbols), dtype=float)
            self.coefficients = [1.0]
            self.basis = [basisfunction(generic_input, position=position)]

        self.ket_sympy_expression = self.get_ket_sympy_expression()
        self.bra_sympy_expression = self.get_bra_sympy_expression()
        self.energy = energy if energy is not None else [0] * len(self.basis)
        self.__name__ = name
        self.bra_state = False
        self.a = None

    # --- Sympy expression assembly ---

    def get_ket_sympy_expression(self):
        result = 0
        for i in range(len(self.coefficients)):
            if isinstance(self.basis[i], (basisfunction, ket)):
                result += self.coefficients[i] * self.basis[i].ket_sympy_expression
            else:
                result += self.coefficients[i] * self.basis[i]
        return result

    def get_bra_sympy_expression(self):
        result = 0
        for i in range(len(self.coefficients)):
            if isinstance(self.basis[i], (basisfunction, ket)):
                result += np.conjugate(self.coefficients[i]) * self.basis[i].bra_sympy_expression
            else:
                result += np.conjugate(self.coefficients[i] * self.basis[i])
        return result

    # --- Arithmetic ---

    def __add__(self, other):
        new_basis = self.basis + other.basis
        new_coefficients = self.coefficients + other.coefficients
        new_energies = self.energy + other.energy
        ret = ket(new_coefficients, basis=new_basis, energy=new_energies)
        if self.autoflatten:
            ret.flatten()
        ret.__name__ = f"{self.__name__} + {other.__name__}"
        return ret

    def __sub__(self, other):
        new_basis = self.basis + other.basis
        new_coefficients = self.coefficients + [-c for c in other.coefficients]
        new_energies = self.energy + other.energy
        ret = ket(new_coefficients, basis=new_basis, energy=new_energies)
        if self.autoflatten:
            ret.flatten()
        ret.__name__ = f"{self.__name__} - {other.__name__}"
        return ret

    def __mul__(self, other):
        if isinstance(other, ket):
            new_basis, new_coefficients, new_energies = [], [], []
            for i in range(len(self.basis)):
                for j in range(len(other.basis)):
                    new_basis.append(self.basis[i] * other.basis[j])
                    new_coefficients.append(self.coefficients[i] * other.coefficients[j])
                    new_energies.append(self.energy[i] * other.energy[j])
            return ket(new_coefficients, basis=new_basis, energy=new_energies)

        if _is_sympy(other):
            new_basis = [
                basisfunction(other * bf.ket_sympy_expression)
                for bf in self.basis
            ]
            return ket(list(self.coefficients), basis=new_basis, energy=list(self.energy))

        return ket(
            [other * c for c in self.coefficients],
            basis=self.basis, energy=self.energy,
        )

    def __rmul__(self, other):
        if _is_sympy(other):
            new_basis = [
                basisfunction(bf.ket_sympy_expression * other)
                for bf in self.basis
            ]
            return ket(list(self.coefficients), basis=new_basis, energy=list(self.energy))
        return ket([other * c for c in self.coefficients], basis=self.basis)

    def __truediv__(self, other):
        assert isinstance(other, (float, int)), "Divisor must be float or int"
        return ket([c / other for c in self.coefficients], basis=self.basis)

    def __matmul__(self, other):
        if isinstance(other, (float, int)):
            return self * other

        if not isinstance(other, ket):
            return NotImplemented

        if self.bra_state:
            return self._inner_product(other)
        else:
            if other.bra_state:
                return _outerprod(self, other)
            return self._direct_product(other)

    def _inner_product(self, other):
        from .integrals import inner_product
        metric = np.zeros((len(self.basis), len(other.basis)), dtype=complex)
        for i in range(len(self.basis)):
            for j in range(len(other.basis)):
                bi, bj = self.basis[i], other.basis[j]
                if isinstance(bi, np.ndarray) and isinstance(bj, np.ndarray):
                    metric[i, j] = np.dot(bi, bj)
                elif isinstance(bi, basisfunction) and isinstance(bj, basisfunction):
                    metric[i, j] = inner_product(bi, bj)
                elif isinstance(bi, basisfunction):
                    metric[i, j] = ket([1.0], basis=[bi]).bra @ bj
                elif isinstance(bj, basisfunction):
                    metric[i, j] = bi.bra @ ket([1.0], basis=[bj])
                else:
                    metric[i, j] = bi.bra @ bj

        if np.linalg.norm(metric.imag) <= 1e-10:
            metric = metric.real
        return np.array(self.coefficients).T @ metric @ np.array(other.coefficients)

    def _direct_product(self, other):
        new_coefficients, new_basis, variable_identities = [], [], []
        for i in range(len(self.basis)):
            for j in range(len(other.basis)):
                bij, sep = relabel_direct(
                    self.basis[i].ket_sympy_expression,
                    other.basis[j].ket_sympy_expression,
                )
                bf = basisfunction(bij)
                bf.position = np.append(self.basis[i].position, other.basis[j].position)
                new_basis.append(bf)
                new_coefficients.append(self.coefficients[i] * other.coefficients[j])
                variable_identities.append(sep)
        ret = ket(new_coefficients, basis=new_basis)
        if self.autoflatten:
            ret.flatten()
        ret.__name__ = self.__name__ + other.__name__
        ret.variable_identities = variable_identities
        return ret

    # --- Flatten (deduplicate basis) ---

    def flatten(self):
        """Remove redundant basis elements using expression hashing."""
        hash_map = {}
        new_coefficients, new_basis, new_energies = [], [], []
        for i in range(len(self.basis)):
            bi = self.basis[i]
            if isinstance(bi, np.ndarray):
                key = hash(bi.tobytes())
            else:
                key = (_expr_hash(bi.ket_sympy_expression), tuple(bi.position))

            if key in hash_map:
                idx = hash_map[key]
                new_coefficients[idx] += self.coefficients[i]
            else:
                hash_map[key] = len(new_basis)
                new_basis.append(bi)
                new_coefficients.append(self.coefficients[i])
                new_energies.append(self.energy[i])
        self.basis = new_basis
        self.coefficients = new_coefficients
        self.energy = new_energies

    # --- Evaluation ---

    def __call__(self, *R, t=None):
        if t is None:
            result = 0
            accessor = (lambda bf, r: np.conjugate(bf(*r))) if self.bra_state else (lambda bf, r: bf(*r))
            for i in range(len(self.basis)):
                result += self.coefficients[i] * accessor(self.basis[i], R)
            return result
        else:
            result = 0
            for i in range(len(self.basis)):
                phase = np.exp(-1j * self.energy[i] * t)
                if self.bra_state:
                    result += np.conjugate(
                        self.coefficients[i] * self.basis[i](*R) * phase
                    )
                else:
                    result += self.coefficients[i] * self.basis[i](*R) * phase
            return result

    # --- Bra property ---

    @property
    def bra(self):
        b = ket(
            [np.conjugate(c) for c in self.coefficients],
            basis=copy.copy(self.basis),
            energy=copy.copy(self.energy),
        )
        b.bra_state = True
        b.__name__ = self.__name__
        return b

    # --- Code generation ---

    def get_ccode(self):
        """Generate GLSL-compatible C code snippets for shader rendering."""
        from .codegen import generate_glsl_snippets
        return generate_glsl_snippets(self)

    def get_wgsl(self):
        """Generate WGSL code snippets for WebGPU rendering."""
        from .codegen import generate_wgsl_snippets
        return generate_wgsl_snippets(self)

    # --- Measurement ---

    def measure(self, observable=None, repetitions=1):
        """
        Simulate quantum measurement.

        Without an observable: position measurement via Metropolis-Hastings.
        With an observable: measurement in the eigenbasis using Born rule.

        Returns array of measurement outcomes.
        """
        if observable is None:
            P_expr = self.get_bra_sympy_expression() * self.get_ket_sympy_expression()
            symbols = get_ordered_symbols(P_expr)
            P = sp.lambdify(symbols, P_expr, "numpy")
            nd = len(symbols)
            sig = 0.1

            r = np.random.multivariate_normal(
                np.zeros(nd), sig * np.eye(nd), repetitions
            ).T
            for _ in range(1000):
                dr = np.random.multivariate_normal(
                    np.zeros(nd), 0.01 * sig * np.eye(nd), repetitions
                ).T
                ratio = np.real(P(*(r + dr))) / np.real(P(*r))
                accept = ratio > np.random.uniform(0, 1, repetitions)
                r[:, accept] += dr[:, accept]
            return r
        else:
            probs = np.zeros(len(observable.eigenstates), dtype=float)
            for i in range(len(observable.eigenstates)):
                probs[i] = np.real(observable.eigenstates[i].bra @ self) ** 2
            probs = probs / probs.sum()
            indices = np.random.choice(len(probs), size=repetitions, p=probs)
            return observable.eigenvalues[indices]

    # --- Visualization ---

    def view(self, mode="splat", web=False, **kwargs):
        """
        Create a browser-based 3D visualization via anywidget.

        web=True uses the original Evince-style colour scheme
        (white background, subtractive blending).
        """
        from .viewer import OrbitalView
        if web or kwargs.get("bg_color") is not None:
            kwargs.setdefault("bg_color", [1.0, 1.0, 1.0])
            kwargs.setdefault("additive", False)
        return OrbitalView(self, mode=mode, **kwargs)

    # --- Display ---

    def _repr_html_(self):
        if len(self.__name__) > 0:
            if self.bra_state:
                return "$ \\langle %s \\vert $" % self.__name__
            return "$ \\vert %s \\rangle $" % self.__name__

        expr = self.get_ket_sympy_expression()
        if self.bra_state:
            return "$ \\langle %s \\vert $" % sp.latex(expr)
        return "$ \\vert %s \\rangle $" % sp.latex(expr)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_sympy(obj):
    return str(type(obj)).split("'")[1].split(".")[0] == "sympy"


def _outerprod(ket_a, bra_b):
    """Outer product |a><b| (returns a callable projector)."""
    class OuterProduct:
        def __init__(self, ket, bra):
            self.ket = ket
            self.bra_ket = bra  # this is a ket with bra_state=True

        def __mul__(self, other):
            coeff = self.bra_ket @ other
            return coeff * self.ket

        def _repr_html_(self):
            return (
                f"$ |{self.ket.__name__}\\rangle"
                f"\\langle {self.bra_ket.__name__}| $"
            )
    return OuterProduct(ket_a, bra_b)
