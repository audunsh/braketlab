"""
braketlab_core: Pyodide-compatible quantum mechanics algebra engine.

A modernized, browser-ready version of Braketlab with Dirac notation,
Gaussian basis functions, analytical integrals, and WebGPU shader codegen.
"""

from .algebra import (
    basisfunction,
    ket,
    operator_expression,
    kinetic_operator,
    onebody_coulomb_operator,
    twobody_coulomb_operator,
    translation,
    differential,
    sympy_operator,
    get_kinetic_operator,
    get_onebody_coulomb_operator,
    get_twobody_coulomb_operator,
    get_translation_operator,
    get_differential_operator,
    get_sympy_operator,
    get_default_variables,
)

from . import basis
from . import basis as basisbank
from . import integrals
from . import codegen

__version__ = "2.0.0a1"

__all__ = [
    "basisfunction",
    "ket",
    "operator_expression",
    "kinetic_operator",
    "onebody_coulomb_operator",
    "twobody_coulomb_operator",
    "translation",
    "differential",
    "sympy_operator",
    "get_kinetic_operator",
    "get_onebody_coulomb_operator",
    "get_twobody_coulomb_operator",
    "get_translation_operator",
    "get_differential_operator",
    "get_sympy_operator",
    "get_default_variables",
    "basis",
    "basisbank",
    "integrals",
    "codegen",
]
