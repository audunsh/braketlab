"""
Operator correctness tests.

Tests kinetic, Coulomb, translation, and differential operators
against known analytical results.
"""

import braketlab as bk
import numpy as np
import pytest

ATOL_1D = 1e-4
ATOL_3D = 5e-2


class TestKineticOperator:
    """Kinetic energy operator: -1/2 nabla^2"""

    def test_ho_ground_state_energy(self):
        """<psi_0|T|psi_0> = 0.25 for omega=1 harmonic oscillator (T = E/2 = 0.25)"""
        psi = bk.basisbank.get_harmonic_oscillator_function(0)
        T = bk.get_kinetic_operator()
        T_psi = T * psi
        kinetic_energy = psi.bra @ T_psi
        assert abs(kinetic_energy - 0.25) < ATOL_1D, (
            f"HO ground state <T> = {kinetic_energy}, expected 0.25"
        )

    def test_ho_first_excited_energy(self):
        """<psi_1|T|psi_1> = 0.75 for omega=1 (T = E_1/2 = 3/4)"""
        psi = bk.basisbank.get_harmonic_oscillator_function(1)
        T = bk.get_kinetic_operator()
        T_psi = T * psi
        kinetic_energy = psi.bra @ T_psi
        assert abs(kinetic_energy - 0.75) < ATOL_1D, (
            f"HO n=1 <T> = {kinetic_energy}, expected 0.75"
        )

    def test_kinetic_hermiticity(self):
        """<a|T|b> == <b|T|a>* (Hermitian)"""
        a = bk.basisbank.get_harmonic_oscillator_function(0)
        b = bk.basisbank.get_harmonic_oscillator_function(1)
        T = bk.get_kinetic_operator()
        lhs = a.bra @ (T * b)
        rhs = np.conj(b.bra @ (T * a))
        assert abs(lhs - rhs) < ATOL_1D, (
            f"Kinetic not Hermitian: {lhs} != conj({rhs})"
        )


class TestTranslationOperator:
    """Translation operator shifts functions"""

    def test_1d_translation(self):
        psi = bk.basisbank.get_harmonic_oscillator_function(0)
        shift = np.array([2.0])
        T_op = bk.get_translation_operator(shift)
        shifted = T_op * psi
        val_original = psi(0.0)
        val_shifted = shifted(2.0)
        assert abs(val_original - val_shifted) < 1e-6, (
            f"Translation failed: psi(0)={val_original} != shifted(2)={val_shifted}"
        )


class TestDifferentialOperator:
    """Differential operator d/dx"""

    def test_first_derivative(self):
        import sympy as sp
        x = sp.Symbol("x_{0; 0}")
        psi = bk.ket(sp.exp(-x**2))
        D = bk.get_differential_operator([1])
        dpsi = D * psi
        val = dpsi(0.0)
        assert abs(val) < 1e-10, (
            f"d/dx exp(-x^2) at x=0 should be 0, got {val}"
        )
        val_off = dpsi(1.0)
        expected = -2.0 * 1.0 * np.exp(-1.0)
        assert abs(val_off - expected) < 1e-4, (
            f"d/dx exp(-x^2) at x=1: {val_off} != {expected}"
        )
