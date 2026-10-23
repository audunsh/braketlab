"""
Algebraic identity tests for the braketlab algebra engine.

Tests linearity, conjugate linearity, positive definiteness,
scalar operations, and structural invariants of bra-ket algebra.
"""

import braketlab as bk
import numpy as np
import pytest

ATOL_1D = 1e-5
ATOL_3D = 5e-2


@pytest.fixture
def ho_states():
    return [bk.basisbank.get_harmonic_oscillator_function(n) for n in range(4)]


@pytest.fixture
def gto_states():
    return [
        bk.basisbank.get_gto(1.0, 0, 0),
        bk.basisbank.get_gto(1.0, 1, 0),
        bk.basisbank.get_gto(1.0, 1, 1),
        bk.basisbank.get_gto(0.5, 0, 0),
    ]


class TestInnerProductLinearity:
    """<a|c> + <b|c> == <a+b|c> (linearity in the bra slot)"""

    def test_1d_linearity(self, ho_states):
        a, b, c = ho_states[0], ho_states[1], ho_states[2]
        lhs = a.bra @ c + b.bra @ c
        rhs = (a + b).bra @ c
        assert abs(lhs - rhs) < ATOL_1D, f"1D linearity failed: {lhs} != {rhs}"

    def test_3d_linearity(self, gto_states):
        a, b, c = gto_states[0], gto_states[1], gto_states[2]
        lhs = a.bra @ c + b.bra @ c
        rhs = (a + b).bra @ c
        assert abs(lhs - rhs) < ATOL_3D, f"3D linearity failed: {lhs} != {rhs}"


class TestConjugateLinearity:
    """<s*a|b> == conj(s) * <a|b>"""

    def test_real_scalar(self, ho_states):
        a, b = ho_states[0], ho_states[1]
        s = 3.0
        lhs = (s * a).bra @ b
        rhs = np.conj(s) * (a.bra @ b)
        assert abs(lhs - rhs) < ATOL_1D, f"Real scalar failed: {lhs} != {rhs}"

    def test_scalar_ket_slot(self, ho_states):
        a, b = ho_states[0], ho_states[1]
        s = 2.5
        lhs = a.bra @ (s * b)
        rhs = s * (a.bra @ b)
        assert abs(lhs - rhs) < ATOL_1D, f"Ket-slot scalar failed: {lhs} != {rhs}"


class TestPositiveDefiniteness:
    """<a|a> >= 0 for any non-zero ket"""

    def test_1d_positive(self, ho_states):
        for psi in ho_states:
            norm_sq = psi.bra @ psi
            assert norm_sq.real > -ATOL_1D, f"Norm^2 = {norm_sq}, expected >= 0"

    def test_3d_positive(self, gto_states):
        for psi in gto_states:
            norm_sq = psi.bra @ psi
            assert norm_sq.real > -ATOL_3D, f"Norm^2 = {norm_sq}, expected >= 0"


class TestKetArithmetic:
    """Addition, subtraction, scalar multiplication"""

    def test_addition_commutative(self, ho_states):
        a, b, c = ho_states[0], ho_states[1], ho_states[2]
        lhs = (a + b).bra @ c
        rhs = (b + a).bra @ c
        assert abs(lhs - rhs) < ATOL_1D, f"a+b != b+a: {lhs} != {rhs}"

    def test_subtraction(self, ho_states):
        a, b = ho_states[0], ho_states[1]
        diff = a - b
        ref = a + (-1.0 * b)
        probe = ho_states[2]
        lhs = diff.bra @ probe
        rhs = ref.bra @ probe
        assert abs(lhs - rhs) < ATOL_1D, f"a-b != a+(-b): {lhs} != {rhs}"

    def test_scalar_division(self, ho_states):
        a = ho_states[0]
        half = a / 2.0
        lhs = half.bra @ a
        rhs = 0.5 * (a.bra @ a)
        assert abs(lhs - rhs) < ATOL_1D, f"Division failed: {lhs} != {rhs}"

    def test_zero_scalar(self, ho_states):
        a = ho_states[0]
        zero_ket = 0.0 * a
        val = zero_ket.bra @ a
        assert abs(val) < ATOL_1D, f"0*|a> should give 0, got {val}"


class TestEvaluation:
    """Ket evaluation (wavefunction call)"""

    def test_1d_evaluation_at_origin(self):
        psi = bk.basisbank.get_harmonic_oscillator_function(0)
        val = psi(0.0)
        expected = (1.0 / np.pi) ** 0.25
        assert abs(val - expected) < 1e-6, f"HO(0) at x=0: {val} != {expected}"

    def test_1d_sum_evaluation(self):
        psi0 = bk.basisbank.get_harmonic_oscillator_function(0)
        psi1 = bk.basisbank.get_harmonic_oscillator_function(1)
        combo = psi0 + psi1
        val = combo(0.0)
        expected = psi0(0.0) + psi1(0.0)
        assert abs(val - expected) < 1e-6, f"Sum evaluation: {val} != {expected}"

    def test_3d_gto_s_orbital_at_origin(self):
        psi = bk.basisbank.get_gto(1.0, 0, 0)
        val = psi(0.0, 0.0, 0.0)
        assert abs(val) > 0.1, f"s-GTO at origin should be nonzero, got {val}"

    def test_3d_gto_p_orbital_at_origin(self):
        psi = bk.basisbank.get_gto(1.0, 1, 0)
        val = psi(0.0, 0.0, 0.0)
        assert abs(val) < 1e-10, f"p_z GTO at origin should be ~0, got {val}"
