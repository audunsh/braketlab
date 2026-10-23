"""
Basis orthonormality tests.

Verifies that standard basis sets satisfy <i|j> = delta_{ij}
within numerical tolerance.
"""

import braketlab as bk
import numpy as np
import pytest

ATOL_1D = 1e-5
ATOL_3D = 5e-2


class TestHarmonicOscillator:
    """1D harmonic oscillator eigenstates"""

    @pytest.mark.parametrize("n", [0, 1, 2, 3])
    def test_normalization(self, n):
        psi = bk.basisbank.get_harmonic_oscillator_function(n)
        norm = psi.bra @ psi
        assert abs(norm - 1.0) < ATOL_1D, (
            f"HO |{n}> norm = {norm}, expected 1.0"
        )

    @pytest.mark.parametrize("n,m", [(0, 1), (0, 2), (1, 2), (1, 3), (0, 3)])
    def test_orthogonality(self, n, m):
        psi_n = bk.basisbank.get_harmonic_oscillator_function(n)
        psi_m = bk.basisbank.get_harmonic_oscillator_function(m)
        overlap = psi_n.bra @ psi_m
        assert abs(overlap) < ATOL_1D, (
            f"HO <{n}|{m}> = {overlap}, expected 0"
        )


class TestGTOs:
    """3D Gaussian type orbitals"""

    @pytest.mark.parametrize("l,m", [(0, 0), (1, 0), (1, 1), (1, -1)])
    def test_normalization(self, l, m):
        psi = bk.basisbank.get_gto(1.0, l, m)
        norm = psi.bra @ psi
        assert abs(norm - 1.0) < ATOL_3D, (
            f"GTO ({l},{m}) norm = {norm}, expected 1.0"
        )

    def test_s_p_orthogonality(self):
        """s and p orbitals with same exponent at same center are orthogonal"""
        s = bk.basisbank.get_gto(1.0, 0, 0)
        p = bk.basisbank.get_gto(1.0, 1, 0)
        overlap = s.bra @ p
        assert abs(overlap) < ATOL_3D, (
            f"GTO <s|p_z> = {overlap}, expected 0"
        )

    def test_p_p_orthogonality(self):
        """p_x and p_z at same center are orthogonal"""
        px = bk.basisbank.get_gto(1.0, 1, 1)
        pz = bk.basisbank.get_gto(1.0, 1, 0)
        overlap = px.bra @ pz
        assert abs(overlap) < ATOL_3D, (
            f"GTO <p_x|p_z> = {overlap}, expected 0"
        )

    def test_different_exponent_overlap(self):
        """Two s-GTOs with different exponents have known overlap"""
        s1 = bk.basisbank.get_gto(1.0, 0, 0)
        s2 = bk.basisbank.get_gto(2.0, 0, 0)
        overlap = s1.bra @ s2
        assert abs(overlap) > 0.01, (
            f"Different exponent s-GTOs should have nonzero overlap, got {overlap}"
        )

    def test_off_center_overlap(self):
        """Regression: off-center GTO overlap from existing tests"""
        psi_1 = bk.basisbank.get_gto(1.0, 1, 0)
        psi_2 = bk.basisbank.get_gto(1.0, 2, 0, position=np.array([1.0, 0.6, 0.3]))
        inner = psi_1.bra @ psi_2
        assert abs(inner + 0.217) < ATOL_3D, (
            f"Off-center overlap = {inner}, expected ~ -0.217"
        )
