"""
Regression tests with locked reference values.

These values were computed with the current braketlab implementation
and serve as a baseline to detect unintended changes during refactoring.
"""

import braketlab as bk
import numpy as np
import pytest

ATOL_1D = 1e-4
ATOL_3D = 5e-2


class TestHORegression:
    """Locked harmonic oscillator values"""

    def test_ho_0_self_overlap(self):
        psi = bk.basisbank.get_harmonic_oscillator_function(0)
        assert abs(psi.bra @ psi - 1.0) < ATOL_1D

    def test_ho_1_self_overlap(self):
        psi = bk.basisbank.get_harmonic_oscillator_function(1)
        assert abs(psi.bra @ psi - 1.0) < ATOL_1D

    def test_ho_2_self_overlap(self):
        psi = bk.basisbank.get_harmonic_oscillator_function(2)
        assert abs(psi.bra @ psi - 1.0) < ATOL_1D

    def test_ho_01_overlap_zero(self):
        p0 = bk.basisbank.get_harmonic_oscillator_function(0)
        p1 = bk.basisbank.get_harmonic_oscillator_function(1)
        assert abs(p0.bra @ p1) < ATOL_1D

    def test_ho_02_overlap_zero(self):
        p0 = bk.basisbank.get_harmonic_oscillator_function(0)
        p2 = bk.basisbank.get_harmonic_oscillator_function(2)
        assert abs(p0.bra @ p2) < ATOL_1D

    def test_ho_0_at_origin(self):
        psi = bk.basisbank.get_harmonic_oscillator_function(0)
        val = psi(0.0)
        expected = (1.0 / np.pi) ** 0.25
        assert abs(val - expected) < 1e-8


class TestGTORegression:
    """Locked GTO values"""

    def test_s_gto_self_overlap(self):
        psi = bk.basisbank.get_gto(1.0, 0, 0)
        assert abs(psi.bra @ psi - 1.0) < ATOL_3D

    def test_p_gto_self_overlap(self):
        psi = bk.basisbank.get_gto(1.0, 1, 0)
        assert abs(psi.bra @ psi - 1.0) < ATOL_3D

    def test_s_p_orthogonal(self):
        s = bk.basisbank.get_gto(1.0, 0, 0)
        p = bk.basisbank.get_gto(1.0, 1, 0)
        assert abs(s.bra @ p) < ATOL_3D

    def test_off_center_regression(self):
        psi_1 = bk.basisbank.get_gto(1.0, 1, 0)
        psi_2 = bk.basisbank.get_gto(1.0, 2, 0, position=np.array([1.0, 0.6, 0.3]))
        inner = psi_1.bra @ psi_2
        assert abs(inner + 0.217) < ATOL_3D

    def test_s_gto_at_origin(self):
        psi = bk.basisbank.get_gto(1.0, 0, 0)
        val = psi(0.0, 0.0, 0.0)
        assert val > 0.5, f"s-GTO at origin = {val}, expected > 0.5"


class TestKetOperations:
    """Locked ket arithmetic results"""

    def test_scaled_ket_overlap(self):
        psi = bk.basisbank.get_harmonic_oscillator_function(0)
        scaled = 2.0 * psi
        val = scaled.bra @ scaled
        assert abs(val - 4.0) < ATOL_1D, f"<2*psi|2*psi> = {val}, expected 4.0"

    def test_sum_ket_overlap(self):
        p0 = bk.basisbank.get_harmonic_oscillator_function(0)
        p1 = bk.basisbank.get_harmonic_oscillator_function(1)
        combo = p0 + p1
        val = combo.bra @ combo
        assert abs(val - 2.0) < ATOL_1D, f"<p0+p1|p0+p1> = {val}, expected 2.0"

    def test_difference_ket_overlap(self):
        p0 = bk.basisbank.get_harmonic_oscillator_function(0)
        p1 = bk.basisbank.get_harmonic_oscillator_function(1)
        diff = p0 - p1
        val = diff.bra @ diff
        assert abs(val - 2.0) < ATOL_1D, f"<p0-p1|p0-p1> = {val}, expected 2.0"
