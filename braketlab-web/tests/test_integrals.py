"""
Tests for the analytical integral engine and expression cache.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from braketlab_core.integrals import (
    overlap_1d,
    overlap_3d_cartesian,
    kinetic_3d_cartesian,
    boys_function,
    inner_product,
    clear_cache,
    _integral_cache,
)
from braketlab_core import basis


class TestBoysFunction:
    def test_n0_T0(self):
        assert abs(boys_function(0, 0.0) - 1.0) < 1e-12

    def test_n0_large_T(self):
        val = boys_function(0, 100.0)
        assert val > 0 and val < 1

    def test_n1_T0(self):
        assert abs(boys_function(1, 0.0) - 1.0 / 3) < 1e-12


class TestOverlap1D:
    def test_s_s_same_center(self):
        """S(0,0) with PA=PB=0 should be 1.0"""
        val = overlap_1d(0, 0, 0.0, 0.0, 1.0)
        assert abs(val - 1.0) < 1e-12

    def test_p_s(self):
        """S(1,0) with PA != 0"""
        val = overlap_1d(1, 0, 0.5, -0.5, 2.0)
        assert abs(val - 0.5) < 1e-10

    def test_s_p(self):
        """S(0,1) with PB != 0"""
        val = overlap_1d(0, 1, 0.5, -0.5, 2.0)
        assert abs(val - (-0.5)) < 1e-10

    def test_p_p_same_center(self):
        """S(1,1) at same center"""
        gamma = 2.0
        val = overlap_1d(1, 1, 0.0, 0.0, gamma)
        expected = 1.0 / (2 * gamma)
        assert abs(val - expected) < 1e-10


class TestOverlap3D:
    def test_s_s_same_center(self):
        """Two identical s-GTOs at origin with exponent 1.0"""
        a, b = 1.0, 1.0
        A = B = np.zeros(3)
        val = overlap_3d_cartesian(a, (0, 0, 0), A, b, (0, 0, 0), B)
        expected = (np.pi / (a + b)) ** 1.5
        assert abs(val - expected) < 1e-10

    def test_s_s_different_exponents(self):
        """Two s-GTOs with different exponents"""
        a, b = 1.0, 2.0
        A = B = np.zeros(3)
        val = overlap_3d_cartesian(a, (0, 0, 0), A, b, (0, 0, 0), B)
        expected = (np.pi / (a + b)) ** 1.5
        assert abs(val - expected) < 1e-10

    def test_s_s_separated(self):
        """Two s-GTOs at different positions"""
        a, b = 1.0, 1.0
        A = np.array([0.0, 0.0, 0.0])
        B = np.array([1.0, 0.0, 0.0])
        val = overlap_3d_cartesian(a, (0, 0, 0), A, b, (0, 0, 0), B)
        gamma = a + b
        AB2 = 1.0
        expected = np.exp(-a * b / gamma * AB2) * (np.pi / gamma) ** 1.5
        assert abs(val - expected) < 1e-10

    def test_p_p_same_center(self):
        """Two p_x GTOs at same center"""
        a = 1.0
        A = B = np.zeros(3)
        val = overlap_3d_cartesian(a, (1, 0, 0), A, a, (1, 0, 0), B)
        assert val > 0, f"p_x overlap should be positive, got {val}"

    def test_s_p_orthogonal_same_center(self):
        """s and p_x at same center should have zero overlap"""
        a = 1.0
        A = B = np.zeros(3)
        val = overlap_3d_cartesian(a, (0, 0, 0), A, a, (1, 0, 0), B)
        assert abs(val) < 1e-10, f"s-px overlap should be 0, got {val}"


class TestKinetic3D:
    def test_s_s_same_center(self):
        """Kinetic integral for two s-GTOs at same center"""
        a = 1.0
        A = B = np.zeros(3)
        T = kinetic_3d_cartesian(a, (0, 0, 0), A, a, (0, 0, 0), B)
        assert T > 0, f"Kinetic energy should be positive, got {T}"


class TestExpressionCache:
    def setup_method(self):
        clear_cache()

    def test_cache_hit(self):
        """Second call with same basis functions should hit cache"""
        psi = basis.get_harmonic_oscillator_function(0)
        b = psi.basis[0]
        inner_product(b, b)
        assert len(_integral_cache) == 1
        inner_product(b, b)
        assert len(_integral_cache) == 1

    def test_cache_different_functions(self):
        """Different basis functions get different cache entries"""
        p0 = basis.get_harmonic_oscillator_function(0)
        p1 = basis.get_harmonic_oscillator_function(1)
        inner_product(p0.basis[0], p0.basis[0])
        inner_product(p0.basis[0], p1.basis[0])
        assert len(_integral_cache) == 2

    def test_clear_cache(self):
        psi = basis.get_harmonic_oscillator_function(0)
        inner_product(psi.basis[0], psi.basis[0])
        assert len(_integral_cache) > 0
        clear_cache()
        assert len(_integral_cache) == 0
