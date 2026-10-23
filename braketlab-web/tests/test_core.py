"""
Tests for braketlab_core: the Pyodide-compatible algebra engine.

Mirrors the test suite from the original braketlab to ensure
feature parity after extraction and cleanup.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import sympy as sp
import pytest

import braketlab_core as bk
from braketlab_core import basis, codegen


ATOL_1D = 1e-5
ATOL_3D = 5e-2


class TestHarmonicOscillator:
    @pytest.mark.parametrize("n", [0, 1, 2, 3])
    def test_normalization(self, n):
        psi = basis.get_harmonic_oscillator_function(n)
        assert abs(psi.bra @ psi - 1.0) < ATOL_1D

    @pytest.mark.parametrize("n,m", [(0, 1), (0, 2), (1, 2)])
    def test_orthogonality(self, n, m):
        psi_n = basis.get_harmonic_oscillator_function(n)
        psi_m = basis.get_harmonic_oscillator_function(m)
        assert abs(psi_n.bra @ psi_m) < ATOL_1D

    def test_evaluation_at_origin(self):
        psi = basis.get_harmonic_oscillator_function(0)
        val = psi(0.0)
        expected = (1.0 / np.pi) ** 0.25
        assert abs(val - expected) < 1e-6


class TestGTOs:
    @pytest.mark.parametrize("l,m", [(0, 0), (1, 0), (1, 1), (1, -1)])
    def test_normalization(self, l, m):
        psi = basis.get_gto(1.0, l, m)
        assert abs(psi.bra @ psi - 1.0) < ATOL_3D

    def test_s_p_orthogonality(self):
        s = basis.get_gto(1.0, 0, 0)
        p = basis.get_gto(1.0, 1, 0)
        assert abs(s.bra @ p) < ATOL_3D

    def test_s_at_origin_nonzero(self):
        psi = basis.get_gto(1.0, 0, 0)
        assert abs(psi(0.0, 0.0, 0.0)) > 0.1

    def test_p_at_origin_zero(self):
        psi = basis.get_gto(1.0, 1, 0)
        assert abs(psi(0.0, 0.0, 0.0)) < 1e-10


class TestAlgebra:
    def test_linearity(self):
        a = basis.get_harmonic_oscillator_function(0)
        b = basis.get_harmonic_oscillator_function(1)
        c = basis.get_harmonic_oscillator_function(2)
        lhs = a.bra @ c + b.bra @ c
        rhs = (a + b).bra @ c
        assert abs(lhs - rhs) < ATOL_1D

    def test_scalar_multiplication(self):
        psi = basis.get_harmonic_oscillator_function(0)
        scaled = 2.0 * psi
        assert abs(scaled.bra @ scaled - 4.0) < ATOL_1D

    def test_addition_norm(self):
        p0 = basis.get_harmonic_oscillator_function(0)
        p1 = basis.get_harmonic_oscillator_function(1)
        combo = p0 + p1
        assert abs(combo.bra @ combo - 2.0) < ATOL_1D

    def test_flatten_efficiency(self):
        psi = basis.get_harmonic_oscillator_function(0)
        combo = psi + psi
        combo.flatten()
        assert len(combo.basis) == 1
        assert abs(combo.coefficients[0] - 2.0) < 1e-10


class TestOperators:
    def test_kinetic_ho_ground(self):
        psi = basis.get_harmonic_oscillator_function(0)
        T = bk.get_kinetic_operator()
        T_psi = T * psi
        ke = psi.bra @ T_psi
        assert abs(ke - 0.25) < ATOL_1D

    def test_translation(self):
        psi = basis.get_harmonic_oscillator_function(0)
        T_op = bk.get_translation_operator(np.array([2.0]))
        shifted = T_op * psi
        assert abs(psi(0.0) - shifted(2.0)) < 1e-6

    def test_differential(self):
        x = sp.Symbol("x_{0; 0}")
        psi = bk.ket(sp.exp(-x**2))
        D = bk.get_differential_operator([1])
        dpsi = D * psi
        assert abs(dpsi(0.0)) < 1e-10


class TestCodegen:
    def test_glsl_snippets(self):
        psi = basis.get_gto(1.0, 0, 0)
        snippets = codegen.generate_glsl_snippets(psi)
        assert len(snippets) == 1
        assert "exp" in snippets[0]

    def test_wgsl_snippets(self):
        psi = basis.get_gto(1.0, 1, 0)
        snippets = codegen.generate_wgsl_snippets(psi)
        assert len(snippets) == 1
        assert "tex.z" in snippets[0] or "tex.x" in snippets[0]

    def test_fragment_shader(self):
        psi = basis.get_gto(1.0, 0, 0)
        shader, nd = codegen.generate_fragment_shader(psi)
        assert nd == 3
        assert "void main()" in shader
        assert "gl_FragColor" in shader

    def test_wgsl_shader(self):
        psi = basis.get_gto(1.0, 0, 0)
        shader, nd = codegen.generate_wgsl_fragment_shader(psi)
        assert nd == 3
        assert "fn main" in shader
        assert "vec4<f32>" in shader

    def test_time_dependent_shader(self):
        p0 = basis.get_harmonic_oscillator_function(0)
        p1 = basis.get_harmonic_oscillator_function(1)
        combo = p0 + p1
        shader, nd = codegen.generate_fragment_shader(combo, time_dependent=True)
        assert "cos" in shader
        assert "sin" in shader
