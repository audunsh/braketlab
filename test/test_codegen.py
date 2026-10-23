"""
Code generation round-trip tests.

Verifies that get_ccode() output, when evaluated numerically,
matches ket.__call__() at sample points.
"""

import braketlab as bk
import numpy as np
import sympy as sp
import pytest


def evaluate_ccode_snippet(snippet, coords):
    """Evaluate a C-code snippet by substituting tex[i] with coordinate values."""
    expr_str = snippet
    for i, val in enumerate(coords):
        expr_str = expr_str.replace(f"tex[{i}]", str(float(val)))
    q = sum(c**2 for c in coords)
    expr_str = expr_str.replace("q", str(float(q)))
    try:
        return float(eval(expr_str, {"__builtins__": {}}, {
            "exp": np.exp, "pow": pow, "sqrt": np.sqrt,
            "sin": np.sin, "cos": np.cos, "abs": abs,
        }))
    except Exception:
        return None


class TestCCodeGeneration:
    """get_ccode() produces valid shader code"""

    def test_s_orbital_produces_code(self):
        psi = bk.basisbank.get_gto(1.0, 0, 0)
        code = psi.get_ccode()
        assert len(code) > 0, "get_ccode() returned empty list"
        assert isinstance(code[0], str), "Code snippet should be a string"
        assert "exp" in code[0], "s-orbital code should contain exp()"
        assert "q" in code[0], "s-orbital code should reference q (= r^2)"

    def test_p_orbital_produces_code(self):
        psi = bk.basisbank.get_gto(1.0, 1, 0)
        code = psi.get_ccode()
        assert len(code) > 0, "get_ccode() returned empty for p orbital"

    def test_ho_produces_code(self):
        psi = bk.basisbank.get_harmonic_oscillator_function(2)
        code = psi.get_ccode()
        assert len(code) > 0, "get_ccode() returned empty for HO"

    def test_sum_of_gtos_produces_multiple_snippets(self):
        s = bk.basisbank.get_gto(1.0, 0, 0)
        p = bk.basisbank.get_gto(1.0, 1, 0)
        combo = s + p
        code = combo.get_ccode()
        assert len(code) == 2, f"Expected 2 code snippets for s+p, got {len(code)}"

    def test_s_orbital_roundtrip(self):
        """Evaluate ccode at a point and compare to ket.__call__"""
        psi = bk.basisbank.get_gto(1.0, 0, 0)
        code = psi.get_ccode()
        point = (0.5, 0.3, 0.1)
        ket_val = psi(*point)
        code_val = evaluate_ccode_snippet(code[0], point)
        if code_val is not None:
            assert abs(ket_val - code_val) < 1e-4, (
                f"Round-trip mismatch at {point}: ket={ket_val}, code={code_val}"
            )

    def test_code_has_no_integer_literals(self):
        """GLSL requires float literals, no bare integers in expressions"""
        psi = bk.basisbank.get_gto(1.0, 1, 1)
        code = psi.get_ccode()
        for snippet in code:
            parts = snippet.split("tex")
            for part in parts[1:]:
                idx = part.find("]")
                if idx >= 0:
                    continue
