import numpy as np

import braketlab as bk


tol_exact = 1e-12
tol_mc = 1e-2


def test_1d_harmonic_oscillator_normality():
    psi_1 = bk.basisbank.get_harmonic_oscillator_function(1)
    inner = psi_1.bra @ psi_1
    expected = 1.0
    error = abs(inner - expected)
    assert error <= tol_exact, f"Harmonic oscillator self-overlap is {inner}," \
        f" should be {expected}."


def test_1d_harmonic_oscillator_orthogonality():
    psi_1 = bk.basisbank.get_harmonic_oscillator_function(1)
    psi_2 = bk.basisbank.get_harmonic_oscillator_function(2)
    inner = psi_1.bra @ psi_2
    expected = 0.0
    error = abs(inner - expected)
    assert error <= tol_exact, f"Harmonic oscillator self-overlap is {inner}," \
        f" should be {expected}."


def test_3d_gto_normality():
    # TODO: set random seed for monte carlo integration such that the tests are
    # consistent, not sometimes passing and sometimes not
    psi_1 = bk.basisbank.get_gto(1.0, 1, 0)
    inner = psi_1.bra @ psi_1
    expected = 1.0
    error = abs(inner - expected)
    assert error <= tol_mc, f"GTO self-overlap is {inner}, should be {expected}."


def test_3d_gto_orthogonality():
    # TODO: set random seed for monte carlo integration such that the tests are
    # consistent, not sometimes passing and sometimes not
    psi_1 = bk.basisbank.get_gto(1.0, 1, 0)
    psi_2 = bk.basisbank.get_gto(1.0, 2, 0)
    inner = psi_1.bra @ psi_2
    expected = 0.0
    error = abs(inner - expected)
    assert error <= tol_mc, f"GTO same center overlap is {inner}, should be {expected}."


def test_3d_gto_off_center_orthogonality():
    # TODO: set random seed for monte carlo integration such that the tests are
    # consistent, not sometimes passing and sometimes not
    psi_1 = bk.basisbank.get_gto(1.0, 1, 0)
    psi_2 = bk.basisbank.get_gto(1.0, 2, 0, position=np.array([1.0, 0.6, 0.3]))
    inner = psi_1.bra @ psi_2
    expected = -0.217
    error = abs(inner - expected)
    assert error <= tol_mc, f"GTO off center overlap is {inner}, should be {expected}."
