import braketlab as bk
import numpy as np

def test_1d_harmonic_oscillator_normality():
    psi_1 = bk.basisbank.get_harmonic_oscillator_function(1)
    inner = np.abs(psi_1.bra@psi_1 - 1.0)
    assert inner<=1e-5, f"Harmonic oscillator self-overlap is {inner}, but should be close to 1."

def test_1d_harmonic_oscillator_orthogonality():
    psi_1 = bk.basisbank.get_harmonic_oscillator_function(1)
    psi_2 = bk.basisbank.get_harmonic_oscillator_function(2)
    inner = np.abs(psi_1.bra@psi_2)
    assert inner<=1e-5, f"Harmonic oscillator overlap is {inner}, but should be below 1e-5."
    

def test_3d_gto_normality():
    psi_1 = bk.basisbank.get_gto(1.0, 1, 0)
    inner = np.abs(psi_1.bra@psi_1 - 1.0)
    assert inner<=1e-2, f"GTO self-overlap is {inner}, but should be close to 1."

def test_3d_gto_orthogonality():
    psi_1 = bk.basisbank.get_gto(1.0, 1, 0)
    psi_2 = bk.basisbank.get_gto(1.0, 2, 0)
    inner = psi_1.bra@psi_2
    assert np.abs(inner)<=1e-2, f"GTO same center overlap is {inner}, but should be below 1e-5."
    
def test_3d_gto_off_center_orthogonality():
    psi_1 = bk.basisbank.get_gto(1.0, 1, 0)
    psi_2 = bk.basisbank.get_gto(1.0, 2, 0, position = np.array([1.0, 0.6, 0.3]))
    inner = psi_1.bra@psi_2
    assert np.abs(inner + .217)<=1e-2, f"GTO off center overlap is {inner}, but should be close to -.2."
    