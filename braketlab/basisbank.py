import sympy as sp
import numpy as np

import braketlab.solid_harmonics as sh
import braketlab as bk
import braketlab.hydrogen as hy
import braketlab.harmonic_oscillator as ho

def get_hydrogen_function(n,l,m, position = np.array([0,0,0])):
    """
    Returns a ket containing the hydrogen eigenfunction with quantum numbers n,l,m
    located at position
    """
    psi = hy.hydrogen_function(n,l,m)
    #vars = list(psi.free_symbols)
    vars = bk.get_ordered_symbols(psi)
    symbols = bk.get_default_variables(0, len(vars))
    for i in range(len(vars)):
        psi = psi.subs(vars[i], symbols[i])

    



    return bk.ket(psi, name = "%i,%i,%i" % (n,l,m), position = position)

def get_harmonic_oscillator_function(n, omega = 1, position = 0):
    """
    Returns a ket containing the harmonic oscillator energy eigenfunction with quantum number n
    located at position
    """
    return bk.ket(ho.psi_ho(n), name = "%i" % n, energy = [omega*(.5+n)], position = np.array([position]))

def get_gto(a,l,m, position = np.array([0,0,0])):
    """
    Returns a ket containing the gaussian type primitive orbital with exponent a, 
    and solid harmonic gaussian angular part defined by l and m
    located at position
    """
    psi = sh.get_Nao(a,l,m)
    vars = list(psi.free_symbols)
    symbols = bk.get_default_variables(0, len(vars))
    for i in range(len(vars)):
        psi = psi.subs(vars[i], symbols[i])
    return bk.ket(psi, name = "\chi_{%i,%i}^{%.2f}" % (l,m,a), position = position)

def get_sto(a,w,l,m, position = np.array([0,0,0])):
    """
    Returns a ket containing the slater type orbital with exponent a, 
    weight w,
    and solid harmonic gaussian angular part defined by l and m
    located at position
    """
    psi = sh.get_sto(a,w,l,m)
    vars = list(psi.free_symbols)
    symbols = bk.get_default_variables(0, len(vars))
    for i in range(len(vars)):
        psi = psi.subs(vars[i], symbols[i])
    return bk.ket(psi, name  = "\chi_{%i,%i}^{%.2f}" % (l,m,a), position = position)

