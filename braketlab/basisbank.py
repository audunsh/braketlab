import sympy as sp
import numpy as np

import braketlab.solid_harmonics as sh
import braketlab as bk
import braketlab.hydrogen as hy
import braketlab.harmonic_oscillator as ho

def get_hydrogen_function(n,l,m):
    return bk.ket(hy.hydrogen_function(n,l,m), name = "%i,%i,%i" % (n,l,m))

def get_harmonic_oscillator_function(n, omega = 1):
    return bk.ket(ho.psi_ho(n), name = "%i" % n, energy = [omega*(.5+n)])

def get_gto(a,l,m):
    return bk.ket(sh.get_Nao(a,l,m), name = "\chi_{%i,%i}^%.2f" % (l,m,a))

def get_sto(a,w,l,m):
    return bk.ket(sh.get_sto(a,w,l,m), name  = "\chi_{%i,%i}^%.2f" % (l,m,a))

