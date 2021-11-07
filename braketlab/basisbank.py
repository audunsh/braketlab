import sympy as sp
import numpy as np

import braketlab.solid_harmonics as sh
import braketlab as bk
import braketlab.hydrogen as hy

def get_hydrogen_function(n,l,m):
    return bk.ket(hy.hydrogen_function(n,l,m))

def get_gto(a,l,m):
    return bk.ket(sh.get_Nao(a,l,m))

def get_sto(a,w,l,m):
    return bk.ket(sh.get_sto(a,w,l,m))

def get_1d_ho():
    pass