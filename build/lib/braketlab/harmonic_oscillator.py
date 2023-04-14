import sympy as sp
import numpy as np


def factorial(n):
    """
    return n!
    """
    return np.prod(np.arange(n)+1)

def hermite(f, z, n):
    """
    The n-th order Hermite polynomial
    """
    return (-1)**n * sp.exp(f**2)*sp.diff( sp.exp(-f**2), z, n)
    
def psi_ho(n, omega = 1, mass= 1, hbar = 1, time = False):
    """
    The n-th normalized harmonic oscillator eigenfunction
    (stationary state if time = false)
    """
    norm_factor = 1/np.sqrt(factorial(n)*2**n) 
    
    x,t = sp.symbols("x t")
    
    prefactor   = (mass*omega/(np.pi*hbar))**.25 
    
    core = sp.exp(-mass*omega*x**2/(2*hbar))
    
    psi = hermite(sp.sqrt(mass*omega/hbar)*x, x, n)
    time_dependence = 1
    if time:
        time_dependence = sp.exp(-sp.I*omega*(n + .5)*t)
    return norm_factor*prefactor*core*psi #*time_dependence