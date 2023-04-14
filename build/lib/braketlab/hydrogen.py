import numpy as np

import sympy as sp

def factorial(n):
    return np.prod(np.arange(n)+1)

def legendre_polynomial(x, n):
    """
    Returns the Legendre polynomial, as 
    """
    return 1.0/(2.0**n*factorial(n))*sp.diff((x**2.0 - 1.0)**n, x, n)

def associated_legendre_function(x, m,l):
    m = np.abs(m)
    return sp.sqrt((1-x**2)**m)*sp.diff(legendre_polynomial(x,l), x, m)

def spherical_harmonics(l,m):
    theta, phi = sp.symbols("theta phi")
    mtag = np.abs(m)
    alf = associated_legendre_function(theta, mtag, l).subs(theta, sp.cos(theta))
    eps = 1
    if m>=0:
        eps = (-1)**m
    return eps*sp.sqrt((2*l+1)*factorial(l-mtag)/(4*sp.pi*factorial(l+mtag)))*alf*sp.exp(sp.I*m*phi)

def solid_harmonic(l,m):
    ylm = spherical_harmonics(l,m)
    theta_,phi_,x,y,z = sp.symbols("theta phi x y z")
    phi = sp.atan2(y,x)
    theta = sp.acos(z / sp.sqrt(x**2.0 + y**2.0 + z**2.0))
    return ylm.subs(theta_, theta).subs(phi_, phi)

def associated_laguerre_polynomial(x, i,alpha):
    """
    Associated Laguerre polynomial constructed by recursion
    (see https://en.wikipedia.org/wiki/Laguerre_polynomials#Generalized_Laguerre_polynomials)
    """

    rr = sp.symbols("r_1")
    return sp.exp(x)*x**-alpha/factorial(i)*(sp.diff(sp.exp(-rr)*rr**(i+alpha), rr, i).subs(rr,x))

    
def hydrogen_radial(n,l,a):
    r = sp.Symbol("r")
    norm = sp.sqrt((2.0/(n*a))**3*factorial(n-l-1)/(2.0*n*factorial(n+l)**3.0))
    return  norm * (2/(n*a)*r)**l * associated_laguerre_polynomial(2/(n*a)*r, n-l-1, 2.0*l+1)*sp.exp(-r/(n*a))

def hydrogen_function(n,l,m):
    x,y,z,r_ = sp.symbols("x y z r")
    r = sp.sqrt(x**2.0 + y**2.0 + z**2.0)
    return hydrogen_radial(n,l,1).subs(r_, r)*solid_harmonic(l,m)