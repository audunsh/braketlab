"""
Real solid harmonic Gaussian basis function,
as presented in chapter 6 of Helgaker, T., Jorgensen, P., & Olsen, J. (2014). Molecular electronic-structure theory. John Wiley & Sons.
"""
import sympy as sp
import numpy as np


def get_default_variables(p, n = 3):
    variables = []
    for i in range(n):
        variables.append(sp.Symbol("x_{%i; %i}" % (p, i)))
    return variables

def binom(a,b):
    #binomial coefficient ( a , b )
    return np.math.factorial(int(a))/(np.math.factorial(int(b))*np.math.factorial(int(a-b)))

def V(m):
    """
    eq. 6.4.50, pink bible (*)
    """
    vm = 0
    if m<0:
        vm = .5
    return vm

def c(l,m,t,u,v):
    """
    eq. 6.4.48, pink bible (*)
    """
    return (-1)**(t + v - V(m)) * (.25)**t * binom(l,t) * binom(l-t, abs(m)+t) * binom(t,u) * binom( abs(m), 2*v )

def N(l,m):
    """
    eq. 6.4.49, pink bible (*)
    """
    return 1/(2**abs(m) * np.math.factorial(l) ) * np.sqrt( 2* np.math.factorial(l + abs(m))*np.math.factorial(l - abs(m)) * (2**(m==0))**-1)

def get_Slm(l,m):
    """
    eq. 6.4.47, pink bible (*)
    """
    x,y,z = sp.symbols("x y z") #get_default_variables(0)
    slm = 0
    for t in range(int(np.floor((l-abs(m))/2))+1):
        for u in range(t +1 ):
            vm = V(m)
            for v in np.arange(vm, np.floor(abs(m)/2 - vm) + vm + 1) :
                slm += c(l,m,t,u,v)*x**int(2*t + abs(m) - 2*(u+v)) * y**int(2*(u+v)) * z**int(l-2*t-abs(m))
    return slm

def get_gto(a,l,m):
    """
    eq. 6.6.15, pink bible (*)
    """
    #x,y,z = get_default_variables(0)
    x,y,z = sp.symbols("x y z")
    return get_Npi(a, l)* N(l,m)*get_Slm(l,m) * sp.exp(-sp.UnevaluatedExpr(a)*(x**2.0 + y**2.0 + z**2.0) )
    
    

def get_Npi(a_i, l):
    """
    Returns the normalization prefactor for S_lm(a_i, r)
    a_i = exponent
    l = angular quantum number
    """
    return (2.0*np.pi)**(-.75) * (4.0*a_i)**(0.75 + l/2.0)  * float(dobfac(2*l - 1))**-.5


def dobfac(n):
    """
    'double' factorial function
    eq. 6.5.10 in pink bible (*)
    """
    if n>=0 and n%2 == 0:
        return np.prod(np.arange(0, n, 2 ) + 2)
    else:
        if n%2==1:
            return np.prod(np.arange(0, n, 2 ) + 1)