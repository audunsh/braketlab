# -*- coding: utf-8 -*-
import numpy as np

import sympy as sp

    

def get_Slm(l, m):
    """
    return the sympy real solid harmonic gaussian S_{lm}(r) 
    as presented in table 6.3 of Helgaker, Jørgensen and Olsen
    (page 211)
    """
    x,y,z = sp.symbols("x y z")
    r = sp.sqrt(x**2.0 + y**2.0 + z**2.0)
    
    assert(l<5), "Only l<=4 permitted"
    assert(l>=0), "Invalid l value"
    assert(np.abs(m)<=l), "Invalid m value"
    
    if l==0:
        if m== 0:
            return x**0
            
    if l==1:
        if m==1:
            return x
        if m==0:
            return z
        if m==-1:
            return y
    if l==2:
        if m==2:
            return (x**2.0 - y**2.0)*sp.sqrt(3.0)/2.0
        if m==1:
            return x*z*sp.sqrt(3.0)
        if m==0:
            return (3.0*z**2.0 - r**2.0)/2.0
        if m==-1:
            return y*z*sp.sqrt(3.0)
        if m==-2:
            return x*y*sp.sqrt(3.0)
    
    if l==3:
        if m==3:
            return x*(x**2.0 - 3.0*y**2.0)*sp.sqrt(5.0/2.0)/2.0
        if m==2:
            return z*(x**2.0 - y**2.0)*sp.sqrt(15.0)/2
        if m==1:
            return x*(5.0*z**2.0 - r**2.0)*sp.sqrt(3.0/2.0)/2.0
        if m==0:
            return z*(2.0*z**2.0 - 3.0*x**2.0 - 3.0*y**2.0)/2.0
        if m==-1:
            return y*(5.0*z**2.0 - r**2.0)*sp.sqrt(3.0/2.0)/2.0
        if m==-2:
            return x*y*z*sp.sqrt(15.0) 
        if m==-3:
            return .5*sp.sqrt(5.0/2.0)*(3*x**2 - y**2)*y
    
    
    if l==4:
        if m==4:
            return (x**4.0 - 6.0*x**2.0*y**2.0 + y**4.0)*sp.sqrt(35.0)/8.0
        if m==3:
            return (x**2.0 - 3.0*y**2.0)*x*z*sp.sqrt(35.0/2.0)/2.0
        if m==2:
            return (7.0*z**2.0 - r**2.0)*(x**2.0 - y**2.0)*sp.sqrt(5.0)/4.0
        if m==1:
            return (7.0*z**2.0 - 3.0*r**2.0)*x*z*sp.sqrt(5.0/2.0)/2.0
        if m==0:
            return (35.0*z**4.0 - 30.0*z**2.0*r**2.0 + 3.0*r**4.0)/8.0
        if m==-1:
            return (7.0*z**2.0 - 3.0*r**2.0)*y*z*sp.sqrt(5.0/2.0)/2.0
        if m==-2:
            return (7.0*z**2.0 - r**2.0)*x*y*sp.sqrt(5.0)/2.0
        if m==-3:
            return (3.0*x**2.0- y**2.0)*y*z*sp.sqrt(35.0/2.0)/2.0
        if m==-4:
            return (x**2.0 - y**2.0)*x*y*sp.sqrt(35.0)/2.0
        
        
    
    
def get_ao(a, l, m):
    """
    return unnormalized 
    solid harmonic gaussian
    for quantum numbers l, m
    a = exponent
    """
    a = np.float(a)
    x,y,z = sp.symbols("x y z")
    slm = get_Slm(l,m)
    return slm*sp.exp(-sp.UnevaluatedExpr(a)*(x**2.0 + y**2.0 + z**2.0))
    
def get_ao_at(pos, a, l, m):
    """
    return unnormalized 
    solid harmonic gaussian
    for quantum numbers l, m
    a = exponent
    """
    a = np.float(a)
    x,y,z = sp.symbols("x y z")
    slm = get_Slm(l,m)
    
    chi = slm*sp.exp(-sp.UnevaluatedExpr(a)*(x**2.0 + y**2.0 + z**2.0))
    chi = chi.subs(x, x-pos[0])
    chi = chi.subs(y, y-pos[1])
    chi = chi.subs(z, z-pos[2])
    
    return chi

def get_Npi(a_i, l):
    """
    Returns the normalization prefactor for S_lm(a_i, r)
    a_i = exponent
    l = angular quantum number
    """
    return (2.0*np.pi)**(-.75) * (4.0*a_i)**(0.75 + l/2.0)
        
def get_Nao(a,l,m):
    """
    return normalized AO in sympy-format
    a = exponent
    l = angular quantum number
    m = magnetic quantum number
    """

    a = np.float(a)
    return get_ao(a,l,m)*get_Npi(a,l)*norm_extra(l)
    
def get_Nao_at(pos, a,l,m):
    """
    return normalized AO in sympy-format
    a = exponent
    l = angular quantum number
    m = magnetic quantum number
    """
    a = np.float(a)
    return get_ao_at(pos, a,l,m)*get_Npi(a,l)*norm_extra(l)


def f(m):
    """
    factorial m!
    """
    return np.float(np.max([np.prod(np.arange(m)+1), 1]))

def norm_extra(l):
    """
    Factor required that is _not_ accounted for
    in eq. 3.3 in LSDalton manual
    """
    return (np.array([1.0,1.0,3.0,15.0,105.0])**-.5)[l]

def get_Nao_lambda(a,l,m):
    """
    return a normalized solid harmonic gaussian
    in numpy lambda format, for convenient evaluation.
    
    Note that every function is centered in (0,0,0)
    translations should be performed retrospectively
    """
    x,y,z = sp.symbols("x y z")
    return sp.lambdify([x,y,z], get_Nao(a,l,m), "numpy")


def contracted_norm(a, w, l):
    """
    Compute normalization factor 
    of contracted basis function
    """ 
    return np.sum(w[:,None]*w[None,:]*(np.sqrt(4.0*a[:,None]*a[None,:])/(a[:,None]+a[None,:]))**(1.5 + l))


def get_contracted(a,w,l,m, representation = "numeric"):
    """
    Generates Solid Harmonic Gaussian lambda functions
    a = exponent
    """
    a = np.float(a)
    S = contracted_norm(a,w,l)
    CGO = 0
    for i in np.arange(a.shape[0]):
        CGO += w[i]*get_Nao(a[i],l,m)/np.sqrt(S)
    
    if representation == "numeric":
        x,y,z = sp.symbols("x y z")
        
        return sp.lambdify([x,y,z], CGO, "numpy")
    if representation == "sympy":
        return CGO


def get_sto(a,w,n,l,m, representation = "sympy"):
    a = np.float(a)
    S = np.sqrt(a**3.0/np.pi)
    S = (2.0*a)**n *np.sqrt(2.0*a/np.math.factorial(2*int(n)))

    x,y,z = sp.symbols("x y z")
    r = sp.sqrt(x**2.0 + y**2.0 + z**2.0)

    sto = S*get_Slm(l,m)*r**(n-1)*sp.exp(-sp.UnevaluatedExpr(a)*r)
    if representation == "numeric":
        return sp.lambdify([x,y,z], sto, "numpy")
        
    if representation == "sympy":
        return sto



def get_contracted_sympy(a,w,l,m):
    """
    Generates Solid Harmonic Gaussian lambda functions
    a = exponents
    
    """
    a = np.float(a)
    S = contracted_norm(a,w,l)
    CGO = 0
    for i in np.arange(a.shape[0]):
        CGO += w[i]*get_Nao(a[i],l,m)/np.sqrt(S)
    
    return CGO
    
def get_contracted_at(pos, a,w,l,m):
    """
    Generates Solid Harmonic Gaussian lambda functions
    a = exponents
    
    """
    S = contracted_norm(a,w,l)
    CGO = 0
    for i in np.arange(a.shape[0]):
        CGO += w[i]*get_Nao_at(pos, a[i],l,m)/np.sqrt(S)
    #print(CGO)
    x,y,z = sp.symbols("x y z")
    
    
    return CGO #sp.lambdify([x,y,z], c*CGO, "numpy")
    