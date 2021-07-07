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
    r = sp.sqrt(x**2 + y**2 + z**2)
    
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
            return (x**2 - y**2)*sp.sqrt(3.0)/2.0
        if m==1:
            return x*z*sp.sqrt(3.0)
        if m==0:
            return (3*z**2 - r**2)/2.0
        if m==-1:
            return y*z*sp.sqrt(3.0)
        if m==-2:
            return x*y*sp.sqrt(3.0)
    
    if l==3:
        if m==3:
            return x*(x**2 - 3*y**2)*sp.sqrt(5/2.0)/2
        if m==2:
            return z*(x**2 - y**2)*sp.sqrt(15)/2
        if m==1:
            return x*(5*z**2 - r**2)*sp.sqrt(3/2.0)/2
        if m==0:
            return z*(5*z**2 - 3*r**2)/2
        if m==-1:
            return y*(5*z**2 - r**2)*sp.sqrt(3/2.0)/2
        if m==-2:
            return x*y*z*sp.sqrt(15) 
        if m==-3:
            return x*y*z*sp.sqrt(15) 
    
    
    if l==4:
        if m==4:
            return (x**4 - 6*x**2*y**2 + y**4)*sp.sqrt(35)/8
        if m==3:
            return (x**2 - 3*y**2)*x*z*sp.sqrt(35/2.0)/2
        if m==2:
            return (7*z**2 - r**2)*(x**2 - y**2)*sp.sqrt(5)/4
        if m==1:
            return (7*z**2 - 3*r**2)*x*z*sp.sqrt(5/2.0)/2
        if m==0:
            return (35*z**4 - 30*z**2*r**2 + 3*r**4)/8
        if m==-1:
            return (7*z**2 - 3*r**2)*y*z*sp.sqrt(5/2.0)/2
        if m==-2:
            return (7*z**2 - r**2)*x*y*sp.sqrt(5)/2
        if m==-3:
            return (3*x**2 - y**2)*y*z*sp.sqrt(35/2.0)/2
        if m==-4:
            return (x**2 - y**2)*x*y*sp.sqrt(35)/2
        
        
    
    
def get_ao(a, l, m):
    """
    return unnormalized 
    solid harmonic gaussian
    for quantum numbers l, m
    a = exponent
    """
    x,y,z = sp.symbols("x y z")
    slm = get_Slm(l,m)
    return slm*sp.exp(-a*(x**2 + y**2 + z**2))
    
def get_ao_at(pos, a, l, m):
    """
    return unnormalized 
    solid harmonic gaussian
    for quantum numbers l, m
    a = exponent
    """
    x,y,z = sp.symbols("x y z")
    slm = get_Slm(l,m)
    
    chi = slm*sp.exp(-a*(x**2 + y**2 + z**2))
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
    return (2*sp.pi)**(-.75) * (4*a_i)**(0.75 + l/2.0)
        
def get_Nao(a,l,m):
    """
    return normalized AO in sympy-format
    a = exponent
    l = angular quantum number
    m = magnetic quantum number
    """
    return get_ao(a,l,m)*get_Npi(a,l)*norm_extra(l)
    
def get_Nao_at(pos, a,l,m):
    """
    return normalized AO in sympy-format
    a = exponent
    l = angular quantum number
    m = magnetic quantum number
    """
    return get_ao_at(pos, a,l,m)*get_Npi(a,l)*norm_extra(l)


def f(m):
    """
    factorial m!
    """
    return np.max([np.prod(np.arange(m)+1), 1])

def norm_extra(l):
    """
    Factor required that is _not_ accounted for
    in eq. 3.3 in LSDalton manual
    """
    return (np.array([1,1,3,15,105])**-.5)[l]

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
    return np.sum(w[:,None]*w[None,:]*(np.sqrt(4*a[:,None]*a[None,:])/(a[:,None]+a[None,:]))**(1.5 + l))


def get_contracted(a,w,l,m, representation = "numeric"):
    """
    Generates Solid Harmonic Gaussian lambda functions
    a = exponent
    """
    S = contracted_norm(a,w,l)
    CGO = 0
    for i in np.arange(a.shape[0]):
        CGO += w[i]*get_Nao(a[i],l,m)/np.sqrt(S)
    
    if representation is "numeric":
        x,y,z = sp.symbols("x y z")
        
        return sp.lambdify([x,y,z], CGO, "numpy")
    if representation is "sympy":
        return CGO


def get_sto(a,w,l,m, representation = "numeric"):
    S = np.sqrt(a**3/np.pi)

    x,y,z = sp.symbols("x y z")
    sto = S*get_Slm(l,m)*sp.exp(-a*sp.sqrt(x**2 + y**2 + z**2))
    if representation is "numeric":
        return sp.lambdify([x,y,z], sto, "numpy")
        
    if representation is "sympy":
        return sto



def get_contracted_sympy(a,w,l,m):
    """
    Generates Solid Harmonic Gaussian lambda functions
    a = exponents
    
    """
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
    