import matplotlib.pyplot as plt

import numpy as np
import os 
import sys
import copy 



from scipy.interpolate import interp1d
import sympy as sp

import braketlab.solid_harmonics as solid_harmonics

from scipy.stats import multivariate_normal




from functools import lru_cache
import warnings

from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator


def plot(*p):
    warnings.warn("replaced by show( ... )", DeprecationWarning, stacklevel=2)
    show(*p)



def show(*p):
    """
    all-purpose vector visualization
    
    Example usage to show the vectors as an image

    a = ket( ... ) 
    b = ket( ... )

    plot(a, b)
    """
    try:
        Nt = 200
        t = np.linspace(-4,4,200)
        Z = np.zeros((Nt, Nt, 3), dtype = float)
        colors = np.random.uniform(0,1,(len(list(p)), 3))
        plt.figure(figsize=(6,6))
        for i in list(p):
            plt.contour(t,t,i(t[:, None], t[None,:]))
        plt.grid()
        plt.show()
            

    except:
        mv = 1
        plt.figure(figsize = (6,6))
        for i in list(p):
            vec_R2 = i.coefficients[0]*i.basis[0].coefficients + i.coefficients[1]*i.basis[1].coefficients
            plt.plot([0, vec_R2[0]], [0, vec_R2[1]], "-")
            
            plt.plot([vec_R2[0]], [vec_R2[1]], "o", color = (0,0,0))
            plt.text(vec_R2[0]+.1, vec_R2[1], "%s" % i.__name__)

            mv = max( mv, max(vec_R2[1], vec_R2[0]) ) 
            
            
        plt.grid()
        plt.xlim(-mv-1,mv+1)
        plt.ylim(-mv-1,mv+1)
        plt.show()

    
def construct_basis(p):
    """
    Build basis from prism object
    """
    
    basis = []
    for atom, pos in zip(p.basis_set, p.atoms):
        for shell in atom:
            for contracted in shell:
                contracted = np.array(contracted)
                l = int(contracted[0,2])
                a = contracted[:, 0]
                w = contracted[:, 1]
                
                for m in range(-l,l+1):
                    bf = w[0]*get_solid_harmonic_gaussian(a[0],l,m, position = [0,0,0])
                    for weights in range(1,len(w)):
                        bf +=  w[i]*get_solid_harmonic_gaussian(a[i],l,m, position = [0,0,0])
                    #print(pos, m, l, a, w)
                    #get_solid_harmonic_gaussian(a,l,m, position = [0,0,0])
                    basis.append( bf )

        #print(" ")
    return basis
    
    





                    
                    

class basisfunction:
    """
    A general class for a basis function in R^n
    
    Keyword arguments:
    sympy_expression -- A sympy expression
    position         -- assumed center of basis function (defaults to vec(0) )
    name             -- (unused)
    domain           -- if None, the domain is R^n
                        if [ [x0, x1], [ y0, y1], ... ] , the domain is finite

    Methods
    normalize        -- Perform numerical normalization of self
    estimate_decay   -- Estimate decay of self, used for importance sampling (currently inactive)
    get_domain(other)-- Returns the intersecting domain of two basis functions


    Example usage:

    x = sympy.Symbol("x")

    x2 = basisfunction(x**2)
    
    x2.normalize()
    
    """
    position = None
    normalization = 1
    domain = None
    __name__ = "\chi"
    
    def __init__(self, sympy_expression, position = None, domain = None, name = "\chi"):
        
        self.__name__ = name
        self.dimension = len(sympy_expression.free_symbols)
        
        self.position = np.array(position)
        
        if position is None:
            self.position = np.zeros(self.dimension, dtype = float)
            
        assert(len(self.position)==self.dimension), "Basis function position contains incorrect number of dimensions (%.i)." % self.dimension


        
        
        # sympy expressions
        self.ket_sympy_expression = translate_sympy_expression(sympy_expression, self.position)
        self.bra_sympy_expression = translate_sympy_expression(sp.conjugate(sympy_expression), self.position)
        
        # numeric expressions
        symbols = np.array(list(sympy_expression.free_symbols))
        l_symbols = np.argsort([i.name for i in symbols])
        symbols = symbols[l_symbols]
        
        self.ket_numeric_expression = sp.lambdify(symbols, self.ket_sympy_expression, "numpy")
        self.bra_numeric_expression = sp.lambdify(symbols, self.bra_sympy_expression, "numpy")
        
        # decay
        self.decay = 1.0
        
            
    def normalize(self, domain = None):
        s_12 = inner_product(self, self)
        self.normalization = s_12**-.5
    
    def estimate_decay(self):
        # estimate standard deviation 
        #todo : proper decay estimate (this one is incorrect)
        
        #x = np.random.multivariate_normal(self.position*0, np.eye(len(self.position)), 1e7)
        #r2 = np.sum(x**2, axis = 1)
        #P = multivariate_normal(mean=self.position*0, cov=np.eye(len(self.position))).pdf(x)
        self.decay = 1 #np.mean(self.numeric_expression(*x.T)*r2*P**-1)**.5
        
    
            
    
    def get_domain(self, other = None):
        if other is None:
            return self.domain
        else:
            domain = self.domain
            if self.domain is not None:
                domain = []
                for i in range(len(self.domain)):
                    domain.append([np.array([self.domain[i].min(), other.domain[i].min()]).max(),
                                   np.array([self.domain[i].max(), other.domain[i].max()]).min()])
                
            return domain
        
    def __call__(self, *r):
        # Evaluate function in coordinates 

        return self.normalization*self.ket_numeric_expression(*r) 
    
    
    
    def __mul__(self, other):
        return basisfunction(self.ket_sympy_expression * other.ket_sympy_expression, 
                   position = .5*(self.position + other.position),
                   domain = self.get_domain(other), name = self.__name__+other.__name__)
    
    def __rmul__(self, other):
        return basisfunction(self.ket_sympy_expression * other.ket_sympy_expression, 
                   position = .5*(self.position + other.position),
                   domain = self.get_domain(other), name = self.__name__+other.__name__)
    
        
    def __add__(self, other):
        return basisfunction(self.ket_sympy_expression + other.ket_sympy_expression, 
                   position = .5*(self.position + other.position),
                   domain = self.get_domain(other))
    
    def __sub__(self, other):
        return basisfunction(self.ket_sympy_expression - other.ket_sympy_expression, 
                   position = .5*(self.position + other.position),
                   domain = self.get_domain(other))
    
    
    def _repr_html_(self):
        
        return "$ %s $" % sp.latex(self.ket_sympy_expression)
    
    
def get_solid_harmonic_gaussian(a,l,m, position = [0,0,0]):
    return basisfunction(solid_harmonics.get_Nao(a,l,m), position = position)                
                
                
class operator():
    """
    A parent class for quantum mechanical operators
    """
    def __init__(self, operator_action, prefactor = 1):
        self.operator_actions = [operator_action]
        self.prefactor = prefactor

    def __mul__(self, other):
        assert(type(other) in [ket, float, int]), "operator cannot act on %s" % type(other)
        if type(other) in [float, int]:
            self.prefactor *= other
        else:
            return_ket = self.operator_actions[-1](other)
            for i in range(1,len(self.operator_actions)):
                return_ket = operator_actions[-i-1](other)
            return self.prefactor*return_ket


def translate_sympy_expression(sympy_expression, translation_vector):
    symbols = np.array(list(sympy_expression.free_symbols))
    l_symbols = np.argsort([i.name for i in symbols])
    shifts = symbols[l_symbols]

    assert(len(shifts)==len(translation_vector)), "Incorrect length of translation vector"

    return_expression = sympy_expression*1

    for i in range(len(shifts)):
        return_expression = return_expression.subs(shifts[i], shifts[i]-translation_vector[i])

    return return_expression


# Operators
class kinetic_operator():
    def __init__(self):
        pass
    
    
    def __mul__(self, other):
        variables = other.basis[0].ket_sympy_expression.free_symbols
        
        #ret = 0
        new_coefficients = other.coefficients
        new_basis = []
        for i in other.basis:
            new_basis_ = 0
            for j in variables:
                new_basis_ += sp.diff(i.ket_sympy_expression,j, 2)
            new_basis.append(basisfunction(new_basis_, position = i.position))
        return ket(-.5*new_coefficients, basis = new_basis)

    def _repr_html_(self):
        return "$ -\\frac{1}{2} \\nabla^2 $" 
                    
        
class onebody_coulomb_operator():
    def __init__(self):
        pass
    
    
    def __mul__(self, other, r = None):
        variables = other.basis[0].ket_sympy_expression.free_symbols
        
        r = 0
        for j in variables:
            r += j**2
        r_inv = r**-.5
        #print(r_inv)
        
        new_coefficients = other.coefficients
        new_basis = []
        for i in other.basis:
            new_basis.append(basisfunction(r_inv*i.ket_sympy_expression, position = i.position))
        return ket(-new_coefficients, basis = new_basis)

    def _repr_html_(self):
        return "$ -\\frac{1}{\\mathbf{r}} $"   

      
class twobody_coulomb_operator():
    def __init__(self):
        pass
    
    
    def __mul__(self, other, r = None):
        variables = other.basis[0].ket_sympy_expression.free_symbols
        
        r = 0
        for j in variables:
            r += j**2
        r_inv = r**-.5
        #print(r_inv)
        
        new_coefficients = other.coefficients
        new_basis = []
        for i in other.basis:
            new_basis.append(basisfunction(r_inv*i.ket_sympy_expression, position = i.position))
        return ket(-new_coefficients, basis = new_basis)

    def _repr_html_(self):
        return "$ -\\frac{1}{\\mathbf{r}} $"  
    
                

def get_standard_basis(n):
    b = np.eye(n)
    basis = []
    for i in range(n):
        basis.append(ket(b[i], basis = b))
    return basis

class ket(object):
    """
    A class for vectors defined on general vector spaces
    Author: Audun Skau Hansen (a.s.hansen@kjemi.uio.no)

    Keyword arguments:
    generic_input        -- if list or numpy.ndarray:
                               if basis is None, returns a cartesian vector
                               else, assumes input to contain coefficients
                            if sympy expression, returns ket([1], basis = [basisfunction(generic_input)])
    name                 -- string, used for labelling and plotting, visual aids
    basis                -- a list of basisfunctions
    position             -- assumed centre of function < |R| > 

    Available operations
    Addition, subtraction, scalar multiplication, inner products, evaluation 

    """
    basis = None
    __name__ = ""
    def __init__(self, generic_input, name = "", basis = None, position = None):
        #assert()

        if type(generic_input) in [np.ndarray, list]:
            self.coefficients = np.array(generic_input) 
            if basis is None:
                self.basis = get_standard_basis(self.coefficients.shape[-1]) #np.eye(self.coefficients.shape[-1])
            else:
                self.basis = basis
        else:
            # assume sympy expression 
            #try:
            if position is None:
                position = np.zeros(len(generic_input.free_symbols), dtype = float)
            self.coefficients = np.ones(1, dtype = float)
            self.basis = [basisfunction(generic_input, position = position)]
            #except:
            #    print("Error: input not understood")

        
        self.__name__ = name
        self.bra_state = False
        self.a = None





    def __mul__(self, other):
        #assert(type(other) in [float, int])
        if type(other) is ket:
            return self.__matmul__(other)
        else:
            return ket(self.coefficients*other, basis = self.basis)

    def __rmul__(self, other):
        return ket(self.coefficients*other, basis = self.basis)
    
    def __matmul__(self, other):
        if type(other) in [float, int]:
            return ket(self.coefficients*other, basis = self.basis)
        
        if type(other) is ket:
            if self.bra_state:
                # Compute < self | other >
                if type(self.basis) is np.ndarray:
                    # Cartesian basis
                    metric = self.basis.T.dot(other.basis)
                else:
                    # L2 basis
                    metric = np.zeros((len(self.basis), len(other.basis)), dtype = np.float)
                    
                    for i in range(len(self.basis)):
                        for j in range(len(other.basis)):
                            if type(self.basis[i]) is basisfunction:
                                if type(other.basis[j]) is basisfunction:
                                    # (basisfunction | basisfunction)
                                    metric[i,j] = inner_product(self.basis[i], other.basis[j])
                                    
                                if other.basis[j] is ket:
                                    # (basisfunction | ket )
                                    metric[i,j] = ket(coefficients = [1.0], basis = [self.basis[i]]).bra@other.basis[j]
                            else:
                                if type(other.basis[j]) is basisfunction:
                                    # ( ket | basisfunction )
                                    metric[i,j] = self.basis[i].bra@ket(coefficients = [1.0], basis = [other.basis[j]])
                                
                                else:
                                    # ( ket | ket )
                                    metric[i,j] = self.basis[i].bra@other.basis[j]
                                
                                
                #else:
                #    metric = np.zeros((len(self.basis), len(other.basis)), dtype = np.float)
                #    for i in range(len(self.basis)):
                #        for j in range(len(other.basis)):
                #            metric[i,j] = inner_product(self.basis[i], other.basis[j])


                return self.coefficients.T.dot(metric.dot(other.coefficients))
            
            else:
                assert(False), "not implemented"
                new_coefficients = []
                new_basis = []
                for i in range(len(self.basis)):
                    for j in range(len(other.basis)):
                        new_basis.append(self.basis[i]*other.basis[j])
                        new_coefficients.append(self.coefficients[i]*other.coefficients[j])
                return ket(new_coefficients, basis = new_basis)


        
    
    def __add__(self, other):
        new_basis = copy.copy(self.basis)
        new_coefficients = copy.copy(self.coefficients)
        
        #found = np.zeros(len(other.basis), dtype = np.bool)
        for i in range(len(other.basis)):
            found = False
            for j in range(len(self.basis)):
                if self.basis[j] == other.basis[i]:
                    new_coefficients[j] += other.coefficients[i]
                    found = True
            if not found:
                new_basis.append(other.basis[i])
                new_coefficients.append(other.coefficients[i])
        return ket(new_coefficients, basis = new_basis)
                
        
    def __sub__(self, other):
        new_basis = copy.copy(self.basis)
        new_coefficients = copy.copy(self.coefficients)
        
        #found = np.zeros(len(other.basis), dtype = np.bool)
        for i in range(len(other.basis)):
            found = False
            for j in range(len(self.basis)):
                if self.basis[j] == other.basis[i]:
                    new_coefficients[j] -= other.coefficients[i]
                    found = True
            if not found:
                new_basis.append(other.basis[i])
                new_coefficients.append(-other.coefficients[i])
        return ket(new_coefficients, basis = new_basis)
    
    
    def __call__(self, *R):
        result = 0
        if self.bra_state:
            for i in range(len(self.basis)):
                result += self.coefficients[i]*self.basis[i](*R)
        else:
            for i in range(len(self.basis)):
                result += self.coefficients[i]*self.basis[i](*R)
        return result
    
    #braket - functions
    
    @property
    def bra(self):
        return self.__a

    ## 
    @bra.setter
    def a(self, var):
        self.__a = copy.copy(self)
        self.__a.bra_state = True
        

    def _repr_html_(self):
        if self.bra_state:
            return "$\\langle %s \\vert$" % self.__name__
        else:
            return "$\\vert %s \\rangle$" % self.__name__
    

@lru_cache(maxsize=100)
def inner_product(b1, b2, operator = None, n_samples = int(1e6), grid = 101):
    """
    Computes the inner product < b1 | b2 >, where bn are instances of basisfunction

    Keyword arguments:
    b1, b2    -- basisfunction objects
    operator  -- obsolete
    n_samples -- number of Monte Carlo samples
    grid      -- number of grid-points in every direction for the 
                 spline control variate

    Returns:
    The inner product as a float

    """
    ri = b1.position
    rj = b2.position

    integrand = lambda *R, \
                       f1 = b1.bra_numeric_expression, \
                       f2 = b2.ket_numeric_expression, \
                       ri = ri, rj = rj:  \
                       f1(*np.array([R[i] - ri[i] for i in range(len(ri))]))*f2(*np.array([R[i] - rj[i] for i in range(len(rj))]))



    ai,aj = b1.decay, b2.decay
    ri,rj = b1.position, b2.position

    R = (ai*ri + aj*rj)/(ai+aj)
    sigma = ai + aj


    return onebody(integrand, np.ones(len(R))*sigma, R, n_samples) #, control_variate = "spline", grid = grid) 
        

def compose_basis(p):
    """
    generate a list of basis functions 
    corresponding to the AO-basis 
    (same ordering and so on)
    """
    basis = []
    for charge in np.arange(p.charges.shape[0]):


        atomic_number = p.charges[charge]
        atom = np.argwhere(p.atomic_numbers==atomic_number)[0,0] #index of basis function


        pos = p.atoms[charge]




        for shell in np.arange(len(p.basis_set[atom])):
            for contracted in np.arange(len(p.basis_set[atom][shell])):
                W = np.array(p.basis_set[atom][shell][contracted])
                w = W[:,1]
                a = W[:,0]
                if shell == 1:
                    for m in np.array([1,-1,0]):
                        basis.append(basis_function([shell,m,a,w], basis_type = "cgto",domain = [[-8,8],[-8,8],[-8,8]], position = pos))
                else:
                    for m in np.arange(-shell, shell+1):
                        basis.append(basis_function([shell,m,a,w], basis_type = "cgto",domain = [[-8,8],[-8,8],[-8,8]], position = pos))
                        


    return basis



def get_control_variate(integrand, loc, a = .6, tmin = 1e-5, extent = 6, grid = 101):
    """
    Generate an nd interpolated control variate

    returns RegularGridInterpolator, definite integral on mesh, mesh points


    Keyword arguments
    integrand    -- an evaluateable function
    loc          -- position offset for integrand
    a            -- grid density decay, 
    tmin         -- 
    extent       --
    grid         -- number of grid points
    """
    
    t = np.linspace(tmin,extent**a,grid)**(a**-1)

    t = np.append(-t[::-1],t)

    R_ = np.ones((loc.shape[0],t.shape[0]))*t[None,:]

    R = np.meshgrid(*(R_ - loc[:, None]), indexing='ij', sparse=True)
    
    data = integrand(*R)
    

    # Integrate
    I0 = rgrid_integrate_nd(t, data)


    #return RegularGridInterpolator(R_, data, bounds_error = False, fill_value = 0), I0, t
    return RegularGridInterpolator(R_-loc[:, None], data, bounds_error = False, fill_value = 0), I0, t



def rgrid_integrate_3d(points, values):
    """
    regular grid integration, 3D
    """
    # volume per cell
    v = np.diff(points)

    v = v[:,None,None]*v[None,:,None]*v[None,None,:]
    
    # weight per cell
    w = values[:-1] + values[1:]
    w = w[:, :-1] + w[:, 1:]
    w = w[:, :, :-1] + w[:, :, 1:]
    w = w/8
    
    return np.sum(w*v)

def rgrid_integrate_nd(points, values):
    """
    Integrate over n dimensions as linear polynomials on a grid

    Keyword arguments:
    points     -- cartesian coordinates of gridpoints
    values     -- values of integrand at gridpoints

    Returns:
    Integral of linearly interpolated integrand
    """


    points = np.diff(points)
    w = ""
    for i in range(len(values.shape)):
        cycle = ""
        for j in range(len(values.shape)):
            if j==i:
                cycle+=":,"
            else:
                cycle+="None,"

        w +="points[%s] * " % cycle[:-1]
    v = eval(w[:-2])

    w = values
    wd= 1

    for i in range(len(values.shape)):
        w = eval("w[%s:-1] + w[%s1:]" % (i*":,", i*":,"))
        wd *= 2
    
    return np.sum(v*w/wd)


def onebody(integrand, sigma, loc, n_samples, control_variate = lambda *r : 0, grid = 101, I0 = 0):
    """
    Monte Carlo (MC) estimate of integral

    Keyword arguments:
    integrand       -- evaluatable function
    sigma           -- standard deviation of normal distribution used
                       for importance sampling
    loc             -- centre used for control variate and importance sampling
    n_sampes        -- number of MC-samples
    control_variate -- evaluatable function
    grid            -- sampling density of spline control variate
    I0              -- analytical integral of control variate

    returns:
    Estimated integral (float)
    """
    if control_variate == "spline":
        
        control_variate, I0, t = get_control_variate(integrand, loc, a = .6, tmin = 1e-5, extent = 6, grid = grid)

    #print("sigma:", sigma)
    #R = np.random.multivariate_normal(loc, np.eye(len(loc))*sigma, n_samples)
    #R = np.random.Generator.multivariate_normal(loc, np.eye(len(loc))*sigma, size=n_samples)
    R = np.random.default_rng().multivariate_normal(loc, np.eye(len(loc))*sigma, n_samples)
    P = multivariate_normal(mean=loc, cov=np.eye(len(loc))*sigma).pdf(R)

    return I0+np.mean((integrand(*R.T)-control_variate(R)) * P**-1)
    

def eri_mci(phi_p, phi_q, phi_r, phi_s, 
            pp = np.array([0,0,0]), 
            pq = np.array([0,0,0]), 
            pr = np.array([0,0,0]), 
            ps = np.array([0,0,0]), 
            N_samples = 1000000, sigma = .5, 
            Pr = np.array([0,0,0]), 
            Qr = np.array([0,0,0]), 
            zeta = 1, 
            eta = 1, 
            auto = False,
            control_variate = lambda x1,x2,x3,x4,x5,x6 : 0):

    """
    Electron repulsion integral estimate using zero-variance Monte Carlo
    """
    
    x = np.random.multivariate_normal([0,0,0,0,0,0], np.eye(6)*sigma, N_samples)
    
    P = multivariate_normal(mean=[0,0,0,0,0,0], cov=np.eye(6)*sigma).pdf 
    
    if auto:
        # estimate mean and variance of orbitals
        X,Y,Z = np.random.uniform(-5,5,(3, 10000))
        P_1 = phi_p(X,Y,Z)*phi_q(X,Y,Z)
        P_2 = phi_r(X,Y,Z)*phi_s(X,Y,Z)
        
        
        
        Pr[0] = np.mean(P_1**2*X)
        Pr[1] = np.mean(P_1**2*Y)
        Pr[2] = np.mean(P_1**2*Z)
        
        Qr[0] = np.mean(P_2**2*X)
        Qr[1] = np.mean(P_2**2*Y)
        Qr[2] = np.mean(P_2**2*Z)
    
    integrand = lambda *R, \
                       phi_p = phi_p, \
                       phi_q = phi_q, \
                       phi_r = phi_r, \
                       phi_s = phi_s, \
                       rp = pp, rq = pq, rr = pq, rs = ps :  \
                       phi_p(R[0] - rp[0], R[1] - rp[1], R[2] - rp[2])* \
                       phi_q(R[0] - rq[0], R[1] - rq[1], R[2] - rq[2])* \
                       phi_r(R[3] - rr[0], R[4] - rr[1], R[5] - rr[2])* \
                       phi_s(R[3] - rs[0], R[4] - rs[1], R[5] - rs[2]) 
                       
    
    
    
        

                     

    
    if control_variate == "spline":
        
        
        
        control_variate,  I0, t = get_control_variate(integrand, loc = np.array([0,0,0,0,0,0]), a = .6, tmin = 1e-5, extent = 6, grid = 11)
        


        
    
    u1 =  x[:, :3]*zeta**-.5 + Pr[:]
    u2 =  x[:, 3:]*eta**-.5  + Qr[:]
    r12 = np.sqrt(np.sum( (u1 - u2)**2, axis = 1))
    
    return np.mean((phi_p(u1[:, 0] - pp[0], u1[:,1] - pp[1], u1[:,2] - pp[2]) * 
                    phi_q(u1[:, 0] - pq[0], u1[:,1] - pq[1], u1[:,2] - pq[2]) *
                    phi_r(u2[:, 0] - pr[0], u2[:,1] - pr[1], u2[:,2] - pr[2]) *
                    phi_s(u2[:, 0] - ps[0], u2[:,1] - ps[1], u2[:,2] - ps[2]) - 
                    control_variate(u1[:, 0], u1[:, 1],u1[:, 2],u2[:, 0],u2[:, 1],u2[:, 2]) ) / 
                   (P(x)*r12) ) * (zeta*eta)**-1.5 

