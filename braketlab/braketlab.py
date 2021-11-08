import matplotlib.pyplot as plt

import numpy as np
import os 
import sys
import copy
from scipy.interpolate.fitpack import splint 




import sympy as sp



#import braketlab.solid_harmonics as solid_harmonics
#import braketlab.hydrogen as hydrogen
import braketlab.basisbank as basisbank
import braketlab.animate as anim




from functools import lru_cache
import warnings

from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator
from scipy import integrate
from scipy.stats import multivariate_normal
from scipy.interpolate import interp1d
from sympy.utilities.runtests import split_list


def plot(*p):
    warnings.warn("replaced by show( ... )", DeprecationWarning, stacklevel=2)
    show(*p)



def show(*p, t=None):
    """
    all-purpose vector visualization
    
    Example usage to show the vectors as an image

    a = ket( ... ) 
    b = ket( ... )

    plot(a, b)
    """
    try:
        Nx = 200
        x = np.linspace(-8,8,200)
        Z = np.zeros((Nx, Nx, 3), dtype = float)
        colors = np.random.uniform(0,1,(len(list(p)), 3))
        plt.figure(figsize=(6,6))
        for i in list(p):
            try:
                plt.contour(x,x,i(x[:, None], x[None,:]))
            except:
                plt.plot(x,i(x) , label=i.__name__)
        plt.grid()
        plt.legend()
        plt.show()
            

    except:
        mv = 1
        plt.figure(figsize = (6,6))
        for i in list(p):
            vec_R2 = i.coefficients[0]*i.basis[0] + i.coefficients[1]*i.basis[1]
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

    Operators acts on kets

    An "operator action" may only act on a basis function directly
    """
    def __init__(self, operator_action, prefactor = 1, special_operator = False):
        if type(operator_action) is list:
            self.operator_actions = operator_action
        else:
            if special_operator:
                self.operator_actions = [[operator_action]]
            
            else:
                # assume operator action is a sympy expression

                self.operator_actions = [[sympy_operator_action(operator_action)]]
        self.prefactor = prefactor

    def __mul__(self, other):
        """
        Multiplication is interpreted as an action on the ket on its right
        """
        assert(type(other) in [operator, ket, float, int]), "operator cannot act on %s" % type(other)
        if type(other) in [float, int]:
            self.prefactor *= other
        if type(other) is operator:
            operator_actions = []
            for i in range(len(self.operator_actions)):
                for j in range(len(other.operator_actions)):
                    operator_actions.append(self.operator_actions[i] + other.operator_actions[j])
            return operator(operator_actions, self.prefactor*other.prefactor)
        else:
            new_basis = []
            for k in range(len(other.basis)):
                # Let the operator act on each basisfunction [k] independently



                return_basis_function = 0

                basis_function = other.basis[k]

                if type(basis_function) is basisfunction:
                    for i in range(len(self.operator_actions)):
                        # The operator may be the sum of several constituent operators [i]

                        product_basisfunction = self.operator_actions[i][-1]*basis_function #.ket_sympy_expression

                        for j in range(1,len(self.operator_actions[i])):
                            # sequence product from the rightmost operator[i][-1-j] in the product
                            product_basisfunction = self.operator_actions[i][-1-j]*product_basisfunction
                        if type(return_basis_function) is int:
                            return_basis_function =  product_basisfunction
                        else:
                            return_basis_function = return_basis_function + product_basisfunction
                else:
                    return_basis_function = self.__mul__(basis_function)

                        
                #print(return_basis_function, basis_function.position)
                
                #bf = basisfunction(return_basis_function)
                #bf.position = basis_function.position

                new_basis.append(return_basis_function)
            
            return ket(other.coefficients, basis = new_basis)
        


    def __add__(self, other):
        """
        Add operators together
        """
        return operator(self.operator_actions + other.operator_actions, self.prefactor)


class sympy_operator_action:
    def __init__(self, sympy_expression):
        self.sympy_expression = sympy_expression
    
    def __mul__(self, other):
        assert(type(other) is basisfunction), "cannot operate on %s" %type(other)
        bs = basisfunction(self.sympy_expression*other.ket_sympy_expression)
        bs.position = other.position
        return bs
        
class translation:
    def __init__(self, translation_vector):
        self.translation_vector = np.array(translation_vector)
        
    def __mul__(self, other):
        assert(type(other) is basisfunction), "cannot translate %s" %type(other)
        new_expression = translate_sympy_expression(other.ket_sympy_expression, self.translation_vector)
        bs = basisfunction(new_expression)
        bs.position = other.position + self.translation_vector
        return bs
    
    
class differential:
    def __init__(self, order):
        self.order = order
        
    def __mul__(self, other):
        assert(type(other) is basisfunction), "cannot differentiate %s" %type(other)
        
        new_expression = 0
        symbols = np.array(list(other.ket_sympy_expression.free_symbols))
        l_symbols = np.argsort([i.name for i in symbols])
        symbols = symbols[l_symbols]
        
        for i in range(len(symbols)):
            new_expression += sp.diff(other.ket_sympy_expression, symbols[i], self.order[i])
        
        bs = basisfunction(new_expression)
        bs.position = other.position
        
        return bs
        
    
    
def get_translation_operator(pos):
    return operator(translation(pos), special_operator = True)

def get_sympy_operator(sympy_expression):
    return operator(sympy_expression)

def get_differential_operator(order):
    return operator(differential(order),special_operator = True)
        

    




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
        return ket([-.5*i for i in new_coefficients], basis = new_basis)

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
        return ket([-1*i for i in new_coefficients], basis = new_basis)

    def _repr_html_(self):
        return "$ -\\frac{1}{\\mathbf{r}} $"   

class twobody_coulomb_operator():
    def __init__(self):
        pass
    
    
    def __mul__(self, other):
        vid = other.variable_identities 
        new_basis = []
        for i in range(len(other.basis)):
            fs1, fs2 = vid[i]
            denom = 0
            for k,l in zip(list(fs1), list(fs2)):
                denom += sp.sqrt((k - l)**2)
            new_basis.append( ket(other.basis[i].ket_sympy_expression/denom) )
        return ket(other.coefficients, basis = new_basis)

    def _repr_html_(self):
        return "$ -\\frac{1}{\\vert \\mathbf{r}_1 - \\mathbf{r}_2 \\vert} $"  
    
class twobody_coulomb_operator_old():
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
    def __init__(self, generic_input, name = "", basis = None, position = None, energy = None):
        self.position = position
        if type(generic_input) in [np.ndarray, list]:
            
            self.coefficients = list(generic_input) 
            self.basis = [i for i in np.eye(len(self.coefficients))]
            if basis is not None:
                self.basis = basis
                
        else:
            # assume sympy expression
            if position is None:
                position = np.zeros(len(generic_input.free_symbols), dtype = float)
            self.coefficients = [1.0]
            self.basis = [basisfunction(generic_input, position = position)]
            
        self.ket_sympy_expression = self.get_ket_sympy_expression()
        self.bra_sympy_expression = self.get_bra_sympy_expression()
        if energy is not None:
            self.energy = energy
        else:
            self.energy = [0 for i in range(len(self.basis))]

        self.__name__ = name
        self.bra_state = False
        self.a = None
        
        
        
        
    """
    Algebraic operators
    """
    def __add__(self, other):
        new_basis = self.basis + other.basis  
        
        new_coefficients = self.coefficients + other.coefficients
        new_energies = self.energy + other.energy
        ret = ket(new_coefficients, basis = new_basis, energy = new_energies)
        ret.flatten()
        ret.__name__ = "%s + %s" % (self.__name__, other.__name__)
        return ret
    
    def __sub__(self, other):
        new_basis = self.basis + other.basis  
        
        new_coefficients = self.coefficients + [-i for i in other.coefficients]
        ret = ket(new_coefficients, basis = new_basis)
        ret.flatten()
        ret.__name__ = "%s - %s" % (self.__name__, other.__name__)
        return ret
    
    def __mul__(self, other):
        if type(other) is ket:
            return self.__matmul__(other)
        else:
            return ket([other*i for i in self.coefficients], basis = self.basis)

    def __rmul__(self, other):
        return ket([other*i for i in self.coefficients], basis = self.basis)
    
    def __truediv__(self, other):
        return ket([i/other for i in self.coefficients], basis = self.basis)
    
    def __matmul__(self, other):
        """
        Inner- and Cartesian products
        """
        if type(other) in [float, int]:
            return self*other
        
        if type(other) is ket:
            if self.bra_state:
                # Compute inner product: < self | other >
                metric = np.zeros((len(self.basis), len(other.basis)), dtype = np.complex)

                for i in range(len(self.basis)):
                    for j in range(len(other.basis)):
                        if type(self.basis[i]) is np.ndarray and type(other.basis[j]) is np.ndarray:
                                metric[i,j] = np.dot(self.basis[i], other.basis[j])

                        else:
                            if type(self.basis[i]) is basisfunction:
                                if type(other.basis[j]) is basisfunction:
                                    # (basisfunction | basisfunction)
                                    metric[i,j] = inner_product(self.basis[i], other.basis[j])

                                if other.basis[j] is ket:
                                    # (basisfunction | ket )
                                    metric[i,j] = ket([1.0], basis = [self.basis[i]]).bra@other.basis[j]
                            else:
                                if type(other.basis[j]) is basisfunction:
                                    # ( ket | basisfunction )
                                    metric[i,j] = self.basis[i].bra@ket([1.0], basis = [other.basis[j]])

                                else:
                                    # ( ket | ket )
                                    metric[i,j] = self.basis[i].bra@other.basis[j]
                
                if np.linalg.norm(metric.imag)<=1e-10:
                    metric = metric.real
                return np.array(self.coefficients).T.dot(metric.dot(np.array(other.coefficients)))
            
            
            else:
                if type(other) is ket:
                    if other.bra_state:
                        return outerprod(self, other)
                        
                    else:

                        new_coefficients = []
                        new_basis = []
                        variable_identities = [] #for potential two-body interactions
                        for i in range(len(self.basis)):
                            for j in range(len(other.basis)):
                                bij, sep = split_variables(self.basis[i].ket_sympy_expression, self.basis[j].ket_sympy_expression)
                                bij = ket(bij)
                                new_basis.append(bij)
                                new_coefficients.append(self.coefficients[i]*other.coefficients[j])
                                variable_identities.append(sep)

                        ret = ket(new_coefficients, basis = new_basis)
                        ret.flatten()
                        ret.__name__ = self.__name__ + other.__name__
                        ret.variable_identities = variable_identities
                        return ret
            
    
    
    def flatten(self):
        """
        Remove redundancies in the expansion of self
        """
        new_coefficients = []
        new_basis = []
        new_energies = []
        found = []
        for i in range(len(self.basis)):
            if i not in found:
                new_coefficients.append(self.coefficients[i])
                new_basis.append(self.basis[i])
                new_energies.append(self.energy[i])
            
                for j in range(i+1, len(self.basis)):
                    #print(i,j,type(self.basis[i]), type(self.basis[j]))
                    if type(self.basis[i]) is np.ndarray:
                        if type(self.basis[j]) is np.ndarray:
                            if np.all(self.basis[i]==self.basis[j]):
                                new_coefficients[i] += self.coefficients[j]
                                found.append(j)
                    else:
                        if self.basis[i].ket_sympy_expression == self.basis[j].ket_sympy_expression:
                            if np.all(self.basis[i].position == self.basis[j].position):
                                new_coefficients[i] += self.coefficients[j]
                                found.append(j)
        self.basis = new_basis
        self.coefficients = new_coefficients
        self.energy = new_energies
        
    def get_ket_sympy_expression(self):
        ret = 0
        for i in range(len(self.coefficients)):
            
            if type(self.basis[i]) in [basisfunction, ket]:
                ret += self.coefficients[i]*self.basis[i].ket_sympy_expression

            else:
                ret += self.coefficients[i]*self.basis[i]
        return ret
    
    def get_bra_sympy_expression(self):
        ret = 0
        for i in range(len(self.coefficients)):
            
            if type(self.basis[i]) in [basisfunction, ket]:
                ret += np.conjugate(self.coefficients[i])*self.basis[i].bra_sympy_expression

            else:
                ret += np.conjugate(self.coefficients[i]*self.basis[i])
        return ret
    
    def __call__(self, *R, t = None):
        if t is None:
            result = 0
            if self.bra_state:
                for i in range(len(self.basis)):
                    result += np.conjugate(self.coefficients[i]*self.basis[i](*R))
            else:
                for i in range(len(self.basis)):
                    result += self.coefficients[i]*self.basis[i](*R)
            return result
        else:
            result = 0
            if self.bra_state:
                for i in range(len(self.basis)):
                    result += np.conjugate(self.coefficients[i]*self.basis[i](*R)*np.exp(-np.complex(0,1)*self.energy[i]*t))
            else:
                for i in range(len(self.basis)):
                    result += self.coefficients[i]*self.basis[i](*R)*np.exp(-np.complex(0,1)*self.energy[i]*t)
            return result

    @property
    def bra(self):
        return self.__a

    @bra.setter
    def a(self, var):
        self.__a = copy.copy(self)
        self.__a.bra_state = True
        

    def _repr_html_(self):
        if self.bra_state:
            return "$\\langle %s \\vert$" % self.__name__
        else:
            return "$\\vert %s \\rangle$" % self.__name__


    def run(self, x = 8*np.linspace(-1,1,100), t = 0, dt = 0.001):
        anim_s = anim.system(self, x, t, dt)
        anim_s.run()

    """
    Measurement
    """

    def measure(self, observable = None, repetitions = 1):
        """
        Make a mesaurement of the observable (hermitian operator)
        """
        if observable is None:
            # Measure position
            P = self.get_bra_sympy_expression()*self.get_ket_sympy_expression()
            symbols = P.free_symbols
            P = sp.lambdify(symbols, P, "numpy")
            nd = len(symbols)
            sig = .1 #variance of initial distribution


            r = np.random.multivariate_normal(np.zeros(nd), sig*np.eye(nd), repetitions )

            # Metropolis-Hastings 
            for i in range(1000):
                dr = np.random.multivariate_normal(np.zeros(nd), 0.01*sig*np.eye(nd), repetitions)
                
                accept = P(r+dr)/P(r) > np.random.uniform(0,1,nd)
                r[accept] += dr[accept]
            return r


def metropolis_hastings(f, N, x0, a):
    """
    Metropolis-Hastings random walk in the function f
    """
    x = np.random.multivariate_normal(x0, a, N)

    
    for i in range(1000):
        dx = np.random.multivariate_normal(x0, a*0.01, N)
        #print(dx.shape)
        accept = f(x+dx)/f(x) > np.random.uniform(0,1,N)
        x[accept] += dx[accept]
    return x




def split_variables(s1,s2):
    # split variables of two sympy expressions
    s1s = list(s1.free_symbols)
    for i in range(len(s1s)):
        s1 = s1.subs(s1s[i], sp.Symbol("x_{1, %i}" % i))

    s2s = list(s2.free_symbols)
    for i in range(len(s2s)):
        s2 = s2.subs(s2s[i], sp.Symbol("x_{2, %i}" % i))
        
    return s1*s2, [s1.free_symbols, s2.free_symbols]
        



def trace(outer):
    assert(len(outer.ket.basis)==len(outer.bra.basis)), "Trace ill-defined."
    return projector(outer.ket, outer.bra)


class outerprod(object):
    def __init__(self, ket, bra):
        self.ket = ket
        self.bra = bra

    def _repr_html_(self):
        return "$$\\sum_{ij} \\vert %s_i \\rangle \\langle %s_j \\vert$$" % (self.ket.__name__, self.bra.__name__)

    def __mul__(self, other):
        if type(other) is ket:
            coefficients = []
            for i in range(len(self.ket.basis)):
                coeff_i = 0
                for j in range(len(self.bra.basis)):
                    if type(self.bra.basis[j]) is ket:
                        coeff_i += self.bra.basis[i].bra@other
                    if type(self.bra.basis[j]) is basisfunction:
                        coeff_i += ket([1], basis = [self.bra.basis[j]]).bra@other
                
                coefficients.append(coeff_i*self.ket.coefficients[i])
            return ket(coefficients, basis = copy.copy(self.ket.basis), energy = copy.copy(self.ket.energy))


class projector(object):
    def __init__(self, ket, bra):
        self.ket = ket
        self.bra = bra

    def _repr_html_(self):
        return "$$\\sum_{i} \\vert %s_i \\rangle \\langle %s_i \\vert$$" % (self.ket.__name__, self.bra.__name__)

    def __mul__(self, other):
        if type(other) is ket:
            coefficients = []
            for i in range(len(self.ket.basis)):
                coeff_i = 0
                if type(self.bra.basis[i]) is ket:
                    coeff_i = self.bra.basis[i].bra@other
                if type(self.bra.basis[i]) is basisfunction:
                    coeff_i = ket([1], basis = [self.bra.basis[i]]).bra@other
                
                coefficients.append(coeff_i*self.ket.coefficients[i])
            return ket(coefficients, basis = copy.copy(self.ket.basis), energy = copy.copy(self.ket.energy))
        
@lru_cache(maxsize=100)
def inner_product(b1, b2, operator = None, n_samples = int(1e7), grid = 101):
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


    variables_b1 = b1.bra_sympy_expression.free_symbols
    variables_b2 = b2.ket_sympy_expression.free_symbols
    if len(variables_b1) == 1 and len(variables_b2) == 1:
        return integrate.quad(integrand, -10,10)[0]




    else:
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

