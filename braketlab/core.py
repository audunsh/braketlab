from operator import ne
from tempfile import gettempdir
import matplotlib.pyplot as plt

import numpy as np
import os 
import sys
import copy
from scipy.interpolate.fitpack import splint 
import evince as ev



import sympy as sp



#import braketlab.solid_harmonics as solid_harmonics
#import braketlab.hydrogen as hydrogen
#import braketlab.basisbank as basisbank

#from braketlab.basisbank import get_hydrogen_function, get_harmonic_oscillator_function, get_gto, get_sto
#import braketlab.animate as anim




from functools import lru_cache
import warnings

from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import RegularGridInterpolator
from scipy import integrate
from scipy.stats import multivariate_normal
from scipy.interpolate import interp1d
#from sympy.utilities.runtests import split_list


def locate(f):
    """
    Determine (numerically) the center and spread of a sympy function
    
    ## Returns 
    
    position (numpy.array)
    standard deviation (numpy.array)
    
    """
    # Sometimes, values of x giving inf or NaN values of f(x) are generated.
    # This reruns the code until valid results are found, up to 10 times
    # There might be a better way to solve this...
    trycounter = 0
    valid_results = False
    while valid_results == False and trycounter <= 10:
        try:
            trycounter += 1
            s = get_ordered_symbols(f)

            nd = len(s)
            fn = sp.lambdify(s, f, "numpy")
            
            xn = np.zeros(nd) # Initial guess of mean at 0
            ss = 100 # initial distribution

            # qnd norm. Draw n20 numbers around 0
            n20 = 1000 # CSG: n20=1000 seems to give an accuracy of >99%
            funcvals = np.array([0])

            #While-loop increasing ss until non-zero values of f(x) are found
            #Could be faster if previously sampled values of x were excluded
            while funcvals.any(0) == False:
                x0 = np.random.multivariate_normal(xn, np.eye(nd)*ss, n20)
                funcvals = fn(*x0.T).real
                ss *= 2 #Sample more broadly if only 0 values of function found
            P = multivariate_normal(mean=xn, cov=np.eye(nd)*ss).pdf(x0)
            n2 = np.mean(fn(*x0.T).real**2*P**-1, axis = -1)**-1
            x_ = np.mean(x0.T*n2*fn(*x0.T).real**2*P**-1, axis = -1)
            assert(n2>1e-15)
            
            n_tot = 0
            mean_estimate = 0

            for i in range(1,100):
                x0 = np.random.multivariate_normal(xn, np.eye(nd)*ss, n20)   
                P = multivariate_normal(mean=xn, cov=np.eye(nd)*ss).pdf(x0)
                n2 = np.mean((fn(*x0.T).real)**2*P**-1, axis = -1)**-1

                assert(n2>1e-15)
                x_ = np.mean(x0.T*n2*fn(*x0.T).real**2*P**-1, axis = -1)

                if i>50:
                    mean_old = mean_estimate
                    mean_estimate = (mean_estimate*n_tot + np.sum(x0.T*n2*fn(*x0.T).real**2*P**-1, axis = -1))/(n_tot+n20)
                    n_tot += n20
                xn = x_

            # determine spread
            n20 *= 1000
                
            sig = .5
            x0 = np.random.multivariate_normal(xn, np.eye(nd)*sig, n20)
            P = multivariate_normal(mean=xn, cov=np.eye(nd)*sig).pdf(x0)

            #first estimate of spread
            n2 = np.mean(fn(*x0.T).real**2*P**-1, axis = -1)**-1
            x_ = np.mean(x0.T**2*n2*fn(*x0.T).real**2*P**-1, axis = -1) - xn.T**2
            # ensure positive non-zero standard-deviation
            # CSG: I did not understand the previous code for ensuring non-zero std dev 
            # and it gave errors. This seems to work. 
            x_ = np.abs(x_)
            i = .5*(2*x_)**-1
            sig = (2*i)**-.5

            
            # recompute spread with better precision
            x0 = np.random.multivariate_normal(xn, np.diag(sig), n20)
            P = multivariate_normal(mean=xn, cov=np.diag(sig)).pdf(x0)
            if np.isnan(P[0]):
                print("passing due to p = NaN, 2nd")
                pass


            n2 = np.mean(fn(*x0.T).real**2*P**-1, axis = -1)**-1
            x_ = np.mean(x0.T**2*n2*fn(*x0.T).real**2*P**-1, axis = -1) - xn.T**2 
            # ensure positive non-zero standard-deviation. 
            # CSG: I did not understand the previous code for ensuring non-zero std dev 
            # and it gave errors. This seems to work. 
            x_ = np.abs(x_)
            i = .5*(2*x_)**-1
            sig = (2*i)**-.5
            print(mean_estimate, sig)
            valid_results = True
        except: 
            print("Error finding center of function. Trying again.")

    
    
    return mean_estimate, sig

def plot(*p):
    warnings.warn("replaced by show( ... )", DeprecationWarning, stacklevel=2)
    show(*p)

def get_cubefile(p, Nx = 60):
    t = np.linspace(-20,20,Nx)
    cubic = p(t[:,None,None], t[None,:,None], t[None,None,:])
    if cubic.dtype == np.complex128:
        cubic = cubic.real

    cube = """CUBE FILE.
     OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z
        3    0.000000    0.000000    0.000000
       %i    1    0.000000    0.000000
       %i    0.000000    1    0.000000
       %i    0.000000    0.000000    1
        """ % (Nx,Nx,Nx)

    for i in range(Nx):
        for j in range(Nx):
            for k in range(Nx):
                cube += "%.4f     " % cubic[i,j,k].real
            cube += "\n"
    #print(cube)
    #f = open("cubeworld.cube", "w")
    #f.write(cube)
    #f.close()
    return cube, cubic.mean(), cubic.max(), cubic.min()

def show(*p, t=None):
    """
    all-purpose vector visualization
    
    Example usage to show the vectors as an image

    a = ket( ... ) 
    b = ket( ... )

    show(a, b)
    """
    mpfig = True
    maxvx_vals = []
    minvx_vals = []
    mvy_vals = []
    sigs = []
    maxvx = 0.0
    for i in list(p):
        spe = i.get_ket_sympy_expression()
        if type(spe) in [np.array, list, np.ndarray]:
            # 1d vector
            if not mpfig:
                mpfig = True
                plt.figure(figsize=(6,6))
            
            
            vec_R2 = i.coefficients[0]*i.basis[0] + i.coefficients[1]*i.basis[1]
            plt.plot([0, vec_R2[0]], [0, vec_R2[1]], "-")
            
            plt.plot([vec_R2[0]], [vec_R2[1]], "o", color = (0,0,0))
            plt.text(vec_R2[0]+.1, vec_R2[1], "%s" % i.__name__)

            maxvx = max( maxvx, max(vec_R2[1], vec_R2[0]) ) 
                
                
            
            
        else:
            vars = list(spe.free_symbols)
            nd = len(vars)
            Nx = 200
            
            
            
            if nd == 1:
                if not mpfig:
                    mpfig = True
                    plt.figure(figsize=(6,6))
                # 4 std. devs. cover the function area
                #Save bounds for each function to find overall lower and upper bound of x-axis
                mean, sig = locate(spe)
                minvx_vals.append(mean-4*sig)
                maxvx_vals.append(mean+4*sig)
                sigs.append(sig)

                mpfig = True
                
            
            if nd == 2:
                x = np.linspace(-8, 8, Nx)
                if not mpfig:
                    mpfig = True
                    plt.figure(figsize=(6,6))
                plt.contour(x,x,i(x[:, None], x[None,:]))
                
            
            if nd == 3:
                """
                cube, cm, cmax, cmin = get_cubefile(i)
                v = py3Dmol.view()
                #cm = cube.mean()
                offs = cmax*.05
                bins = np.linspace(cm-offs,cm+offs, 2)

                for i in range(len(bins)):

                    di = int((255*i/len(bins)))
                    
                    v.addVolumetricData(cube, "cube", {'isoval':bins[i], 'color': '#%02x%02x%02x' % (255 - di, di, di), 'opacity': 1.0})
                v.zoomTo()
                v.show()
                """

                import k3d
                import SimpleITK as sitk

                #psi = bk.basisbank.get_hydrogen_function(5,2,2)
                #psi = bk.basisbank.get_gto(4,2,0)
                x = np.linspace(-1,1,100)*80
                img = i(x[None,None,:], x[None,:,None], x[:,None,None])

                #Nc = 3

                #colormap = interp1d(np.linspace(0,1,Nc), np.random.uniform(0,1,(3, Nc)))
                #embryo = k3d.volume(img.astype(np.float32), 
                #                    color_map=np.array(k3d.basic_color_maps.BlackBodyRadiation, dtype=np.float32),
                #                    opacity_function = np.linspace(0,1,30)[::-1]**.1)

                orb_pos = k3d.volume(img.astype(np.float32), 
                                    color_map=np.array(k3d.basic_color_maps.Gold, dtype=np.float32),
                                    opacity_function = np.linspace(0,1,30)[::-1]**.2)

                orb_neg = k3d.volume(-1*img.astype(np.float32), 
                                    color_map=np.array(k3d.basic_color_maps.Blues, dtype=np.float32),
                                    opacity_function = np.linspace(0,1,30)[::-1]**.2)
                plot = k3d.plot()
                plot += orb_pos
                plot += orb_neg
                plot.display()






    if mpfig:
        plt.grid()        
        if nd == 1:
            minvx = min(minvx_vals)
            maxvx = max(maxvx_vals)
            # "mean(+/-)4*sig" determines x-axis length, so higher sig => more points needed
            # A minimum of 200 points are plotted. Necessary in case of small sig
            Nx = max([200, int(500*max(sigs))])       
            x = np.linspace(minvx, maxvx, Nx)
            for i in list(p):  
                mvy_vals.append(abs(max(i(x))))
                mvy_vals.append(abs(min(i(x))))
                plt.plot(x,i(x), label=i.__name__)    
            mvy = max(mvy_vals)
            
        if nd == 2:
            maxvx = max(x)
            mvy = maxvx

        plt.ylim(-1.1*mvy, 1.1*mvy)
        plt.legend()
        plt.show()
        

    


def view(*p, t=None):
    """
    all-purpose vector visualization
    
    Example usage to show the vectors as an image

    a = ket( ... ) 
    b = ket( ... )

    plot(a, b)
    """
    plt.figure(figsize=(6,6))
    try:
        maxvx = 8
        Nx = 200
        x = np.linspace(-maxvx, maxvx, Nx)
        Z = np.zeros((Nx, Nx, 3), dtype = float)
        colors = np.random.uniform(0,1,(len(list(p)), 3))
        
        for i in list(p):
            try:
                plt.contour(x,x,i(x[:, None], x[None,:]))
            except:
                plt.plot(x,i(x) , label=i.__name__)
        plt.grid()
        plt.legend()
        #plt.show()
            

    except:
        maxvx = 1
        #plt.figure(figsize = (6,6))
        for i in list(p):
            vec_R2 = i.coefficients[0]*i.basis[0] + i.coefficients[1]*i.basis[1]
            plt.plot([0, vec_R2[0]], [0, vec_R2[1]], "-")
            
            plt.plot([vec_R2[0]], [vec_R2[1]], "o", color = (0,0,0))
            plt.text(vec_R2[0]+.2, vec_R2[1], "%s" % i.__name__)

            maxvx = max( maxvx, max(vec_R2[1], vec_R2[0]) ) 
            
            
        plt.grid()
        plt.xlim(-1.1*maxvx, 1.1*maxvx)
        plt.ylim(-1.1*maxvx, 1.1*maxvx)
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
                    
                    #get_solid_harmonic_gaussian(a,l,m, position = [0,0,0])
                    basis.append( bf )

        
    return basis
    
    





                    
                    

class basisfunction:
    """
    # A general class for a basis function in $\mathbb{R}^n$
    
    ## Keyword arguments:

    | Argument      | Description |
    | ----------- | ----------- |
    | sympy_expression      | A sympy expression       |
    | position   | assumed center of basis function (defaults to $\mathbf{0}$ )        |
    | name   | (unused)        |
    | domain   |if None, the domain is R^n, if [ [x0, x1], [ y0, y1], ... ] , the domain is finite      |


    ## Methods

    | Method      | Description |
    | ----------- | ----------- |
    | normalize      | Perform numerical normalization of self       |
    | estimate_decay   | Estimate decay of self, used for importance sampling (currently inactive)        |
    | get_domain(other)   | Returns the intersecting domain of two basis functions        |


    ## Example usage:

    ```
    x = sympy.Symbol("x")
    x2 = basisfunction(x**2)
    x2.normalize()
    ```

        
    
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
        """
        Set normalization factor $N$ of self ($\chi$) so that $\langle \chi \\vert \chi \\rangle = 1$.
        """
        s_12 = inner_product(self, self)
        self.normalization = s_12**-.5


    def locate(self):
        """
        Locate and determine spread of self
        """
        self.position, self.decay = locate(self.ket_sympy_expression)
    
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
        """
        Evaluate function in coordinates ```*r``` (arbitrary dimensions).

        ## Returns
        The basisfunction $\chi$ evaluated in the coordinates provided in the array(s) ```*r```:
        $\int_{\mathbb{R}^n} \delta(\mathbf{r} - \mathbf{r'}) \chi(\mathbf{r'}) d\mathbf{r'}$
        """

        return self.normalization*self.ket_numeric_expression(*r) 
    
    
    
    def __mul__(self, other):
        """
        Returns a basisfunction $\chi_{a*b}(\mathbf{r})$, where
        $\chi_{a*b}(\mathbf{r}) = \chi_a(\mathbf{r}) \chi_b(\mathbf{r})$
        """
        return basisfunction(self.ket_sympy_expression * other.ket_sympy_expression, 
                   position = .5*(self.position + other.position),
                   domain = self.get_domain(other), name = self.__name__+other.__name__)
    
    def __rmul__(self, other):
        return basisfunction(self.ket_sympy_expression * other.ket_sympy_expression, 
                   position = .5*(self.position + other.position),
                   domain = self.get_domain(other), name = self.__name__+other.__name__)
    
        
    def __add__(self, other):
        """
        Returns a basisfunction  $\chi_{a+b}(\mathbf{r})$, where
        $\chi_{a+b}(\mathbf{r}) = \chi_a(\mathbf{r}) + \chi_b(\mathbf{r})$
        """
        return basisfunction(self.ket_sympy_expression + other.ket_sympy_expression, 
                   position = .5*(self.position + other.position),
                   domain = self.get_domain(other))
    
    def __sub__(self, other):
        """
        Returns a basisfunction  $\chi_{a-b}(\mathbf{r})$, where
        $\chi_{a-b}(\mathbf{r}) = \chi_a(\mathbf{r}) - \chi_b(\mathbf{r})$
        """
        return basisfunction(self.ket_sympy_expression - other.ket_sympy_expression, 
                   position = .5*(self.position + other.position),
                   domain = self.get_domain(other))
    
    
    def _repr_html_(self):
        """
        Returns a latex-formatted string to display the mathematical expression of the basisfunction. 
        """
        return "$ %s $" % sp.latex(self.ket_sympy_expression)
    
    
#def get_solid_harmonic_gaussian(a,l,m, position = [0,0,0]):
#    return basisfunction(solid_harmonics.get_Nao(a,l,m), position = position)                

class operator_expression(object):
    """
    # A class for algebraic operator manipulations

    instantiate with a list of list of operators

    ## Example

    ```operator([[a, b], [c,d]], [1,2]]) = 1*ab + 2*cd ```

    """
    def __init__(self, ops, coefficients = None):
        self.ops = ops
        if issubclass(type(ops),operator):
            self.ops = [[ops]]

        self.coefficients = coefficients
        if coefficients is None:
            self.coefficients = np.ones(len(self.ops))
    
    def __mul__(self, other):
        """
        # Operator multiplication
        """
        if type(other) is operator:
            new_ops = []
            for i in self.ops:
                for j in other.ops:
                    new_ops.append(i+j)
            return operator(new_ops).flatten()
        else:
            return self.apply(other)
    
    def __add__(self, other):
        """
        # Operator addition
        """
        new_ops = self.ops + other.ops
        new_coeffs = self.coefficients + other.coefficients
        return operator_expression(new_ops, new_coeffs).flatten()

    def __sub__(self, other):
        """
        # Operator subtraction
        """
        new_ops = self.ops + other.ops
        new_coeffs = self.coefficients + [-1*i for i in other.coefficients]
        return operator_expression(new_ops, new_coeffs).flatten()
    
    def flatten(self):
        """
        # Remove redundant terms
        """
        new_ops = []
        new_coeffs = []
        found = []
        for i in range(len(self.ops)):
            if i not in found:
                new_ops.append(self.ops[i])
                new_coeffs.append(1)
                for j in range(i+1, len(self.ops)):
                    if self.ops[i]==self.ops[j]:
                        print("flatten:", i,j, self.ops[i], self.ops[j])
                        #self.coefficients[i] += 1
                        found.append(j)
                        new_coeffs[-1] += self.coefficients[j]

        return operator_expression(new_ops, new_coeffs)
    
    def apply(self, other_ket):
        """
        # Apply operator to ket

        $\hat{\Omega} \vert a \rangle =  \vert a' \rangle $

        ## Returns

        A new ket

        """
        ret = 0
        for i in range(len(self.ops)):
            ret_term = other_ket*1
            for j in range(len(self.ops[i])):
                ret_term = self.ops[i][-j]*ret_term
            if i==0:
                ret = ret_term
            else:
                ret = ret + ret_term
        return ret
    
    def _repr_html_(self):
        """
        Returns a latex-formatted string to display the mathematical expression of the operator. 
        """
        ret = ""
        for i in range(len(self.ops)):
            
            if np.abs(self.coefficients[i]) == 1:
                if self.coefficients[i]>0:
                    ret += "+" 
                else:
                    ret += "-"
            else:
                if self.coefficients[i]>0:
                    ret += "+ %.2f" % self.coefficients[i]
                else:
                    ret += "%.2f" % self.coefficients[i]
            for j in range(len(self.ops[i])):
                ret += "$\\big{(}$" + self.ops[i][j]._repr_html_() + "$\\big{)}$"
                
                
        return ret

class operator(object):
    """
    Parent class for operators
    """
    def __init__(self):
        pass

class sympy_operator_action:
    def __init__(self, sympy_expression):
        self.sympy_expression = sympy_expression
    
    def __mul__(self, other):
        assert(type(other) is basisfunction), "cannot operate on %s" %type(other)
        bs = basisfunction(self.sympy_expression*other.ket_sympy_expression)
        bs.position = other.position
        return ket( bs) 

class sympy_operator(operator):
    def __init__(self, sympy_expression):
        self.sympy_expression = sympy_expression
    
    def __mul__(self, other):
        #assert(type(other) is bk.core.basisfunction), "cannot operate on %s" %type(other)
        
        return self.sympy_expression*other 
        
class translation(operator):
    def __init__(self, translation_vector):
        self.translation_vector = np.array(translation_vector)
        
    def __mul__(self, other):
        #assert(type(other) is basisfunction), "cannot translate %s" %type(other)
        new_expression = translate_sympy_expression(other.get_ket_sympy_expression(), self.translation_vector)
        #bs = basisfunction(new_expression)
        #if other.position is not None:
        #    bs.position = other.position + self.translation_vector
        return ket( new_expression )
    
    
class differential(operator):
    def __init__(self, order, variables = None):
        self.order = order
        self.variables = variables
        
    def __mul__(self, other):
        #assert(type(other) is basisfunction), "cannot differentiate %s" %type(other)
        
        new_expression = 0
        if self.variables is None:
            symbols = np.array(list(other.get_ket_sympy_expression().free_symbols))
            l_symbols = np.argsort([i.name for i in symbols])
            symbols = symbols[l_symbols]
            
            for i in range(len(symbols)):
                new_expression += sp.diff(other.get_ket_sympy_expression(), symbols[i], self.order[i])
        else:
            for i in range(len(self.variables)):
                new_expression += sp.diff(other.get_ket_sympy_expression(), self.variables[i], self.order[i])
        bs = basisfunction(new_expression)
        #bs.position = other.position
        
        return ket( new_expression)
        
    

    

        

    




def translate_sympy_expression(sympy_expression, translation_vector):
    symbols = np.array(list(sympy_expression.free_symbols))
    l_symbols = np.argsort([i.name for i in symbols])
    shifts = symbols[l_symbols]

    assert(len(shifts)==len(translation_vector)), "Incorrect length of translation vector"

    return_expression = sympy_expression*1

    for i in range(len(shifts)):
        if np.abs(translation_vector[i])>=20.0:
            return_expression = return_expression.subs(shifts[i], shifts[i]-sp.UnevaluatedExpr(translation_vector[i]))
        else:
            return_expression = return_expression.subs(shifts[i], shifts[i]-translation_vector[i])

    return return_expression




# Operators
class kinetic_operator(operator):
    def __init__(self, p = None):
        self.p = p
        if p is not None:
            self.variables = get_default_variables(p)
    
    
    def __mul__(self, other):
        if self.p is None:
            self.variables = other.basis[0].ket_sympy_expression.free_symbols
        
        #ret = 0
        new_coefficients = other.coefficients
        new_basis = []
        for i in other.basis:
            new_basis_ = 0
            for j in self.variables:
                new_basis_ += sp.diff(i.ket_sympy_expression,j, 2)
            new_basis.append(basisfunction(new_basis_))
            new_basis[-1].position = i.position
        return ket([-.5*i for i in new_coefficients], basis = new_basis)

    def _repr_html_(self):
        return "$ -\\frac{1}{2} \\nabla^2 $" 

def get_translation_operator(pos):
    return operator_expression(translation(pos))# , special_operator = True)

def get_sympy_operator(sympy_expression):
    return operator_expression(sympy_expression)

def get_differential_operator(order):
    return operator_expression(differential(order)) #,special_operator = True)

def get_onebody_coulomb_operator(position = np.array([0,0,0.0]), Z = 1.0, p = None, variables = None):
    return operator_expression(onebody_coulomb_operator(position, Z = Z, p = p, variables = variables))
                    
def get_twobody_coulomb_operator(p1=0,p2=1):
    return operator_expression(twobody_coulomb_operator(p1,p2))

def get_kinetic_operator(p = None):
    return operator_expression(kinetic_operator(p = p))

def get_default_variables(p, n = 3):
    variables = []
    for i in range(n):
        variables.append(sp.Symbol("x_{%i; %i}" % (p, i)))
    return variables



class onebody_coulomb_operator(operator):
    def __init__(self, position = np.array([0.0,0.0,0.0]), Z = 1.0, p = None, variables = None):
        self.position = position
        self.Z = Z
        self.p = p
        self.variables = variables
        if p is not None:
            self.variables = get_default_variables(self.p, len(position))
            r = 0
            for j in range(len(self.variables)):
                r += (self.variables[j]-self.position[j])**2
            self.r_inv = r**-.5
        
    
    
    def __mul__(self, other, r = None):
        variables = self.variables
        if self.variables is None:
            #variables = other.basis[0].ket_sympy_expression.free_symbols
            symbols = np.array(list(other.basis[0].ket_sympy_expression.free_symbols))
            
            """
            symbols_particles = [int(x.name.split("{")[1].split(";")[0]) for x in symbols]
            particle_symbols = []
            for i in range(len(symbols)):
                if symbols_particles[i] == self.p:
                    particle_symbols.append(symbols[i])
            print("part_s:", particle_symbols)
            symbols = particle_symbols        
            """
            l_symbols = np.argsort([i.name for i in symbols])
            variables = symbols[l_symbols]

        

        
            r = 0
            for j in range(len(variables)):
                r += (variables[j]-self.position[j])**2
            self.r_inv = r**-.5
        
        new_coefficients = other.coefficients
        new_basis = []
        for i in other.basis:
            new_basis.append(basisfunction(self.r_inv*i.ket_sympy_expression))#, position = i.position+self.position))
        return ket([-self.Z*i for i in new_coefficients], basis = new_basis)

    def _repr_html_(self):
        if self.position is None:
            return "$ -\\frac{1}{\\mathbf{r}} $"   
        else:
            return "$ -\\frac{1}{\\vert \\mathbf{r} - (%f, %f, %f) \\vert }$" % (self.position[0], self.position[1], self.position[2]) 

def twobody_denominator(p1, p2, ndim):
    v_1 = get_default_variables(p1, ndim)
    v_2 = get_default_variables(p2, ndim)
    ret = 0
    for i in range(ndim):
        ret += (v_1[i] - v_2[i])**2
    return ret**.5
    

class twobody_coulomb_operator(operator):
    def __init__(self, p1 = 0, p2 = 1, ndim = 3):
        self.p1 = p1
        self.p2 = p2
        self.ndim = ndim

    
    
    def __mul__(self, other):
        #vid = other.variable_identities 
        #if vid is None:
        #    assert(False), "unable to determine variables of ket"
        new_basis = 0
        for i in range(len(other.basis)):
            #new_basis += other.coefficients[i]*apply_twobody_operator(other.basis[i].ket_sympy_expression, self.p1, self.p2)
            new_basis += other.coefficients[i]*other.basis[i].ket_sympy_expression/twobody_denominator(self.p1, self.p2, self.ndim)
        
        return ket(new_basis)

    def _repr_html_(self):
        return "$ -\\frac{1}{\\vert \\mathbf{r}_1 - \\mathbf{r}_2 \\vert} $" 

class twobody_coulomb_operator_older(operator):
    def __init__(self):
        pass
    
    
    def __mul__(self, other):
        vid = other.variable_identities 
        if vid is None:
            assert(False), "unable to determine variables of ket"
        new_basis = []
        for i in range(len(other.basis)):
            fs1, fs2 = vid[i]
            denom = 0
            for k,l in zip(list(fs1), list(fs2)):
                denom += (k - l)**2

            new_basis.append( ket(other.basis[i].ket_sympy_expression/np.sqrt(denom) ) )
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

    ## Keyword arguments:

    | Method      | Description |
    | ----------- | ----------- |
    | generic_input      | if list or numpy.ndarray:  if basis is None, returns a cartesian vector else, assumes input to contain coefficients. If sympy expression, returns ket([1], basis = [basisfunction(generic_input)])    |
    | name   | a string, used for labelling and plotting, visual aids        |
    | basis   | a list of basisfunctions       |
    | position   | assumed centre of function $\langle  \\vert \hat{\mathbf{r}} \\vert \\rangle$.     |
    | energy   | if this is an eigenstate of a Hamiltonian, it's eigenvalue may be fixed at initialization     |

    ## Operations  
    For kets B and A and scalar c
    
    | Operation      | Description |
    | ----------- | ----------- |
    | A + B | addition |
    | A - C | subtraction |
    |  A * c   |  scalar multiplication   |
    |  A / c  |   division by a scalar |
    |  A * B   |  pointwise product   |
    |  A.bra*B |  inner product  |
    |  A.bra@B |  inner product  |
    |  A @ B   |  cartesian product  |
    |  A(x)   |   $\int_R^n \delta(x - x') f(x') dx'$ evaluate function at x  |    

    """
    def __init__(self, generic_input, name = "", basis = None, position = None, energy = None, autoflatten = True):
        """
        ## Initialization of a ket
        




        """
        self.autoflatten = autoflatten
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
        if self.autoflatten:
            ret.flatten()
        ret.__name__ = "%s + %s" % (self.__name__, other.__name__)
        return ret
    
    def __sub__(self, other):
        new_basis = self.basis + other.basis  
        
        new_coefficients = self.coefficients + [-i for i in other.coefficients]
        new_energies = self.energy + other.energy
        ret = ket(new_coefficients, basis = new_basis, energy = new_energies)
        if self.autoflatten:
            ret.flatten()
        ret.__name__ = "%s - %s" % (self.__name__, other.__name__)
        return ret
    
    def __mul__(self, other):
        if type(other) is ket:
            new_basis = []
            new_coefficients = []
            new_energies = []
            for i in range(len(self.basis)):
                for j in range(len(other.basis)):
                    new_basis.append(self.basis[i]*other.basis[j])
                    new_coefficients.append(self.coefficients[i]*other.coefficients[j])
                    new_energies.append(self.energy[i]*other.energy[j]) 



            #return self.__matmul__(other)
            return ket(new_coefficients, basis = new_basis, energy = new_energies)
        else:
            if str(type(other)).split("'")[1].split(".")[0] == "sympy":
                new_basis = []
                new_coefficients = []
                new_energies = []
                for i in range(len(self.basis)):                    
                    new_basis.append(basisfunction(other*self.basis[i].ket_sympy_expression))
                    new_coefficients.append(self.coefficients[i])
                    new_energies.append(self.energy[i]) 
                return ket(new_coefficients, basis = new_basis, energy = new_energies)
            else:
                return ket([other*i for i in self.coefficients], basis = self.basis, energy=self.energy)

    def __rmul__(self, other):
        if str(type(other)).split("'")[1].split(".")[0] == "sympy":
            new_basis = []
            new_coefficients = []
            new_energies = []
            for i in range(len(self.basis)):                    
                new_basis.append(basisfunction(self.basis[i].ket_sympy_expression*other))
                new_coefficients.append(self.coefficients[i])
                new_energies.append(self.energy[i]) 
            return ket(new_coefficients, basis = new_basis, energy = new_energies)
        else:
            return ket([other*i for i in self.coefficients], basis = self.basis)
    
    def __truediv__(self, other):
        assert(type(other) in [float, int]), "Divisor must be float or int"
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
                                #bij, sep = split_variables(self.basis[i].ket_sympy_expression, other.basis[j].ket_sympy_expression)
                                bij, sep = relabel_direct(self.basis[i].ket_sympy_expression, other.basis[j].ket_sympy_expression)
                                #bij = ket(bij)
                                bij = basisfunction(bij) #, position = other.basis[j].position)
                                bij.position = np.append(self.basis[i].position, other.basis[j].position)
                                new_basis.append(bij)
                                new_coefficients.append(self.coefficients[i]*other.coefficients[j])
                                variable_identities.append(sep)


                        ret = ket(new_coefficients, basis = new_basis)
                        if self.autoflatten:
                            ret.flatten()
                        ret.__name__ = self.__name__ + other.__name__
                        ret.variable_identities = variable_identities
                        return ret

    def set_position(self, position):
        for i in range(len(self.basis)):
            pass
            
    
    
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

    def get_ccode(self):
        """
        Generate a WebGL-shader code snippet
        for Evince rendering (experimental)
        """
        code_snippets = []
        for i in range(len(self.coefficients)):
            if type(self.basis[i]) in [basisfunction, ket]:
                # get term (with energy self.energy[i])
                ret_i = self.coefficients[i]*self.basis[i].ket_sympy_expression 

                

                # replace standard symbols with WebGL specific variables
                symbol_list = get_ordered_symbols(ret_i)
                for i in range(len(symbol_list)):
                    #ret_i = ret_i.replace(symbol_list[i], sp.UnevaluatedExpr(sp.symbols("tex[%i]" % i)))
                    ret_i = ret_i.replace(symbol_list[i], sp.symbols("tex[%i]" % i))


                # substitute r^2 and pi with WebGL-friendly expressions
                simp_ret = ret_i.subs(get_r2_sp(ret_i), sp.symbols("q")).simplify().subs(sp.pi, np.pi)

                # replace all integers (up to 20) with floats
                #for j in range(20):
                #    simp_ret.subs(sp.Integer(j), sp.Float(j))

                # generate C code
                shadercode_i = sp.ccode(simp_ret)

                # workaround (for now, fix later)
                for j in range(20):
                    shadercode_i = shadercode_i.replace(" %i)" %j, " %i.0)" %j)
                    shadercode_i = shadercode_i.replace(" %i," %j, " %i.0," %j)

                # append vector component to code snippets
                code_snippets.append(shadercode_i)
            
        return code_snippets 
        
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

        #Ri = *np.array([R[i] - self.position[i] for i in range(len(self.position))])

        #Ri = np.array([R[i] - self.position[i] for i in range(len(self.position))], dtype = object)
        
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


    #def run(self, x = 8*np.linspace(-1,1,100), t = 0, dt = 0.001):
    #    anim_s = anim.system(self, x, t, dt)
    #    anim_s.run()

    """
    Measurement
    """
    def measure(self, observable = None, repetitions = 1):
        """
        Make a mesaurement of the observable (hermitian operator)

        Measures by default the continuous distribution as defined by self.bra*self
        """
        if observable is None:
            # Measure position
            P = self.get_bra_sympy_expression()*self.get_ket_sympy_expression()
            symbols = get_ordered_symbols(P)
            P = sp.lambdify(symbols, P, "numpy")
            nd = len(symbols)
            sig = .1 #variance of initial distribution


            r = np.random.multivariate_normal(np.zeros(nd), sig*np.eye(nd), repetitions ).T
            # Metropolis-Hastings 
            for i in range(1000):
                dr = np.random.multivariate_normal(np.zeros(nd), 0.01*sig*np.eye(nd), repetitions).T
                #print(dr.shape, r.shape,P(*(r+dr))/P(*r))

                accept = P(*(r+dr))/P(*r) > np.random.uniform(0,1,repetitions)
                #print(accept)
                r[:,accept] += dr[:,accept]
            return r
        else:
            #assert(False), "Arbitrary measurements not yet implemented"

            # get coefficients 
            P = np.zeros(len(observable.eigenstates), dtype = float)
            for i in range(len(observable.eigenstates)):
                P[i] = (observable.eigenstates[i].bra@self)**2
                

            distribution = discrete_metropolis_hastings(P, n_samples = repetitions)

            return observable.eigenvalues[distribution]

    def view(self, web = False, squared = False, n_concentric = 100):
        """
        Create an Evince viewer (using ipywidgets) 

        """
        nd = len(self.bra_sympy_expression.free_symbols)
        blend_factor = 1.0
        if nd>2:
            blend_factor = 0.1
        if web:
            self.m = ev.BraketView(self, additive = False, bg_color = [1.0, 1.0, 1.0], blender='    gl_FragColor = vec4(.9*csR - csI,  .9*abs(csR) + csI, -1.0*csR - csI, %f)' % blend_factor, squared = squared, n_concentric=n_concentric) 

            #self.m = ev.BraketView(self, bg_color = [1.0, 1.0, 1.0], additive = False, blender = '    gl_FragColor = gl_FragColor + vec4(.2*csR, .1*csR + .1*csI, -.1*csR, .1)', squared = squared)
        else:
            self.m = ev.BraketView(self, additive = True, bg_color = [0.0,0.0,0.0], blender='    gl_FragColor = vec4(.9*csR ,  csI, -1.0*csR, %f)' % blend_factor, squared = squared, n_concentric=n_concentric) 

            #self.m = ev.BraketView(self, additive = True, squared = squared)
        return self.m

def discrete_metropolis_hastings(P, n_samples = 10000, n_iterations = 100000, stepsize = None, T = 0.001):
    """
    Perform a random walk in the discrete distribution P (array)
    """
    #ensure normality
    n = np.sum(P)
    
    Px = interp1d(np.linspace(0,1,len(P)), P/n)
    
    x = np.random.uniform(0,1,n_samples)
    
    if stepsize is None:
        #set stepsize proportional to discretization
        
        stepsize = .5*len(P)**-1
    
    for i in range(n_iterations):
        dx = np.random.normal(0,stepsize, n_samples)
        xdx = x + dx

        # periodic boundaries
        xdx[xdx<0] += 1
        xdx[xdx>1] -= 1
        
        #if Px(xdx)>Px(x):
            
        
        accept = np.exp(-(Px(xdx)-Px(x))/T) < np.random.uniform(0,1,n_samples)
        
        x[accept] = xdx[accept]
        
    return np.array(x*len(P), dtype = int)

def metropolis_hastings(f, N, x0, a):
    """
    Metropolis-Hastings random walk in the function f
    """
    x = np.random.multivariate_normal(x0, a, N)

    
    for i in range(1000):
        dx = np.random.multivariate_normal(x0, a*0.01, N)
        
        accept = f(x+dx)/f(x) > np.random.uniform(0,1,N)
        x[accept] += dx[accept]
    return x

def get_particles_in_expression(s):
    symbols = get_ordered_symbols(s)
    particles = []
    for i in symbols:
        particles.append( int(i.name.split("{")[1].split(";")[0] ) )
    particles = np.array(particles)
    return np.unique(particles)

def get_ordered_symbols(sympy_expression):
    symbols = np.array(list(sympy_expression.free_symbols))
    l_symbols = np.argsort([i.name for i in symbols])
    return symbols[l_symbols]

def substitute_sequence(s, var_i, var_f):
    for i in range(len(var_i)):
        s = s.subs(var_i[i], var_f[i])

    return s

def relabel_direct(s1,s2):
    p1 = get_particles_in_expression(s1)
    p_max = p1.max() + 1
    p2 = get_particles_in_expression(s2)
    for i in p2:
        if i in p1:
            
            s2 = substitute_sequence(s2, get_default_variables(i), get_default_variables(p_max))
            p_max += 1
    return s1*s2, get_ordered_symbols(s1*s2)





def split_variables(s1,s2):
    """
    make a product where 
    """

    # gather particles in first symbols
    s1s = get_ordered_symbols(s1)
    for i in range(len(s1s)):
        s1 = s1.subs(s1s[i], sp.Symbol("x_{0; %i}" % i))

    s2s = get_ordered_symbols(s2)
    for i in range(len(s2s)):
        s2 = s2.subs(s2s[i], sp.Symbol("x_{1; %i}" % i))
        
    return s1*s2, get_ordered_symbols(s1*s2)
        
def get_r2(p):
    """
    extract the r^2 equivalent term from the ket p
    """
    r_it = list(p.ket_sympy_expression.free_symbols)
    r2 = 0
    for i in r_it:
        r2 += i**2.0
    return r2

def get_r2_sp(p):
    """
    extract the r^2 equivalent term from the sympy expression p
    """
    r_it = get_ordered_symbols(p)
    r2 = 0
    for i in r_it:
        r2 += i**2.0
    return r2

def parse_symbol(x):
    """
    Parse a symbol of the form 
    
    x_{i;j}
    
    Return a list
    
    [i,j]
    """
    strspl = str(x).split("{")[1].split("}")[0].split(";")
    return [int(i) for i in strspl]

def map_expression(sympy_expression, x1=0, x2=1):
    """
    Map out the free symbols of sympy_expressions
    in order to determine 
    
    z[p, x] 
    
    where p = [0,1] is particle x1 and x2, while
    x is their cartesian component
    """
    map_ = {x1:0, x2:1}
    s = sympy_expression.free_symbols
    n = int(len(s)/2)
    z = np.zeros((2, n), dtype = object)
    for i in s:
        j,k = parse_symbol(i) #particle, coordinate
        z[map_[j], k] = i
    return z, n

def get_twobody_denominator(sympy_expression, p1, p2):
    """
    For a sympy_expression of arbitrary dimensionality,
    generate the coulomb operator
    
    1/sqrt( r_{p1, p2} )
    
    assuming that the symbols are of the form "x_{pn, x_i}"
    where x_i is the cartesian vector component
    """
    mex, n = map_expression(sympy_expression, p1, p2)

    denom = 0
    for i in range(n):
        denom += (mex[0,i] - mex[1,i])**2
        
    return sp.sqrt(denom)




def apply_twobody_operator(sympy_expression, p1, p2):
    """
    Generate the sympy expression 
    
    sympy_expression / | x_p1 - x_p2 |
    """
    return sympy_expression/get_twobody_denominator(sympy_expression, p1, p2)



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
def inner_product(b1, b2, operator = None, n_samples = int(1e6), grid = 101, sigma = None):
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
    ri = b1.position*0
    rj = b2.position*0

    integrand = lambda *R, \
                       f1 = b1.bra_numeric_expression, \
                       f2 = b2.ket_numeric_expression, \
                       ri = ri, rj = rj:  \
                       f1(*np.array([R[i] - ri[i] for i in range(len(ri))]))*f2(*np.array([R[i] - rj[i] for i in range(len(rj))]))
                       #f1(*np.array([R[i] - ri[i] for i in range(len(ri))]))*f2(*np.array([R[i] - rj[i] for i in range(len(rj))]))


    variables_b1 = b1.bra_sympy_expression.free_symbols
    variables_b2 = b2.ket_sympy_expression.free_symbols
    if len(variables_b1) == 1 and len(variables_b2) == 1:
        return integrate.quad(integrand, -np.inf,np.inf)[0]
    else:
        ai,aj = b1.decay, b2.decay
        ri,rj = b1.position, b2.position

        R = (ai*ri + aj*rj)/(ai+aj)
        if sigma is None:
            sigma = .5*(ai + aj)
        #print("R, sigma:", R, sigma)
    
        return onebody(integrand, np.ones(len(R))*sigma, R, n_samples) #, control_variate = "spline", grid = grid) 
    """
    else:

        R, sigma = locate(b1.bra_sympy_expression*b2.ket_sympy_expression)

        return onebody(integrand, np.ones(len(R))*sigma, R, n_samples) #, control_variate = "spline", grid = grid) 

    """

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

def sphere_distribution(N_samples, scale = 1):
    theta = np.random.uniform(0,2*np.pi, N_samples)
    phi = np.arccos(np.random.uniform(-1,1, N_samples))
    r = np.random.exponential(scale, N_samples)
    
    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)
    return np.array([x,y,z])

def sphere_pdf(x, scale =1):
    r = np.sqrt(np.sum(x**2, axis= 0))
    return np.exp(-x/scale)/scale #/scale


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

    
    #R = np.random.multivariate_normal(loc, np.eye(len(loc))*sigma, n_samples)
    #R = np.random.Generator.multivariate_normal(loc, np.eye(len(loc))*sigma, size=n_samples)

    #sig = np.eye(len(loc))*sigma
    sig = np.diag(sigma)
    R = np.random.default_rng().multivariate_normal(loc, sig, n_samples)
    P = multivariate_normal(mean=loc, cov=sig).pdf(R)

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

