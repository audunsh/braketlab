import sympy as sp
import numpy as np

import braketlab.solid_harmonics as sh
import braketlab.real_solid_harmonics as rsh
import braketlab.hydrogen as hy
import braketlab.harmonic_oscillator as ho

from braketlab.core import ket, get_ordered_symbols, get_default_variables

def extract_pyscf_basis(mo):
    """
    Get a braketlab basis from a pyscf mol object

    ## Arguments 

    |mo | pySCF molecule object |

    ## Returns

    A list containing braketlab basis functions


    ## Example usage

    ```python
    from pyscf import gto
    mol = gto.Mole()
    mol.build(atom = '''O 0 0 0; H  0 1 0; H 0 0 1''',basis = 'sto-3g')
    basis = extract_pyscf_basis(mol)
    ```

    """
    contracted_aos = []
    for contr in range(mo.nbas):
        l = mo.bas_angular(contr)
        contr_coeff = mo.bas_ctr_coeff(contr)
        exponent = mo.bas_exp(contr)
        #position = mo.
        
        if l == 1:
            
            for m in [1, -1,0]:
                #for b in range(len(contr_coeff[a]))
                #a = 0
                for b in range(len(contr_coeff[0])):
                    a = 0
                    #print(l,m,"    ", exponent[a], contr_coeff[a][b], np.array(mo.bas_coord(contr)))
                    chi_lm = contr_coeff[a][b]*bk.basisbank.get_gto(exponent[0], l, m, position = np.array(mo.bas_coord(contr)))
                    for  a in range(1, len(exponent)):

                        #print(l,m,"    ", exponent[a], "     ", contr_coeff[a][0])
                        chi_lm += contr_coeff[a][b]*bk.basisbank.get_gto(exponent[a], l, m,  position = np.array(mo.bas_coord(contr)))
                    contracted_aos.append(chi_lm)
        else:
            for m in range(-l, l+1):
                for b in range(len(contr_coeff[0])):
                    a = 0
                    #print(l,m,"    ", exponent[a], contr_coeff[a][0], mo.bas_coord(contr))
                    chi_lm = contr_coeff[a][b]*bk.basisbank.get_gto(exponent[0],l,m, position = np.array(mo.bas_coord(contr)))
                    for  a in range(1, len(exponent)):
                        #print(l,m,"    ", exponent[a], "     ", contr_coeff[a][0])
                        chi_lm += contr_coeff[a][b]*bk.basisbank.get_gto(exponent[a], l,m, position = np.array(mo.bas_coord(contr)))

                    contracted_aos.append(chi_lm)
    return contracted_aos


def get_default_variables(p, n = 3):
    variables = []
    for i in range(n):
        variables.append(sp.Symbol("x_{%i; %i}" % (p, i)))
    return variables



def get_hydrogen_function(n,l,m, position = np.array([0,0,0])):
    """
    Returns a ket containing the hydrogen eigenfunction with quantum numbers n,l,m
    located at position
    """
    psi = hy.hydrogen_function(n,l,m)
    #vars = list(psi.free_symbols)
    vars = get_ordered_symbols(psi)
    symbols = get_default_variables(0, len(vars))
    for i in range(len(vars)):
        psi = psi.subs(vars[i], symbols[i])

    



    return ket(psi, name = "%i,%i,%i" % (n,l,m), position = position)

def get_harmonic_oscillator_function(n, omega = 1, position = 0):
    """
    Returns a ket containing the harmonic oscillator energy eigenfunction with quantum number n
    located at position
    """
    psi = ho.psi_ho(n)
    symbols = np.array(list(psi.free_symbols))
    l_symbols = np.argsort([i.name for i in symbols])
    symbols = symbols[l_symbols]
    #vars = list(psi.free_symbols)
    vars = get_default_variables(0, len(symbols))
    for i in range(len(vars)):
        psi = psi.subs(symbols[i], vars[i])

    return ket(psi, name = "%i" % n, energy = [omega*(.5+n)], position = np.array([position]))

def get_gto(a,l,m, position = np.array([0,0,0])):
    """
    Returns a ket containing the gaussian type primitive orbital with exponent a, 
    and solid harmonic gaussian angular part defined by l and m
    located at position
    """
    psi = rsh.get_gto(a,l,m)

    

    symbols = np.array(list(psi.free_symbols))
    l_symbols = np.argsort([i.name for i in symbols])
    symbols = symbols[l_symbols]
    #vars = list(psi.free_symbols)
    vars = get_default_variables(0, len(symbols))
    for i in range(len(vars)):
        psi = psi.subs(symbols[i], vars[i])
    


    return ket(psi, name = "\chi_{%i,%i}^{%.2f}" % (l,m,a), position = position)

def get_sto(a,w,n,l,m, position = np.array([0,0,0])):
    """
    Returns a ket containing the slater type orbital with exponent a, 
    weight w,
    and solid harmonic gaussian angular part defined by l and m
    located at position
    """
    psi = sh.get_sto(a,w,n,l,m)

    symbols = np.array(list(psi.free_symbols))
    l_symbols = np.argsort([i.name for i in symbols])
    symbols = symbols[l_symbols]
    #vars = list(psi.free_symbols)
    vars = get_default_variables(0, len(symbols))
    for i in range(len(vars)):
        psi = psi.subs(symbols[i], vars[i])

    #vars = list(psi.free_symbols)
    #symbols = bk.get_default_variables(0, len(vars))
    #for i in range(len(vars)):
    #    psi = psi.subs(vars[i], symbols[i])
    return ket(psi, name  = "\chi_{%i,%i}^{%.2f}" % (l,m,a), position = position)

