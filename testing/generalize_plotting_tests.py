import braketlab as bk 
import sympy as sp
import numpy as np 

x, y = bk.get_default_variables(1,2)
x = sp.symbols('x')
y = sp.symbols('y')

a = bk.ket( [ 2,4], name = "a")
b = bk.ket( [-2,3], name = "b")

bk.show_old(a, b, a-b, a+b)

N = 50
sigma = 20
k = 5

c = bk.ket((N*sp.exp(-1/2*sigma*x**2)), name = "c")
bk.show(c)
d = bk.ket( x*sp.exp(-.2*(x**2 + y**2) ), name = "d")
bk.show(d)

e = bk.ket((2*N*sp.exp(-1/2*sigma*x**2)), name = "e")
bk.show(c, e)

