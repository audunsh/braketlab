import braketlab as bk 
import sympy as sp
import numpy as np 
import time



"""
Testing code in tutorial
"""
x,y,z = bk.get_default_variables(1,3)

#Two functions in one plot
a = bk.ket( x*sp.exp(-x**2), name = "a" )
b = bk.ket( sp.exp(-2*x**2), name = "b" )
#bk.show(a,b)

#Two dimensions
a = bk.ket( x*sp.exp(-.2*(x**2 + y**2) ), name = "a") # A 2D ket
#bk.show(a)

#Three dimensions.
#Couldn't test on my machine, but should not be affected by the changes
#a = bk.ket( x*sp.exp(-.01*(x**2 + y**2 + z**2)), name = "a") # A 3D ket
#bk.show(a) #visualize with https://github.com/K3D-tools/K3D-jupyter

#Vectors
a = bk.ket( [ 2,4], name = "a")
b = bk.ket( [-2,3], name = "b")

#bk.show_old(a, b, a-b, a+b)

#Abstract kets
a = bk.ket( y*sp.exp(-2*(x**2 + y**2))/(1+.1*x**2) )
b = bk.ket( .1*(y**2+x**2)*sp.sin(x**2)*sp.exp(-.5*y**2 -.5*x**2))

#bk.show(.1*a+2*b)

#Orthonormality
psi = bk.ket( 2*sp.cos(2*x) * sp.exp(-.2*x**2), name = "$\\psi$" ) # <- some ket in Hilbert space

psi_normalized = (psi.bra@psi)**-.5*psi #normalization

psi_normalized.__name__ = "$\\psi_n$"
#bk.show(psi, psi_normalized)

psi_a = bk.ket( 5*sp.cos(2*x) * sp.exp(-.1*x**2), name = "$\\psi_a$" ) 
psi_b = bk.ket( 5*sp.sin(1*x) * sp.exp(-.1*x**2), name = "$\\psi_b$" ) 

#bk.show(psi_a, psi_b)

#Outer products and operators


psi_a = bk.ket( sp.exp(-.1*x**2), name = "psi_a")
psi_b = bk.ket( x*sp.exp(-.2*x**2), name = "psi_b")
#bk.show(psi_a, psi_b)


ab = psi_a@psi_b
#bk.show(ab)

#Translation operator
psi = bk.ket( sp.exp(-4*(x+3)**2))
T = bk.get_translation_operator(np.array([2.1]))

Tpsi = T*psi
TTpsi = T*Tpsi
TTTpsi = T*TTpsi
TTTTpsi = T*TTTpsi
#bk.show(psi, Tpsi, TTpsi, TTTpsi, TTTTpsi)

#Differential operator
a = bk.ket( sp.exp(-x**2), name = "a(x)")

D = bk.get_differential_operator(order = [1])

Da = D*a
Da.__name__ = "$\\frac{d}{dx} a(x)$"

#bk.show(a, Da)

#Diff. operator in 2D
a = bk.ket( x*sp.exp(-(x**2 + y**2)**.5))

D = bk.get_differential_operator(order = [1,1])

D2a = D*a

#bk.show(a)
#bk.show(D2a)

"""
Testing specific to most recent changes
"""

t0 = time.time_ns()


a = 1
b = 50
c = 1
k = bk.ket(a*sp.exp(-(x-b)**2/2*c**2), name="k")
#k = (k.bra@k)**(-0.5)*k
b = 3
l = bk.ket(a*sp.exp(-(x-b)**2/2*c**2), name="l")
b = 9
m = bk.ket(a*sp.exp(-(x-b)**2/2*c**2), name="m")
b = -5000
n = bk.ket(a*sp.exp(-(x-b)**2/2*c**2), name="n")



bk.show(k, l, m, n)
t1 = time.time_ns()

total = t1-t0
print("time [s] = ",total/1e9)
#Usually takes about 0.5-1 s per function plotted

b = 0
c = 50
o = bk.ket(a*sp.exp(-(x-b)**2/2*c**2), name="o")
#bk.show(o)




