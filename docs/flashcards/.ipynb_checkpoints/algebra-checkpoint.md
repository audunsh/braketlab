<html>
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js" integrity="sha256-Ae2Vz/4ePdIu6ZyI/5ZGsYnb+m0JlOmKPjt6XZ9JJkA=" crossorigin="anonymous"></script>
</html>

# Algebra

---

Intances of the BraketLab ```ket```-class are able to do most of the things you would expect if them: they can be added and subtracted, multiplied in various ways with scalars and each other, projected into other bases, formed into inner products, acted upon by operators and much more. In addition, they have a ```.view()```, which yields a visualization in the *representation* they were defined. (BraketLab kets are defined in a given representation, not from operators, and as such may have a conceptual difference from the Dirac-kets you find in books).

Let's defined two harmonic oscillator functions in position representation to illustrate the algebra:


```python
import braketlab as bk

p = bk.basisbank.get_harmonic_oscillator_function(1)
q = bk.basisbank.get_harmonic_oscillator_function(2)
```

## Addition and subtraction


```python
(p+q).view(web = True)
```




    BraketView(additive=False, ao=[1], bg_color=[1.0, 1.0, 1.0], fragment_shader='uniform vec3 user_color;\nunifor…




```python
(p-q).view(web = True)
```




    BraketView(additive=False, ao=[1], bg_color=[1.0, 1.0, 1.0], fragment_shader='uniform vec3 user_color;\nunifor…



## Scalar multiplication


```python
(2.0*p).view(web = True)
```




    BraketView(additive=False, ao=[1], bg_color=[1.0, 1.0, 1.0], fragment_shader='uniform vec3 user_color;\nunifor…



## Other products

### Inner product

The inner product can be defined in various ways. In BraketLab it is either the <a href="">dot product</a> for vectors in Euclidean space:

\begin{equation}
\langle p \vert q  \rangle := \sum_{i} p_i q_i,
\end{equation}

or the <a href="https://mathworld.wolfram.com/InnerProduct.html">Hermitian inner product</a> for the vector space of complex functions of arbitrary dimensionality:

\begin{equation}
\langle p \vert q  \rangle := \int_{\mathbb{R}^N} p(\mathbf{x})^* q(\mathbf{x}) d\mathbf{x}.
\end{equation}

You may compute it as follows:*


```python
p.bra@q #should be zero for the orthogonal eigenstates of the Harmonic Oscillator
```




    0.0



**Note** that the inner-product of BraketLab is approximated numerically for complex functions, using quadrature in one dimension and Monte Carlo-integration in higher dimensions. 

### Pointwise product

The <a href="https://en.wikipedia.org/wiki/Pointwise_product">pointwise product</a> for two functions $p$ and $q$ is

\begin{equation}
(p \cdot q)(x) = p(x) \cdot q(x)
\end{equation}

In BraketLab, the corresponding operation is


```python
(p*q).view(web = True)
```




    BraketView(additive=False, ao=[1], bg_color=[1.0, 1.0, 1.0], fragment_shader='uniform vec3 user_color;\nunifor…



### Direct product

For convenience, the direct product in BraketLab is defined to be

\begin{equation}
p(\mathbf{x}) \times q(\mathbf{x}) = p(\mathbf{x}_1) q(\mathbf{x}_2).
\end{equation}

The forking of the variable $\mathbf{x}$ is done to simplify treatment of many-body systems and separable Hamiltonians, which was some of the original focus of BraketLab. It is computed as follows:


```python
(p@q).view()
```




    BraketView(ao=[1], bg_color=[0.0, 0.0, 0.0], fragment_shader='uniform vec3 user_color;\nuniform float time;\n\…



### Outer product

The outer product is special to linear algebra, but is still useful within BraketLab for many reasons. For our purpose, you may view it as

\begin{equation}
\vert p \rangle \langle q \vert 
\end{equation}

and compute it by


```python
P = p@q.bra
P
```




$$\sum_{ij} \vert 1_i \rangle \langle 2_j \vert$$



The *bra* in the expression above is more than simply a complex conjugate ket - it is an instruction to integrate whatever it encounter to its right. This can be understood from the following operations (make sure you agree that the first operation yields a blank window while the second do not:


```python
(P*p).view(web = True)
```




    BraketView(additive=False, ao=[1], bg_color=[1.0, 1.0, 1.0], fragment_shader='uniform vec3 user_color;\nunifor…




```python
(P*q).view(web = True)
```




    BraketView(additive=False, ao=[1], bg_color=[1.0, 1.0, 1.0], fragment_shader='uniform vec3 user_color;\nunifor…




```python

```
