from scipy.integrate import quad
from numpy.polynomial.chebyshev import Chebyshev
import numpy as np
import quimb.tensor as qtn

# Define the coefficients
A = np.array([1, 2])  # a1 and a2
B = np.array([4, 3])  # b1 and b2


# Define the Chebyshev polynomials g0(x) = T_0(x) and g1(x) = T_1(x)
g0 = Chebyshev.basis(1)  # T_0(x), which is 1
g1 = Chebyshev.basis(2)  # T_1(x), which is x

# Define the functions f(x) and h(x) as linear combinations of g0 and g1
def f(x):
    return A[0] * g0(x) + A[1] * g1(x)

def h(x):
    return B[0] * g0(x) + B[1] * g1(x)

# Define the product of f(x) and h(x) with the weighting factor
def weighted_product(x):
    weight = 2 / (np.pi * np.sqrt(1 - x**2))
    return weight * f(x) * h(x)

# Perform numerical integration of the weighted product from -1 to 1
integral, error = quad(func=weighted_product, a=-1, b=1, epsabs=1e-12, epsrel=1e-12)
print(f"Numerical integral of the weighted product of f(x) and h(x) from -1 to 1: {integral}")

print(A @ B.T)



