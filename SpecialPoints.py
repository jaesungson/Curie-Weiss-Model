import numpy as np
from scipy.integrate import quad
from scipy import stats
import math
import warnings


# The fourth derivative of the H function
def H4(beta, p, x):
    s = 1 / (1 - x ** 2)
    u = -(8 * x ** 2 * s ** 3 + 2 * s ** 2)
    if p < 4:
        return u
    else:
        return beta * p * (p - 1) * (p - 2) * (p - 3) * x ** (p - 4) + u


def densityF(x, s, t, beta, h, p, m):
    return math.exp(H4(beta, p, m)/24 * x**4 + (s*p*m**(p-1) + t)*x)

def normConstant(s, t, beta, h, p, m):
    result = quad(densityF, -math.inf, math.inf, args=(s, t, beta, h, p, m))
    # abs error grows with value of the integral; so, we are not concerned about it here
    return result[0]


def meanF(s, t, beta, h, p, m):
    c = normConstant(s, t, beta, h, p, m)
    def integrand(x):
        return x * densityF(x, s, t, beta, h, p, m) / c
    result = quad(integrand, -math.inf, math.inf)
    if result[1] < 1e-3:
        return result[0]
    else:
        return warnings.warn("Convergence Error")

def G1(t, beta, h, p, m):
    mean = meanF(0, t, beta, h, p, m)
    c = normConstant(0, 0, beta, h, p, m)
    result = quad(densityF, -math.inf, mean, args=(0, 0, beta, h, p, m))
    if result[1] < 1e-3:
        return result[0]/c
    else:
        return warnings.warn("Convergence Error")

def G2(t, beta, h, p, m):
    mean = meanF(t, 0, beta, h, p, m)
    c = normConstant(0, 0, beta, h, p, m)
    result = quad(densityF, -math.inf, mean, args=(0, 0, beta, h, p, m))
    if result[1] < 1e-3:
        return result[0]/c
    else:
        return warnings.warn("Convergence Error")


