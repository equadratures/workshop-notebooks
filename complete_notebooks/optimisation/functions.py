import numpy as np
from scipy.optimize import rosen

def Mccormick(s):
    return np.sin(s[0] + s[1]) + (s[0] - s[1])**2 - 1.5*s[0] + 2.5*s[1] + 1.0

def Himmelblau(s):
    return (s[0]**2 +s[1] - 11.0)**2 + (s[0] + s[1]**2 - 7.0)**2

def StyblinskiTang(s):
    n = s.size
    f = 0
    for i in range(n):
        f += 0.5 * (s[i]**4 - 16.0*s[i]**2 + 5.0*s[i])
    return f

def Rosenbrock(s):
	return rosen(s)