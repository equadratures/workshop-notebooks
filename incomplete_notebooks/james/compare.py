from scipy.optimize import minimize, Bounds
import numpy as np
import equadratures as eq
from trustregion import TrustRegion 

np.random.seed(0)

def compare_optimisation(fun, s0):
    dims = s0.size
    TR = TrustRegion(fun)
    s0 = np.zeros(dims)
    sopt, fopt = TR.trust_region(s0)
    print("Using our trust-region method, an optimal value of {} was found after {} function evaluations".format(fopt, TR.num_evals))
    
    methods = ['COBYLA', 'trust-constr']
    for method in methods:
        if method == 'COBYLA':
            cons = []
            for factor in range(dims):
                l = {'type': 'ineq',
                     'fun': lambda s, lb=-1.0, i=factor: s[i] - lb}
                u = {'type': 'ineq',
                     'fun': lambda s, ub=1.0, i=factor: ub - s[i]}
                cons.append(l)
                cons.append(u)
            sol = minimize(fun, s0, method=method, constraints=cons)
        else:
            bounds = Bounds(-np.ones(dims), np.ones(dims))
            sol = minimize(fun, s0, method=method, bounds=bounds)
        print("Using {}, an optimal value of {} was found after {} function evaluations".format(method, sol['fun'], sol['nfev']))
