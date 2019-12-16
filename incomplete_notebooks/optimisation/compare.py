from scipy.optimize import minimize, Bounds, shgo, differential_evolution
import numpy as np
import equadratures as eq

def compare_optimisation(fun, s0, bounds):
    dims = s0.size
    Opt = eq.Optimisation(method='trust-region')
    Opt.add_objective(custom={'function': fun})
    Opt.add_bounds(bounds[0], bounds[1])
    sol = Opt.optimise(s0)
    print("Using the trust-region method, an optimal value of {} was found after {} function evaluations".format(sol['fun'], sol['nfev']))

    bounds_2 =[]
    for i in range(dims):
        bounds_2.append((bounds[0][i], bounds[1][i]))
    sol = differential_evolution(fun, bounds_2)
    print("Using differential evolution, an optimal value of {} was found after {} function evaluations".format(sol['fun'], sol['nfev']))

    cons = []
    for factor in range(len(bounds)):
        lower, upper = bounds_2[factor]
        l = {'type': 'ineq',
             'fun': lambda x, lb=lower, i=factor: x[i] - lb}
        u = {'type': 'ineq',
             'fun': lambda x, ub=upper, i=factor: ub - x[i]}
        cons.append(l)
        cons.append(u)
    sol = minimize(fun, s0, method='COBYLA', constraints=cons, tol=1.0e-8, options={'rhobeg': 1.0, 'maxiter': 100000, 'disp': False, 'catol': 1.0e-8})
    print("Using COBYLA, an optimal value of {} was found after {} function evaluations".format(sol['fun'], sol['nfev']))