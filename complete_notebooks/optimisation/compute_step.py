from equadratures import *
import numpy as np 

def compute_step(s_old,my_poly,del_k,a,b): 
####################################################################################
    # WRITE YOUR CODE HERE 
    # Add objectives and constraints to the optimisation problem
    Opt = Optimisation(method='TNC')
    Opt.add_objective(poly=my_poly)
    Opt.add_bounds(np.maximum(a, s_old-del_k),np.minimum(b, s_old+del_k))
    sol = Opt.optimise(s_old)
####################################################################################
    return sol['x'], sol['fun']
