from equadratures import *
import numpy as np 

def compute_step(s_old,my_poly,del_k):
    Opt = Optimisation(method='SLSQP')
####################################################################################
    # WRITE YOUR CODE HERE 
    # Add objectives and constraints to the optimisation problem
    # Approximately 3 lines
    Opt.add_objective(poly=my_poly)
    Opt.add_bounds(-np.ones(s_old.size),np.ones(s_old.size))
    Opt.add_linear_ineq_con(np.eye(s_old.size), s_old-del_k*np.ones(s_old.size), s_old+del_k*np.ones(s_old.size))
####################################################################################
    sol = Opt.optimise(s_old)
    s_new = sol['x']
    m_new = sol['fun']
    return s_new, m_new
