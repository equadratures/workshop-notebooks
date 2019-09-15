import numpy as np
import scipy as sp
import equadratures as eq

from build_model import build_model
from compute_step import compute_step

class TrustRegion:
    
    def __init__(self,function,constraints=None):
        np.random.seed(0)
        self.function = function
        
        self.num_evals = 0
        self.S = np.array([])
        self.f = np.array([])
        self.consecutive_unsuccess = 0
#   Evaluates a given point and stores solutions in database and counts number of function evaluations thus far
#   Inputs:
#           x:      point to evaluate
#   Outputs:
#           f:      solution of f(x)
    def blackbox_evaluation(self,s):
        f = self.function(s.flatten())
        self.num_evals += 1
        if self.S.size == 0 and self.f.size == 0:
            self.S = s.reshape(1,-1)
            self.f = np.array([[f]])
        elif self.S.size != 0 and self.f.size != 0:
            self.S = np.vstack((self.S,s.reshape(1,-1)))
            self.f = np.vstack((self.f,np.array([f])))
        else:
            raise ValueError('The arrays of solutions and their corresponding function values are not equivalent!')
        return f
#   Finds points for regression/interpolation
#   Inputs:
#           xk:     current iterate
#           fk:     function values of fk
#           rk:     radius of points allowed for regression/interpolation
#           qoi:    dictionary of quantity of interest
#   Outputs: 
#           Y:      set of regression/interpolation points
#           fY:     function values at points of Y
    def regression_set(self, s_old, f_old, del_k):
#       Copy database of solutions and remove the current iterate
        S_hat = np.copy(self.S)
        f_hat = np.copy(self.f)
        ind_not_current = np.where(np.linalg.norm(S_hat-s_old,axis=1) >= 1.0e-14)[0]
        S_hat = S_hat[ind_not_current,:]
        f_hat = f_hat[ind_not_current]
#       Remove points outside the trust-region
        ind_within_TR = np.where(np.linalg.norm(S_hat-s_old,axis=1) <= del_k)[0]
        S_hat = S_hat[ind_within_TR,:]
        f_hat = f_hat[ind_within_TR]
#       If Yhat does not contain at least q points, uniformly generate q points with a d-dimensional hypercube of radius rk around centre 
        while S_hat.shape[0] < int(1.2*np.ceil(self.q)):
            tmp = np.random.randn(self.n)
            s = s_old + (del_k*np.random.rand()**(1.0/self.n) / np.linalg.norm(tmp)) * tmp
            S_hat = np.vstack((S_hat, s))
            f_hat = np.vstack((f_hat, self.blackbox_evaluation(s)))
#       Centre and scale points
        S_hat -= s_old
        DelS = max(np.linalg.norm(S_hat, axis=1))
        S_hat = (1.0/DelS)*S_hat
#       Initialise regression/interpolation points and their corresponding function evaluations
        S = np.zeros(self.n).reshape(1,-1)
        f = np.array([[f_old]])
#       Find well-poised points
        S,f,S_hat,f_hat = self.well_poised_LU(S,f,S_hat,f_hat)
#       Include all of the left-over points
        S = np.vstack((S,S_hat))
        f = np.vstack((f,f_hat))
#       Unscale rand uncentre points
        S = DelS*S +s_old       
#       Evaluate newly generated regression/interpolation points which do not have an evaluation value
        for j in range(f.shape[0]):
            if np.isinf(f[j]):
                f[j,0] = self.blackbox_evaluation(S[j,:])
                
#        _, ind_unique = np.unique(S.round(decimals=10), axis=0, return_index=True)
#        S = S[ind_unique, :]
#        f = f[ind_unique] 
        return S, f
#   Finds a well-poised set of points for regression/interpolation (see "Geometry of interpolation sets in derivative free optimization" by Conn)
#   Inputs:
#           Y:      current set of regression/interpolation points
#           fY:     function values at points of Y
#           Yhat:   current set of points to choose from
#           fYhat:  function values at points of Yhat
#           qoi:    dictionary of quantity of interest
#   Outputs: 
#           Y:      new set of regression/interpolation points
#           fY:     function values at points of Ytmp
#           Yhat:   new set of points to choose from
#           fYhat:  function values at points of Yhat
    def well_poised_LU(self,S,f,S_hat,f_hat):
#       Poised constant of algorithm
        psi = 0.25
#       Generate natural monomial basis
        Base = eq.Basis('total-order', orders=np.tile([2], self.n))
        basis = Base.get_basis()[:,range(self.n-1, -1, -1)]
        def natural_basis_function(x, basis):
            phi = np.zeros(basis.shape[0])
            for j in range(basis.shape[0]):
                phi[j] = 1.0
                for k in range(basis.shape[1]):
                    phi[j] *= (x[k]**basis[j,k]) / sp.special.factorial(basis[j,k])
            return phi
        phi_function = lambda x: natural_basis_function(x, basis)
#       Initialise U matrix of LU factorisation of M matrix (see Conn et al.)
        U = np.zeros((self.q,self.q))
#       Initialise the first row of U to the e1 basis vector which corresponds to solution with all zeros
        U[0,0] = 1.0
#       Perform the LU factorisation algorithm for the rest of the points
        for k in range(1,self.q):
            v = np.zeros(self.q)
            for j in range(k):
                v[j] = -U[j,k] / U[j,j]
            v[k] = 1.0
#           If there are still points to choose from, find if points meet criterion. If so, use the index to choose 
#           point with given index to be next point in regression/interpolation set
            if S_hat.size != 0:
                M = self.natural_basis_matrix(S_hat,v,phi_function)
                index2 = np.argmax(M)
                if M[index2] < psi:
                    index2 = None
            else:
                index2 = None
#           If index exists, choose the point with that index and delete it from possible choices
            if index2 is not None:
                s = S_hat[index2,:].flatten()
                S = np.vstack((S,s))
                f = np.vstack((f,f_hat[index2].flatten()))
                S_hat = np.delete(S_hat, index2, 0)
                f_hat = np.delete(f_hat, index2, 0)
                phi = phi_function(s.flatten())
#           If index doesn't exist, solve an optimisation point to find the point in the range which best satisfies criterion
            else:
                s = sp.optimize.minimize(lambda x: -abs(np.dot(v,phi_function(x.flatten()))), np.zeros(self.n), method='COBYLA',constraints={'type':'ineq', 'fun': lambda x: 1.0 - np.dot(x.T,x)},options={'disp': False})['x'].flatten()
                S = np.vstack((S,s))
                f = np.vstack((f,np.array([np.inf])))
                phi = phi_function(s.flatten())
#           Update U factorisation in LU algorithm
            U[k,k] = np.dot(v,phi)
            for i in range(k+1,self.q):
                U[k,i] += phi[i]
                for j in range(k):
                    U[k,i] -= (phi[j]*U[j,i])/U[j,j]
        return S,f,S_hat,f_hat
#   Finds absolute value of a vector multiplied by the Vandermonde matrix using natural basis of monomial terms (see "Geometry of interpolation sets in derivative free optimization" by Conn)
#   Inputs:
#           X:      set of points
#           v:      vector to multipy Vandermonde matrix by
#           deg:    degree of expansion
#   Outputs: 
#           Mv_abs: absolute value of a vector multiplied by the Vandermonde matrix 
    def natural_basis_matrix(self,S,v,phi):  
        M = []
        for i in range(S.shape[0]):
            M.append(phi(S[i,:].flatten()))
        M = np.array(M)
        Mv_abs = np.absolute(np.dot(M,v))
        return Mv_abs
##   Builds the regression/interpolation model given regression/interpolation points and their values
##   Inputs:
##           Y:      regression/interpolation points
##           f:      function values at points
##   Outputs: 
##           m_k:  ridge function model at iteration k
#    def build_model(self,S,f,del_k):
#        myParameters = [eq.Parameter(distribution='uniform', lower=S[0,i] - del_k, upper=S[0,i] + del_k, order=2) for i in range(S.shape[1])]
#        myBasis = eq.Basis('total-order')
#        my_poly = eq.Poly(myParameters, myBasis, method='compressive-sensing', sampling_args={'sample-points':S, 'sample-outputs':f})
#        my_poly.set_model()
#        return my_poly
##   Computes the solution to the trust-region subproblem, subject to box constraints
##   Inputs:
##           xk:     initial point
##           m_k:     current ridge function model
##           delk:   initial trust-region radius
##   Outputs: 
##           x1:     found optimal solution
##           f1:     model value at optimal 
#    def compute_step(self,s_old,my_poly,del_k):
#        Opt = eq.Optimisation(method='SLSQP')
#        Opt.add_objective(poly=my_poly)
#        Opt.add_bounds(-np.ones(s_old.size),np.ones(s_old.size))
#        Opt.add_linear_ineq_con(np.eye(s_old.size), s_old-del_k*np.ones(s_old.size), s_old+del_k*np.ones(s_old.size))
#        sol = Opt.optimise_poly(s_old)
#        s_new = sol['x']
#        m_new = sol['fun']
#        return s_new, m_new
#   Computes criticality measure to determine if model is optimal wrt some constraints (see "NOWPAC: A provably convergent derivative-free nonlinear optimizer with path-augmented constraints" by Augustin)
#   Inputs:
#           m_k:    model at current iterate
#           xk:     current iterate
#           del_k:  trust-region radius
#   Outputs: 
#           alpha_k:    criticality measure
    def compute_criticality_measure(self,my_poly,s_old,del_k):
        g_k = my_poly.get_polyfit_grad(s_old).flatten()
        crit = sp.optimize.minimize(lambda s: np.dot(g_k,s-s_old), np.zeros_like(s_old), method='COBYLA',constraints=[{'type':'ineq', 'fun': lambda s: 1.0 - s},{'type':'ineq', 'fun': lambda s: s + 1.0},{'type':'ineq', 'fun': lambda s: del_k*np.ones_like(s_old) - s + s_old},{'type':'ineq', 'fun': lambda s: del_k*np.ones_like(s_old) + s - s_old}],options={'disp': False})['fun']
        alpha_k = abs(crit) / del_k
        return alpha_k
#   Performs the ridge function trust-region algorithm
#   Inputs:
#           x0:         initial point
#           del_ref:    initial trust-region radius
#   Outputs: 
#           x0:         found optimal solution
#           f0:         objective value at optimal
    def trust_region(self, s_old, del_k = 1.0, eta0 = 0.0, eta1 = 0.5, gam0 = 0.01, gam1 = 1.2, epsilon_c = 1.0e-2, delkmin = 1.0e-8, delkmax = 1.5):
        self.n = s_old.size
        self.p = self.n + 1
#        self.q = 2*self.n + 1
        self.q = int(sp.special.comb(self.n+2, 2))
        itermax = 500
#       Make the first black-box function call and initialise the database of solutions and labels
        f_old = self.blackbox_evaluation(s_old)
#       Construct the regression set
        S, f = self.regression_set(s_old,f_old,del_k)
#       Construct the model and evaluate at current point
        my_poly = build_model(S,f,del_k)
#       Begin algorithm
        for i in range(itermax):
#           If trust-region radius is less than minimum, break loop
            if del_k < delkmin:
                break
            m_old = np.asscalar(my_poly.get_polyfit(s_old)[0])
#           If gradient of model is very small, need to check the validity of the model
            s_new, m_new = compute_step(s_old,my_poly,del_k)
            f_new = self.blackbox_evaluation(s_new)
########################################################## Check acceptance  ######################################################
####################################################################################################################################
            if m_new >= m_old:
                del_k *= gam0
                continue
#           Calculate trust-region factor
            rho_k = (f_old - f_new) / (m_old - m_new)
            if rho_k >= eta1:
                s_old = s_new
                f_old = f_new
                S, f = self.regression_set(s_old,f_old,del_k)
                my_poly = build_model(S,f,del_k)
                alpha_k = self.compute_criticality_measure(my_poly, s_old, del_k)
                if alpha_k <= epsilon_c:
                    del_k *= gam0
                else:
                    del_k = min(gam1*del_k,delkmax)
            elif rho_k > eta0: 
                s_old = s_new
                f_old = f_new
                S, f = self.regression_set(s_old,f_old,del_k)
                my_poly = build_model(S,f,del_k)
                alpha_k = self.compute_criticality_measure(my_poly, s_old, del_k)
                if alpha_k <= epsilon_c:
                    del_k *= gam0
            else:
                del_k *= gam0
        return s_old, f_old
