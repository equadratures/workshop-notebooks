import numpy as np
from equadratures import *
from scipy.special import comb, factorial
from scipy import optimize

from build_model import build_model
from compute_step import compute_step

class TrustRegion:
    
    def __init__(self, function):
        self.function = function
        
        self.num_evals = 0
        self.S = np.array([])
        self.f = np.array([])

#   Evaluates a given point and stores solutions in database and counts number of function evaluations thus far
#   Inputs:
#           x:      point to evaluate
#   Outputs:
#           f:      solution of f(x)
    def _blackbox_evaluation(self,s):
        """
        Evaluates the point s for ``trust-region`` method
        """
        f = self.function(s.flatten())
        self.num_evals += 1
        if self.f.size == 0:
            self.S = s.reshape(1,-1)
            self.f = np.array([[f]])
        else:
            self.S = np.vstack((self.S,s.reshape(1,-1)))
            self.f = np.vstack((self.f,np.array([f]))) 
        return f
        
    @staticmethod
    def _remove_point_from_set(S, f, s_old, tol):
        ind_current = np.where(np.linalg.norm(S-s_old, axis=1, ord=np.inf) <= tol)[0]
        S = np.delete(S, ind_current, 0)
        f = np.delete(f, ind_current, 0)
        return S, f
    @staticmethod
    def _remove_points_within_TR_from_set(S, f, s_old, del_k):
        ind_within_TR = np.where(np.linalg.norm(S-s_old, axis=1, ord=np.inf) <= del_k)[0]
        S = S[ind_within_TR,:]
        f = f[ind_within_TR]
        return S, f
    
    def _sample_set(self, s_old, f_old, del_k, method='new', S=None, f=None, num=1):
#       Poised constant of algorithm
        if method == 'new':
            S_hat = np.copy(self.S)
            f_hat = np.copy(self.f)
            S_hat, f_hat = self._remove_point_from_set(S_hat, f_hat, s_old, del_k*1.0e-8)
            S_hat, f_hat = self._remove_points_within_TR_from_set(S_hat, f_hat, s_old, del_k)
                
            S = s_old.reshape(1,-1)
            f = np.array([f_old])
            S, f = self._improve_poisedness_LU(S, f, S_hat, f_hat, s_old, del_k, 'new')
        elif method == 'replace':
            if max(np.linalg.norm(S-s_old, axis=1, ord=np.inf)) > self.epsilon*del_k:
                ind_distant = np.argmax(np.linalg.norm(S-s_old, axis=1, ord=np.inf))
                S = np.delete(S, ind_distant, 0)
                f = np.delete(f, ind_distant, 0)
            else:
                S_hat = np.copy(S)
                f_hat = np.copy(f)
                S_hat, fhat = self._remove_point_from_set(S_hat, f_hat, s_old, del_k*1.0e-8)
                
                S = s_old.reshape(1,-1)
                f = np.array([f_old])
                S, f = self._improve_poisedness_LU(S, f, S_hat, f_hat, s_old, del_k, 'improve')
        elif method == 'improve':
            S_hat = np.copy(S)
            f_hat = np.copy(f)
            S_hat, f_hat = self._remove_point_from_set(S_hat, f_hat, s_old, del_k*1.0e-8)
            
            if max(np.linalg.norm(S_hat-s_old, axis=1, ord=np.inf)) > self.epsilon*del_k:
                ind_distant = np.argmax(np.linalg.norm(S_hat-s_old, axis=1, ord=np.inf))
                S_hat = np.delete(S_hat, ind_distant, 0)
                f_hat = np.delete(f_hat, ind_distant, 0)
                
            S = s_old.reshape(1,-1)
            f = np.array([f_old])
            S, f = self._improve_poisedness_LU(S, f, S_hat, f_hat, s_old, del_k, 'improve', num)
        return S, f
    
    def _improve_poisedness_LU(self, S, f, S_hat, f_hat, s_old, del_k, method=None, num=1):
        if S_hat.shape[0] > 1:
            Del_S = max(np.linalg.norm(S_hat-s_old, axis=1, ord=np.inf))
        else:
            Del_S = del_k
        Base = Basis('total-order', orders=np.tile([2], self.n))
        basis = Base.get_basis()[:,range(self.n-1, -1, -1)]
        def natural_basis_function(s, basis):
            s = (s - s_old) / Del_S
            try:
                m,n = s.shape
            except:
                m = 1
                s = s.reshape(1,-1)
            phi = np.zeros((m, basis.shape[0]))
            for i in range(m):
                for k in range(basis.shape[0]):
                    phi[i,k] = np.prod(np.divide(np.power(s[i,:], basis[k,:]), factorial(basis[k,:])))
            if m == 1:
                return phi.flatten()
            else:
                return phi
        phi_function = lambda s: natural_basis_function(s, basis)
#       Initialise U matrix of LU factorisation of M matrix (see Conn et al.)
        U = np.zeros((self.q,self.q))
        U[0,0] = 1.0
#       Perform the LU factorisation algorithm for the rest of the points
        for k in range(1, self.q):
            index = None
#            index2 = None
            v = np.zeros(self.q)
            for j in range(k):
                v[j] = -U[j,k] / U[j,j]
            v[k] = 1.0
            
#           If there are still points to choose from, find if points meet criterion. If so, use the index to choose 
#           point with given index to be next point in regression/interpolation set
            if f_hat.size > 0:
                M = np.absolute(np.array([np.dot(phi_function(S_hat),v)]).flatten())
                index = np.argmax(M)
#                print(phi_function(Xhat))
                if abs(M[index]) < 1.0e-3:
                    index = None
                elif method =='new':
                    if M[index] < self.psi:
                        index = None
                elif method == 'improve':
                    if f.size == self.q - num:
                        if M[index] < self.psi:
                            index = None
#           If index exists, choose the point with that index and delete it from possible choices
            if index is not None:
                s = S_hat[index,:]
                S = np.vstack((S, s))
                f = np.vstack((f, f_hat[index]))
                S_hat = np.delete(S_hat, index, 0)
                f_hat = np.delete(f_hat, index, 0)
#           If index doesn't exist, solve an optimisation point to find the point in the range which best satisfies criterion
            else:
                s = self._find_new_point(v, phi_function, del_k, s_old)
                S = np.vstack((S, s))
                f = np.vstack((f, self._blackbox_evaluation(s)))
#           Update U factorisation in LU algorithm
            phi = phi_function(s)
            U[k,k] = np.dot(v, phi)
            for i in range(k+1,self.q):
                U[k,i] += phi[i]
                for j in range(k):
                    U[k,i] -= (phi[j]*U[j,i]) / U[j,j]
        return S, f
    
    def _find_new_point(self, v, phi_function, del_k, s_old):
        obj1 = lambda s: np.dot(v, phi_function(s))
        obj2 = lambda s: -np.dot(v, phi_function(s))
        if self.bounds is not None:
            bounds_l = np.maximum(self.bounds[0], s_old-del_k)
            bounds_u = np.minimum(self.bounds[1], s_old+del_k)
        else:
            bounds_l = s_old-del_k
            bounds_u = s_old+del_k
        bounds = []
        for i in range(self.n):
            bounds.append((bounds_l[i], bounds_u[i]))
        res1 = optimize.minimize(obj1, s_old, method='TNC', bounds=bounds, options={'disp': False, 'maxiter': 500})
        res2 = optimize.minimize(obj2, s_old, method='TNC', bounds=bounds, options={'disp': False, 'maxiter': 500})
        if abs(res1['fun']) > abs(res2['fun']):
            return res1['x']
        else:
            return res2['x']

    # def _build_model(self,S,f):
    #     """
    #     Constructs quadratic model for ``trust-region`` method
    #     """
    #     myParameters = [Parameter(distribution='uniform', lower=np.min(S[:,i]), upper=np.max(S[:,i]), order=2) for i in range(self.n)]
    #     myBasis = Basis('total-order')
    #     my_poly = Poly(myParameters, myBasis, method='least-squares', sampling_args={'mesh': 'user-defined', 'sample-points':S, 'sample-outputs':f})
    #     my_poly.set_model()
    #     return my_poly

    # def _compute_step(self,s_old,my_poly,del_k):
    #     """
    #     Solves the trust-region subproblem for ``trust-region`` method
    #     """
    #     Opt = Optimisation(method='TNC')
    #     Opt.add_objective(poly=my_poly)
    #     if self.bounds is not None:
    #         bounds_l = np.maximum(self.bounds[0], s_old-del_k)
    #         bounds_u = np.minimum(self.bounds[1], s_old+del_k)
    #     else:
    #         bounds_l = s_old-del_k
    #         bounds_u = s_old+del_k
    #     Opt.add_bounds(bounds_l,bounds_u)
    #     sol = Opt.optimise(s_old)
    #     s_new = sol['x']
    #     m_new = sol['fun']
    #     return s_new, m_new
    
    @staticmethod
    def _choose_best(X, f):
        ind_min = np.argmin(f)
        x_0 = X[ind_min,:]
        f_0 = np.asscalar(f[ind_min])
        return x_0, f_0

    def trust_region(self, s_old, lower_bound=None, upper_bound=None, del_k = 0.25, eta0 = 0.1, eta1 = 0.7, gam0 = 0.25, gam1 = 1.5, omega = 0.6, delmin = 1.0e-5, delmax = 1.0, max_evals=1000):
        """
        Computes optimum using the ``trust-region`` method
        """
        self.n = s_old.size
        self.psi = 0.25
        self.epsilon = 1.2
        self.p = self.n + 1
        self.q = int(comb(self.n+2, 2))
        if lower_bound is None:
            lower_bound = -np.inf*np.ones(self.n)
        if upper_bound is None:
            upper_bound = np.inf*np.ones(self.n)
        self.bounds = [lower_bound, upper_bound]
        itermax = 10000
        # Make the first black-box function call and initialise the database of solutions and labels
        f_old = self._blackbox_evaluation(s_old)
        # Construct the regression set
        S, f = self._sample_set(s_old, f_old, del_k)
        # Construct the model and evaluate at current point
        s_old, f_old = self._choose_best(self.S, self.f)
        for i in range(itermax):
            # If trust-region radius is less than minimum, break loop
            if len(self.f) >= max_evals or del_k < delmin:
                break
            my_poly = build_model(S, f)
            m_old = np.asscalar(my_poly.get_polyfit(s_old))
            s_new, m_new = compute_step(s_old,my_poly,del_k,lower_bound,upper_bound)
            # Safety step implemented in BOBYQA
            if np.linalg.norm(s_new - s_old, ord=np.inf) < 0.01*del_k:
                del_k *= omega
                S, f = self._sample_set(s_old, f_old, del_k, 'improve', S, f)
                s_old, f_old = self._choose_best(S, f)
                continue
            f_new = self._blackbox_evaluation(s_new)
            # Calculate trust-region factor
            rho_k = (f_old - f_new) / (m_old - m_new)
            S = np.vstack((S, s_new))
            f = np.vstack((f, f_new))
            s_old, f_old = self._choose_best(S, f)
            if rho_k >= eta1:
                del_k = min(gam1*del_k,delmax)
                S, f = self._sample_set(s_old, f_old, del_k, 'replace', S, f)
            elif rho_k > eta0:
                S, f = self._sample_set(s_old, f_old, del_k, 'replace', S, f)
            else:
                if max(np.linalg.norm(S-s_old, axis=1, ord=np.inf)) <= self.epsilon*del_k:
                    del_k *= gam0
                    S, f = self._sample_set(s_old, f_old, del_k, 'replace', S, f)
                else:
                    S, f = self._sample_set(s_old, f_old, del_k, 'improve', S, f)
        s_old, f_old = self._choose_best(S, f)
        return s_old, f_old
