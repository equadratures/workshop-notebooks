from equadratures import *

def build_model(S,f):
####################################################################################
    # WRITE YOUR CODE HERE 
    # Define Poly object with name 'my_poly' with 'least-squares' method
    myParameters = [Parameter(distribution='uniform', lower=np.min(S[:,i]), upper=np.max(S[:,i]), order=2) \
                    for i in range(S.shape[1])]
    myBasis = Basis('total-order')
    my_poly = Poly(myParameters, myBasis, method='least-squares', \
                   sampling_args={'mesh': 'user-defined', 'sample-points':S, 'sample-outputs':f})
    my_poly.set_model()
####################################################################################
    
    return my_poly
