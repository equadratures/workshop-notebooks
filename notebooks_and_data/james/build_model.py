from equadratures import *

def build_model(S,f,del_k):
####################################################################################
    # WRITE YOUR CODE HERE 
    # Define Poly object with name 'my_poly' with 'least-squares' method
    # Approximately 3 lines
    myParameters = [Parameter(distribution='uniform', lower=S[0,i] - del_k, upper=S[0,i] + del_k, order=2) for i in range(S.shape[1])]
    myBasis = Basis('total-order')
    my_poly = Poly(myParameters, myBasis, method='least-squares', sampling_args={'sample-points':S, 'sample-outputs':f})
####################################################################################
    my_poly.set_model()
    return my_poly
