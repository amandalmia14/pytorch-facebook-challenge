import numpy as np

# a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    _cross_entropy = -np.sum(Y*np.log(P)+(1-Y)*np.log(1-P))
    return _cross_entropy
