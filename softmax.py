import numpy as np

# a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    _exp_val = np.exp(L)
    _sum = sum(_exp_val)
    _sm_output = []
    for i in _exp_val:
        _sm_output.append(i*1.0/_sum)
    return _sm_output
