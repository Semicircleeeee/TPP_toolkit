import numpy as np
from numpy import ndarray
from kernel import compute_kernel_value

def HP_intensity(t, para, History = None):
    """
    t : current time
    % Parameters of Hawkes processes
    % para.mu: base exogenous intensity
    % para.A: coefficients of impact function
    % para.kernel: 'exp', 'gauss'
    % para.w: bandwith of kernel
    """
    mu = para['mu'][:]
    
    if History is not None:
        
        time = History[0, :]
        index = time <= t
        time = time[index]
        event = History[1, index]

        basis = compute_kernel_value(t - time[:], para)
        a = para['A'][event, :, :]

        for i in range(para['A'].shape[2]):
            mu[i] = mu[i] + np.sum(basis * a[:, :, i])

    mu[mu < 0] = 0 # Retain the entries that is bigger than 0.


