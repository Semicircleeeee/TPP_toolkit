import numpy as np
from numpy import ndarray
from scipy.special import erf

## More like a intensity function rather than a kernel??
def compute_kernel_value(dt: ndarray, para: dict) -> ndarray: 
    landmark = para['landmark']
    w = para['w']
    """
    Compute the value of the kernel function.
    
    Args: 
        dt (ndarray): Time difference in ndarray with dimension (data.dim, time_length).
        para (dict): A dictionary containing all parameters, include
            - 'landmark', center of the marking point
            - 'kernel', kernel tp
            - 'w', weight parameters in the kernel function
    Return:
        ans (ndarray): The kernel value on all time difference.
    """
    assert dt.shape[-1] == landmark.shape[0] # Test weather they have the same data dimension.
    # TODO: Figure out the dimension of dt and complete it in the annotation.
    distance = np.abs(dt[:, np.newaxis] - landmark[np.newaxis, :]) 
    ans = np.array(dt)

    if para['kernel'] == 'exp':
        ans = w * np.exp(-w * distance) # PDF of exponential distribution ??
        ans[ans > 1] = 0 # Prevent Overflow bt setting values bigger than 1 to be 1.

    elif para['kernel'] == 'gauss':
        ans = np.exp(-(distance ** 2)) / (2 * (w ** 2)) / (np.sqrt(2 * np.pi) * w) ## PDF of gaussian distribution

    else:
        raise ValueError('Invalid Kernel')
    return ans

def compute_TVHP_kernel_value():
    raise NotImplementedError

def kernel_value_approx(dt: ndarray, para: dict):
    g = np.zeros((dt.shape[0], para['g'].shape[1])) # initialize kernel value like array

    m = g.shape[0]

    nums = np.ceil(dt / para['dt']).astype(np.int32) # Scale up or unify the time distance.
    # dict['g'] is a pre-computed dict-like array
    for i in range(dt.shape[0]):
        g[i, :] = para['g'][nums[i] - 1] if nums[i] <= m - 1 else 0
    
    return g

def kernel_integration_approx(dt: ndarray, para: dict):
    g = np.zeros((dt.shape[0], para['g'].shape[1])) # initialize kernel value like array

    m = g.shape[0]

    nums = np.ceil(dt / para['dt']).astype(np.int32) # Scale up or unify the time distance.
    # dict['g'] is a pre-computed dict-like array
    for i in range(dt.shape[0]):
        g[i, :] = np.sum(para['g'][:nums[i] - 1] * para['dt']) if nums[i] <= m - 1 else np.sum(para['g'] * para['dt'])
    
    return g

def kernel_intergration(dt, para):
    landmark = para['landmark']

    distance = dt[:, np.newaxis] - landmark[np.newaxis, :]

    landmark = landmark[np.newaxis, :] #reshape landmark

    if para['kernel'] == 'exp':
        g = 1 - np.exp(-para['w'] * (distance - landmark))
        g[g < 0 ] = 0
    elif para['kernel'] == 'gauss':
        g = 0.5 * ( erf(distance / (np.sqrt(2) * para['w'])) + erf(landmark / (np.sqrt(2) * para['w'])) )
        #TODO: Find why is this expression
    else:
        raise ValueError('Invalid Kernel')
    return g
    

def compute_tv_kernel_integration(tj: ndarray, option: dict, upper, lower) -> ndarray:
    """
    Compute the approximate intensity/kernel integration.
    
    Args:
        tj (ndarray): Time points of events?? TODO: Figure out what tj indicates
        option (dict): The params include:
            - 'landmark' Vector of reference point
            - 'sigmaA' paraA
            - 'sigmaT' paraT
    """
    tj = np.array(tj)
    landmark = np.array(option['landmark'])
    
    dt = landmark[:, np.newaxis] - tj[np.newaxis, :]

    weight = np.sqrt(np.pi / 2) * option['sigmaA'] * \
             np.exp(0.5 * option['sigmaA']**2 * option['sigmaT']**2 - \
                    option['sigmaT'] * dt)
    
    upp = (upper - (landmark - option['sigmaA']**2 * option['sigmaT'])) / (np.sqrt(2) * option['sigmaA'])
    low = (lower - (landmark - option['sigmaA']**2 * option['sigmaT'])) / (np.sqrt(2) * option['sigmaA'])
    
    phi = erf(upp) - erf(low)
    
    kg_t = weight * phi
    
    return kg_t


def similarity(f1, f2 ,sigma):
    """
    Compute the similarity between two vectors f1 and f2
    """
    if f1 is None or f2 is None:
        weight = 1
    else:
        dt = np.linalg.norm(f1 - f2)**2
        weight = np.exp(- dt / (2 * sigma ** 2)) # Gaussian Kernel

    return weight
    

def SoftThreshold(A: ndarray, thres, tp = 'S'):
    if tp == 'S':
        # Using direct abs value
        temp = A
        s = np.sign(temp)
        temp = (np.abs(temp) - thres)
        temp[temp < 0] = 0
        z = s * temp
    elif tp == 'LR':
        z = np.zeros(A.shape)
        # LR threshold using singular value
        for i in range(A.shape[1]):
            temp = A[:, i, :]
            temp = np.reshape(temp, (A.shape[0], A.shape[2]))
            (Ut, St, Vt) = np.linalg.svd(temp)
            St = St - thres
            St[St < 0] = 0
            reconstructed = np.dot(Ut, np.dot(np.diag(St), Vt))
            z[:, i:, :] = reconstructed.reshape(A.shape[0], 1, A.shape[2])
    elif tp == 'GS':
        raise NotImplementedError
    else:
        raise ValueError('')