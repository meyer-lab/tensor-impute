"""
Tensor decomposition methods
"""
import numpy as np
from scipy.optimize import minimize
from .cmtf import calcR2X
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.cp_tensor import CPTensor, cp_normalize
from .impute_helper import createCube

tl.set_backend('numpy')


def reorient_factors(tensorFac):
    """ This function ensures that factors are negative on at most one direction. """
    for jj in range(1, len(tensorFac)):
        # Calculate the sign of the current factor in each component
        means = np.sign(np.mean(tensorFac[jj], axis=0))

        # Update both the current and last factor
        tensorFac[0] *= means[np.newaxis, :]
        tensorFac[jj] *= means[np.newaxis, :]
    return tensorFac


def buildTensors(pIn, tensor, r):
    """ Use parameter vector to build CP tensors. """
    nn = np.cumsum(tensor.shape) * r
    A = np.reshape(pIn[:nn[0]], (tensor.shape[0], r))
    B = np.reshape(pIn[nn[0]:nn[1]], (tensor.shape[1], r))
    C = np.reshape(pIn[nn[1]:], (tensor.shape[2], r))

    return CPTensor((None, [A, B, C]))


def cost(pIn, tensor, tmask, r):
    tensF = buildTensors(pIn, tensor, r)
    cost = np.linalg.norm(tl.cp_to_tensor(tensF, mask=1 - tmask) - tensor) # Tensor cost
    cost += 1e-9 * np.linalg.norm(pIn)
    return cost


def perform_CP_DO(tensorOrig=None, r=6):
    """ Perform CP decomposition. """
    if tensorOrig is None:
        tensorOrig = createCube()

    tensorIn = tensorOrig.copy()
    tmask = np.isnan(tensorIn)
    tensorIn[tmask] = 0.0

    CPinit = parafac(tensorIn.copy(), r, mask=tmask, n_iter_max=50, orthogonalise=10)
    x0 = np.concatenate((np.ravel(CPinit.factors[0]), np.ravel(CPinit.factors[1]), np.ravel(CPinit.factors[2])))

    rgs = (tensorIn, tmask, r)
    res = minimize(cost, x0, method='L-BFGS-B', jac=False, args=rgs, options={"maxiter": 50000})
    tensorFac = buildTensors(res.x, tensorIn, tmask, r)
    tensorFac = cp_normalize(tensorFac)

    # Reorient the later tensor factors
    tensorFac.factors = reorient_factors(tensorFac.factors)

    R2X = calcR2X(tensorOrig, tensorFac)

    for ii in range(3):
        tensorFac.factors[ii] = np.array(tensorFac.factors[ii])

    return tensorFac, R2X