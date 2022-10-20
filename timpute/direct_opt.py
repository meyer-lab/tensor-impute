"""
Tensor decomposition methods
"""
import numpy as np
import jax.numpy as jnp
from jax import jit, grad
from jax.config import config
from scipy.optimize import minimize
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.cp_tensor import CPTensor, cp_normalize
from .impute_helper import createCube

tl.set_backend('numpy')
config.update("jax_enable_x64", True)

def calcR2X(tensorIn, tensorFac):
    """ Calculate R2X. """
    tErr = np.nanvar(tl.cp_to_tensor(tensorFac) - tensorIn)
    return 1.0 - (tErr) / (np.nanvar(tensorIn))


def reorient_factors(tensorFac):
    """ This function ensures that factors are negative on at most one direction. """
    for jj in range(1, len(tensorFac)):
        # Calculate the sign of the current factor in each component
        means = np.sign(np.mean(tensorFac[jj], axis=0))

        # Update both the current and last factor
        tensorFac[0] *= means[np.newaxis, :]
        tensorFac[jj] *= means[np.newaxis, :]
    return tensorFac


def buildTensors(pIn, tensor, tmask, r):
    """ Use parameter vector to build CP tensors. """
    nn = np.cumsum(tensor.shape) * r
    A = jnp.reshape(pIn[:nn[0]], (tensor.shape[0], r))
    B = jnp.reshape(pIn[nn[0]:nn[1]], (tensor.shape[1], r))
    C = jnp.reshape(pIn[nn[1]:], (tensor.shape[2], r))

    return CPTensor((None, [A, B, C]))


def cost(pIn, tensor, tmask, r):
    tl.set_backend('jax')
    tensF = buildTensors(pIn, tensor, tmask, r)
    cost = jnp.linalg.norm(tl.cp_to_tensor(tensF, mask=1 - tmask) - tensor) # Tensor cost
    cost += 1e-9 * jnp.linalg.norm(pIn)
    tl.set_backend('numpy')
    return cost


def perform_CP_DO(tensorOrig=None, r=6):
    """ Perform CP decomposition. """
    if tensorOrig is None:
        tensorOrig = createCube()

    tensorIn = tensorOrig.copy()
    tmask = np.isnan(tensorIn)
    tensorIn[tmask] = 0.0

    cost_jax = jit(cost)
    cost_grad = jit(grad(cost, 0))

    def costt(*args):
        return np.array(cost_jax(*args))

    def gradd(*args):
        return np.array(cost_grad(*args))

    CPinit = parafac(tensorIn.copy(), r, mask=tmask, n_iter_max=50, orthogonalise=10)
    x0 = np.concatenate((np.ravel(CPinit.factors[0]), np.ravel(CPinit.factors[1]), np.ravel(CPinit.factors[2])))

    rgs = (tensorIn, tmask, r)
    res = minimize(costt, x0, method='L-BFGS-B', jac=gradd, args=rgs, options={"maxiter": 50000})
    tensorFac = buildTensors(res.x, tensorIn, tmask, r)
    tensorFac = cp_normalize(tensorFac)

    # Reorient the later tensor factors
    tensorFac.factors = reorient_factors(tensorFac.factors)

    R2X = calcR2X(tensorOrig, tensorFac)

    for ii in range(3):
        tensorFac.factors[ii] = np.array(tensorFac.factors[ii])

    return tensorFac, R2X