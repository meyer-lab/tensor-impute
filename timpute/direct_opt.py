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
from .cmtf import calcR2X

tl.set_backend('numpy')
config.update("jax_enable_x64", True)

def khatri_rao_double(a, b):
    a = jnp.asarray(a)
    b = jnp.asarray(b)

    assert (a.ndim == 2 and b.ndim == 2)
    assert (a.shape[1] == b.shape[1])

    c = a[..., :, jnp.newaxis, :] * b[..., jnp.newaxis, :, :]
    return c.reshape((-1,) + c.shape[2:])

def khatri_rao(mats):
    if len(mats) == 1:
        return mats[0]
    if len(mats) >= 2:
        return khatri_rao_double(mats[0], khatri_rao(mats[1:]))

def factors_to_tensor(factors):
    shape = [ff.shape[0] for ff in factors]
    unfold = jnp.dot(factors[0], khatri_rao(factors[1:]).T)
    return unfold.reshape(shape)


def reorient_factors(tensorFac):
    """ This function ensures that factors are negative on at most one direction. """
    for jj in range(1, len(tensorFac)):
        # Calculate the sign of the current factor in each component
        means = np.sign(np.mean(tensorFac[jj], axis=0))

        # Update both the current and last factor
        tensorFac[0] *= means[np.newaxis, :]
        tensorFac[jj] *= means[np.newaxis, :]
    return tensorFac


def buildTensors(pIn, r, tshape):
    """ Use parameter vector to build CP tensors. """
    nn = np.cumsum(tshape) * r
    return [x.reshape(tshape[i], r) for i, x in enumerate(jnp.split(pIn, nn)) if i < len(nn)]


def cost(pIn, tensor, tmask, r):
    tensF = buildTensors(pIn, r, tensor.shape)
    cost = jnp.linalg.norm((factors_to_tensor(tensF) - tensor) * (1-tmask)) # Tensor cost
    cost += 1e-9 * jnp.linalg.norm(pIn)
    return cost


def perform_CP_DO(tensorOrig=None, r=6):
    """ Perform CP decomposition. """
    if tensorOrig is None:
        tensorOrig = createCube()

    tensorIn = tensorOrig.copy()
    tmask = np.isnan(tensorIn)
    tensorIn[tmask] = 0.0

    cost_jax = jit(cost, static_argnums=(3))
    cost_grad = jit(grad(cost, 0), static_argnums=(3))

    def costt(*args):
        return np.array(cost_jax(*args))

    def gradd(*args):
        return np.array(cost_grad(*args))

    CPinit = parafac(tensorIn.copy(), r, mask=tmask, n_iter_max=50, orthogonalise=10)
    x0 = np.concatenate((np.ravel(CPinit.factors[0]), np.ravel(CPinit.factors[1]), np.ravel(CPinit.factors[2])))

    rgs = (tensorIn, tmask, r)
    res = minimize(costt, x0, method='L-BFGS-B', jac=gradd, args=rgs, options={"maxiter": 50000})
    tensorFac = CPTensor((None, buildTensors(res.x, r, tensorIn.shape)))
    tensorFac = cp_normalize(tensorFac)

    # Reorient the later tensor factors
    tensorFac.factors = reorient_factors(tensorFac.factors)

    R2X = calcR2X(tensorOrig, tensorFac)

    for ii in range(3):
        tensorFac.factors[ii] = np.array(tensorFac.factors[ii])

    return tensorFac, R2X