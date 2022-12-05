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
from initialize_fac import initialize_fac
from tensorly.cp_tensor import CPTensor, cp_normalize
from .impute_helper import createCube

tl.set_backend('numpy')
config.update("jax_enable_x64", True)


def do_khatri_rao(a, b):
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
        return do_khatri_rao(mats[0], khatri_rao(mats[1:]))


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

class do_callback():
    def __init__(self, callback, r, shape):
        self.callback = callback
        self.r = r
        self.shape = shape
        if self.callback.track_runtime:
            self.callback.begin()

    def __call__(self, x):
        tensorFac = CPTensor((None, buildTensors(x, self.r, self.shape)))
        tensorFac = cp_normalize(tensorFac)
        tensorFac.factors = reorient_factors(tensorFac.factors)
        for ii in range(3):
            tensorFac.factors[ii] = np.array(tensorFac.factors[ii])
        self.callback(tensorFac)


def perform_CP_DO(tensorOrig=None, r=6, maxiter=50, callback=None):
    """ Perform CP decomposition. """
    if tensorOrig is None:
        tensorOrig = createCube()
    if callback:
        temp_callback = do_callback(callback, r, tensorOrig.shape)
        temp_callback
    else:
        temp_callback = None

    tensorIn = tensorOrig.copy()
    tmask = np.isnan(tensorIn)
    tensorIn[tmask] = 0.0

    cost_jax = jit(cost, static_argnums=(3))
    cost_grad = jit(grad(cost, 0), static_argnums=(3))

    def costt(*args):
        return np.array(cost_jax(*args))

    def gradd(*args):
        return np.array(cost_grad(*args))

    CPinit = initialize_fac(tensorIn.copy(), r)
    x0 = np.concatenate(tuple([np.ravel(CPinit.factors[ii]) for ii in range(np.ndim(tensorIn))]))

    rgs = (tensorIn, tmask, r)
    res = minimize(costt, x0, method='L-BFGS-B', jac=gradd, args=rgs, options={"maxiter":maxiter}, callback=temp_callback)
    tensorFac = CPTensor((None, buildTensors(res.x, r, tensorIn.shape)))
    tensorFac = cp_normalize(tensorFac)

    # Reorient the later tensor factors
    tensorFac.factors = reorient_factors(tensorFac.factors)

    for ii in range(np.ndim(tensorIn)):
        tensorFac.factors[ii] = np.array(tensorFac.factors[ii])

    return tensorFac