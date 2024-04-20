"""
Tensor decomposition methods
"""
import numpy as np
from scipy.optimize import minimize
import tensorly as tl
from .initialization import initialize_fac
from .tracker import Tracker
from tensorly.cp_tensor import cp_normalize, cp_lstsq_grad
from .generateTensor import generateTensor

tl.set_backend("numpy")


def buildTensors(pIn: np.ndarray, tshape: tuple):
    """Use parameter vector to build CP tensors."""
    r = int(pIn.size / np.sum(tshape))
    nn = np.cumsum(tshape) * r
    return (None, [x.reshape(-1, r) for x in np.split(pIn, nn[0:-1])])


def cost(pIn, tensor: np.ndarray, tmask: np.ndarray):
    tensF = buildTensors(pIn, tensor.shape)
    grad, costt = cp_lstsq_grad(tensF, tensor, return_loss=True, mask=1 - tmask)
    gradd = np.concatenate([g.flatten() for g in grad.factors])
    return costt, gradd


class do_callback:
    def __init__(self, callback:Tracker, shape:tuple):
        self.callback = callback
        self.shape = shape

    def __call__(self, x):
        tensorFac = cp_normalize(buildTensors(x, self.shape))
        self.callback(tensorFac)


def perform_DO(
    tensorOrig:np.ndarray=None,
    rank:int=6,
    n_iter_max:int=5_000,
    callback=None,
    init=None,
    **kwargs
) -> tl.cp_tensor.CPTensor:
    """Perform CP decomposition."""
    if tensorOrig is None:
        tensorOrig = generateTensor('unknown')
    if init == None:
        init = initialize_fac(tensorOrig, rank)
    if callback:
        temp_callback = do_callback(callback, tensorOrig.shape)
    else:
        temp_callback = None

    tensorIn = tensorOrig.copy()
    tmask = np.isnan(tensorIn)
    tensorIn[tmask] = 0.0

    x0 = np.concatenate([f.flatten() for f in init.factors])

    res = minimize(
        cost,
        x0,
        method="L-BFGS-B",
        jac=True,
        args=(tensorIn, tmask),
        options={"maxiter": n_iter_max},
        tol=tol,
        callback=temp_callback
    )

    return cp_normalize(buildTensors(res.x, tensorIn.shape))
