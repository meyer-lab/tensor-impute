"""
Tensor decomposition methods
"""
import numpy as np
from scipy.optimize import minimize
import tensorly as tl
from .initialize_fac import initialize_fac
from tensorly.cp_tensor import CPTensor, cp_normalize, cp_lstsq_grad
from .test.simulated_tensors import createUnknownRank

tl.set_backend('numpy')


def buildTensors(pIn, r, tshape):
    """ Use parameter vector to build CP tensors. """
    nn = np.cumsum(tshape) * r
    return [x.reshape(tshape[i], r) for i, x in enumerate(np.split(pIn, nn)) if i < len(nn)]


def cost(pIn, tensor, tmask, r):
    tensF = buildTensors(pIn, r, tensor.shape)
    grad, costt = cp_lstsq_grad((None, tensF), tensor, return_loss=True, mask=tmask)
    gradd = np.concatenate([g.flatten() for g in grad[1]])
    return costt, gradd


class do_callback():
    def __init__(self, callback, r, shape):
        self.callback = callback
        self.r = r
        self.shape = shape

    def __call__(self, x):
        tensorFac = CPTensor((None, buildTensors(x, self.r, self.shape)))
        tensorFac = cp_normalize(tensorFac)
        for ii in range(len(self.shape)):
            tensorFac.factors[ii] = np.array(tensorFac.factors[ii])
        self.callback(tensorFac)


def perform_DO(tensorOrig=None, rank=6, n_iter_max=50, callback=None, init=None, mask=None) -> tl.cp_tensor.CPTensor:
    """ Perform CP decomposition. """
    if tensorOrig is None: tensorOrig = createUnknownRank()
    if init==None: init=initialize_fac(tensorOrig, rank)
    if callback:
        callback(init)
        temp_callback = do_callback(callback, rank, tensorOrig.shape)
    else: temp_callback = None

    tensorIn = tensorOrig.copy()
    tmask = np.isnan(tensorIn)
    tensorIn[tmask] = 0.0

    x0 = np.concatenate(tuple([np.ravel(init.factors[ii]) for ii in range(np.ndim(tensorIn))]))

    rgs = (tensorIn, tmask, rank)
    res = minimize(cost, x0, method='L-BFGS-B', jac=True, args=rgs, options={"maxiter":n_iter_max}, callback=temp_callback)
    tensorFac = CPTensor((None, buildTensors(res.x, rank, tensorIn.shape)))
    tensorFac = cp_normalize(tensorFac)

    for ii in range(np.ndim(tensorIn)):
        tensorFac.factors[ii] = np.array(tensorFac.factors[ii])

    return tensorFac