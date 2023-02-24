"""
Censored Least Squares
"""

import numpy as np
from tensorly.tenalg import svd_interface
import tensorly as tl
from tensorly.tenalg import khatri_rao
from .initialize_fac import initialize_fac
from copy import deepcopy
from tqdm import tqdm
from sklearn.linear_model import Ridge


tl.set_backend('numpy')


def buildMat(tFac):
    """ Build the matrix in CMTF from the factors. """
    if hasattr(tFac, 'mWeights'):
        return tFac.factors[0] @ (tFac.mFactor * tFac.mWeights).T
    return tFac.factors[0] @ tFac.mFactor.T


def calcR2X(tFac, tIn=None, mIn=None, calcError=False, mask=None):
    """ Calculate R2X. Optionally it can be calculated for only the tensor or matrix. """
    assert (tIn is not None) or (mIn is not None)

    vTop, vBottom = 0.0, 0.0

    if tIn is not None:
        tOrig = np.copy(tIn)
        if mask is not None:
            tTens = tl.cp_to_tensor(tFac)*mask
            tOrig = tIn*mask
        tMask = np.isfinite(tOrig)
        tIn = np.nan_to_num(tOrig)
        vTop += np.linalg.norm(tTens * tMask - tOrig)**2.0
        vBottom += np.linalg.norm(tIn)**2.0
    if mIn is not None:
        mMask = np.isfinite(mIn)
        recon = tFac if isinstance(tFac, np.ndarray) else buildMat(tFac)
        mIn = np.nan_to_num(mIn)
        vTop += np.linalg.norm(recon * mMask - mIn)**2.0
        vBottom += np.linalg.norm(mIn)**2.0

    if calcError: return vTop / vBottom
    else: return 1 - vTop / vBottom

def tensor_degFreedom(tFac) -> int:
    """ Calculate the degrees of freedom within a tensor factorization. """
    deg = np.sum([f.size for f in tFac.factors])

    if hasattr(tFac, 'mFactor'):
        deg += tFac.mFactor.size

    return deg


def reorient_factors(tFac):
    """ This function ensures that factors are negative on at most one direction. """
    # Flip the types to be positive
    tMeans = np.sign(np.mean(tFac.factors[2], axis=0))
    tFac.factors[1] *= tMeans[np.newaxis, :]
    tFac.factors[2] *= tMeans[np.newaxis, :]

    # Flip the cytokines to be positive
    rMeans = np.sign(np.mean(tFac.factors[1], axis=0))
    tFac.factors[0] *= rMeans[np.newaxis, :]
    tFac.factors[1] *= rMeans[np.newaxis, :]

    if hasattr(tFac, 'mFactor'):
        tFac.mFactor *= rMeans[np.newaxis, :]
    return tFac


def sort_factors(tFac):
    """ Sort the components from the largest variance to the smallest. """
    tensor = deepcopy(tFac)

    # Variance separated by component
    norm = np.copy(tFac.weights)
    for factor in tFac.factors:
        norm *= np.sum(np.square(factor), axis=0)

    # Add the variance of the matrix
    if hasattr(tFac, 'mFactor'):
        norm += np.sum(np.square(tFac.factors[0]), axis=0) * np.sum(np.square(tFac.mFactor), axis=0) * tFac.mWeights

    order = np.flip(np.argsort(norm))
    tensor.weights = tensor.weights[order]
    tensor.factors = [fac[:, order] for fac in tensor.factors]
    np.testing.assert_allclose(tl.cp_to_tensor(tFac), tl.cp_to_tensor(tensor), atol=1e-9)

    if hasattr(tFac, 'mFactor'):
        tensor.mFactor = tensor.mFactor[:, order]
        tensor.mWeights = tensor.mWeights[order]
        np.testing.assert_allclose(buildMat(tFac), buildMat(tensor), atol=1e-9)

    return tensor


def delete_component(tFac, compNum):
    """ Delete the indicated component. """
    tensor = deepcopy(tFac)
    compNum = np.array(compNum, dtype=int)

    # Assert that component # don't exceed range, and are unique
    assert np.amax(compNum) < tensor.rank
    assert np.unique(compNum).size == compNum.size

    tensor.rank -= compNum.size
    tensor.weights = np.delete(tensor.weights, compNum)

    if hasattr(tFac, 'mFactor'):
        tensor.mFactor = np.delete(tensor.mFactor, compNum, axis=1)
        tensor.mWeights = np.delete(tensor.mWeights, compNum)

    tensor.factors = [np.delete(fac, compNum, axis=1) for fac in tensor.factors]
    return tensor


def censored_lstsq(A: np.ndarray, B: np.ndarray, uniqueInfo=None, alpha=None) -> np.ndarray:
    """Solves least squares problem subject to missing data.
    Note: uses a for loop over the missing patterns of B, leading to a
    slower but more numerically stable algorithm
    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))
    """
    X = np.empty((A.shape[1], B.shape[1]))
    # Missingness patterns
    if uniqueInfo is None:
        unique, uIDX = np.unique(np.isfinite(B), axis=1, return_inverse=True)
    else:
        unique, uIDX = uniqueInfo

    for i in range(unique.shape[1]):
        uI = uIDX == i
        uu = np.squeeze(unique[:, i])

        Bx = B[uu, :]

        if alpha is None:
            X[:, uI] = np.linalg.lstsq(A[uu, :], Bx[:, uI], rcond=-1)[0]
        else:
            clf = Ridge(alpha=alpha, fit_intercept=False)
            clf.fit(A[uu, :], Bx[:, uI])
            X[:, uI] = clf.coef_.T
    return X.T


def cp_normalize(tFac):
    """ Normalize the factors using the inf norm. """
    for i, factor in enumerate(tFac.factors):
        scales = np.linalg.norm(factor, ord=np.inf, axis=0)
        tFac.weights *= scales
        if i == 0 and hasattr(tFac, 'mFactor'):
            mScales = np.linalg.norm(tFac.mFactor, ord=np.inf, axis=0)
            tFac.mWeights = scales * mScales
            tFac.mFactor /= mScales

        tFac.factors[i] /= scales

    return tFac

  
def perform_CLS(tOrig, rank=6, alpha=None, tol=1e-6, n_iter_max=50, progress=False, callback=None, init=Nonem mask=None):
    """ Perform CP decomposition. """

    if init==None: tFac = initialize_fac(tOrig.copy(), rank)
    else: tFac = init
    if callback: callback(tFac)
    
    # Pre-unfold
    unfolded = [tl.unfold(tOrig, i) for i in range(tOrig.ndim)]
    R2X_last = -np.inf
    tFac.R2X = calcR2X(tFac, tOrig)

    # Precalculate the missingness patterns
    uniqueInfo = [np.unique(np.isfinite(B.T), axis=1, return_inverse=True) for B in unfolded]

    tq = tqdm(range(n_iter_max), disable=(not progress))
    for i in tq:
        # Solve on each mode
        for m in range(len(tFac.factors)):
            kr = khatri_rao(tFac.factors, skip_matrix=m)
            tFac.factors[m] = censored_lstsq(kr, unfolded[m].T, uniqueInfo[m], alpha=alpha)
        
        R2X_last = tFac.R2X
        tFac.R2X = calcR2X(tFac, tOrig)
        tq.set_postfix(R2X=tFac.R2X, delta=tFac.R2X - R2X_last, refresh=False)
        # assert tFac.R2X > 0.0
        if callback: callback(tFac)

        # if tFac.R2X - R2X_last < tol:
        #     break

    tFac = cp_normalize(tFac)
    tFac = reorient_factors(tFac)

    if rank > 1:
        tFac = sort_factors(tFac)

    return tFac
