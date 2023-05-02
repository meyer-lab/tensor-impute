"""
Censored Least Squares
"""

import numpy as np
import tensorly as tl
from tensorly.cp_tensor import cp_normalize
from tensorly.tenalg import khatri_rao
from .initialize_fac import initialize_fac
from tqdm import tqdm
from sklearn.linear_model import Ridge


tl.set_backend('numpy')


def calcR2X(tFac, tIn, calcError=False, mask=None):
    """ Calculate R2X. Optionally it can be calculated for only the tensor or matrix. """
    vTop, vBottom = 0.0, 0.0

    tOrig = np.copy(tIn)
    if mask is not None:
        recons_tFac = tl.cp_to_tensor(tFac)*mask
        tOrig = tOrig*mask
    else:
        recons_tFac = tl.cp_to_tensor(tFac)
    tMask = np.isfinite(tOrig)
    tOrig = np.nan_to_num(tOrig)
    vTop += np.linalg.norm(recons_tFac * tMask - tOrig)**2.0
    vBottom += np.linalg.norm(tOrig)**2.0

    if calcError: return vTop / vBottom
    else: return 1 - vTop / vBottom


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

  
def perform_CLS(tOrig, rank=6, alpha=None, tol=1e-6, n_iter_max=50, progress=False, callback=None, init=None, mask=None):
    """ Perform CP decomposition. """

    if init==None: tFac = initialize_fac(tOrig, rank)
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

    return tFac
