"""
Censored Least Squares
"""

import numpy as np
import tensorly as tl
from tensorly.cp_tensor import cp_normalize
from tensorly.tenalg import khatri_rao
from .initialization import initialize_fac
from .impute_helper import calcR2X, reorient_factors
from tqdm import tqdm
from sklearn.linear_model import Ridge


tl.set_backend('numpy')

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

  
def perform_CLS(tOrig,
                rank=6,
                init=None, 
                alpha=None,
                tol=1e-6,
                n_iter_max=50,
                progress=False,
                callback=None,
                **kwargs
)  -> tl.cp_tensor.CPTensor:
    """ Perform CP decomposition. """

    if init==None: tFac = initialize_fac(tOrig, rank)
    else:
        tFac = init
        tFac_last = init
    
    # Pre-unfold
    unfolded = [tl.unfold(tOrig, i) for i in range(tOrig.ndim)]
    R2X_last = -np.inf
    tFac.R2X = calcR2X(tFac, tOrig)

    # Precalculate the missingness patterns
    uniqueInfo = [np.unique(np.isfinite(B.T), axis=1, return_inverse=True) for B in unfolded]

    tq = tqdm(range(n_iter_max), disable=(not progress))
    for _ in tq:
        # Solve on each mode
        for m in range(len(tFac.factors)):
            kr = khatri_rao(tFac.factors, skip_matrix=m)
            tFac.factors[m] = censored_lstsq(kr, unfolded[m].T, uniqueInfo[m], alpha=alpha)
        
        R2X_last = tFac.R2X
        tFac.R2X = calcR2X(tFac, tOrig)
        tq.set_postfix(R2X=tFac.R2X, delta=tFac.R2X - R2X_last, refresh=False)

        if callback: callback(tFac)
        if tFac.R2X - R2X_last < tol:
            break


    tFac = cp_normalize(tFac)
    tFac = reorient_factors(tFac)
    tFac.R2X = calcR2X(tFac, tOrig)

    return tFac
