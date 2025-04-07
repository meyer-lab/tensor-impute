"""
Censored Least Squares
"""

from copy import deepcopy

import numpy as np
import tensorly as tl
from scipy.linalg import solve as sp_solve
from tensorly.cp_tensor import cp_flip_sign, cp_normalize
from tensorly.tenalg import khatri_rao
from tqdm import tqdm

from .initialization import initialize_fac
from .linesearch import Nesterov

tl.set_backend("numpy")


def ridge_solve_cholesky(X, y, alpha: float):
    # w = inv(X^t X + alpha*Id) * X.T y
    A = X.T @ X
    Xy = X.T @ y

    A.flat[:: X.shape[1] + 1] += alpha
    return sp_solve(A, Xy, assume_a="pos", overwrite_a=True)


def censored_lstsq(
    A: np.ndarray, B: np.ndarray, uniqueInfo=None, alpha=None
) -> np.ndarray:
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
        uu = unique[:, i]

        Bx = B[uu, :]

        if alpha is None:
            X[:, uI] = np.linalg.lstsq(A[uu, :], Bx[:, uI], rcond=None)[0]
        else:
            X[:, uI] = ridge_solve_cholesky(A[uu, :], Bx[:, uI], alpha=alpha)
    return X.T


def perform_CLS(
    tOrig: np.ndarray,
    rank: int = 6,
    init=None,
    alpha=None,
    tol=1e-6,
    n_iter_max=50,
    verbose=False,
    callback=None,
    **kwargs,
) -> tl.cp_tensor.CPTensor:
    """Perform CP decomposition."""

    if init is None:
        tFac = initialize_fac(tOrig, rank)
    else:
        tFac = init

    # Pre-unfold
    unfolded = [tl.unfold(tOrig, i) for i in range(tOrig.ndim)]
    R2X_last = -np.inf

    linesrc = Nesterov()
    fac, R2X, jump = linesrc.perform(tFac.factors, tOrig)
    tFac.factors = fac

    # Precalculate the missingness patterns
    uniqueInfo = [
        np.unique(np.isfinite(B.T), axis=1, return_inverse=True) for B in unfolded
    ]

    tq = tqdm(range(n_iter_max), disable=(not verbose))
    for _ in tq:
        R2X_last = R2X
        # Solve on each mode
        tFac_old = deepcopy(tFac)
        for m in range(len(tFac.factors)):
            kr = khatri_rao(tFac.factors, skip_matrix=m)
            tFac.factors[m] = censored_lstsq(
                kr, unfolded[m].T, uniqueInfo[m], alpha=alpha
            )

        if verbose is True:
            print(len(fac))
            print(fac[0].shape, fac[1].shape, fac[2].shape)

        fac, R2X, jump = linesrc.perform(tFac.factors, tOrig)

        if R2X - R2X_last < tol:
            tFac = tFac_old
            break

        tq.set_postfix(R2X=R2X, delta=R2X - R2X_last, jump=jump, refresh=False)

        if callback:
            callback(tFac)

    tFac = cp_normalize(tFac)
    tFac = cp_flip_sign(tFac)
    tFac.R2X = R2X

    return tFac
