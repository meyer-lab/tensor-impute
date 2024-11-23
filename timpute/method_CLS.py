"""
Censored Least Squares
"""

import numpy as np
import tensorly as tl
from tensorly.cp_tensor import cp_flip_sign, cp_normalize
from tensorly.tenalg import khatri_rao
from scipy.linalg import solve as sp_solve
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
    rank: int=6,
    init=None,
    alpha=None,
    tol: float=1e-6,
    n_iter_max: int=50,
    progress=False,
    callback=None,
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

    tq = tqdm(range(n_iter_max), disable=(not progress))
    for _ in tq:
        # Solve on each mode
        for m in range(len(tFac.factors)):
            kr = khatri_rao(tFac.factors, skip_matrix=m)
            tFac.factors[m] = censored_lstsq(
                kr, unfolded[m].T, uniqueInfo[m], alpha=alpha
            )

        R2X_last = R2X
        fac, R2X, jump = linesrc.perform(tFac.factors, tOrig)
        tFac.factors = fac

        tq.set_postfix(
            R2X=R2X, delta=R2X - R2X_last, jump=jump, refresh=False
        )

        if callback:
            callback(tFac)
        if R2X - R2X_last < tol:
            break

    tFac = cp_normalize(tFac)
    tFac = cp_flip_sign(tFac)
    tFac.R2X = R2X

    return tFac
