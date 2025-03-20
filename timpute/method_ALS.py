import numpy as np

import tensorly as tl
from tensorly.cp_tensor import (
    cp_to_tensor,
)
from tensorly.cp_tensor import cp_normalize
from tqdm import tqdm
from .linesearch import Nesterov
from tensorly.tenalg.core_tenalg.mttkrp import unfolding_dot_khatri_rao
from .initialization import initialize_fac
from .impute_helper import calcR2X


def perform_ALS(
    tOrig,
    rank,
    n_iter_max=50,
    init=None,
    tol=1e-6,
    callback=None,
    progress=False,
) -> tl.cp_tensor.CPTensor:
    """CANDECOMP/PARAFAC decomposition via alternating least squares (ALS)
    Computes a rank-`rank` decomposition of `tensor` [1]_ such that:

    ``tensor = [|weights; factors[0], ..., factors[-1] |]``.

    Parameters
    ----------
    tensor : ndarray
    rank  : int
        Number of components.
    n_iter_max : int
        Maximum number of iteration
    init : {'svd', 'random', CPTensor}, optional
        Type of factor matrix initialization.
        If a CPTensor is passed, this is directly used for initalization.
        See `initialize_factors`.
    tol : float, optional
        (Default: 1e-6) Relative reconstruction error tolerance. The
        algorithm is considered to have found the global minimum when the
        reconstruction error is less than `tol`.

    Returns
    -------
    CPTensor : (weight, factors)
        * weights : 1D array of shape (rank, )

          * all ones if normalize_factors is False (default)
          * weights of the (normalized) factors otherwise

        * factors : List of factors of the CP decomposition element `i` is of shape ``(tensor.shape[i], rank)``

    errors : list
        A list of reconstruction errors at each iteration of the algorithms.

    References
    ----------
    .. [1] T.G.Kolda and B.W.Bader, "Tensor Decompositions and Applications", SIAM
           REVIEW, vol. 51, n. 3, pp. 455-500, 2009.
    .. [2] Tomasi, Giorgio, and Rasmus Bro. "PARAFAC and missing values."
           Chemometrics and Intelligent Laboratory Systems 75.2 (2005): 163-180.
    .. [3] R. Bro, "Multi-Way Analysis in the Food Industry: Models, Algorithms, and
           Applications", PhD., University of Amsterdam, 1998
    """
    tensor = np.nan_to_num(tOrig)
    mask = np.isfinite(tOrig)

    if init is None:
        tFac = initialize_fac(tensor, rank, method="random")
    else:
        tFac = init

    linesrc = Nesterov()
    fac, R2X, jump = linesrc.perform(tFac.factors, tOrig)
    tFac.R2X = R2X
    tFac.factors = fac

    tq = tqdm(range(n_iter_max), disable=(not progress))
    for _ in tq:
        weights, factors = tFac

        # Update the tensor based on the mask
        low_rank_component = cp_to_tensor((weights, factors))
        tensor = tensor * mask + low_rank_component * (1 - mask)

        for mode in range(np.ndim(tensor)):
            pseudo_inverse = np.ones((rank, rank))
            for i, factor in enumerate(factors):
                if i != mode:
                    pseudo_inverse = pseudo_inverse * np.dot(factor.T, factor)
            pseudo_inverse = (
                np.reshape(weights, (-1, 1))
                * pseudo_inverse
                * np.reshape(weights, (1, -1))
            )
            mttkrp = unfolding_dot_khatri_rao(tensor, (weights, factors), mode)

            factor = np.linalg.solve(pseudo_inverse.T, mttkrp.T).T
            factors[mode] = factor

        R2X_last = tFac.R2X

        fac, R2X, jump = linesrc.perform(tFac.factors, tOrig)

        if R2X - R2X_last < tol:
            break

        tFac.R2X = R2X
        tFac.factors = fac

        tq.set_postfix(
            R2X=tFac.R2X, delta=tFac.R2X - R2X_last, jump=jump, refresh=False
        )
        # assert tFac.R2X > 0.0

        if callback:
            callback(tFac)

    tFac = cp_normalize(tFac)
    tFac.R2X = calcR2X(tFac, tOrig)

    return tFac
