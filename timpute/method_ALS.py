import numpy as np

import tensorly as tl
from tensorly.cp_tensor import (
    cp_to_tensor,
    CPTensor,
)
from tensorly.tenalg.core_tenalg.mttkrp import unfolding_dot_khatri_rao
from .initialization import initialize_fac


def perform_ALS(
    tensor,
    rank,
    n_iter_max=100,
    init=None,
    tol=1e-6,
    mask=None,
    callback=None,
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
    mask : ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else. Note:  if tensor is
        sparse, then mask should also be sparse with a fill value of 1 (or
        True). Allows for missing values [2]_

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
    tensor = np.nan_to_num(tensor)

    if init is None:
        init = initialize_fac(tensor, rank, method="random")

    weights, factors = init

    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)

    for iteration in range(n_iter_max):
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

            factor = np.solve(pseudo_inverse.T, mttkrp.T).T
            factors[mode] = factor

        # Calculate the current unnormalized error
        low_rank_component = cp_to_tensor((weights, factors))

        # Update the tensor based on the mask
        if mask is not None:
            tensor = tensor * mask + low_rank_component * (1 - mask)
            norm_tensor = tl.norm(tensor, 2)

        unnorml_rec_error = tl.norm(tensor - low_rank_component, 2)

        if tol:
            rec_error = unnorml_rec_error / norm_tensor
            rec_errors.append(rec_error)

        if callback is not None:
            cp_tensor = CPTensor((weights, factors))
            callback(cp_tensor)

        if tol is not None:
            if iteration >= 1:
                rec_error_decrease = rec_errors[-2] - rec_errors[-1]

                if abs(rec_error_decrease) < tol:
                    break

    cp_tensor = CPTensor((weights, factors))

    return cp_tensor
