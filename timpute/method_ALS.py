import numpy as np
import warnings

import tensorly as tl
from tensorly.cp_tensor import (
    cp_to_tensor,
    CPTensor,
    cp_norm,
    cp_normalize,
    validate_cp_rank,
)
from tensorly.tenalg.core_tenalg.mttkrp import unfolding_dot_khatri_rao
from .initialization import initialize_fac

# Authors: Jean Kossaifi <jean.kossaifi+tensors@gmail.com>
#          Chris Swierczewski <csw@amazon.com>
#          Sam Schneider <samjohnschneider@gmail.com>
#          Aaron Meurer <asmeurer@gmail.com>

# License: BSD 3 clause


def sparsify_tensor(tensor, card):
    """Zeros out all elements in the `tensor` except `card` elements with maximum absolute values.

    Parameters
    ----------
    tensor : ndarray
    card : int
        Desired number of non-zero elements in the `tensor`

    Returns
    -------
    ndarray of shape tensor.shape
    """
    if card >= np.prod(tensor.shape):
        return tensor
    bound = tl.sort(tl.abs(tensor), axis=None)[-card]

    return tl.where(
        tl.abs(tensor) < bound, tl.zeros(tensor.shape, **tl.context(tensor)), tensor
    )


def error_calc(tensor, norm_tensor, weights, factors, sparsity, mask, mttkrp=None):
    r"""Perform the error calculation. Different forms are used here depending upon
    the available information. If `mttkrp=None` or masking is being performed, then the
    full tensor must be constructed. Otherwise, the mttkrp is used to reduce the calculation cost.

    Parameters
    ----------
    tensor : tensor
    norm_tensor : float
        The l2 norm of tensor.
    weights : tensor
        The current CP weights
    factors : tensor
        The current CP factors
    sparsity : float or int
        Whether we allow for a sparse component
    mask : bool
        Whether masking is being performed.
    mttkrp : tensor or None
        The mttkrp product, if available.

    Returns
    -------
    unnorml_rec_error : float
        The unnormalized reconstruction error.
    tensor : tensor
        The tensor, in case it has been updated by masking.
    norm_tensor: float
        The tensor norm, in case it has been updated by masking.
    """

    # If we have to update the mask we already have to build the full tensor
    if (mask is not None) or (mttkrp is None):
        low_rank_component = cp_to_tensor((weights, factors))

        # Update the tensor based on the mask
        if mask is not None:
            tensor = tensor * mask + low_rank_component * (1 - mask)
            norm_tensor = tl.norm(tensor, 2)

        if sparsity:
            sparse_component = sparsify_tensor(tensor - low_rank_component, sparsity)
        else:
            sparse_component = 0.0

        unnorml_rec_error = tl.norm(tensor - low_rank_component - sparse_component, 2)
    else:
        if sparsity:
            low_rank_component = cp_to_tensor((weights, factors))
            sparse_component = sparsify_tensor(tensor - low_rank_component, sparsity)

            unnorml_rec_error = tl.norm(
                tensor - low_rank_component - sparse_component, 2
            )
        else:
            # ||tensor - rec||^2 = ||tensor||^2 + ||rec||^2 - 2*<tensor, rec>
            factors_norm = cp_norm((weights, factors))

            # mttkrp and factor for the last mode. This is equivalent to the
            # inner product <tensor, factorization>
            iprod = tl.sum(tl.sum(mttkrp * tl.conj(factors[-1]), axis=0))
            unnorml_rec_error = tl.sqrt(
                tl.abs(norm_tensor**2 + factors_norm**2 - 2 * iprod)
            )

    return unnorml_rec_error, tensor, norm_tensor


def perform_ALS(
    tensor,
    rank,
    n_iter_max=100,
    init=None,
    svd="truncated_svd",
    normalize_factors=False,
    orthogonalise=False,
    tol=1e-8,
    random_state=None,
    verbose=0,
    return_errors=False,
    sparsity=None,
    l2_reg=0,
    mask=None,
    cvg_criterion="abs_rec_error",
    fixed_modes=None,
    svd_mask_repeats=5,
    linesearch=False,
    callback=None,
)  -> tl.cp_tensor.CPTensor:
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
    svd : str, default is 'truncated_svd'
        function to use to compute the SVD, acceptable values in tensorly.SVD_FUNS
    normalize_factors : if True, aggregate the weights of each factor in a 1D-tensor
        of shape (rank, ), which will contain the norms of the factors
    tol : float, optional
        (Default: 1e-6) Relative reconstruction error tolerance. The
        algorithm is considered to have found the global minimum when the
        reconstruction error is less than `tol`.
    random_state : {None, int, np.random.RandomState}
    verbose : int, optional
        Level of verbosity
    return_errors : bool, optional
        Activate return of iteration errors
    mask : ndarray
        array of booleans with the same shape as ``tensor`` should be 0 where
        the values are missing and 1 everywhere else. Note:  if tensor is
        sparse, then mask should also be sparse with a fill value of 1 (or
        True). Allows for missing values [2]_
    cvg_criterion : {'abs_rec_error', 'rec_error'}, optional
       Stopping criterion for ALS, works if `tol` is not None.
       If 'rec_error',  ALS stops at current iteration if ``(previous rec_error - current rec_error) < tol``.
       If 'abs_rec_error', ALS terminates when `|previous rec_error - current rec_error| < tol`.
    sparsity : float or int
        If `sparsity` is not None, we approximate tensor as a sum of low_rank_component and sparse_component, where low_rank_component = cp_to_tensor((weights, factors)). `sparsity` denotes desired fraction or number of non-zero elements in the sparse_component of the `tensor`.
    fixed_modes : list, default is None
        A list of modes for which the initial value is not modified.
        The last mode cannot be fixed due to error computation.
    svd_mask_repeats: int
        If using a tensor with masked values, this initializes using SVD multiple times to
        remove the effect of these missing values on the initialization.
    linesearch : bool, default is False
        Whether to perform line search as proposed by Bro [3].

    Returns
    -------
    CPTensor : (weight, factors)
        * weights : 1D array of shape (rank, )

          * all ones if normalize_factors is False (default)
          * weights of the (normalized) factors otherwise

        * factors : List of factors of the CP decomposition element `i` is of shape ``(tensor.shape[i], rank)``
        * sparse_component : nD array of shape tensor.shape. Returns only if `sparsity` is not None.

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

    rank = validate_cp_rank(tl.shape(tensor), rank=rank)

    if return_errors:
        DeprecationWarning(
            "return_errors argument will be removed in the next version of TensorLy. Please use a callback function instead."
        )

    if orthogonalise and not isinstance(orthogonalise, int):
        orthogonalise = n_iter_max

    if linesearch:
        acc_pow = 2.0  # Extrapolate to the iteration^(1/acc_pow) ahead
        acc_fail = 0  # How many times acceleration have failed
        max_fail = 4  # Increase acc_pow with one after max_fail failure

    
    if init is None: init = initialize_fac(tensor,rank)
    weights, factors = init

    rec_errors = []
    norm_tensor = tl.norm(tensor, 2)
    if l2_reg:
        Id = tl.eye(rank, **tl.context(tensor)) * l2_reg
    else:
        Id = 0

    if fixed_modes is None:
        fixed_modes = []

    if fixed_modes == list(range(tl.ndim(tensor))):  # Check If all modes are fixed
        cp_tensor = CPTensor(
            (weights, factors)
        )  # No need to run optimization algorithm, just return the initialization
        return cp_tensor

    if tl.ndim(tensor) - 1 in fixed_modes:
        warnings.warn(
            "You asked for fixing the last mode, which is not supported.\n The last mode will not be fixed. Consider using tl.moveaxis()"
        )
        fixed_modes.remove(tl.ndim(tensor) - 1)
    modes_list = [mode for mode in range(tl.ndim(tensor)) if mode not in fixed_modes]

    if sparsity:
        sparse_component = tl.zeros_like(tensor)
        if isinstance(sparsity, float):
            sparsity = int(sparsity * np.prod(tensor.shape))
        else:
            sparsity = int(sparsity)

    for iteration in range(n_iter_max):
        if orthogonalise and iteration <= orthogonalise:
            factors = [
                tl.qr(f)[0] if min(tl.shape(f)) >= rank else f
                for i, f in enumerate(factors)
            ]

        if linesearch and iteration % 2 == 0:
            factors_last = [tl.copy(f) for f in factors]
            weights_last = tl.copy(weights)

        if verbose > 1:
            print("Starting iteration", iteration + 1)
        for mode in modes_list:
            if verbose > 1:
                print("Mode", mode, "of", tl.ndim(tensor))

            pseudo_inverse = tl.tensor(np.ones((rank, rank)), **tl.context(tensor))
            for i, factor in enumerate(factors):
                if i != mode:
                    pseudo_inverse = pseudo_inverse * tl.dot(
                        tl.conj(tl.transpose(factor)), factor
                    )
            pseudo_inverse += Id
            pseudo_inverse = (
                tl.reshape(weights, (-1, 1))
                * pseudo_inverse
                * tl.reshape(weights, (1, -1))
            )
            mttkrp = unfolding_dot_khatri_rao(tensor, (weights, factors), mode)

            factor = tl.transpose(
                tl.solve(tl.conj(tl.transpose(pseudo_inverse)), tl.transpose(mttkrp))
            )
            factors[mode] = factor
            if normalize_factors and mode != modes_list[-1]:
                weights, factors = cp_normalize((weights, factors))

        # Will we be performing a line search iteration
        if linesearch and iteration % 2 == 0 and iteration > 5:
            line_iter = True
        else:
            line_iter = False

        # Calculate the current unnormalized error if we need it
        if (tol or return_errors) and not line_iter:
            unnorml_rec_error, tensor, norm_tensor = error_calc(
                tensor, norm_tensor, weights, factors, sparsity, mask, mttkrp
            )
        else:
            if mask is not None:
                tensor = tensor * mask + tl.cp_to_tensor(
                    (weights, factors), mask=1 - mask
                )

        # Start line search if requested.
        if line_iter:
            jump = iteration ** (1.0 / acc_pow)

            new_weights = weights_last + (weights - weights_last) * jump
            new_factors = [
                factors_last[ii] + (factors[ii] - factors_last[ii]) * jump
                for ii in range(tl.ndim(tensor))
            ]

            new_rec_error, new_tensor, new_norm_tensor = error_calc(
                tensor, norm_tensor, new_weights, new_factors, sparsity, mask
            )

            if (new_rec_error / new_norm_tensor) < rec_errors[-1]:
                factors, weights = new_factors, new_weights
                tensor, norm_tensor = new_tensor, new_norm_tensor
                unnorml_rec_error = new_rec_error
                acc_fail = 0

                if verbose:
                    print(f"Accepted line search jump of {jump}.")
            else:
                unnorml_rec_error, tensor, norm_tensor = error_calc(
                    tensor, norm_tensor, weights, factors, sparsity, mask, mttkrp
                )
                acc_fail += 1

                if verbose:
                    print(f"Line search failed for jump of {jump}.")

                if acc_fail == max_fail:
                    acc_pow += 1.0
                    acc_fail = 0

                    if verbose:
                        print("Reducing acceleration.")

        if (tol or return_errors) and not line_iter:
            rec_error = unnorml_rec_error / norm_tensor
            rec_errors.append(rec_error)

        if callback is not None:
            cp_tensor = CPTensor((weights, factors))

            if sparsity:
                sparse_component = sparsify_tensor(
                    tensor - cp_to_tensor((weights, factors)), sparsity
                )
                retVal = callback((cp_tensor, sparse_component), rec_error)
            else:
                retVal = callback(cp_tensor)

        if tol is not None:
            if iteration >= 1:
                rec_error_decrease = rec_errors[-2] - rec_errors[-1]

                if verbose:
                    print(
                        f"iteration {iteration}, reconstruction error: {rec_error}, decrease = {rec_error_decrease}, unnormalized = {unnorml_rec_error}"
                    )

                if cvg_criterion == "abs_rec_error":
                    stop_flag = abs(rec_error_decrease) < tol
                elif cvg_criterion == "rec_error":
                    stop_flag = rec_error_decrease < tol
                else:
                    raise TypeError("Unknown convergence criterion")

                if stop_flag:
                    if verbose:
                        print(f"PARAFAC converged after {iteration} iterations")
                    break

            else:
                if verbose:
                    print(f"reconstruction error={rec_errors[-1]}")
        if normalize_factors:
            weights, factors = cp_normalize((weights, factors))

    cp_tensor = CPTensor((weights, factors))

    if sparsity:
        sparse_component = sparsify_tensor(
            tensor - cp_to_tensor((weights, factors)), sparsity
        )
        cp_tensor = (cp_tensor, sparse_component)

    if return_errors:
        return cp_tensor, rec_errors
    else:
        return cp_tensor