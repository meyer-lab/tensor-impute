import numpy as np
import tensorly as tl
from tensorly.random import random_cp
from tensorly.tenalg import svd_interface


class IterativeSVD(object):
    def __init__(
        self,
        rank,
        convergence_threshold=1e-7,
        max_iters=500,
        random_state=None,
        min_value=None,
        max_value=None,
        verbose=False,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.rank = rank
        self.max_iters = max_iters
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose
        self.random_state = random_state

    def clip(self, X):
        """
        Clip values to fall within any global or column-wise min/max constraints
        """
        X = np.asarray(X)
        if self.min_value is not None:
            X[X < self.min_value] = self.min_value
        if self.max_value is not None:
            X[X > self.max_value] = self.max_value
        return X

    def prepare_input_data(self, X):
        """
        Check to make sure that the input matrix and its mask of missing
        values are valid. Returns X and missing mask.
        """
        if X.dtype != "f" and X.dtype != "d":
            X = X.astype(float)

        assert X.ndim == 2
        missing_mask = np.isnan(X)
        assert not missing_mask.all()
        return X, missing_mask

    def fit_transform(self, X, y=None):
        """
        Fit the imputer and then transform input `X`
        Note: all imputations should have a `fit_transform` method,
        but only some (like IterativeImputer in sklearn) also support inductive
        mode using `fit` or `fit_transform` on `X_train` and then `transform`
        on new `X_test`.
        """
        X_original, missing_mask = self.prepare_input_data(X)
        observed_mask = ~missing_mask
        X_filled = X_original.copy()
        X_filled[missing_mask] = 0.0
        assert isinstance(X_filled, np.ndarray)
        X_result = self.solve(X_filled, missing_mask)
        assert isinstance(X_result, np.ndarray)
        X_result = self.clip(np.asarray(X_result))
        X_result[observed_mask] = X_original[observed_mask]
        return X_result

    def _converged(self, X_old, X_new, missing_mask):
        F32PREC = np.finfo(np.float32).eps
        # check for convergence
        old_missing_values = X_old[missing_mask]
        new_missing_values = X_new[missing_mask]
        difference = old_missing_values - new_missing_values
        ssd = np.sum(difference**2)
        old_norm_squared = (old_missing_values**2).sum()
        # edge cases
        if old_norm_squared == 0 or (old_norm_squared < F32PREC and ssd > F32PREC):
            return False
        else:
            return (ssd / old_norm_squared) < self.convergence_threshold

    def solve(self, X, missing_mask):
        observed_mask = ~missing_mask
        X_filled = X
        for i in range(self.max_iters):
            curr_rank = self.rank
            self.U, S, V = svd_interface(
                matrix=X_filled, n_eigenvecs=curr_rank, random_state=self.random_state
            )
            X_reconstructed = self.U @ np.diag(S) @ V
            X_reconstructed = self.clip(X_reconstructed)

            # Masked mae
            mae = np.mean(np.abs(X[observed_mask] - X_reconstructed[observed_mask]))

            if self.verbose:
                print("[IterativeSVD] Iter %d: observed MAE=%0.6f" % (i + 1, mae))
            converged = self._converged(
                X_old=X_filled, X_new=X_reconstructed, missing_mask=missing_mask
            )
            X_filled[missing_mask] = X_reconstructed[missing_mask]
            if converged:
                break
        return X_filled


def initialize_fac(tensor: np.ndarray, rank: int, method="svd"):
    """Initialize factors used in `parafac`.
    Parameters
    ----------
    tensor : ndarray
    rank : int
    Returns
    -------
    factors : CPTensor
        An initial cp tensor.
    """
    if method == "random":
        return random_cp(shape=tensor.shape, rank=rank, normalise_factors=False)

    factors = [np.ones((tensor.shape[i], rank)) for i in range(tensor.ndim)]
    contain_missing = np.sum(~np.isfinite(tensor)) > 0

    # SVD init mode whose size is larger than rank
    for mode in range(tensor.ndim):
        if tensor.shape[mode] >= rank:
            unfold = tl.unfold(tensor, mode)
            if contain_missing:
                si = IterativeSVD(rank)
                unfold = si.fit_transform(unfold)

            factors[mode] = svd_interface(matrix=unfold, n_eigenvecs=rank, flip=True)[0]
        else:  # tensor.shape[mode] < rank
            unfold = tl.unfold(tensor, mode)
            if contain_missing:
                si = IterativeSVD(tensor.shape[mode])
                unfold = si.fit_transform(unfold)  # unfold (tensor.shape[mode] x ...)

            svd_factor = svd_interface(matrix=unfold, n_eigenvecs=rank, flip=True)[0]
            factors[mode][:, 0 : tensor.shape[mode]] = svd_factor

    return tl.cp_tensor.CPTensor((None, factors))
