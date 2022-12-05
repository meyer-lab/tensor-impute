import numpy as np
from tensorly import svd_interface
import tensorly as tl
from .SVD_impute import IterativeSVD

def initialize_fac(tensor: np.ndarray, rank: int):
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
    factors = [np.ones((tensor.shape[i], rank)) for i in range(tensor.ndim)]
    contain_missing = (np.sum(~np.isfinite(tensor)) > 0)

    # SVD init mode whose size is larger than rank
    for mode in range(tensor.ndim):
        if tensor.shape[mode] >= rank:
            unfold = tl.unfold(tensor, mode)
            if contain_missing:
                si = IterativeSVD(rank)
                unfold = si.fit_transform(unfold)

            factors[mode] = svd_interface(matrix=unfold, n_eigenvecs=rank, flip=True)[0]
        else: # tensor.shape[mode] < rank
            unfold = tl.unfold(tensor, mode)
            if contain_missing:
                si = IterativeSVD(tensor.shape[mode])
                unfold = si.fit_transform(unfold) # unfold (tensor.shape[mode] x ...)

            svd_factor = svd_interface(matrix=unfold, n_eigenvecs=rank, flip=True)[0]
            factors[mode][:,0:tensor.shape[mode]] = svd_factor

    return tl.cp_tensor.CPTensor((None, factors))