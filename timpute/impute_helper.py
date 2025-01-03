import numpy as np
import tensorly as tl


def calcR2X(tFac: tl.cp_tensor.CPTensor, tIn: np.ndarray, calcError=False, mask: np.ndarray=None) -> float:
    """ Calculate R2X. Optionally it can be calculated for only the tensor or matrix.
    Mask is for imputation and must be of same shape as tIn/tFac, with 0s indicating artifically dropped values
    """
    vTop, vBottom = 0.0, 1e-8
    vTop, vBottom = 0.0, 1e-8

    if mask is None:
        tOrig = np.copy(tIn)
        recons_tFac = tl.cp_to_tensor(tFac)
    else:
        tOrig = np.copy(tIn)*mask
        recons_tFac = tl.cp_to_tensor(tFac)*mask
        
    tMask = np.isfinite(tOrig)
    tOrig = np.nan_to_num(tOrig)
    vTop += np.linalg.norm(recons_tFac * tMask - tOrig)**2.0
    vBottom += np.linalg.norm(tOrig)**2.0

    if calcError: return vTop / vBottom
    else: return 1 - vTop / vBottom

def entry_drop(tensor:np.ndarray, drop:int, dropany=False, seed:int=None):
    """
    Drops random values within a tensor. Finds a bare minimum cube before dropping values to ensure PCA remains viable.

    Parameters
    ----------
    tensor : ndarray
        Takes a tensor of any shape. Preference for at least two values present per chord.
    drop : int
        To set a percentage, multiply np.sum(np.isfinite(tensor)) by the percentage
        to find the relevant drop value, rounding to nearest int.

    Returns
    -------
    mask : ndarray (boolean)
        artificial missingness mask
        0 = indicates artificial missingness added
        1 = indicates original data was untouched (regardless of true missingness status)
    """
    # Track chords for each mode to ensure bare minimum cube covers each chord at least once

    if seed != None:
        np.random.seed(seed)

    # withhold base missingness from entry dropping
    if dropany: idxs = np.argwhere(np.isfinite(tensor))
    else:
        midxs = np.zeros((tensor.ndim, max(tensor.shape)))
        for i in range(tensor.ndim):
            midxs[i] = [1 for n in range(tensor.shape[i])] + [0 for m in range(len(midxs[i]) - tensor.shape[i])]
        modecounter = np.arange(tensor.ndim)

        # Remove bare minimum cube idxs from droppable values
        idxs = np.argwhere(np.isfinite(tensor))
        while np.sum(midxs) > 0:
            removable = False
            ran = np.random.choice(idxs.shape[0], 1)
            ranidx = idxs[ran][0]
            counter = 0
            for i in ranidx:
                if midxs[modecounter[counter], i] > 0:
                    removable = True
                midxs[modecounter[counter], i] = 0
                counter += 1
            if removable == True:
                idxs = np.delete(idxs, ran, axis=0)
        assert idxs.shape[0] >= drop

    # Drop values
    data_pattern = np.ones_like(tensor) # capture missingness pattern
    dropidxs = idxs[np.random.choice(idxs.shape[0], drop, replace=False)]
    dropidxs = [tuple(dropidxs[i]) for i in range(dropidxs.shape[0])]
    for i in dropidxs:
        tensor[i] = np.nan
        data_pattern[i] = 0
    
    # missingness pattern holds 0s where dropped, 1s if left alone (does not guarantee nonmissing)

    return np.array(data_pattern, dtype=bool)

def chord_drop(tensor: np.ndarray, drop: int, seed: int=None):
    """
    Removes chords along axis = 0 of a tensor.

    Parameters
    ----------
    tensor : ndarray
        Takes a tensor of any shape.
    drop : int
        To set a percentage, multiply tensor.shape[0] by the percentage
        to find the relevant drop value, rounding to nearest int.

    Returns
    -------
    mask : ndarray (boolean)
        artificial missingness mask
        0 = indicates artificial missingness added
        1 = indicates original data was untouched (regardless of true missingness status)
    """

    if seed != None: np.random.seed(seed)

    # Drop chords based on random idxs
    data_pattern = np.ones_like(tensor) # capture missingness pattern

    chordlen = tensor.shape[0]
    for _ in range(drop):
        idxs = np.argwhere(np.isfinite(tensor))
        chordidx = np.delete(idxs[np.random.choice(idxs.shape[0], 1)][0], 0)
        dropidxs = [tuple(np.insert(chordidx, 0, i)) for i in range(chordlen)]
        for d in dropidxs:
            if tensor[d] != np.nan:
                data_pattern[d] = 0  
            tensor[d] = np.nan

    # missingness pattern holds 0s where dropped, 1s if left alone (may be inherently missing or undropped)
    
    return np.array(data_pattern, dtype=bool)

def kronecker_mat_ten(matrices, X):
    for k in range(len(matrices)):
        M = matrices[k]
        X = tl.tenalg.mode_dot(X, M, k)
    return X

# Shortcut to invert singular values.
# Given a vector of singular values, returns the inverted matrix
def invert_sing(s):
    return np.diag(1.0 / s)

def corcondia_3d(tOrig, tFac):
    # Weights are not important since normalize_factors is false by default
    #  so the weights will be all ones.
    _, X_approx_ks = tFac.factors
    k = tFac.rank

    A, B, C = X_approx_ks
    Ua, Sa, Va = np.linalg.svd(A, full_matrices=False)
    Ub, Sb, Vb = np.linalg.svd(B, full_matrices=False)
    Uc, Sc, Vc = np.linalg.svd(C, full_matrices=False)

    inverted = [invert_sing(x) for x in (Sa, Sb, Sc)]

    part1 = kronecker_mat_ten([Ua.T, Ub.T, Uc.T], tOrig)
    # TODO: line below can be omitted if we multiply the k-th column
    #  of Ua, Ub and Uc by the inverse of the corresponding k-th singular value, that is 1/σ_k.
    #  Then, the previous line will need to use the updated Ua, Ub and Uc
    # (note by Yorgos)
    part2 = kronecker_mat_ten(inverted, part1)
    G = kronecker_mat_ten([Va.T, Vb.T, Vc.T], part2)

    for i in range(k):
        G[:,:,i] = G[:,:,i] / G[i,i,i]

    T = np.zeros((k, k, k))
    for i in range(k):
        T[i,i,i] = 1

    return 100 * (1 - ((G-T)**2).sum() / float(k))