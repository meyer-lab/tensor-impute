import numpy as np
import tensorly as tl

def calcR2X(tFac: tl.cp_tensor.CPTensor, tIn: np.ndarray, calcError=False, mask: np.ndarray=None) -> float:
    """ Calculate R2X. Optionally it can be calculated for only the tensor or matrix.
    Mask is for imputation and must be of same shape as tIn/tFac, with 0s indicating artifically dropped values
    """
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
    
    return np.array(data_pattern, dtype=bool)

def chord_drop(tensor: np.ndarray, drop: int, seed: int=None):
    """
    Removes chords along axis = 0 of a tensor.

    Parameters
    ----------
    tensor : ndarray
        Takes a tensor of any shape.
    drop : int
        To set a percentage, multiply int(tensor.size/tensor.shape[0]) by the percentage
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
        dropidxs = []
        for i in range(chordlen):
            dropidxs.append(tuple(np.insert(chordidx, 0, i)))
        for i in range(chordlen):
            if tensor[dropidxs[i]] != np.nan:
                data_pattern[dropidxs[i]] = 0  
            tensor[dropidxs[i]] = np.nan

    return np.array(data_pattern, dtype=bool)