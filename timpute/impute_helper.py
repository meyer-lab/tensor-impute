import numpy as np

def create_missingness(tensor, drop):
    """
    Creates missingness for a full tensor. Slgihtly faster than entry_drop()
    """
    idxs = np.argwhere(np.isfinite(tensor))
    dropidxs = idxs[np.random.choice(idxs.shape[0], drop, replace=False)]
    dropidxs = tuple(dropidxs.T)
    tensor[dropidxs] = np.nan


def entry_drop(tensor, drop, seed=None):
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
    dropped_tensor : ndarray
        tensor is modified with missing chords
    mask : ndarray (boolean)
        artificial missingness mask
    """
    # Track chords for each mode to ensure bare minimum cube covers each chord at least once

    if seed != None:
        np.random.seed(seed)

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
    dropidxs = [tuple(dropidxs[i]) for i in range(drop)]
    dropped_tensor = np.copy(tensor)
    for i in dropidxs:
        dropped_tensor[i] = np.nan
        data_pattern[i] = 0
    
    return dropped_tensor, np.array(data_pattern, dtype=bool)

def chord_drop(tensor, drop, seed=None):
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
    dropped_tensor : ndarray
        tensor is modified with missing chords
    mask : ndarray (boolean)
        artificial missingness mask
    """

    if seed != None:
        np.random.seed(seed)

    # Drop chords based on random idxs
    data_pattern = np.ones_like(tensor) # capture missingness pattern

    chordlen = tensor.shape[0]
    dropped_tensor = np.copy(tensor)
    for _ in range(drop):
        idxs = np.argwhere(np.isfinite(tensor))
        chordidx = np.delete(idxs[np.random.choice(idxs.shape[0], 1)][0], 0, -1)
        dropidxs = []
        for i in range(chordlen):
            dropidxs.append(tuple(np.insert(chordidx, 0, i).T))
        for i in range(chordlen):
            if tensor[dropidxs[i]] != np.nan:
                data_pattern[dropidxs[i]] = 0  
            dropped_tensor[dropidxs[i]] = np.nan  

    return dropped_tensor, np.array(data_pattern, dtype=bool)