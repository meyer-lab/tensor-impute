import os

import numpy as np

from ..decomposition import Decomposition
from ..tracker import Tracker


def runImputation(data:np.ndarray,
                  max_rr:int,
                  impType:str,
                  savename:str, 
                  method,
                  callback = True,
                  save = True,
                  printRuntime = False,
                  **kwargs):
    assert impType == 'entry' or impType == 'chord'
    decomposition = Decomposition(data, max_rr)
    if callback is True:
        tracker = Tracker(data)
        tracker.begin()
    else:
        tracker = None
    
    # run method & save
    # include **kwargs include: repeat=reps, drop=drop_perc, init=init_type, callback_r=max_rr
    decomposition.imputation(type=impType, method=method, callback=tracker, printRuntime=printRuntime, **kwargs)
    if tracker:
        tracker.combine()

    if save is True:
        if os.path.isdir(savename) is False: os.makedirs(savename)
        decomposition.save(f"./{savename}{impType}-{method.__name__}.decomposition")
        if tracker:
            tracker.save(f"./{savename}{impType}-{method.__name__}.tracker")

    return decomposition, tracker


def loadImputation(impType, method, savename):
    decomposition = Decomposition()
    tracker = Tracker()

    decomposition.load(f"./{savename}{impType}-{method.__name__}.decomposition")
    tracker.load(f"./{savename}{impType}-{method.__name__}.tracker")

    return decomposition, tracker