import os
import numpy as np
from ..decomposition import Decomposition
from ..tracker import Tracker


def runImputation(data:np.ndarray,
                  min_rr:int,
                  max_rr:int,
                  impType:str,
                  method,
                  dataname = None,
                  savename:str = None,
                  printRuntime:bool = False,
                  **kwargs):
    
    assert impType == 'entry' or impType == 'chord'
    
    decomposition = Decomposition(data=data, dataname=dataname, min_rr=min_rr, max_rr=max_rr)
    tracker = Tracker(data)
    tracker.begin()
    
    # run method & save
    # include **kwargs include: repeat=reps, drop=drop_perc, init=init_type, callback_r=max_rr
    decomposition.imputation(imp_type=impType, method=method, callback=tracker, printRuntime=printRuntime, **kwargs)
    tracker.combine()

    if savename is not None:
        if os.path.isdir(savename) is False:
            os.makedirs(savename)
        decomposition.save(f"./{savename}{impType}-{method.__name__}.decomposition")
        tracker.save(f"./{savename}{impType}-{method.__name__}.tracker")

        _,tmp = loadImputation(impType,method,savename)
        print(tmp.total_array)

    return decomposition, tracker


def loadImputation(impType, method, savename, callback=True):
    decomposition = Decomposition()
    decomposition.load(f"./{savename}{impType}-{method.__name__}.decomposition")

    if callback is True:
        tracker = Tracker()
        tracker.load(f"./{savename}{impType}-{method.__name__}.tracker")
    else:
        tracker = None

    return decomposition, tracker