import os
import numpy as np
from time import time
from ..decomposition import Decomposition, MultiDecomp
from ..tracker import Tracker, MultiTracker

from ..method_ALS import perform_ALS
from ..method_CLS import perform_CLS
from ..method_DO import perform_DO
METHODS = (perform_DO, perform_ALS, perform_CLS)
METHODNAMES = ["DO","ALS","CLS"]
SAVENAMES = ["zohar", "alter", "hms", "coh_response"]
DATANAMES = ['Covid serology', 'HIV serology', 'DyeDrop profiling', 'BC cytokine']
LINESTYLES = ('dashdot', (0,(1,1)), 'solid', (3,(3,1,1,1,1,1)), 'dotted', (0,(5,1)))
DROPS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)


def runImputation(data:np.ndarray,
                  max_rr:int,
                  impType:str,
                  savename:str, 
                  method,
                  save = True,
                  printRuntime = False,
                  **kwargs):
    assert impType == 'entry' or impType == 'chord'
    start = time()
    decomposition = Decomposition(data, max_rr)
    tracker = Tracker(data)
    
    # run method & save
    tracker.begin()
    # include **kwargs include: repeat=reps, drop=drop_perc, init=init_type, callback_r=max_rr
    decomposition.imputation(type=impType, method=method, callback=tracker, printRuntime=printRuntime, **kwargs)
    tracker.combine()

    if save is True:
        if os.path.isdir(savename) is False: os.makedirs(savename)
        decomposition.save(f"./{savename}{impType}-{method.__name__}.decomposition")
        tracker.save(f"./{savename}{impType}-{method.__name__}.tracker")

    # print(f"{method.__name__} runtime for components 1-{max_rr}: {time()-start}")

    return decomposition, tracker
    
def runMultiImputation(data:np.ndarray, number:int,
                       max_rr:int,
                       impType:str,
                       savename:str, 
                       method,
                       printRuntime = False,
                       **kwargs):
    assert impType == 'entry' or impType == 'chord'
    if number == 0:
        tracker = MultiTracker()
        if impType == 'entry':
            decomposition = MultiDecomp(chord=False)
        if impType == 'chord':
            decomposition = MultiDecomp(entry=False)
    else:
        decomposition, tracker = loadMultiImputation(impType,method,savename)

    tmpDec, tmpTra = runImputation(data=data, max_rr=max_rr, impType=impType, method=method, savename=savename, save=False, printRuntime=printRuntime, **kwargs)
    
    decomposition(tmpDec)
    tracker(tmpTra)

    decomposition.save(f"./{savename}{impType}-{method.__name__}.multiDecomposition")
    tracker.save(f"./{savename}{impType}-{method.__name__}.multiTracker")



def loadImputation(impType, method, savename):
    decomposition = Decomposition()
    tracker = Tracker()

    decomposition.load(f"./{savename}{impType}-{method.__name__}.decomposition")
    tracker.load(f"./{savename}{impType}-{method.__name__}.tracker")

    return decomposition, tracker

def loadMultiImputation(impType, method, savename):
    decomposition = MultiDecomp()
    tracker = MultiTracker()

    decomposition.load(f"./{savename}{impType}-{method.__name__}.multiDecomposition")
    tracker.load(f"./{savename}{impType}-{method.__name__}.multiTracker")

    return decomposition, tracker
