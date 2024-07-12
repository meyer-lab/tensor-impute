import os
import numpy as np
from ..decomposition import Decomposition
from ..tracker import Tracker

from ..method_ALS import perform_ALS
from ..method_CLS import perform_CLS
from ..method_DO import perform_DO
METHODS = (perform_DO, perform_ALS, perform_CLS)
METHODNAMES = ["DO","ALS","CLS"]
NEWMETHODNAMES = ["DO","ALS-SI","C-ALS"]
SAVENAMES = ["zohar", "alter", "hms", "coh_response"]
DATANAMES = ['SARS-COV-19 serology', 'HIV serology', 'DyeDrop profiling', 'BC cytokine']
LINESTYLES = ('dashdot', (0,(1,1)), 'solid', (3,(3,1,1,1,1,1)), 'dotted', (0,(5,1)))
DROPS = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)

TITLE_FONTSIZE = 14
SUBTITLE_FONTSIZE = 8
TEXT_FONTSIZE = 6


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