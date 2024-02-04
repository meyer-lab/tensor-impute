import psutil
# import resource
import numpy as np
from .runImputation import *
from ..generateTensor import generateTensor

# poetry run python -m timpute.figures.dataUsage

def testMemory(dataname, method, max_comp):
    # generate tensor
    orig = generateTensor(type=dataname)

    # set nonmissing case
    drop_perc = 0
    run = "nonmissing/"

    impType = 'entry'
    start = time()
    decomposition = Decomposition(orig, max_comp)
    tracker = Tracker(orig)
    
    # run method & save
    tracker.begin()
    # include **kwargs include: repeat=reps, drop=drop_perc, init=init_type
    decomposition.profile_imputation(type=impType, method=method, callback=tracker, drop=drop_perc)
    tracker.combine()

    
    print(f"finished {dataname}, {run}{impType} for {method.__name__} in {time()-start} seconds")
    print(f"required {psutil.virtual_memory().used / 1024**2} MB\n")


if __name__ == "__main__":
    max_comps = [10,10,10,20]
    # resource.setrlimit(resource.RLIMIT_AS, (int(1e9), int(1e9)))
    for n, dat in enumerate(SAVENAMES):
        for m in METHODS:
            testMemory(dat, m, max_comps[n])