import numpy as np
import os

from .runImputation import *
from ..generateTensor import generateTensor
from ..plot import *
from ..common import *
import pickle

# output
from time import time
from datetime import datetime
from pytz import timezone
import inspect
import resource

# poetry run python -m timpute.figures.realdatafigs  

def real_data(datalist=SAVENAMES, max_comps=[10,10,10,20]):
    assert len(datalist) == len(max_comps)
    
    seed = 1
    init_type = 'random'
    reps = 20

    for i,dataset in enumerate(datalist):
        dirname = f"timpute/figures/cache/{dataset}"
        if os.path.isdir(dirname) is False: os.makedirs(dirname)
        # stdout = open(f"{dirname}/output.txt", 'a')

        orig = generateTensor(type=dataset)
        max_component = max_comps[i]
        
        # stdout.write(f"\n\n===================\nStarting new run for [{inspect.currentframe().f_code.co_name}]\nTimestamp: {datetime.now(timezone('US/Pacific'))}\n")
        savename = "/"
        folder = dirname+savename
        np.random.seed(seed)

        # stdout.write("--- BEGIN NONMISSING ---\n")
        print("--- BEGIN NONMISSING ---")
        drop_perc = 0.0
        run = "nonmissing/"

        if os.path.isdir(folder+run) is False: os.makedirs(folder+run)
        impType = 'entry'
        for m in METHODS:
            # start = time()
            runImputation(data=orig, max_rr=max_component, impType=impType, savename=folder+run, method=m, printRuntime=True,
                        repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*i, tol=1e-6)
            # stdout.write(f"finished {dataset}, {run}{impType} for {m.__name__} in {time()-start} seconds\n")
        

        for i in DROPS:
            print(f"--- BEGIN MISSING ({i}) ---")
            # stdout.write(f"--- BEGIN MISSING ({i}) ---\n")
            drop_perc = i
            run = f"drop_{i}/"

            if os.path.isdir(folder+run) is False: os.makedirs(folder+run)
            for m in METHODS:
                # start = time()
                impType = 'entry'
                runImputation(data=orig, max_rr=max_component, impType=impType, savename=folder+run, method=m, printRuntime=True,
                            repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*i, tol=1e-6)
                # stdout.write(f"finished {dataset}, {run}{impType} for {m.__name__} in {time()-start} seconds\n")

                # start = time()
                impType = 'chord'
                runImputation(data=orig, max_rr=max_component, impType=impType, savename=folder+run, method=m, printRuntime=True,
                            repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*i, tol=1e-6)
                # stdout.write(f"finished {dataset}, {run}{impType} for {m.__name__} in {time()-start} seconds\n")           


def bestComps(drop=0.1, impType = "entry", datalist=SAVENAMES):
    """
    determines  best component for all methods of all datasets for a given imputation type
    returns in the form of {'data' : {'method' : #}}
    """
    bestComp = dict()
    for data in datalist:
        folder = f"timpute/figures/cache/{data}/drop_{drop}/"
        if drop == 0:
            folder = f"timpute/figures/cache/{data}/nonmissing/"
        tmp = dict()
        for n,m in enumerate(METHODS):
            run, _ = loadImputation(impType, m, folder)
            if impType == "entry":
                tmp[METHODNAMES[n]] = np.median(run.entry_imputed, axis=0).argmin()+1
            elif impType == "chord":
                tmp[METHODNAMES[n]] = np.median(run.chord_imputed, axis=0).argmin()+1
            else:
                raise ValueError(f'impType "{impType}" not recognized')
        bestComp.update({data : tmp})
    return bestComp

if __name__ == "__main__":
    for i in ['entry', 'chord']:
        data = dict()
        for d in DROPS:
            data[d] = bestComps(d, i)
        with open(f'./timpute/figures/cache/bestComps_{i}.pickle', 'wb') as handle:
            pickle.dump(data, handle)
        
