import numpy as np
import os

from .figure_helper import *
from ..generateTensor import generateTensor
from ..plot import *
from ..common import *
import pickle

# poetry run python -m timpute.figures.figure_data  

def real_data(datalist=["zohar", "alter", "hms", "coh_response"], max_comps=[10,10,10,20]):
    assert len(datalist) == len(max_comps)
    
    seed = 1
    init_type = 'random'
    reps = 20

    for i,dataset in enumerate(datalist):
        dirname = f"timpute/figures/cache/{dataset}"
        if os.path.isdir(dirname) is False: os.makedirs(dirname)

        orig = generateTensor(type=dataset)
        max_component = max_comps[i]
        
        savename = "/"
        folder = dirname+savename
        np.random.seed(seed)

        print("--- BEGIN NONMISSING ---")
        # stdout.write("--- BEGIN NONMISSING ---\n")
        drop_perc = 0.0
        run = "nonmissing/"

        if os.path.isdir(folder+run) is False: os.makedirs(folder+run)
        impType = 'entry'
        for m in METHODS:
            runImputation(data=orig, max_rr=max_component, impType=impType, savename=folder+run, method=m, printRuntime=True,
                        repeat=reps, drop=drop_perc, init=init_type, seed=seed*i, tol=1e-6)

        for i in DROPS:
            print(f"--- BEGIN MISSING ({i}) ---")
            drop_perc = i
            run = f"drop_{i}/"

            if os.path.isdir(folder+run) is False: os.makedirs(folder+run)
            for m in METHODS:
                impType = 'entry'
                runImputation(data=orig, max_rr=max_component, impType=impType, savename=folder+run, method=m, printRuntime=True,
                            repeat=reps, drop=drop_perc, init=init_type, seed=seed*i, tol=1e-6)

                impType = 'chord'
                runImputation(data=orig, max_rr=max_component, impType=impType, savename=folder+run, method=m, printRuntime=True,
                            repeat=reps, drop=drop_perc, init=init_type, seed=seed*i, tol=1e-6)


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


def chordMasking(datalist=SAVENAMES, max_comps=[10,10,10,20], drops = (0.05, 0.1, 0.2, 0.3, 0.4)):
    assert len(datalist) == len(max_comps)

    init_type = 'random'
    impType = 'chord'
    reps = 20
    seed = 1

    for i,dataset in enumerate(datalist):
        folder = f"timpute/figures/cache/modeComparison/{dataset}/"
        max_component = max_comps[i]
        orig = generateTensor(type=dataset, shape = (10,20,30)) if dataset == 'random' else generateTensor(type=dataset)

        for d in drops:
            print(f"--- BEGIN {dataset}: {int(d*100)}% MISSING ---")
            for mode in range(orig.ndim):
                run = f"drop_{d}/mode_{mode}/"
                if os.path.isdir(folder+run) is False:os.makedirs(folder+run)
                for m in METHODS:
                    runImputation(data=orig, max_rr=max_component, impType=impType, savename=folder+run, method=m, printRuntime=True,
                                  callback=False, repeat=reps, drop=d, init=init_type, seed=seed*i, tol=1e-6, chord_mode=mode)

if __name__ == "__main__":
    # for i in ['entry', 'chord']:
    #     data = dict()
    #     for d in DROPS:
    #         data[d] = bestComps(d, i)
    #     with open(f'./timpute/figures/cache/bestComps_{i}.pickle', 'wb') as handle:
    #         pickle.dump(data, handle)
    chordMasking()