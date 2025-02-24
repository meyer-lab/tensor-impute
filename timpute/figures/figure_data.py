import os
import numpy as np
from .figure_helper import runImputation, loadImputation
from ..generateTensor import generateTensor
from . import METHODS, METHODNAMES, SAVENAMES, DROPS

import argparse

# poetry run python -m timpute.figures.figure_data


def real_data(
    datalist=SAVENAMES,
    start_comps=[1, 1, 1, 1],
    max_comps=[20, 20, 20, 20],
    reps=20,
    drops=DROPS,
    methods=METHODS,
    nonmissing=True,
    savedir="timpute/figures/cache",
):
    assert len(datalist) == len(max_comps)

    seed = 1
    init_type = "random"

    for i, dataset in enumerate(datalist):
        dirname = os.path.join(savedir, f"{dataset}")
        if os.path.isdir(dirname) is False:
            os.makedirs(dirname)

        orig = generateTensor(type=dataset)
        min_component = start_comps[i]
        max_component = max_comps[i]

        savename = "/"
        folder = dirname + savename
        np.random.seed(seed)

        if nonmissing is True:
            print("--- BEGIN NONMISSING ---")
            # stdout.write("--- BEGIN NONMISSING ---\n")
            drop_perc = 0.0
            run = "nonmissing/"

            if os.path.isdir(folder + run) is False:
                os.makedirs(folder + run)
            impType = "entry"
            for m in methods:
                runImputation(
                    data=orig,
                    dataname=dataset,
                    min_rr=min_component,
                    max_rr=max_component,
                    impType=impType,
                    savename=folder + run,
                    method=m,
                    printRuntime=True,
                    repeat=reps,
                    drop=drop_perc,
                    init=init_type,
                    seed=seed * i,
                    tol=1e-6,
                )

        for i in drops:
            print(f"--- BEGIN MISSING ({i}) ---")
            drop_perc = i
            run = f"drop_{i}/"

            if os.path.isdir(folder + run) is False:
                os.makedirs(folder + run)
            for m in methods:
                impType = "entry"
                runImputation(
                    data=orig,
                    dataname=dataset,
                    min_rr=min_component,
                    max_rr=max_component,
                    repeat=reps,
                    method=m,
                    impType=impType,
                    drop=drop_perc,
                    init=init_type,
                    seed=seed * i,
                    tol=1e-6,
                    savename=folder + run,
                    printRuntime=True,
                )

                impType = "chord"
                runImputation(
                    data=orig,
                    dataname=dataset,
                    min_rr=min_component,
                    max_rr=max_component,
                    repeat=reps,
                    method=m,
                    impType=impType,
                    drop=drop_perc,
                    init=init_type,
                    seed=seed * i,
                    tol=1e-6,
                    savename=folder + run,
                    printRuntime=True,
                )

    return True


def bestComps(drop=0.1, impType="entry", datalist=SAVENAMES):
    """
    determines  best component for all methods of all datasets for a given imputation type
    returns in the form of {'data' : {'method' : #}}
    """
    bestComp = dict()
    for data in datalist:
        folder = f"timpute/figures/revision_cache/{data}/drop_{drop}/"
        if drop == 0:
            folder = f"timpute/figures/revision_cache/{data}/nonmissing/"
        tmp = dict()
        for n, m in enumerate(METHODS):
            run, _ = loadImputation(impType, m, folder)
            if impType == "entry":
                tmp[METHODNAMES[n]] = np.median(run.entry_imputed, axis=0).argmin() + 1
            elif impType == "chord":
                tmp[METHODNAMES[n]] = np.median(run.chord_imputed, axis=0).argmin() + 1
            else:
                raise ValueError(f'impType "{impType}" not recognized')
        bestComp.update({data: tmp})
    return bestComp


def chordMasking(
    datalist=SAVENAMES, max_comps=[10, 10, 10, 20], drops=(0.05, 0.1, 0.2, 0.3, 0.4)
):
    assert len(datalist) == len(max_comps)

    init_type = "random"
    impType = "chord"
    reps = 20
    seed = 1

    for i, dataset in enumerate(datalist):
        folder = f"timpute/figures/cache/modeComparison/{dataset}/"
        max_component = max_comps[i]
        orig = (
            generateTensor(type=dataset, shape=(10, 20, 30))
            if dataset == "random"
            else generateTensor(type=dataset)
        )

        for d in drops:
            print(f"--- BEGIN {dataset}: {int(d*100)}% MISSING ---")
            for mode in range(orig.ndim):
                run = f"drop_{d}/mode_{mode}/"
                if os.path.isdir(folder + run) is False:
                    os.makedirs(folder + run)
                for m in METHODS:
                    runImputation(
                        data=orig,
                        max_rr=max_component,
                        impType=impType,
                        savename=folder + run,
                        method=m,
                        printRuntime=True,
                        callback=False,
                        repeat=reps,
                        drop=d,
                        init=init_type,
                        seed=seed * i,
                        tol=1e-6,
                        chord_mode=mode,
                    )
