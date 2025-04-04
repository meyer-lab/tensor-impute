import os
import pickle
from os.path import join

import numpy as np

from ..generateTensor import generateTensor
from . import DROPS, METHODNAMES, METHODS, SAVENAMES
from .figure_helper import loadImputation, runImputation

# poetry run python -m timpute.figures.figure_data


def sim_data(
    true_ranks=(5, 25),
    shape=(50, 50, 50),
    decomp_ranks=(1, 30),
    reps=20,
    drops=DROPS,
    methods=METHODS,
    nonmissing=True,
    savedir="timpute/figures/cache/simulated",
):
    seed = 1
    init_type = "random"
    min_component, max_component = decomp_ranks

    for i, true_r in enumerate(np.arange(true_ranks[0], true_ranks[1] + 1)):
        dataset = f"rank_{true_r}"
        dirname = join(savedir, dataset)
        if os.path.isdir(dirname) is False:
            os.makedirs(dirname)

        tensor, factors = generateTensor(
            "tensorly",
            shape=shape,
            r=true_r,
            noise_scale=1,
        )

        with open(join(dirname, "CPfactors.pickle"), "wb") as output_file:
            pickle.dump(factors, output_file)

        if nonmissing is True:
            print("--- BEGIN NONMISSING ---")
            # stdout.write("--- BEGIN NONMISSING ---\n")
            drop_perc = 0.0

            run = "nonmissing/"
            folder = join(dirname, run)
            if os.path.isdir(folder) is False:
                os.makedirs(folder)

            impType = "entry"
            for m in methods:
                runImputation(
                    data=tensor,
                    dataname=dataset,
                    min_rr=min_component,
                    max_rr=max_component,
                    impType=impType,
                    savename=folder,
                    method=m,
                    printRuntime=True,
                    repeat=reps,
                    drop=drop_perc,
                    init=init_type,
                    seed=seed * i,
                    tol=1e-6,
                )

        for drop_perc in drops:
            print(f"--- BEGIN MISSING ({i}) ---")
            run = f"drop_{drop_perc}/"
            folder = join(dirname, run)

            if os.path.isdir(folder) is False:
                os.makedirs(folder)
            for m in methods:
                impType = "entry"
                runImputation(
                    data=tensor,
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
                    savename=folder,
                    printRuntime=True,
                )

                impType = "chord"
                runImputation(
                    data=tensor,
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
                    savename=folder,
                    printRuntime=True,
                )

    return True


def real_data(
    datalist=SAVENAMES,
    start_comps=(1, 1, 1, 1),
    max_comps=(20, 20, 20, 20),
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
        dirname = join(savedir, f"{dataset}")
        if os.path.isdir(dirname) is False:
            os.makedirs(dirname)

        orig = generateTensor(type=dataset)
        min_component = start_comps[i]
        max_component = max_comps[i]

        if nonmissing is True:
            print("--- BEGIN NONMISSING ---")
            # stdout.write("--- BEGIN NONMISSING ---\n")
            drop_perc = 0.0
            run = "nonmissing/"
            folder = join(dirname, run)

            if os.path.isdir() is False:
                os.makedirs(folder)
            impType = "entry"
            for m in methods:
                runImputation(
                    data=orig,
                    dataname=dataset,
                    min_rr=min_component,
                    max_rr=max_component,
                    impType=impType,
                    savename=folder,
                    method=m,
                    printRuntime=True,
                    repeat=reps,
                    drop=drop_perc,
                    init=init_type,
                    seed=seed * i,
                    tol=1e-6,
                )

        for drop_perc in drops:
            print(f"--- BEGIN MISSING ({i}) ---")
            run = f"drop_{drop_perc}/"
            folder = join(dirname, run)

            if os.path.isdir(folder) is False:
                os.makedirs(folder)
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
                    savename=folder,
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
                    savename=folder,
                    printRuntime=True,
                )

    return True


def bestComps(
    drop=0.1, impType="entry", datalist=SAVENAMES, imputed=True, outputData=False
):
    """
    determines best component for all methods of all datasets for a given setting
    returns in the form of {'data' : {'method' : #}}
    """
    bestComp = dict()
    dirname = "timpute/figures/revision_cache"

    for data in datalist:
        data_folder = join(dirname, data)
        # print(data_folder)

        if drop == 0:
            folder = join(data_folder, "nonmissing/")
            # print(folder)
            tmp = dict()
            for n, m in enumerate(METHODS):
                run, _ = loadImputation(impType, m, folder)
                tmp[METHODNAMES[n]] = int(
                    np.median(run.entry_total, axis=0).argmin() + 1
                )
            bestComp[data] = tmp

        elif drop in DROPS:
            folder = join(data_folder, f"drop_{drop}/")
            # print(folder)

            tmp = dict()
            for n, m in enumerate(METHODS):
                run, _ = loadImputation(impType, m, folder)
                if imputed is True:

                    if outputData is False:
                        if impType == "entry":
                            tmp[METHODNAMES[n]] = int(
                                np.median(run.entry_imputed, axis=0).argmin() + 1
                            )
                        elif impType == "chord":
                            tmp[METHODNAMES[n]] = int(
                                np.median(run.chord_imputed, axis=0).argmin() + 1
                            )
                        else:
                            raise ValueError(f'impType "{impType}" not recognized')

                    else:
                        if impType == "entry":
                            tmp[METHODNAMES[n]] = np.median(
                                run.entry_imputed, axis=0
                            ).min()
                        elif impType == "chord":
                            tmp[METHODNAMES[n]] = np.median(
                                run.chord_imputed, axis=0
                            ).min()
                        else:
                            raise ValueError(f'impType "{impType}" not recognized')

                else:
                    if outputData is False:
                        if impType == "entry":
                            tmp[METHODNAMES[n]] = int(
                                np.median(run.entry_total, axis=0).argmin() + 1
                            )
                        elif impType == "chord":
                            tmp[METHODNAMES[n]] = int(
                                np.median(run.chord_total, axis=0).argmin() + 1
                            )
                        else:
                            raise ValueError(f'impType "{impType}" not recognized')
                    else:
                        if impType == "entry":
                            tmp[METHODNAMES[n]] = np.median(
                                run.entry_total, axis=0
                            ).min()
                        elif impType == "chord":
                            tmp[METHODNAMES[n]] = np.median(
                                run.chord_total, axis=0
                            ).min()
                        else:
                            raise ValueError(f'impType "{impType}" not recognized')

            bestComp[data] = tmp

        else:
            raise ValueError(f'drop percentage "{drop}" not recognized')

    return bestComp


def bestSimComps(
    drop=0.1, impType="entry", first=5, last=25, imputed=True, outputData=False
):
    bestComp = dict()
    dirname = "timpute/figures/revision_cache/simulated"

    for rank in np.arange(first, last + 1):
        run = f"rank_{rank}"
        data_folder = join(dirname, run)

        if drop == 0:
            folder = join(data_folder, "nonmissing/")
            tmp = dict()
            for n, m in enumerate(METHODS):
                run, _ = loadImputation(impType, m, folder)
                tmp[METHODNAMES[n]] = int(
                    np.median(run.entry_total, axis=0).argmin() + 1
                )
            bestComp[rank] = tmp

        elif drop in DROPS:
            folder = join(data_folder, f"drop_{drop}/")
            tmp = dict()
            for n, m in enumerate(METHODS):
                run, _ = loadImputation(impType, m, folder)
                if imputed is True:
                    if outputData is False:
                        if impType == "entry":
                            tmp[METHODNAMES[n]] = int(
                                np.median(run.entry_imputed, axis=0).argmin() + 1
                            )
                        elif impType == "chord":
                            tmp[METHODNAMES[n]] = int(
                                np.median(run.chord_imputed, axis=0).argmin() + 1
                            )
                        else:
                            raise ValueError(f'impType "{impType}" not recognized')
                    else:
                        if impType == "entry":
                            tmp[METHODNAMES[n]] = np.median(
                                run.entry_imputed, axis=0
                            ).min()
                        elif impType == "chord":
                            tmp[METHODNAMES[n]] = np.median(
                                run.chord_imputed, axis=0
                            ).min()
                        else:
                            raise ValueError(f'impType "{impType}" not recognized')

                else:
                    if outputData is False:
                        if impType == "entry":
                            tmp[METHODNAMES[n]] = int(
                                np.median(run.entry_total, axis=0).argmin() + 1
                            )
                        elif impType == "chord":
                            tmp[METHODNAMES[n]] = int(
                                np.median(run.chord_total, axis=0).argmin() + 1
                            )
                        else:
                            raise ValueError(f'impType "{impType}" not recognized')
                    else:
                        if impType == "entry":
                            tmp[METHODNAMES[n]] = np.median(
                                run.entry_total, axis=0
                            ).min()
                        elif impType == "chord":
                            tmp[METHODNAMES[n]] = np.median(
                                run.chord_total, axis=0
                            ).min()
                        else:
                            raise ValueError(f'impType "{impType}" not recognized')

            bestComp[rank] = tmp

        else:
            raise ValueError(f'drop percentage "{drop}" not recognized')

    return bestComp


def chordMasking(
    datalist=SAVENAMES, max_comps=(10, 10, 10, 20), drops=(0.05, 0.1, 0.2, 0.3, 0.4)
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
