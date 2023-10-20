import numpy as np
import os

from .runImputation import *
from ..generateTensor import generateTensor
from ..plot import *
from ..common import *

# output
from time import time
from datetime import datetime
from pytz import timezone
import inspect

# poetry run python -m timpute.figures.realdatafigs  

def real_data(datalist=["zohar", "alter", "hms", "coh_response"], max_comps=[10,10,10,20]):
    assert len(datalist) == len(max_comps)
    
    seed = 1
    init_type = 'random'
    reps = 20

    for i,dataset in enumerate(datalist):
        dirname = f"timpute/figures/cache/{dataset}"
        if os.path.isdir(dirname) is False: os.makedirs(dirname)
        # stdout = open(f"{dirname}/output.txt", 'a')

        orig = generateTensor(type=dataset)
        print(orig.shape)
        max_component = max_comps[i]
        
        # stdout.write(f"\n\n===================\nStarting new run for [{inspect.currentframe().f_code.co_name}]\nTimestamp: {datetime.now(timezone('US/Pacific'))}\n")
        savename = "/"
        folder = dirname+savename
        np.random.seed(seed)

        print("--- BEGIN NONMISSING ---")
        # stdout.write("--- BEGIN NONMISSING ---\n")
        drop_perc = 0.0
        run = "nonmissing/"

        if os.path.isdir(folder+run) is False: os.makedirs(folder+run)
        impType = 'entry'
        for m in methods:
            start = time()
            runImputation(data=orig, max_rr=max_component, impType=impType, savename=folder+run, method=m, printRuntime=True,
                        repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100, tol=1e-6)
            # stdout.write(f"finished {dataset}, {run}{impType} for {m.__name__} in {time()-start} seconds\n")
        

        for i in drops:
            print(f"--- BEGIN MISSING ({i}) ---")
            # stdout.write(f"--- BEGIN MISSING ({i}) ---\n")
            drop_perc = i
            run = f"drop_{i}/"

            if os.path.isdir(folder+run) is False: os.makedirs(folder+run)
            impType = 'entry'
            for m in methods:
                start = time()
                runImputation(data=orig, max_rr=max_component, impType=impType, savename=folder+run, method=m, printRuntime=True,
                            repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100, tol=1e-6)
                # stdout.write(f"finished {dataset}, {run}{impType} for {m.__name__} in {time()-start} seconds\n")

            if os.path.isdir(folder+run) is False: os.makedirs(folder+run)
            impType = 'chord'
            for m in methods:
                start = time()
                runImputation(data=orig, max_rr=max_component, impType=impType, savename=folder+run, method=m, printRuntime=True,
                            repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100, tol=1e-6)
                # stdout.write(f"finished {dataset}, {run}{impType} for {m.__name__} in {time()-start} seconds\n")           


def bestComps(drop=0.1, impType = 'entry', datalist=["zohar", "alter", "hms", "coh_response"]):
    bestComp = dict()
    for data in enumerate(datalist):
        folder = f"timpute/figures/cache/{data}/drop_{drop}/"
        tmplist = list()
        for m in methods:
            run, _ = loadImputation(impType, m, folder)
            if impType == 'entry': tmplist.append(np.median(run.entry_imputed, axis=0).argmin())
            if impType == 'chord': tmplist.append(np.median(run.chord_imputed, axis=0).argmin())
            else:
                raise ValueError(f'impType "{impType}" not recognized')
        bestComp.update({data : tmplist})
    return bestComp


def dataset_info(datalist=["zohar", "alter", "hms", "coh_response"]):
    
    dirname = f"timpute/figures"
    stdout = open(f"{dirname}/dataset_info.txt", 'w')
    for i,data in enumerate(datalist):
        stdout.write(f"\n--{datanames[i]}--\n")
        tensor = generateTensor(data)
        stdout.write(f"shape: {tensor.shape}\n")
        stdout.write(f"missingness: {np.sum(np.isnan(tensor))/tensor.size}\n")

        tmp = generateTensor(data)
        for _ in range(tensor.ndim):
            count = 0
            for j in range(tmp.shape[1]):
                for k in range(tmp.shape[2]):
                    if np.any(np.isnan(tmp[:,j,k])):
                        count += 1
            stdout.write(f"{tmp.shape[0]}: {count/(tmp.shape[1]*tmp.shape[2])*100}% missingness | ")
            tmp = np.moveaxis(tmp, 0, -1)


def explore_expanded(datalist=["zohar", "hms", "alter", "coh_response"]):

    for shift, data in enumerate(datalist):
        ax, f = getSetup((8*len(drops),16), (2,len(drops)))
        row = len(drops)
        for col,d in enumerate(drops):
            folder = f"timpute/figures/cache/{data}/drop_{d}/"
            impType = 'entry' 
            for mID, m in enumerate(methods):
                run, _ = loadImputation(impType, m, folder)

                comps = np.arange(1,run.entry_imputed.shape[1]+1)
                label = f"{m.__name__}"
                total_errbar = np.vstack((abs(np.percentile(run.entry_total,25,0) - np.nanmedian(run.entry_total,0)),
                                        abs(np.percentile(run.entry_total,75,0) - np.nanmedian(run.entry_total,0))))
                ax[col].errorbar(comps, np.median(run.entry_total,0), yerr=total_errbar,label=label, ls='solid', color=rgbs(mID, 0.7), alpha=0.5)

                label = f"{m.__name__} Imputed Error"
                imp_errbar = np.vstack((abs(np.percentile(run.entry_imputed,25,0) - np.nanmedian(run.entry_imputed,0)),
                                        abs(np.percentile(run.entry_imputed,75,0) - np.nanmedian(run.entry_imputed,0))))
                e = ax[col].errorbar(comps, np.median(run.entry_imputed,0), yerr=imp_errbar, label=label, ls='dashed', color=rgbs(mID, 0.7), alpha=0.5)
                e[-1][0].set_linestyle('dashed')

                ax[col].legend()
                ax[col].set_xlabel("Number of Components")
                ax[col].set_ylabel("Error")
                ax[col].set_xticks([x for x in comps])
                ax[col].set_xticklabels([x for x in comps])
                ax[col].set_title(f"{data} dataset, {int(d*100)}% entry missingness vs component")
                # ax[col].set_yscale("log")


            impType = 'chord'
            for mID, m in enumerate(methods):
                run, _ = loadImputation(impType, m, folder)

                comps = np.arange(1,run.chord_imputed.shape[1]+1)
                label = f"{m.__name__}"
                total_errbar = np.vstack((abs(np.percentile(run.chord_total,25,0) - np.nanmedian(run.chord_total,0)),
                                        abs(np.percentile(run.chord_total,75,0) - np.nanmedian(run.chord_total,0))))
                ax[col+row].errorbar(comps, np.median(run.chord_total,0), yerr=total_errbar,label=label, ls='solid', color=rgbs(mID, 0.7), alpha=0.5)

                label = f"{m.__name__} Imputed Error"
                imp_errbar = np.vstack((abs(np.percentile(run.chord_imputed,25,0) - np.nanmedian(run.chord_imputed,0)),
                                        abs(np.percentile(run.chord_imputed,75,0) - np.nanmedian(run.chord_imputed,0))))
                e = ax[col+row].errorbar(comps, np.median(run.chord_imputed,0), yerr=imp_errbar, label=label, ls='dashed', color=rgbs(mID, 0.7), alpha=0.5)
                e[-1][0].set_linestyle('dashed')

                ax[col+row].legend()
                ax[col+row].set_xlabel("Number of Components")
                ax[col+row].set_ylabel("Error")
                ax[col+row].set_xticks([x for x in comps])
                ax[col+row].set_xticklabels([x for x in comps])
                ax[col+row].set_title(f"{data} dataset, {int(d*100)}% chord missingness vs component")
                # ax[col+row].set_yscale("log")
        
        f.savefig(f'timpute/figures/{data}_expanded.png', bbox_inches="tight", format='png')


def explore(datalist=["zohar", "hms", "alter", "coh_response"]):
    # if you want an offset between lines, add a ` + (mID*0.1-0.1)` after comps

    rowLen = 4
    ax, f = getSetup((8*rowLen,8*len(datalist)), (len(datalist),rowLen))
    for shift, data in enumerate(datalist):
        row = shift*rowLen
        folder = f"timpute/figures/cache/{data}/nonmissing/"
        impType = 'entry'
        for mID, m in enumerate(methods):
            run, _ = loadImputation(impType, m, folder)

            comps = np.arange(1,run.entry_imputed.shape[1]+1)
            label = f"{m.__name__}"
            total_errbar = np.vstack((abs(np.percentile(run.entry_total,25,0) - np.nanmedian(run.entry_total,0)),
                                    abs(np.percentile(run.entry_total,75,0) - np.nanmedian(run.entry_total,0))))
            e = ax[0+row].errorbar(comps, np.median(run.entry_total,0), yerr=total_errbar,label=label, ls='solid', color=rgbs(mID, 0.7), alpha=0.5)

            ax[0+row].legend()
            ax[0+row].set_xlabel("Number of Components")
            ax[0+row].set_ylabel("Error")
            ax[0+row].set_xticks([x for x in comps])
            ax[0+row].set_xticklabels([x for x in comps])
            ax[0+row].set_title(f"{data} dataset, decomposition vs component")
            # ax[0+row].set_yscale("log")
        

        folder = f"timpute/figures/cache/{data}/drop_0.1/"
        impType = 'entry' 
        for mID, m in enumerate(methods):
            run, _ = loadImputation(impType, m, folder)

            comps = np.arange(1,run.entry_imputed.shape[1]+1)
            label = f"{m.__name__}"
            total_errbar = np.vstack((abs(np.percentile(run.entry_total,25,0) - np.nanmedian(run.entry_total,0)),
                                    abs(np.percentile(run.entry_total,75,0) - np.nanmedian(run.entry_total,0))))
            ax[1+row].errorbar(comps, np.median(run.entry_total,0), yerr=total_errbar,label=label, ls='solid', color=rgbs(mID, 0.7), alpha=0.5)

            label = f"{m.__name__} Imputed Error"
            imp_errbar = np.vstack((abs(np.percentile(run.entry_imputed,25,0) - np.nanmedian(run.entry_imputed,0)),
                                    abs(np.percentile(run.entry_imputed,75,0) - np.nanmedian(run.entry_imputed,0))))
            e = ax[1+row].errorbar(comps, np.median(run.entry_imputed,0), yerr=imp_errbar, label=label, ls='dashed', color=rgbs(mID, 0.7), alpha=0.5)
            e[-1][0].set_linestyle('dashed')

            ax[1+row].legend()
            ax[1+row].set_xlabel("Number of Components")
            ax[1+row].set_ylabel("Error")
            ax[1+row].set_xticks([x for x in comps])
            ax[1+row].set_xticklabels([x for x in comps])
            ax[1+row].set_title(f"{data} dataset, 10% entry missingness vs component")
            # ax[1+row].set_yscale("log")


        impType = 'chord'
        for mID, m in enumerate(methods):
            run, _ = loadImputation(impType, m, folder)

            comps = np.arange(1,run.chord_imputed.shape[1]+1)
            label = f"{m.__name__}"
            total_errbar = np.vstack((abs(np.percentile(run.chord_total,25,0) - np.nanmedian(run.chord_total,0)),
                                    abs(np.percentile(run.chord_total,75,0) - np.nanmedian(run.chord_total,0))))
            ax[2+row].errorbar(comps, np.median(run.chord_total,0), yerr=total_errbar,label=label, ls='solid', color=rgbs(mID, 0.7), alpha=0.5)

            label = f"{m.__name__} Imputed Error"
            imp_errbar = np.vstack((abs(np.percentile(run.chord_imputed,25,0) - np.nanmedian(run.chord_imputed,0)),
                                    abs(np.percentile(run.chord_imputed,75,0) - np.nanmedian(run.chord_imputed,0))))
            e = ax[2+row].errorbar(comps, np.median(run.chord_imputed,0), yerr=imp_errbar, label=label, ls='dashed', color=rgbs(mID, 0.7), alpha=0.5)
            e[-1][0].set_linestyle('dashed')

            ax[2+row].legend()
            ax[2+row].set_xlabel("Number of Components")
            ax[2+row].set_ylabel("Error")
            ax[2+row].set_xticks([x for x in comps])
            ax[2+row].set_xticklabels([x for x in comps])
            ax[2+row].set_title(f"{data} dataset, 10% chord missingness vs component")
            # ax[2+row].set_yscale("log")


        impType = 'chord'
        for mID, m in enumerate(methods):
            bestImpErr = list()
            bestTotErr = list()
            for d in drops:
                folder = f"timpute/figures/cache/{data}/drop_{d}/"
                run, _ = loadImputation(impType, m, folder)
                bestImpErr.append(np.min(np.median(run.chord_imputed,0)))
                bestTotErr.append(np.min(np.median(run.chord_total,0)))

            label = f"{m.__name__} (imputed)"
            ax[3+row].plot([str(x) for x in drops], bestImpErr, '-o', label=label, color=rgbs(mID, 0.7))
            label = f"{m.__name__} (total)"
            ax[3+row].plot([str(x) for x in drops], bestTotErr, '-^', label=label, color=rgbs(mID, 0.7))

            ax[3+row].legend()
            ax[3+row].set_xlabel("Chord Drop Percent")
            ax[3+row].set_ylabel("Median Error")
            ax[3+row].set_title(f"{data} dataset, lowest median error by chord missingness percentage")

        # must adjust `rowLen``
        # for mID, m in enumerate(methods):
        #     _, tracker = loadImputation(impType, m, folder)

        #     label = f"{m.__name__} Imputed Error"
        #     imputed_errbar = np.vstack((-(np.percentile(tracker.imputed_array,25,0) - np.nanmedian(tracker.imputed_array,0)),
        #                                 np.percentile(tracker.imputed_array,75,0) - np.nanmedian(tracker.imputed_array,0),))
        #     e = ax[3+row].errorbar(np.arange(tracker.imputed_array.shape[1]) + (0.1-mID*0.1), np.nanmedian(tracker.imputed_array,0), label=label, color=rgbs(mID,0.7),
        #                         yerr = imputed_errbar, ls='dashed', errorevery=5)
        #     e[-1][0].set_linestyle('dashed')

        #     label = f"{m.__name__} Total Error"
        #     total_errbar = np.vstack((-(np.percentile(tracker.total_array,25,0) - np.nanmedian(tracker.total_array,0)),
        #                                 np.percentile(tracker.total_array,75,0) - np.nanmedian(tracker.total_array,0)))
        #     ax[3+row].errorbar(np.arange(tracker.total_array.shape[1]), np.nanmedian(tracker.total_array,0), label=label, color=rgbs(mID,0.7),
        #                         yerr=total_errbar, ls = 'solid', errorevery=5)

        #     ax[3+row].legend()
        #     ax[3+row].set_xlim((0, tracker.total_array.shape[1]))
        #     ax[3+row].set_xlabel('Iteration')
        #     ax[3+row].set_ylabel('Error')
        

    subplotLabel(ax)
    f.savefig('timpute/figures/realdat_figures.png', bbox_inches="tight", format='png')

real_data(["hms"],[10])
