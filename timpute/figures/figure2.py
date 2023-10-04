import numpy as np
import os

# output
from tqdm import tqdm
from datetime import datetime
from pytz import timezone
import inspect

from .runImputation import *
from ..generateTensor import generateTensor
from ..method_ALS import perform_ALS
from ..method_CLS import perform_CLS
from ..method_DO import perform_DO
from ..plot import *
from ..common import *

# run `poetry run python -m timpute.figures.figure2` from root
methods = [perform_CLS]

"""
Figure 2 - Intro to Imputation Metrics
    * uses only CLS method
    a) run method on full tensors, for 3 distributions at rank 8/20 [component vs total error, line graph w/ errbar]
    b) impute @ default - best component [component vs imputed error, line graph w/ errbar]
    c) impute @ default - method behavior [iteration vs imputed error, line graph w/ errbar]
    d) impute @ default - runtime to imputed threshold, after reaching fitted threshold of ?? [count of runtime, histogram]

DEFAULTS:
    - NUMPY SEED: 1
    - INITIALIZATION: random factors
    - TRUE RANK OF SIMULATED DATA: 8
    - PERCENT MISSINGNESS: 10% (0.1)
    - UNIQUE MISSINGNESS PATTERNS PER TENSOR: 20
    - TENSORS PER SIMULATED "RUN": 10
    
"""
def fig_2a_data():
    """ 2a) run CLS method on full tensors (3 dist, 2 ranks) """

    seed = 1
    drop_perc = 0.0
    init_type = 'random'
    max_component = 10
    reps = 20
    n_simulated = 10
    dirname = f"timpute/figures/cache/figure2"
    stdout = open(f"{dirname}/output.txt", 'a+')
    stdout.write(f"\n\n===================\nStarting new run for [{inspect.currentframe().f_code.co_name}]\nTimestamp: {datetime.now(timezone('US/Pacific'))}")

    savename = "/simulated-nonmissing/"
    folder = dirname+savename
    np.random.seed(seed)

    run = "gamma_r8/"
    if os.path.isdir(folder+run) is False: os.makedirs(folder+run)
    impType = 'entry'
    for tens in tqdm(range(n_simulated), file=stdout, miniters=int(n_simulated/5), desc=f"Decomposing {n_simulated} tensors ({run})"):
        orig = generateTensor(type='known', r=8, distribution='gamma')
        for m in methods:
            runMultiImputation(data=orig, number=tens, max_rr=max_component, impType=impType, savename=folder+run, method=m,
                               repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)
    
    run = "logistic_r8/"
    if os.path.isdir(folder+run) is False: os.makedirs(folder+run)
    impType = 'entry'
    for tens in tqdm(range(n_simulated), file=stdout, miniters=int(n_simulated/5), desc=f"Decomposing {n_simulated} tensors ({run})"):
        orig = generateTensor(type='known', r=8, distribution='logistic')
        for m in methods:
            runMultiImputation(data=orig, number=tens, max_rr=max_component, impType=impType, savename=folder+run, method=m,
                               repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)
    
    run = "normal_r8/"
    if os.path.isdir(folder+run) is False: os.makedirs(folder+run)
    impType = 'entry'
    for tens in tqdm(range(n_simulated), file=stdout, miniters=int(n_simulated/5), desc=f"Decomposing {n_simulated} tensors ({run})"):
        orig = generateTensor(type='known', r=8, distribution='normal')
        for m in methods:
            runMultiImputation(data=orig, number=tens, max_rr=max_component, impType=impType, savename=folder+run, method=m,
                               repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)
            
    run = "gamma_r20/"
    if os.path.isdir(folder+run) is False: os.makedirs(folder+run)
    impType = 'entry'
    for tens in tqdm(range(n_simulated), file=stdout, miniters=int(n_simulated/5), desc=f"Decomposing {n_simulated} tensors ({run})"):
        orig = generateTensor(type='known', r=20, distribution='gamma')
        for m in methods:
            runMultiImputation(data=orig, number=tens, max_rr=max_component, impType=impType, savename=folder+run, method=m,
                               repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)
    
    run = "logistic_r20/"
    if os.path.isdir(folder+run) is False: os.makedirs(folder+run)
    impType = 'entry'
    for tens in tqdm(range(n_simulated), file=stdout, miniters=int(n_simulated/5), desc=f"Decomposing {n_simulated} tensors ({run})"):
        orig = generateTensor(type='known', r=20, distribution='logistic')
        for m in methods:
            runMultiImputation(data=orig, number=tens, max_rr=max_component, impType=impType, savename=folder+run, method=m,
                               repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)
    
    run = "normal_r20/"
    if os.path.isdir(folder+run) is False: os.makedirs(folder+run)
    impType = 'entry'
    for tens in tqdm(range(n_simulated), file=stdout, miniters=int(n_simulated/5), desc=f"Decomposing {n_simulated} tensors ({run})"):
        orig = generateTensor(type='known', r=20, distribution='normal')
        for m in methods:
            runMultiImputation(data=orig, number=tens, max_rr=max_component, impType=impType, savename=folder+run, method=m,
                               repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)
    

def fig_2bcd_data():
    """ 2b/c/d) run CLS method to impute @ default (3 dist, 2 ranks) """

    seed = 1
    drop_perc = 0.1
    init_type = 'random'
    max_component = 10
    reps = 20
    n_simulated = 10
    dirname = f"timpute/figures/cache/figure2"
    stdout = open(f"{dirname}output.txt", 'a+')
    stdout.write(f"\n\n===================\nStarting new run for [{inspect.currentframe().f_code.co_name}]\nTimestamp: {datetime.now(timezone('US/Pacific'))}")

    savename = "/simulated-imputed/"
    folder = dirname+savename
    np.random.seed(seed)

    run = "gamma_r8/"
    if os.path.isdir(folder+run) is False: os.makedirs(folder+run)
    impType = 'entry'
    for tens in tqdm(range(n_simulated), file=stdout, miniters=int(n_simulated/5), desc=f"Decomposing {n_simulated} tensors ({run})"):
        orig = generateTensor(type='known', r=8, distribution='gamma')
        for m in methods:
            runMultiImputation(data=orig, number=tens, max_rr=max_component, impType=impType, savename=folder+run, method=m,
                               repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)
    
    run = "logistic_r8/"
    if os.path.isdir(folder+run) is False: os.makedirs(folder+run)
    impType = 'entry'
    for tens in tqdm(range(n_simulated), file=stdout, miniters=int(n_simulated/5), desc=f"Decomposing {n_simulated} tensors ({run})"):
        orig = generateTensor(type='known', r=8, distribution='logistic')
        for m in methods:
            runMultiImputation(data=orig, number=tens, max_rr=max_component, impType=impType, savename=folder+run, method=m,
                               repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)
    
    run = "normal_r8/"
    if os.path.isdir(folder+run) is False: os.makedirs(folder+run)
    impType = 'entry'
    for tens in tqdm(range(n_simulated), file=stdout, miniters=int(n_simulated/5), desc=f"Decomposing {n_simulated} tensors ({run})"):
        orig = generateTensor(type='known', r=8, distribution='normal')
        for m in methods:
            runMultiImputation(data=orig, number=tens, max_rr=max_component, impType=impType, savename=folder+run, method=m,
                               repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)
            
    run = "gamma_r20/"
    if os.path.isdir(folder+run) is False: os.makedirs(folder+run)
    impType = 'entry'
    for tens in tqdm(range(n_simulated), file=stdout, miniters=int(n_simulated/5), desc=f"Decomposing {n_simulated} tensors ({run})"):
        orig = generateTensor(type='known', r=20, distribution='gamma')
        for m in methods:
            runMultiImputation(data=orig, number=tens, max_rr=max_component, impType=impType, savename=folder+run, method=m,
                               repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)
    
    run = "logistic_r20/"
    if os.path.isdir(folder+run) is False: os.makedirs(folder+run)
    impType = 'entry'
    for tens in tqdm(range(n_simulated), file=stdout, miniters=int(n_simulated/5), desc=f"Decomposing {n_simulated} tensors ({run})"):
        orig = generateTensor(type='known', r=20, distribution='logistic')
        for m in methods:
            runMultiImputation(data=orig, number=tens, max_rr=max_component, impType=impType, savename=folder+run, method=m,
                               repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)
    
    run = "normal_r20/"
    if os.path.isdir(folder+run) is False: os.makedirs(folder+run)
    impType = 'entry'
    for tens in tqdm(range(n_simulated), file=stdout, miniters=int(n_simulated/5), desc=f"Decomposing {n_simulated} tensors ({run})"):
        orig = generateTensor(type='known', r=20, distribution='normal')
        for m in methods:
            runMultiImputation(data=orig, number=tens, max_rr=max_component, impType=impType, savename=folder+run, method=m,
                               repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)



def fig_2():
    ax,f = getSetup((12,12), (2,2))
    dirname = f"timpute/figures/cache/figure2"

    savename = "/simulated-nonmissing/"
    folder = dirname+savename
    impType = 'entry'

    linestyles = ['solid', (0,(1,1)), 'dashdot', (0,(5,1)), 'dotted', (3,(3,1,1,1,1,1))]
    #             solid / dot (tight) / dashdot /  dash  / dot (wide) /  dash dotdot
    comparison = ["gamma_r8", "logistic_r8", "normal_r8", "gamma_r20", "logistic_r20", "normal_r20"]

    # ----- FIGURE 2A -----

    for ls, exp in enumerate(comparison):
        for mID, m in enumerate(methods):
            run, _ = loadMultiImputation(impType, m, folder+exp+'/')

            comps = np.arange(1,run.entry_imputed.shape[1]+1)
            label = f"{exp} Total Error"
            total_errbar = np.vstack((abs(np.percentile(run.entry_total,25,0) - np.nanmedian(run.entry_total,0)),
                                    abs(np.percentile(run.entry_total,75,0) - np.nanmedian(run.entry_total,0))))
            ax[0].errorbar(comps, np.median(run.entry_total,0), yerr=total_errbar,label=label, ls=linestyles[ls], color=rgbs(mID, 0.7), markersize=10/3, alpha=0.5)

            ax[0].legend()
            ax[0].set_xlabel("Number of Components")
            ax[0].set_ylabel("Error")
            ax[0].set_xticks([x for x in comps])
            ax[0].set_xticklabels([x for x in comps])
            ax[0].set_yscale("log")
            ax[0].set_ylim(10**-3.5,1)
    
    savename = "/simulated-imputed/"
    folder = dirname+savename

    

    for ls, exp in enumerate(comparison):
        for mID, m in enumerate(methods):
            # ----- FIGURE 2B -----

            run, tracker = loadMultiImputation(impType, m, folder+exp+'/')

            comps = np.arange(1,run.entry_imputed.shape[1]+1)
            label = f"{exp} Total Error"
            total_errbar = np.vstack((abs(np.percentile(run.entry_total,25,0) - np.nanmedian(run.entry_total,0)),
                                    abs(np.percentile(run.entry_total,75,0) - np.nanmedian(run.entry_total,0))))
            ax[1].errorbar(comps, np.median(run.entry_total,0), yerr=total_errbar,label=label, ls=linestyles[ls], color=rgbs(mID, 0.7), alpha=0.5)


            label = f"{exp} Imputed Error"
            imp_errbar = np.vstack((abs(np.percentile(run.entry_imputed,25,0) - np.nanmedian(run.entry_imputed,0)),
                                    abs(np.percentile(run.entry_imputed,75,0) - np.nanmedian(run.entry_imputed,0))))
            ax[1].errorbar(comps, np.median(run.entry_imputed,0), yerr=imp_errbar,label=label, ls=linestyles[ls], color=rgbs(mID+1, 0.7), alpha=0.5)

            ax[1].legend()
            ax[1].set_xlabel("Number of Components")
            ax[1].set_ylabel("Error")
            ax[1].set_xticks([x for x in comps])
            ax[1].set_xticklabels([x for x in comps])
            ax[1].set_yscale("log")
            ax[1].set_ylim(10**-3.5,1)

            # ----- FIGURE 2C -----

            label = f"{exp} Imputed Error"
            imp_errbar = np.vstack((-(np.percentile(tracker.imputed_array,25,0) - np.nanmedian(tracker.imputed_array,0)),
                                        np.percentile(tracker.imputed_array,75,0) - np.nanmedian(tracker.imputed_array,0),))
            ax[2].errorbar(np.arange(tracker.imputed_array.shape[1]), np.nanmedian(tracker.imputed_array,0), label=label, color=rgbs(mID+1, 0.7),
                             yerr = imp_errbar, ls=linestyles[ls], alpha=0.5, errorevery=5)


            label = f"{exp} Total Error"
            total_errbar = np.vstack((-(np.percentile(tracker.total_array,25,0) - np.nanmedian(tracker.total_array,0)),
                                        np.percentile(tracker.total_array,75,0) - np.nanmedian(tracker.total_array,0)))
            ax[2].errorbar(np.arange(tracker.total_array.shape[1]), np.nanmedian(tracker.total_array,0), label=label, color=rgbs(mID, 0.7),
                        yerr=total_errbar, ls=linestyles[ls], alpha=0.5, errorevery=5)

            ax[2].legend()
            ax[2].set_xlim((0, tracker.total_array.shape[1]))
            ax[2].set_xlabel('Iteration')
            ax[2].set_ylabel('Error')
            ax[2].set_yscale("log")
            ax[2].set_ylim(10**-3.5,10**0.5)

            # ----- FIGURE 2D -----

            if ls%3 == 0:
                thresholds = tracker.time_thresholds(0.1, True)
                label = f"{exp} ({len(thresholds)})"
                ax[3].hist(thresholds, label=label, fc=rgbs(ls,0.2), edgecolor=rgbs(ls), bins=50, range=0.1)
                ax[3].axvline(np.mean(thresholds), color=rgbs(ls), linestyle='dashed', linewidth=1)

                ax[3].legend(loc='upper right')
                ax[3].set_xlabel('Runtime')
                ax[3].set_ylabel('Count')
                ax[3].yaxis.set_major_locator(mtick.MaxNLocator(integer=True))

    
    f.savefig('timpute/figures/figure2.png', bbox_inches="tight", format='png')
        

# fig_2a_data()
# fig_2bcd_data()
fig_2()