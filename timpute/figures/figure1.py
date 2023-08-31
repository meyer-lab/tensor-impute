import numpy as np
import os

from .runImputation import *
from ..generateTensor import generateTensor
from ..method_ALS import perform_ALS
from ..method_CLS import perform_CLS
from ..method_DO import perform_DO
from ..plot import *
from ..common import *

# run `poetry run python -m timpute.figures.figure1` from root
methods = [perform_DO, perform_ALS, perform_CLS]

def figure1(mini = False):
    drop_perc = 0.0
    init_type = 'random'
    max_component = 8
    reps = 20
    n_simulated = 10
    dirname = f"timpute/figures/saves/figure1"
    seed = 1

    print("--- BEGAN SIMULATED ---")
    np.random.seed(seed)
    savename = "/simulated/"
    if mini:
        savename += "mini/"
        n_simulated = 3
        reps = 5
    if os.path.isdir(dirname+savename) is False: os.makedirs(dirname+savename)

    impType = 'entry'
    for tens in range(n_simulated):
        orig = generateTensor('known')
        for m in methods:
            runMultiImputation(data=orig, number=tens, max_rr=max_component, impType=impType, savename=dirname+savename, method=m,
                               repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)
        print(f"Tensor {tens+1} completed.")

    #################################
    
    print("--- BEGAN ZOHAR ---")
    savename = "/zohar/"
    orig = generateTensor('zohar')
    if mini:
        savename += "mini/"
        reps = 5
    if os.path.isdir(dirname+savename) == False: os.makedirs(dirname+savename)

    impType = 'entry'
    for m in methods:
        runImputation(orig, max_component, impType, m, dirname+savename, save=True,
                      repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)
    
    #################################
    
    print("--- BEGAN HMS ---")
    savename = "/mills/"
    orig = generateTensor('hms')
    if mini:
        savename += "mini/"
        reps = 5
    if os.path.isdir(dirname+savename) == False: os.makedirs(dirname+savename)

    impType = 'entry'
    for m in methods:
        runImputation(orig, max_component, impType, m, dirname+savename, save=True,
                      repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)





def figure1_graph(mini = False):
    ax,f = getSetup((12,6), (1,2))
    dirname = f"timpute/figures/saves/figure1"

    savename = "/simulated/"
    if mini: savename += "mini/"
        
    impType = 'entry'
    for mID, m in enumerate(methods):
        run, callback = loadMultiImputation(impType, m, dirname+savename)
        # figure 1a
        q2x_plot(ax[0], methodname = m.__name__,
                imputed_arr = run.entry_imputed, fitted_arr = run.entry_fitted, total_arr = run.entry_total,
                plot_impute = False, plot_total = True,
                showLabels = True,
                offset = -0.1 + (mID*0.1),
                color = rgbs(mID, 0.7),
                total_fmt='D', total_fmtScale=1/1.5)
        # figure 1b
        iteration_plot(ax[1], methodname = m.__name__,
                       tracker = callback,
                       plot_impute = False, plot_total = True,
                       showLabels = True,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       total_ls = 'solid')


    savename = "/zohar/"
    if mini: savename += "mini/"

    impType = 'entry'
    for mID, m in enumerate(methods):
        run, callback = loadImputation(impType, m, dirname+savename)

        # figure 1a
        q2x_plot(ax[0], methodname = m.__name__,
                imputed_arr = run.entry_imputed, fitted_arr = run.entry_fitted, total_arr = run.entry_total,
                plot_impute = False, plot_total = True,
                showLabels = False,
                offset = -0.1 + (mID*0.1),
                color = rgbs(mID, 0.7),
                total_fmt='.', total_fmtScale=1.5)
        # figure 1b
        iteration_plot(ax[1], methodname = m.__name__,
                       tracker = callback,
                       plot_impute = False, plot_total = True,
                       showLabels = False,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       total_ls = (0,(1, 1)))
    

    savename = "/mills/"
    if mini: savename += "mini/"

    impType = 'entry'
    for mID, m in enumerate(methods):
        run, callback = loadImputation(impType, m, dirname+savename)

        # figure 1a
        q2x_plot(ax[0], methodname = m.__name__,
                imputed_arr = run.entry_imputed, fitted_arr = run.entry_fitted, total_arr = run.entry_total,
                plot_impute = False, plot_total = True,
                showLabels = False,
                showLegend = True,
                offset = -0.1 + (mID*0.1),
                color = rgbs(mID, 0.7),
                logbound = -4,
                total_fmt='^', total_fmtScale=1)
        # figure 1b
        iteration_plot(ax[1], methodname = m.__name__,
                       tracker = callback,
                       plot_impute = False, plot_total = True,
                       showLabels = False,
                       showLegend = True,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       logbound = -4,
                       total_ls = (0,(4, 1)))


    if mini: dirname += "-mini"
    f.savefig(dirname+'.png', bbox_inches="tight", format='png')

# figure1(mini=True)
# figure1_graph(mini=True)

# figure1()
# figure1_graph()