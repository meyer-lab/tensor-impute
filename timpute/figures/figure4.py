import numpy as np
import os

from .runImputation import *
from ..generateTensor import generateTensor
from ..method_ALS import perform_ALS
from ..method_CLS import perform_CLS
from ..method_DO import perform_DO
from ..plot import *
from ..common import *

# run `poetry run python -m timpute.figures.figure4` from root
methods = [perform_DO, perform_ALS, perform_CLS]

def figure4():
    drop_perc = 0.1
    max_component = 8
    reps = 20
    n_simulated = 10
    seed = 4

    dirname = f"timpute/figures/saves/figure4"

    print("--- BEGAN SIMULATED ---")
    tens_type = 'simulated'
    init_type = 'random'
    savename = f"/{tens_type}/{init_type}/"
    if os.path.isdir(dirname+savename) is False: os.makedirs(dirname+savename)

    impType = 'entry'
    for tens in range(n_simulated):
        orig = generateTensor('known')
        for m in methods:
            runMultiImputation(data=orig, number=tens, max_rr=max_component, impType=impType, savename=dirname+savename, method=m,
                               repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)
        print(f"Tensor {tens+1} completed.")

    impType = 'chord'
    for tens in range(n_simulated):
        orig = generateTensor('known')
        for m in methods:
            runMultiImputation(data=orig, number=tens, max_rr=max_component, impType=impType, savename=dirname+savename, method=m,
                               repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)
        print(f"Tensor {tens+1} completed.")

    
    init_type = 'svd'
    savename = f"/{tens_type}/{init_type}/"
    if os.path.isdir(dirname+savename) is False: os.makedirs(dirname+savename)

    impType = 'entry'
    for tens in range(n_simulated):
        orig = generateTensor('known')
        for m in methods:
            runMultiImputation(data=orig, number=tens, max_rr=max_component, impType=impType, savename=dirname+savename, method=m,
                               repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)
        print(f"Tensor {tens+1} completed.")

    impType = 'chord'
    for tens in range(n_simulated):
        orig = generateTensor('known')
        for m in methods:
            runMultiImputation(data=orig, number=tens, max_rr=max_component, impType=impType, savename=dirname+savename, method=m,
                               repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)
        print(f"Tensor {tens+1} completed.")

    #################################

    print("--- BEGAN ZOHAR ---")
    orig = generateTensor('zohar')

    tens_type = 'zohar'
    init_type = 'random'
    savename = f"/{tens_type}/{init_type}/"
    if os.path.isdir(dirname+savename) == False: os.makedirs(dirname+savename)

    impType = 'entry'
    for m in methods:
        runImputation(orig, max_component, impType, m, dirname+savename, save=True,
                      repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)
    
    impType = 'chord'
    for m in methods:
        runImputation(orig, max_component, impType, m, dirname+savename, save=True,
                      repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)    
    

    init_type = 'svd'
    savename = f"/{tens_type}/{init_type}/"
    if os.path.isdir(dirname+savename) == False: os.makedirs(dirname+savename)

    impType = 'entry'
    for m in methods:
        runImputation(orig, max_component, impType, m, dirname+savename, save=True,
                      repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)
    
    impType = 'chord'
    for m in methods:
        runImputation(orig, max_component, impType, m, dirname+savename, save=True,
                      repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)
    
    #################################

    print("--- BEGAN MILLS ---")
    orig = generateTensor('hms')

    tens_type = 'mills'
    init_type = 'random'
    savename = f"/{tens_type}/{init_type}/"
    if os.path.isdir(dirname+savename) == False: os.makedirs(dirname+savename)

    impType = 'entry'
    for m in methods:
        runImputation(orig, max_component, impType, m, dirname+savename, save=True,
                      repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)
    
    impType = 'chord'
    for m in methods:
        runImputation(orig, max_component, impType, m, dirname+savename, save=True,
                      repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)  
    

    init_type = 'svd'
    savename = f"/{tens_type}/{init_type}/"
    if os.path.isdir(dirname+savename) == False: os.makedirs(dirname+savename)

    impType = 'entry'
    for m in methods:
        runImputation(orig, max_component, impType, m, dirname+savename, save=True,
                      repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)
    
    impType = 'chord'
    for m in methods:
        runImputation(orig, max_component, impType, m, dirname+savename, save=True,
                      repeat=reps, drop=drop_perc, init=init_type, callback_r=max_component, seed=seed*100)
    


def figure4_graph_expanded():
    ax, f = getSetup((54,24), (4,9))
    dirname = f"timpute/figures/saves/figure4"

    ################ SIMULATED ################  

    tens_type = 'simulated'
    init_type = 'random'
    savename = f"/{tens_type}/{init_type}/"
    
    impType = 'entry'
    for mID, m in enumerate(methods):
        run, callback = loadMultiImputation(impType, m, dirname+savename)
        q2x_plot(ax[0], methodname = m.__name__,
                imputed_arr = run.entry_imputed, fitted_arr = run.entry_fitted, total_arr = run.entry_total,
                showLegend = True,
                offset = -0.1 + (mID*0.1),
                color = rgbs(mID, 0.7),
                logbound = -4)
        iteration_plot(ax[1], methodname = m.__name__,
                       tracker = callback,
                       showLegend = True,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       logbound = -4)
        runtime_plot(ax[2],methodname = m.__name__,
                     tracker = callback,
                     plotTotal = True,
                     threshold = 0.1,
                     timebound = (0,0.01),
                     color = rgbs(mID, 0.2))
    
    impType = 'chord'
    for mID, m in enumerate(methods):
        run, callback = loadMultiImputation(impType, m, dirname+savename)
        q2x_plot(ax[9], methodname = m.__name__,
                imputed_arr = run.chord_imputed, fitted_arr = run.chord_fitted, total_arr = run.chord_total,
                showLegend = True,
                offset = -0.1 + (mID*0.1),
                color = rgbs(mID, 0.7),
                logbound = -4)
        iteration_plot(ax[10], methodname = m.__name__,
                       tracker = callback,
                       showLegend = True,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       logbound = -4)
        runtime_plot(ax[11],methodname = m.__name__,
                     tracker = callback,
                     plotTotal = True,
                     threshold = 0.1,
                     timebound = (0,0.01),
                     color = rgbs(mID, 0.2))
    
    init_type = 'svd'
    savename = f"/{tens_type}/{init_type}/"

    impType = 'entry'
    for mID, m in enumerate(methods):
        run, callback = loadMultiImputation(impType, m, dirname+savename)
        q2x_plot(ax[18], methodname = m.__name__,
                imputed_arr = run.entry_imputed, fitted_arr = run.entry_fitted, total_arr = run.entry_total,
                showLegend = True,
                offset = -0.1 + (mID*0.1),
                color = rgbs(mID, 0.7),
                logbound = -4)
        iteration_plot(ax[19], methodname = m.__name__,
                       tracker = callback,
                       showLegend = True,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       logbound = -4)
        runtime_plot(ax[20],methodname = m.__name__,
                     tracker = callback,
                     plotTotal = True,
                     threshold = 0.1,
                     timebound = (0,0.01),
                     color = rgbs(mID, 0.2))
    
    impType = 'chord'
    for mID, m in enumerate(methods):
        run, callback = loadMultiImputation(impType, m, dirname+savename)
        q2x_plot(ax[27], methodname = m.__name__,
                imputed_arr = run.chord_imputed, fitted_arr = run.chord_fitted, total_arr = run.chord_total,
                showLegend = True,
                offset = -0.1 + (mID*0.1),
                color = rgbs(mID, 0.7),
                logbound = -4)
        iteration_plot(ax[28], methodname = m.__name__,
                       tracker = callback,
                       showLegend = True,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       logbound = -4)
        runtime_plot(ax[29],methodname = m.__name__,
                     tracker = callback,
                     plotTotal = True,
                     threshold = 0.1,
                     timebound = (0,0.01),
                     color = rgbs(mID, 0.2))
        
    ################ ZOHAR ################

    tens_type = 'zohar'
    init_type = 'random'
    savename = f"/{tens_type}/{init_type}/"
    
    impType = 'entry'
    for mID, m in enumerate(methods):
        run, callback = loadImputation(impType, m, dirname+savename)
        q2x_plot(ax[0+3], methodname = m.__name__,
                imputed_arr = run.entry_imputed, fitted_arr = run.entry_fitted, total_arr = run.entry_total,
                showLegend = True,
                offset = -0.1 + (mID*0.1),
                color = rgbs(mID, 0.7),
                logbound = -2)
        iteration_plot(ax[1+3], methodname = m.__name__,
                       tracker = callback,
                       showLegend = True,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       logbound = -2)
        runtime_plot(ax[2+3],methodname = m.__name__,
                     tracker = callback,
                     plotTotal = True,
                     threshold = 0.1,
                     timebound = (0,0.5),
                     color = rgbs(mID, 0.2))
    
    impType = 'chord'
    for mID, m in enumerate(methods):
        run, callback = loadImputation(impType, m, dirname+savename)
        q2x_plot(ax[9+3], methodname = m.__name__,
                imputed_arr = run.chord_imputed, fitted_arr = run.chord_fitted, total_arr = run.chord_total,
                showLegend = True,
                offset = -0.1 + (mID*0.1),
                color = rgbs(mID, 0.7),
                logbound = -2)
        iteration_plot(ax[10+3], methodname = m.__name__,
                       tracker = callback,
                       showLegend = True,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       logbound = -2)
        runtime_plot(ax[11+3],methodname = m.__name__,
                     tracker = callback,
                     plotTotal = True,
                     threshold = 0.1,
                     timebound = (0,0.5),
                     color = rgbs(mID, 0.2))
    
    init_type = 'svd'
    savename = f"/{tens_type}/{init_type}/"

    impType = 'entry'
    for mID, m in enumerate(methods):
        run, callback = loadImputation(impType, m, dirname+savename)
        q2x_plot(ax[18+3], methodname = m.__name__,
                imputed_arr = run.entry_imputed, fitted_arr = run.entry_fitted, total_arr = run.entry_total,
                showLegend = True,
                offset = -0.1 + (mID*0.1),
                color = rgbs(mID, 0.7),
                logbound = -2)
        iteration_plot(ax[19+3], methodname = m.__name__,
                       tracker = callback,
                       showLegend = True,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       logbound = -2)
        runtime_plot(ax[20+3],methodname = m.__name__,
                     tracker = callback,
                     plotTotal = True,
                     threshold = 0.1,
                     timebound = (0,0.5),
                     color = rgbs(mID, 0.2))
    
    impType = 'chord'
    for mID, m in enumerate(methods):
        run, callback = loadImputation(impType, m, dirname+savename)
        q2x_plot(ax[27+3], methodname = m.__name__,
                imputed_arr = run.chord_imputed, fitted_arr = run.chord_fitted, total_arr = run.chord_total,
                showLegend = True,
                offset = -0.1 + (mID*0.1),
                color = rgbs(mID, 0.7),
                logbound = -2)
        iteration_plot(ax[28+3], methodname = m.__name__,
                       tracker = callback,
                       showLegend = True,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       logbound = -2)
        runtime_plot(ax[29+3],methodname = m.__name__,
                     tracker = callback,
                     plotTotal = True,
                     threshold = 0.1,
                     timebound = (0,0.5),
                     color = rgbs(mID, 0.2))
        
    ################ MILLS ################  

    tens_type = 'mills'
    init_type = 'random'
    savename = f"/{tens_type}/{init_type}/"
    
    impType = 'entry'
    for mID, m in enumerate(methods):
        run, callback = loadImputation(impType, m, dirname+savename)
        q2x_plot(ax[0+6], methodname = m.__name__,
                imputed_arr = run.entry_imputed, fitted_arr = run.entry_fitted, total_arr = run.entry_total,
                showLegend = True,
                offset = -0.1 + (mID*0.1),
                color = rgbs(mID, 0.7),
                logbound = -2)
        iteration_plot(ax[1+6], methodname = m.__name__,
                       tracker = callback,
                       showLegend = True,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       logbound = -2)
        runtime_plot(ax[2+6],methodname = m.__name__,
                     tracker = callback,
                     plotTotal = True,
                     threshold = 0.1,
                     timebound = (0,0.3),
                     color = rgbs(mID, 0.2),
                     printvalues=True)
    
    impType = 'chord'
    for mID, m in enumerate(methods):
        run, callback = loadImputation(impType, m, dirname+savename)
        q2x_plot(ax[9+6], methodname = m.__name__,
                imputed_arr = run.chord_imputed, fitted_arr = run.chord_fitted, total_arr = run.chord_total,
                showLegend = True,
                offset = -0.1 + (mID*0.1),
                color = rgbs(mID, 0.7),
                logbound = -2)
        iteration_plot(ax[10+6], methodname = m.__name__,
                       tracker = callback,
                       showLegend = True,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       logbound = -2)
        runtime_plot(ax[11+6],methodname = m.__name__,
                     tracker = callback,
                     plotTotal = True,
                     threshold = 0.1,
                     timebound = (0,0.3),
                     color = rgbs(mID, 0.2))
    
    init_type = 'svd'
    savename = f"/{tens_type}/{init_type}/"

    impType = 'entry'
    for mID, m in enumerate(methods):
        run, callback = loadImputation(impType, m, dirname+savename)
        q2x_plot(ax[18+6], methodname = m.__name__,
                imputed_arr = run.entry_imputed, fitted_arr = run.entry_fitted, total_arr = run.entry_total,
                showLegend = True,
                offset = -0.1 + (mID*0.1),
                color = rgbs(mID, 0.7),
                logbound = -2)
        iteration_plot(ax[19+6], methodname = m.__name__,
                       tracker = callback,
                       showLegend = True,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       logbound = -2)
        runtime_plot(ax[20+6],methodname = m.__name__,
                     tracker = callback,
                     plotTotal = True,
                     threshold = 0.1,
                     timebound = (0,0.3),
                     color = rgbs(mID, 0.2))
    
    impType = 'chord'
    for mID, m in enumerate(methods):
        run, callback = loadImputation(impType, m, dirname+savename)
        q2x_plot(ax[27+6], methodname = m.__name__,
                imputed_arr = run.chord_imputed, fitted_arr = run.chord_fitted, total_arr = run.chord_total,
                showLegend = True,
                offset = -0.1 + (mID*0.1),
                color = rgbs(mID, 0.7),
                logbound = -2)
        iteration_plot(ax[28+6], methodname = m.__name__,
                       tracker = callback,
                       showLegend = True,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       logbound = -2)
        runtime_plot(ax[29+6],methodname = m.__name__,
                     tracker = callback,
                     plotTotal = True,
                     threshold = 0.1,
                     timebound = (0,0.3),
                     color = rgbs(mID, 0.2))
    

    f.savefig(dirname+'-expanded.png', bbox_inches="tight", format='png')



# figure4()
figure4_graph_expanded()
