import numpy as np
import os

from .runImputation import *
from ..generateTensor import generateTensor
from ..method_ALS import perform_ALS
from ..method_CLS import perform_CLS
from ..method_DO import perform_DO
from ..plot import *
from ..common import *

# run `poetry run python -m timpute.figures.figure2-3` from root
methods = [perform_DO, perform_ALS, perform_CLS]

def figure23(simulated=True, zohar=True, mills=True):
    drop_perc = 0.1
    init_type = 'random'
    max_component = 8
    reps = 20
    n_simulated = 10
    dirname = f"timpute/figures/saves/figure2"
    seed = 2

    if simulated is True:
        print("--- BEGAN SIMULATED ---")
        savename = "/simulated/"
        if os.path.isdir(dirname+savename) is False: os.makedirs(dirname+savename)

        impType = 'entry'
        callback_rs = [8,8,8]
        for tens in range(n_simulated):
            orig = generateTensor('known')
            for i,m in enumerate(methods):
                runMultiImputation(data=orig, number=tens, max_rr=max_component, impType=impType, savename=dirname+savename, method=m,
                                repeat=reps, drop=drop_perc, init=init_type, callback_r=callback_rs[i], seed=seed*100)
            print(f"Tensor {tens+1} completed.")
        
        impType = 'chord'
        callback_rs = [7,2,7]
        for tens in range(n_simulated):
            orig = generateTensor('known')
            for i,m in enumerate(methods):
                runMultiImputation(data=orig, number=tens, max_rr=max_component, impType=impType, savename=dirname+savename, method=m,
                                repeat=reps, drop=drop_perc, init=init_type, callback_r=callback_rs[i], seed=seed*100)
            print(f"Tensor {tens+1} completed.")

    #################################
    
    if zohar is True:
        print("--- BEGAN ZOHAR ---")
        orig = generateTensor('zohar')

        savename = "/zohar/"
        if os.path.isdir(dirname+savename) == False: os.makedirs(dirname+savename)

        impType = 'entry'
        callback_rs = [7,7,7]
        for i,m in enumerate(methods):
            runImputation(orig, max_component, impType, m, dirname+savename, save=True,
                        repeat=reps, drop=drop_perc, init=init_type, callback_r=callback_rs[i], seed=seed*100)
        
        impType = 'chord'
        callback_rs = [6,2,6]
        for i,m in enumerate(methods):
            runImputation(orig, max_component, impType, m, dirname+savename, save=True,
                        repeat=reps, drop=drop_perc, init=init_type, callback_r=callback_rs[i], seed=seed*100)
        
    #################################
    
    if mills is True:
        print("--- BEGAN MILLS ---")
        orig = generateTensor('hms')

        savename = "/mills/"
        if os.path.isdir(dirname+savename) == False: os.makedirs(dirname+savename)

        impType = 'entry'
        callback_rs = [7,5,6]
        for i,m in enumerate(methods):
            runImputation(orig, max_component, impType, m, dirname+savename, save=True,
                        repeat=reps, drop=drop_perc, init=init_type, callback_r=callback_rs[i], seed=seed*100)
        
        impType = 'chord'
        callback_rs = [7,4,7]
        for i,m in enumerate(methods):
            runImputation(orig, max_component, impType, m, dirname+savename, save=True,
                        repeat=reps, drop=drop_perc, init=init_type, callback_r=callback_rs[i], seed=seed*100)
        


def figure2_graph():
    ax, f = getSetup((12,6), (1,2))
    dirname = f"timpute/figures/saves/figure2"

    savename = "/simulated/"
    
    for mID, m in enumerate(methods):
        impType = 'entry'
        run, _ = loadMultiImputation(impType, m, dirname+savename)
        # figure 2a
        q2x_plot(ax[0], methodname = m.__name__,
                imputed_arr = run.entry_imputed, fitted_arr = run.entry_fitted, total_arr = run.entry_total,
                plot_impute = True, plot_total = False,
                showLegend = True,
                offset = -0.1 + (mID*0.1),
                color = rgbs(mID, 0.7),
                logbound = -4)
        
        impType = 'chord'
        run, _ = loadMultiImputation(impType, m, dirname+savename)
        # figure 2b
        q2x_plot(ax[1], methodname = m.__name__,
                imputed_arr = run.chord_imputed, fitted_arr = run.chord_fitted, total_arr = run.chord_total,
                plot_impute = True, plot_total = False,
                showLegend = False,
                offset = -0.1 + (mID*0.1),
                color = rgbs(mID, 0.7),
                logbound = -4)

    f.savefig(dirname+'.png', bbox_inches="tight", format='png')
    
    #################################

    # figure 2c
    savename = "/zohar/"
    data = dict()

    impType = 'entry'
    for mID, m in enumerate(methods):
        tmplist = list()
        run, _ = loadImputation(impType, m, dirname+savename)
        impMatrix = run.entry_imputed
        impDist = impMatrix[:, np.median(impMatrix, axis=0).argmin()]
        tmplist.append(impDist)
        data[m.__name__] = tmplist
        print(np.median(impMatrix, axis=0).argmin())
        # 7, 7, 7

    impType = 'chord'
    for mID, m in enumerate(methods):
        run, _ = loadImputation(impType, m, dirname+savename)
        impMatrix = run.chord_imputed
        impDist = impMatrix[:, np.median(impMatrix, axis=0).argmin()]
        data[m.__name__].append(impDist)
        print(np.median(impMatrix, axis=0).argmin())
        # 6, 2, 6       
    
    savename = "/mills/"

    impType = 'entry'
    for mID, m in enumerate(methods):
        tmplist = list()
        run, _ = loadImputation(impType, m, dirname+savename)
        impMatrix = run.entry_imputed
        impDist = impMatrix[:, np.median(impMatrix, axis=0).argmin()]
        tmplist.append(impDist)
        data[m.__name__] = tmplist
        print(np.median(impMatrix, axis=0).argmin())
        # 7, 4, 7

    impType = 'chord'
    for mID, m in enumerate(methods):
        run, _ = loadImputation(impType, m, dirname+savename)
        impMatrix = run.chord_imputed
        impDist = impMatrix[:, np.median(impMatrix, axis=0).argmin()]
        data[m.__name__].append(impDist)
        print(np.median(impMatrix, axis=0).argmin())
        # 6, 3, 5

    plt.figure()
    ticks = ['Zohar (entry)', 'Zohar (chord)', 'Mills (entry)', 'Mills (chord)']

    bar_spacing = 0.3
    bar_width = 0.25
    exp_spacing = 2
    bpDO = plt.boxplot(data['perform_DO'], positions=np.array(range(len(data['perform_DO']))) * exp_spacing - bar_spacing, sym='', widths=bar_width)
    bpALS = plt.boxplot(data['perform_ALS'], positions=np.array(range(len(data['perform_ALS']))) * exp_spacing, sym='', widths=bar_width)
    bpCLS = plt.boxplot(data['perform_CLS'], positions=np.array(range(len(data['perform_CLS']))) * exp_spacing + bar_spacing, sym='', widths=bar_width)
    set_boxplot_color(bpDO, rgbs(0))
    set_boxplot_color(bpALS, rgbs(1))
    set_boxplot_color(bpCLS, rgbs(2))
    plt.xticks(range(0, len(ticks) * exp_spacing, exp_spacing), ticks)

    for i,m in enumerate(methods):
        plt.plot([], c=rgbs(i), label=m.__name__)
    plt.legend()

    plt.savefig(dirname+'c.png', bbox_inches="tight", format='png')



def figure3_graph():
    ax, f = getSetup((12,36), (6,2))
    dirname = f"timpute/figures/saves/figure2"

    savename = "/simulated/"
    
    impType = 'entry'
    for mID, m in enumerate(methods):
        _, callback = loadMultiImputation(impType, m, dirname+savename)
        # figure 3a
        iteration_plot(ax[0], methodname = m.__name__,
                       tracker = callback,
                       plot_impute = False, plot_total = True,
                       showLegend = True,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       logbound = -4)
        # figure 3b
        runtime_plot(ax[1],methodname = m.__name__,
                     tracker = callback,
                     plotTotal = True,
                     threshold = 0.01,
                     timebound = (0,0.15),
                     color = rgbs(mID, 0.2))

    impType = 'chord'
    for mID, m in enumerate(methods):
        _, callback = loadMultiImputation(impType, m, dirname+savename)
        # figure 3c
        iteration_plot(ax[2], methodname = m.__name__,
                       tracker = callback,
                       plot_impute = False, plot_total = True,
                       showLegend = True,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       logbound = -4)
        # figure 3d
        runtime_plot(ax[3],methodname = m.__name__,
                     tracker = callback,
                     plotTotal = True,
                     threshold = 0.01,
                     timebound = (0,0.15),
                     color = rgbs(mID, 0.2))
    
    #################################

    savename = "/zohar/"
    
    impType = 'chord'
    for mID, m in enumerate(methods):
        run, callback = loadImputation(impType, m, dirname+savename)
        # figure 3e
        iteration_plot(ax[4], methodname = m.__name__,
                       tracker = callback,
                       plot_impute = False, plot_total = True,
                       showLegend = True,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       logbound = -2)
        # figure 3f
        runtime_plot(ax[5], methodname= m.__name__,
                     tracker = callback,
                     plotTotal = True,
                     threshold = 0.05,
                     timebound = (0,0.2),
                     color = rgbs(mID, 0.2))


    # #################################

    savename = "/mills/"

    impType = 'chord'
    for mID, m in enumerate(methods):
        run, callback = loadImputation(impType, m, dirname+savename)
        # figure 3g
        iteration_plot(ax[6], methodname = m.__name__,
                       tracker = callback,
                       plot_impute = False, plot_total = True,
                       showLegend = True,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       logbound = -2)
        # figure 3h
        runtime_plot(ax[7], methodname= m.__name__,
                     tracker = callback,
                     plotTotal = True,
                     threshold = 0.05,
                     timebound = (0,0.2),
                     color = rgbs(mID, 0.2))

    dirname = f"timpute/figures/saves/figure3"
    f.savefig(dirname+'.png', bbox_inches="tight", format='png')
    

    
def figure3_graph_expanded():
    ax, f = getSetup((12,36), (6,2))
    dirname = f"timpute/figures/saves/figure2"
    
    ################ SIMULATED ################
    savename = "/simulated/"
        
    impType = 'entry'
    for mID, m in enumerate(methods):
        _, callback = loadMultiImputation(impType, m, dirname+savename)
        iteration_plot(ax[0], methodname = m.__name__,
                       tracker = callback,
                       plot_impute = False, plot_total = True,
                       showLegend = True,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       logbound = -4)
        runtime_plot(ax[1],methodname = m.__name__,
                     tracker = callback,
                     plotTotal = True,
                     threshold = 0.01,
                     timebound = (0,0.15),
                     color = rgbs(mID, 0.2))
    
    impType = 'chord'
    for mID, m in enumerate(methods):
        _, callback = loadMultiImputation(impType, m, dirname+savename)
        iteration_plot(ax[2], methodname = m.__name__,
                       tracker = callback,
                       plot_impute = False, plot_total = True,
                       showLegend = True,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       logbound = -4)
        runtime_plot(ax[3],methodname = m.__name__,
                     tracker = callback,
                     plotTotal = True,
                     threshold = 0.01,
                     timebound = (0,0.15),
                     color = rgbs(mID, 0.2))
    
    ################ ZOHAR ################
    savename = "/zohar/"

    impType = 'entry'
    for mID, m in enumerate(methods):
        run, callback = loadImputation(impType, m, dirname+savename)
        iteration_plot(ax[4], methodname = m.__name__,
                       tracker = callback,
                       plot_impute = False, plot_total = True,
                       showLegend = True,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       logbound = -2)
        runtime_plot(ax[5], methodname= m.__name__,
                     tracker = callback,
                     plotTotal = True,
                     threshold = 0.05,
                     timebound = (0,3),
                     color = rgbs(mID, 0.2))
    
    impType = 'chord'
    for mID, m in enumerate(methods):
        run, callback = loadImputation(impType, m, dirname+savename)
        iteration_plot(ax[6], methodname = m.__name__,
                       tracker = callback,
                       plot_impute = False, plot_total = True,
                       showLegend = True,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       logbound = -2)
        runtime_plot(ax[7], methodname= m.__name__,
                     tracker = callback,
                     plotTotal = True,
                     threshold = 0.05,
                     timebound = (0,0.2),
                     color = rgbs(mID, 0.2))


    ################ MILLS ################
    savename = "/mills/"

    impType = 'entry'
    for mID, m in enumerate(methods):
        run, callback = loadImputation(impType, m, dirname+savename)
        iteration_plot(ax[8], methodname = m.__name__,
                       tracker = callback,
                       plot_impute = False, plot_total = True,
                       showLegend = True,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       logbound = -2)
        runtime_plot(ax[9], methodname= m.__name__,
                     tracker = callback,
                     plotTotal = True,
                     threshold = 0.05,
                     timebound = (0,3),
                     color = rgbs(mID, 0.2))
    
    impType = 'chord'
    for mID, m in enumerate(methods):
        run, callback = loadImputation(impType, m, dirname+savename)
        iteration_plot(ax[10], methodname = m.__name__,
                       tracker = callback,
                       plot_impute = False, plot_total = True,
                       showLegend = True,
                       offset = -0.4 + (mID*0.4),
                       color = rgbs(mID, 0.7),
                       logbound = -2)
        runtime_plot(ax[11], methodname= m.__name__,
                     tracker = callback,
                     plotTotal = True,
                     threshold = 0.05,
                     timebound = (0,0.2),
                     color = rgbs(mID, 0.2))

    dirname = f"timpute/figures/saves/figure3"
    f.savefig(dirname+'-expanded.png', bbox_inches="tight", format='png')

# figure23()
figure2_graph()
# figure3_graph_expanded()