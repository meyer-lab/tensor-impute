import numpy as np
from timpute.decomposition import Decomposition, MultiDecomp
from timpute.tracker import tracker
from timpute.common import *
from timpute.plot import *
from time import process_time
import os
from copy import deepcopy

from .test.simulated_tensors import createKnownRank, createUnknownRank, createNoise
from tensordata.atyeo import data as atyeo
from tensordata.zohar import data as zohar
from tensordata.alter import data as alter
from .initialization import initialize_fac
from .method_DO import perform_DO
from .method_ALS import perform_ALS
from .method_CLS import perform_CLS


""" Exploratory Analysis Graphs """

def generateTensor(type=None, r=6, shape=(10,10,10), scale=2, distribution='gamma', par=2, missingness=0.1, noise_scale=50):
    """ Tensor options: 'zohar', 'atyeo', 'alter', 'unknown', 'known', defaulting to 'known' """
    if type == 'zohar': return zohar().to_numpy()
    elif type == 'atyeo': return atyeo().to_numpy()
    elif type == 'alter': return alter()['Fc'].to_numpy()
    elif type == 'unknown':
            temp = createUnknownRank(drop_perc=missingness, size=shape, distribution=distribution, scale=scale, par=par)
            return createNoise(temp,noise_scale)
    elif type == 'known':
            temp = createKnownRank(drop_perc=missingness, size=shape, rank=r, distribution=distribution, scale=scale, par=par)
            return createNoise(temp,noise_scale)
    else:
            temp = createKnownRank(drop_perc=missingness, size=shape, rank=r, distribution=distribution, scale=scale, par=par)
            return createNoise(temp,noise_scale)

def compare_imputation(tensor=None, init='svd', alpha=None, methods=[perform_DO,perform_ALS,perform_CLS],
                       impute_type='entry', impute_r=6, impute_reps=5, impute_perc=0.25, impute_mode=0,
                       f_size=(12,6), save=None, printRuntime=True):
    # run all methods
    if tensor is None: tensor = generateTensor()
    
    dirname = 'methodruns/'+save
    if os.path.isdir(dirname) == False: os.makedirs(dirname)

    ax, f = getSetup(f_size, (2,len(methods)))
    methodID = 0
    start = process_time()

    for m in methods:
        mstart = process_time()
        # instantiate objects
        track = tracker(tensor,track_runtime=True)
        decomp = Decomposition(tensor, method=m, max_rr=impute_r)

        # run imputation
        # TODO: update with regularization when reg=True
        if impute_type=='entry':
            drop = int(impute_perc*np.sum(np.isfinite(tensor)))
            if alpha is not None and m == perform_CLS:
                decomp.Q2X_entry(drop=drop, repeat=impute_reps, callback=track, init=init, alpha=alpha)
            else:
                decomp.Q2X_entry(drop=drop, repeat=impute_reps, callback=track, init=init)

        if impute_type=='chord':
            drop = int(impute_perc*tensor.size/tensor.shape[impute_mode])
            if drop < 1: drop = 1
            decomp.Q2X_chord(drop=drop, repeat=impute_reps, callback=track, init=init)
        track.combine()
        if printRuntime: print(m.__name__ + ": " + str(process_time()-mstart))

        # plot components vs imputed/fitted error
        plotID = methodID
        if impute_type == 'entry': q2xentry(ax[plotID], decomp, methodname = m.__name__, detailed=True)
        elif impute_type == 'chord': q2xchord(ax[plotID], decomp, methodname = m.__name__, detailed=True)
        plotID = methodID + 3
        track.plot_iteration(ax[plotID])

        methodID += 1

        # save for inspection
        if save is not None:
            decomp.save('./'+dirname+'/' + m.__name__ + '-imputations')
            track.save('./'+dirname+'/' + m.__name__ + '-iters')
    
    if printRuntime: print("Total runtime: " + str(process_time()-start)+'\n')
    if save is not None: f.savefig('./'+dirname+'/' + "imputation_results", bbox_inches="tight")
    return f 

def l2_comparison(tensor=None, init='svd', alpha=[1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2],
                       impute_type='entry', impute_r=[3,4,5,6], impute_reps=5, impute_perc=0.25, impute_mode=0,
                       f_size=(12,6), save=None, printRuntime=True):
    if tensor is None: tensor = generateTensor()

    dirname = 'methodruns/'+save
    if os.path.isdir(dirname) == False: os.makedirs(dirname)

    ax, f = getSetup(f_size, (2,len(alpha)))
    start = process_time()

    for rr in impute_r:
        rstart = process_time()
        decomp = Decomposition(tensor, method=perform_CLS, max_rr=rr)

        for a in alpha:
            if impute_type=='entry':
                drop = int(impute_perc*np.sum(np.isfinite(tensor)))
                decomp.Q2X_entry(drop=drop, repeat=impute_reps, init=init, alpha=alpha, single=True)
                l2_plot(ax,decomp,methodname="perform_CLS", alpha=a, comp=rr)

            if impute_type=='chord':
                drop = int(impute_perc*tensor.size/tensor.shape[impute_mode])
                decomp.Q2X_chord(drop=drop, repeat=impute_reps, init=init, alpha=alpha, single=True)
                l2_plot(ax,decomp,methodname="perform_CLS", alpha=a, comp=rr)
        
        
         
        if printRuntime: print(rr + " components: " + str(process_time()-rstart))

def regraph(save=None, fname="new_imputation_results", impute_type='entry', methods=[perform_DO,perform_ALS,perform_CLS], f_size=(12,6)):
    assert(save is not None)
    ax, f = getSetup(f_size, (2,3))
    methodID = 0
    dirname = os.getcwd()+'/methodruns/'+save
    os.chdir(dirname)


    for m in methods:
        decomp = Decomposition(np.ndarray((0)))
        track = tracker(np.ndarray((0)))
        decomp.load(m.__name__ + '-imputations')
        track.load(m.__name__ + '-iters')

        # plot components vs imputed/fitted error
        plotID = methodID
        if impute_type == 'entry': q2xentry(ax[plotID], decomp, methodname = m.__name__, detailed=True)
        elif impute_type == 'chord': q2xchord(ax[plotID], decomp, methodname = m.__name__, detailed=True)
        plotID = methodID + 3
        track.plot_iteration(ax[plotID])

        methodID = methodID + 1
    
    f.savefig(fname, bbox_inches="tight")

    return f



""" Figure Helper Functions """

methods = [perform_CLS, perform_ALS, perform_DO]

def rgbs(color = 0, transparency = None):
    color_rgbs = [(0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
                  (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
                  (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
                  (0.8352941176470589, 0.3686274509803922, 0.0),
                  (0.8, 0.47058823529411764, 0.7372549019607844),
                  (0.792156862745098, 0.5686274509803921, 0.3803921568627451),
                  (0.984313725490196, 0.6862745098039216, 0.8941176470588236),
                  (0.5803921568627451, 0.5803921568627451, 0.5803921568627451),
                  (0.9254901960784314, 0.8823529411764706, 0.2),
                  (0.33725490196078434, 0.7058823529411765, 0.9137254901960784)]
    if transparency is not None:
        preTran = list(color_rgbs[color])
        preTran.append(transparency)
        return tuple(preTran)
    else: return color_rgbs[color]

def sim_data(tensortype = 'known', name = 'simulated', tSize = (10,10,10),
             init = 'svd', methods = methods,
             useCallback=True, best_comp = [6,6,6],
             impute_perc = 0.1, impute_reps = 1, tensor_samples = 20,
             impEntry = True, impChord = True,
             seed = 5, printRuntime = True):
    """
    Generates a figure of method for `tensor_samples` tensors, each run `impute_reps` times for components 1 to 6 (max_rr).
    Identical initializations for each method run per tensor via np.random.seed.
    """
    assert init == 'svd' or init == 'random'
    assert impChord or impEntry
    np.random.seed(seed)
    max_rr = 6

    dirname = f"figures/{name}_{impute_perc}"
    if os.path.isdir(dirname) == False: os.makedirs(dirname)

    # for each tensor
    tstart = process_time()
    for i in range(tensor_samples):
        # generate tensor
        tensor = generateTensor(tensortype, max_rr, shape=tSize, missingness=0)

        for j, m in enumerate(methods):
            # initialize objects
            decomp = Decomposition(tensor, max_rr)
            if i == 0:
                if useCallback:
                    m_track = tracker(tensor,track_runtime=True)
                else:
                    m_track = None
                    for i in range(len(best_comp)): best_comp[i] = None
            else:
                m_decomp.load(f"./{dirname}/{m.__name__}-decomp")
                if useCallback:
                    m_track.load(f"./{dirname}/{m.__name__}-track")
                    m_track.new()
                    
            # run imputation, tracking for chords
            if impEntry and impChord:
                decomp.imputation(method=m, type='entry', drop=impute_perc, repeat=impute_reps, init=init)
                decomp.imputation(method=m, type='chord', drop=impute_perc, repeat=impute_reps, init=init, callback=m_track, callback_r=best_comp[j])
            elif not impChord:
                decomp.imputation(method=m, type='entry', drop=impute_perc, repeat=impute_reps, init=init, callback=m_track, callback_r=best_comp[j])
            elif not impEntry:
                decomp.imputation(method=m, type='chord', drop=impute_perc, repeat=impute_reps, init=init, callback=m_track, callback_r=best_comp[j])

            # save runs
            if i == 0:
                m_decomp = MultiDecomp(decomp, impEntry, impChord)
            else:
                m_decomp(decomp)

            m_decomp.save(f"./{dirname}/{m.__name__}-decomp")
            if useCallback: m_track.save(f"./{dirname}/{m.__name__}-track")

        # print runtime every 20% of the way there
        if (printRuntime and (i+1)%round(tensor_samples*0.2) == 0):
            print(f"Average runtime for {i+1} tensors: {(process_time()-tstart)/(i+1)} seconds")

    if printRuntime: print(f"{process_time()-tstart} seconds elapsed for figure {dirname}") # print total runtime

    return m_decomp, m_track

def full_graph(dirname, ax, ax_start, plot_total = False, showLegend=False,
                    logComp = True, logTrack = True, logbound=-3.5, endbound=1):
    """
    Generate graphs based on runs from sim_data().
    Requires saving runs from sim_data in MultiDecomp and tracker objects, with the proper naming (see outermost loop).
    a) Entry imputation for each method, component vs error
    b) Chord imputation for each method, component vs error
    c) Chord imputation for specified components (see sim_data(best_comp = ))
    d) Runtimes for specified components
    e) Number that hit threshold for specified components
    """

    m_decomp = MultiDecomp()
    m_track = tracker()

    # plot components vs imputed/fitted error
    for mID, m in enumerate(methods):
        m_decomp.load(f"./{dirname}/{m.__name__}-decomp")
        m_track.load(f"./{dirname}/{m.__name__}-track")
        m_track.combine()

        print(m.__name__)
        # plot graphs
        q2x_plot(ax[ax_start], m.__name__, m_decomp.entry_imputed, m_decomp.entry_fitted, m_decomp.entry_total, color=rgbs(mID, transparency=0.8),
                 plot_total=plot_total, offset=mID, log=logComp, logbound=logbound, endbound=endbound, showLegend=showLegend)
        q2x_plot(ax[ax_start+1], m.__name__, m_decomp.chord_imputed, m_decomp.chord_fitted, m_decomp.chord_total, color=rgbs(mID, transparency=0.8),
                 plot_total=plot_total, offset=mID, log=logComp, logbound=logbound, endbound=endbound, showLegend=showLegend)
        m_track.plot_iteration(ax[ax_start+2], color=rgbs(mID, transparency=0.8),
                               plot_total=plot_total, offset=mID, log=logTrack, logbound=logbound)

def single_graph(figname, ax, ax_start, plot_total=False, use_tracker=True, showLegend=False,
                    logbound=-3.5, logComp = True, logTrack = True, type='chord'):
    """
    Generate graphs based on runs from sim_data() (only chooses chord OR entry imputation).
    Requires saving runs from sim_data in MultiDecomp and tracker objects, with the proper naming (see outermost loop).
    a) Entry/Chord imputation (henceforth, "Imputation") for each method, component vs error
    b) Imputation for specified components (see sim_data(best_comp = ))
    c) Runtimes for specified components
    d) Number that hit threshold for specified components

    * typically used to create a row for SVD & random initialized decompositions
    """
    assert type == 'entry' or type == 'chord'
    dirname = f"figures/{figname}"
    m_decomp = MultiDecomp()
    m_track = tracker()
    transparency = 0.7

    # plot components vs imputed/fitted error
    for mID,m in enumerate(methods):
        m_decomp.load(f"./{dirname}/{m.__name__}-decomp")
        if type == 'entry': q2x_plot(ax[ax_start], m.__name__, m_decomp.entry_imputed, m_decomp.entry_fitted, m_decomp.entry_total, color=rgbs(mID, transparency),
                                     plot_total=plot_total, offset=mID, log=logComp, logbound=logbound,  showLegend=showLegend)
        if type == 'chord': q2x_plot(ax[ax_start], m.__name__, m_decomp.chord_imputed, m_decomp.chord_fitted, m_decomp.chord_total, color=rgbs(mID, transparency),
                                     plot_total=plot_total, offset=mID, log=logComp, logbound=logbound,  showLegend=showLegend)

        if use_tracker:
            m_track.load(f"./{dirname}/{m.__name__}-track")
            m_track.combine()
            m_track.plot_iteration(ax[ax_start+1], plot_total=plot_total, offset=mID, log=logTrack, logbound=logbound, color=rgbs(mID, transparency))

def comp_graph(figname, ax, ax_start, plot_total=False, showLegend=False,
                   logComp = True, logbound=-3.5, type='entry'):
    """
    Generate graphs based on runs from sim_data() (only chooses chord OR entry imputation).
    Requires saving runs from sim_data in MultiDecomp object, with the proper naming (see outermost loop).
    a) Entry/Chord imputation for each method, component vs error
    """
    assert type == 'entry' or type == 'chord'
    dirname = f"figures/{figname}"
    m_decomp = MultiDecomp()
    transparency = 0.7

    # plot components vs imputed/fitted error
    for mID,m in enumerate(methods):
        m_decomp.load(f"./{dirname}/{m.__name__}-decomp")
        if type == 'entry': q2x_plot(ax[ax_start], m.__name__, m_decomp.entry_imputed, m_decomp.entry_fitted, m_decomp.entry_total, showLegend=showLegend,
                                     plot_total=plot_total, offset=mID, log=logComp, logbound=logbound, color=rgbs(mID, transparency))
        if type == 'chord': q2x_plot(ax[ax_start], m.__name__, m_decomp.chord_imputed, m_decomp.chord_fitted, m_decomp.chord_total, showLegend=showLegend,
                                       plot_total=plot_total, offset=mID, log=logComp, logbound=logbound, color=rgbs(mID, transparency))

def runtime_graph(dirname, ax, ax_start, threshold=0.1, timebound=0.1,
                  graph_runtime=True, graph_unmet=True):
    """
    Generate graphs based on runs from sim_data() (only chooses chord OR entry imputation).
    Requires saving runs from sim_data in tracker objects, with the proper naming (see outermost loop).
    May choose one or both of these plots
    a) Runtimes for specified components
    b) Number that hit threshold for specified components
    """

    assert graph_runtime or graph_unmet
    m_track = tracker()
    unmet = list()
    methodnames = [m.__name__ for m in methods]

    for mID,m in enumerate(methods):
        m_track.load(f"./{dirname}/{m.__name__}-track")
        m_track.combine()
        thresholds = m_track.time_thresholds(threshold)
        ax[ax_start].hist(thresholds, label=f"{m.__name__} ({len(thresholds)})", fc=rgbs(mID, transparency=0.25), edgecolor=rgbs(mID), bins=50, range=(0,timebound))
        print(np.mean(thresholds))
        ax[ax_start].axvline(np.mean(thresholds), color=rgbs(mID), linestyle='dashed', linewidth=1)
        unmet.append(m_track.unmet_thresholds(threshold))

    if graph_runtime:
        ax[ax_start].legend(loc='upper right')
        ax[ax_start].set_xlabel('Runtime')
        ax[ax_start].set_ylabel('Count')
        if graph_unmet:
            unmet[:] = [x / m_track.imputed_array.shape[0] * 100 for x in unmet]
            ax[ax_start+1].bar(methodnames, unmet)
            ax[ax_start+1].set_xlabel('Method')
            ax[ax_start+1].set_ylabel('Percent Unmet')
            ax[ax_start+1].yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    elif graph_unmet:
        ax[ax_start].bar(methodnames, unmet)
        ax[ax_start].set_xlabel('Method')
        ax[ax_start].set_ylabel('Count')

    
