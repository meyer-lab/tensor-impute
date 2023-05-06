import numpy as np
import tensorly as tl
from timpute.decomposition import Decomposition, MultiDecomp
from timpute.tracker import tracker
from timpute.common import *
from timpute.plot import *
from time import process_time
import os

from timpute.test.simulated_tensors import createKnownRank, createUnknownRank, createNoise
from tensordata.atyeo import data as atyeo
from tensordata.zohar import data as zohar
from tensordata.alter import data as alter
from timpute.initialize_fac import initialize_fac
from timpute.direct_opt import perform_DO
from timpute.tensorly_als import perform_ALS
from timpute.cmtf import perform_CLS


""" Exploratory Analysis Graphs """

def generateTensor(type=None, r=6, shape=(20,25,30), scale=2, distribution='gamma', par=2, missingness=0.1, noise_scale=50):
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
        track.plot_iteration(ax[plotID], methodname=m.__name__)

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
        track.plot_iteration(ax[plotID], methodname=m.__name__)

        methodID = methodID + 1
    
    f.savefig(fname, bbox_inches="tight")

    return f



""" Figure Base Functions """



methods = [perform_CLS, perform_ALS, perform_DO]

def sim_data(name = None, tSize = (10,10,10), useCallback=True, best_comp = [6,6,6],
             impute_perc = 0.1, init = 'svd', impEntry = True, impChord = True,
             tensor_samples = 5, impute_reps = 5,
             seed = 5, printRuntime = True):
    """ Generates a figure of method for `tensor_samples` tensors, each run `impute_reps` times. Identical initializations for each method's run per tensor."""
    assert init == 'svd' or init == 'random'
    assert impChord or impEntry
    np.random.seed(seed)
    max_rr = 6

    dirname = f"figures/simulated_{name}_{impute_perc}"
    if os.path.isdir(dirname) == False: os.makedirs(dirname)

    # for each tensor
    tstart = process_time()
    for i in range(tensor_samples):
        # generate tensor
        tensor = generateTensor('known',r=max_rr,shape=tSize,missingness=0)
        entry_drop = int(impute_perc*np.sum(np.isfinite(tensor)))
        chord_drop = int(impute_perc*tensor.size/tensor.shape[0])
        inits = [[initialize_fac(tensor, rr, init) for _ in range(impute_reps)] for rr in range(1,max_rr+1)]
        for j, m in enumerate(methods):
            # initialize objects
            decomp = Decomposition(tensor, method=m, max_rr=max_rr)
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
                decomp.Q2X_entry(drop=chord_drop, repeat=impute_reps, init=inits)
                decomp.Q2X_chord(drop=chord_drop, repeat=impute_reps, init=inits, callback=m_track, callback_r=best_comp[j])
            elif not impChord: decomp.Q2X_entry(drop=entry_drop, repeat=impute_reps, init=inits, callback=m_track, callback_r=best_comp[j])
            elif not impEntry: decomp.Q2X_chord(drop=chord_drop, repeat=impute_reps, init=inits, callback=m_track, callback_r=best_comp[j])

            # save runs
            if i == 0: m_decomp = MultiDecomp(decomp, impEntry, impChord)
            else: m_decomp(decomp)
            m_decomp.save(f"./{dirname}/{m.__name__}-decomp")
            if useCallback: m_track.save(f"./{dirname}/{m.__name__}-track")

        # print runtime every 20% of the way there
        if (printRuntime and (i+1)%round(tensor_samples*0.2) == 0):
            print(f"Average runtime for {i+1} tensors: {(process_time()-tstart)/(i+1)} seconds")

    if printRuntime: print(f"{process_time()-tstart} seconds elapsed for figure {dirname}") # print total runtime

    return m_decomp, m_track

def comp_iter_graph(figname, f_size = (12,9),
                    logComp = True, logTrack = True, logbound=-3.5,
                    save = True, saveFormat = 'png'):
    if save: assert saveFormat == 'png' or saveFormat == 'svg' or saveFormat == 'jpg' or saveFormat == 'jpeg' or saveFormat == 'pdf'
    dirname = f"figures/{figname}"
    ax, f = getSetup(f_size, (3,len(methods)))
    m_decomp = MultiDecomp()
    m_track = tracker()

    # plot components vs imputed/fitted error
    for methodID,m in enumerate(methods):
        m_decomp.load(f"./{dirname}/{m.__name__}-decomp")
        m_track.load(f"./{dirname}/{m.__name__}-track")
        m_track.combine()

        # plot graphs
        q2x_plot(ax[methodID], m.__name__, m_decomp.entry_imputed, m_decomp.entry_fitted, m_decomp.entry_total, log=logComp, logbound=logbound)
        q2x_plot(ax[methodID+3], m.__name__, m_decomp.chord_imputed, m_decomp.chord_fitted, m_decomp.chord_total, log=logComp, logbound=logbound)
        m_track.plot_iteration(ax[methodID+6], methodname=m.__name__, log=logTrack, logbound=logbound)
    
    subplotLabel(ax)
    if save: f.savefig(f"./{dirname}/{figname}.{saveFormat}", bbox_inches="tight", format=saveFormat)
    return f

def comp_init_graph(figname, ax = None, ax_start = None,
                    logbound=-3.5, logComp = True, logTrack = True, type='entry'):
    """ only run entry graph, comparing for each method by initialization """
    assert type == 'entry' or type == 'chord'
    assert ax is not None and ax_start is not None
    dirname = f"figures/{figname}"
    
    m_decomp = MultiDecomp()
    m_track = tracker()

    # plot components vs imputed/fitted error
    for methodID,m in enumerate(methods):
        m_decomp.load(f"./{dirname}/{m.__name__}-decomp")
        m_track.load(f"./{dirname}/{m.__name__}-track")
        m_track.combine()

        if type == 'entry': q2x_plot(ax[methodID+ax_start], m.__name__, m_decomp.entry_imputed, m_decomp.entry_fitted, m_decomp.entry_total, log=logComp, logbound=logbound)
        elif type == 'chord': q2x_plot(ax[methodID+ax_start], m.__name__, m_decomp.chord_imputed, m_decomp.chord_fitted, m_decomp.chord_total, log=logComp, logbound=logbound)
        m_track.plot_iteration(ax[methodID+ax_start+3], methodname=m.__name__, log=logTrack, logbound=logbound)

def comp_dim_graph(figname, ax = None, ax_start = None,
                   logComp = True, logbound=-3.5, type='entry'):
    """ only run entry graph, NO TRACKER, comparing for each case by initialization """
    assert type == 'entry' or type == 'chord'
    assert ax is not None and ax_start is not None
    dirname = f"figures/{figname}"
    
    m_decomp = MultiDecomp()

    # plot components vs imputed/fitted error
    for methodID,m in enumerate(methods):
        m_decomp.load(f"./{dirname}/{m.__name__}-decomp")
        if type == 'entry': q2x_plot(ax[methodID+ax_start], m.__name__, m_decomp.entry_imputed, m_decomp.entry_fitted, m_decomp.entry_total, log=logComp, logbound=logbound)
        elif type == 'chord': q2x_plot(ax[methodID+ax_start], m.__name__, m_decomp.chord_imputed, m_decomp.chord_fitted, m_decomp.chord_total, log=logComp, logbound=logbound)


""" Figure Data/Graphs """



def sim10_data(best_comp = [6,2,6], printRuntime=True):
    sim_data(name="", tSize = (10,10,10), best_comp = best_comp, impute_perc = 0.1, printRuntime=printRuntime, tensor_samples=50, impute_reps=10, seed=5)

def sim10_figure(f_size = (12,9), logComp=True, logTrack=True, logbound=-3.5, save=True, saveFormat='png'):
    if save: assert saveFormat == 'png' or saveFormat == 'svg' or saveFormat == 'jpg' or saveFormat == 'jpeg' or saveFormat == 'pdf'
    figname = "simulated_0.1"
    return comp_iter_graph(figname=figname, f_size=f_size, logComp=logComp, logTrack=logTrack, logbound=logbound, save=save, saveFormat=saveFormat)
    

def sim25_data(best_comp = [6,2,5], printRuntime=True):
    sim_data(name="", tSize = (10,10,10), best_comp = best_comp, impute_perc = 0.25, printRuntime=printRuntime, tensor_samples=50, impute_reps=10, seed=5)

def sim25_figure(f_size = (12,9), logComp=True, logTrack=True, logbound=-3.5, save=True, saveFormat='png'):
    if save: assert saveFormat == 'png' or saveFormat == 'svg' or saveFormat == 'jpg' or saveFormat == 'jpeg' or saveFormat == 'pdf'
    figname = "simulated_0.25"
    return comp_iter_graph(figname=figname, f_size=f_size, logComp=logComp, logTrack=logTrack, logbound=logbound, save=save, saveFormat=saveFormat)


def siminit_data(best_comp = [6,2,6], printRuntime=True):
    sim_data(name='init/svd_sim', tSize = (10,10,10), best_comp=best_comp, printRuntime=printRuntime, tensor_samples=50, impute_reps=10, seed=5, init='svd', impChord=False)
    sim_data(name='init/random_sim', tSize = (10,10,10), best_comp=best_comp, printRuntime=printRuntime, tensor_samples=50, impute_reps=10, seed=5, init='random', impChord=False)

def siminit_figure(f_size = (24,9), logComp=True, logTrack=True, logbound=-4.5, save=True, saveFormat='png'):
    if save: assert saveFormat == 'png' or saveFormat == 'svg' or saveFormat == 'jpg' or saveFormat == 'jpeg' or saveFormat == 'pdf'
    ax, f = getSetup(f_size,(2,len(methods)*2))

    figname = "simulated_init/svd_sim_0.1"
    comp_init_graph(figname=figname, ax=ax, ax_start=0, logComp=logComp, logTrack=logTrack, logbound=logbound)
    figname = "simulated_init/random_sim_0.1"
    comp_init_graph(figname=figname, ax=ax, ax_start=6, logComp=logComp, logTrack=logTrack, logbound=logbound)

    subplotLabel(ax)
    if save: f.savefig(f"./figures/simulated_init/simulated_init.{saveFormat}", bbox_inches="tight", format=saveFormat)
    return f


def simdim_data(best_comp = [6,6,6], printRuntime=True):
    sim_data(name='dims/3D_case', tSize = (20,25,20), best_comp=best_comp, printRuntime=printRuntime, tensor_samples=20, impute_reps=5, seed=5, impChord=False)
    sim_data(name='dims/4D_case', tSize = (10,10,10,10), best_comp=best_comp, printRuntime=printRuntime, tensor_samples=20, impute_reps=5, seed=5, impChord=False)
    sim_data(name='dims/5D_case', tSize = (10,5,5,5,8), best_comp=best_comp, printRuntime=printRuntime, tensor_samples=20, impute_reps=5, seed=5, impChord=False)
    
def simdim_figure(f_size = (12,12), logComp=True, logbound=[-6,-8,-8], save=True, saveFormat='png'):
    if save: assert saveFormat == 'png' or saveFormat == 'svg' or saveFormat == 'jpg' or saveFormat == 'jpeg' or saveFormat == 'pdf'
    ax, f = getSetup(f_size,(3,len(methods)))
    
    figname = "simulated_dims/3D_case_0.1"
    comp_dim_graph(figname=figname, ax=ax, ax_start=0, logComp=logComp, logbound=logbound[0])
    figname = "simulated_dims/4D_case_0.1"
    comp_dim_graph(figname=figname, ax=ax, ax_start=3, logComp=logComp, logbound=logbound[1])
    figname = "simulated_dims/5D_case_0.1"
    comp_dim_graph(figname=figname, ax=ax, ax_start=6, logComp=logComp, logbound=logbound[2])

    subplotLabel(ax)
    if save: f.savefig(f"./figures/simulated_dims/simulated_dims.{saveFormat}", bbox_inches="tight", format=saveFormat)
    return f


def zohar_data(best_comp = [6,4,3], impute_perc = 0.1, impute_reps=50, seed=5):
    """ Generates a figure of method for `tensor_samples` tensors, each run `impute_reps` times. Identical initializations for each method's run per tensor."""
    np.random.seed(seed)
    max_rr = 6

    dirname = f"figures/zohar_{impute_perc}"
    if os.path.isdir(dirname) == False: os.makedirs(dirname)

    # for each tensor
    # generate tensor
    tensor = generateTensor('zohar')
    entry_drop = int(impute_perc*np.sum(np.isfinite(tensor)))
    chord_drop = int(impute_perc*tensor.size/tensor.shape[0])
    inits = [[initialize_fac(tensor,rr) for _ in range(impute_reps)] for rr in range(1,max_rr+1)]
    for j, m in enumerate(methods):
        # initialize objects
        decomp = Decomposition(tensor, method=m, max_rr=max_rr)
        m_track = tracker(tensor,track_runtime=True)
        
        # run imputation, tracking for chords
        decomp.Q2X_entry(drop=entry_drop, repeat=impute_reps, init=inits)
        decomp.Q2X_chord(drop=chord_drop, repeat=impute_reps, init=inits, callback=m_track, callback_r=best_comp[j])

        # save runs
        m_decomp = MultiDecomp(decomp)
        m_decomp.save(f"./{dirname}/{m.__name__}-decomp")
        m_track.save(f"./{dirname}/{m.__name__}-track")
    
    print(f"{process_time()} seconds elapsed for figure {dirname}")

    return m_decomp, m_track

def zohar_figure(f_size = (12,9), impute_perc=0.1, logComp=True, logTrack=True, logbound=-4, save=True, saveFormat='png'):
    if save: assert saveFormat == 'png' or saveFormat == 'svg' or saveFormat == 'jpg' or saveFormat == 'jpeg' or saveFormat == 'pdf'
    figname = f"zohar_{impute_perc}"
    return comp_iter_graph(figname=figname, f_size=f_size, logComp=logComp, logTrack=logTrack, logbound=logbound, save=save, saveFormat=saveFormat)