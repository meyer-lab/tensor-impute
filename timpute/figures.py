import numpy as np
import tensorly as tl
from timpute.decomposition import Decomposition, MultiDecomp
from timpute.tracker import tracker
from timpute.common import *
from timpute.plot import *
import time
import os

from timpute.test.simulated_tensors import createKnownRank, createUnknownRank, createNoise
from tensordata.atyeo import data as atyeo
from tensordata.zohar import data as zohar
from tensordata.alter import data as alter
from timpute.initialize_fac import initialize_fac
from timpute.direct_opt import perform_DO
from timpute.tensorly_als import perform_ALS
from timpute.cmtf import perform_CLS


def generateTensor(type=None, r=6, shape=(20,25,30), scale=2, distribution='gamma', par=2, missingness=0.1, noise_scale=50):
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
    start = time.time()

    for m in methods:
        mstart = time.time()
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
        if printRuntime: print(m.__name__ + ": " + str(time.time()-mstart))

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
    
    if printRuntime: print("Total runtime: " + str(time.time()-start)+'\n')
    if save is not None: f.savefig('./'+dirname+'/' + "imputation_results", bbox_inches="tight")
    return f 



def l2_comparison(tensor=None, init='svd', alpha=[1e-4,1e-3,1e-2,1e-1,1e0,1e1,1e2],
                       impute_type='entry', impute_r=[3,4,5,6], impute_reps=5, impute_perc=0.25, impute_mode=0,
                       f_size=(12,6), save=None, printRuntime=True):
    if tensor is None: tensor = generateTensor()

    dirname = 'methodruns/'+save
    if os.path.isdir(dirname) == False: os.makedirs(dirname)

    ax, f = getSetup(f_size, (2,len(alpha)))
    start = time.time()

    for rr in impute_r:
        rstart = time.time()
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
        
        
         
        if printRuntime: print(rr + " components: " + str(time.time()-rstart))

    
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

def q2x_plot(ax, imputed_arr:np.ndarray, fitted_arr:np.ndarray, methodname:str):
    comps = np.arange(1,imputed_arr.shape[1])
    imputed_df = pd.DataFrame(imputed_arr).T
    fitted_df = pd.DataFrame(fitted_arr).T

    imputed_df.index = comps
    imputed_df['mean'] = imputed_df.mean(axis=1)
    imputed_df['sem'] = imputed_df.sem(axis=1)
    imputed_means = imputed_df['mean']
    imputed_sem = imputed_df['sem']
    ax.scatter(comps + 0.05, imputed_means, color='C0', s=10, label=methodname+' Imputed Error')
    ax.errorbar(comps + 0.05, imputed_means, yerr=imputed_sem, fmt='none', ecolor='C0')
    
    fitted_df.index = comps
    fitted_df['mean'] = fitted_df.mean(axis=1)
    fitted_df['sem'] = fitted_df.sem(axis=1)
    fitted_means = fitted_df['mean']
    fitted_sem = fitted_df['sem']
    ax.scatter(comps, fitted_means, color='C1', s=10, label=methodname+' Fitted Error')
    ax.errorbar(comps, fitted_means, yerr=fitted_sem, fmt='none', ecolor='C1')
    ax.set_ylabel("Imputation Error")

    ax.set_xlabel("Number of Components")
    ax.set_xticks([x for x in comps])
    ax.set_xticklabels([x for x in comps])
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")



def figure1(tensor_samples=50, impute_reps=5, impute_perc = 0.1, printRuntime=True):
    """ Generates a figure of method for `tensor_samples` tensors, each run `impute_reps` times. Identical initializations for each method's run per tensor."""
    f_size = (12,6)
    methods = [perform_ALS,perform_CLS]
    dirname = 'figures/figure_1'

    if os.path.isdir(dirname) == False: os.makedirs(dirname)
    ax, f = getSetup(f_size, (2,len(methods)))

    # for each tensor
    for i in range(tensor_samples):
        # generate tensor
        tensor = generateTensor('known',r=6,shape=(10,10,10))
        drop = int(impute_perc*np.sum(np.isfinite(tensor)))                                     # TODO: adjust for entry/chord
        inits = [initialize_fac(tensor,6) for _ in range(impute_reps)]

        tstart = time.time()
        for m in methods:
            # initialize objects
            decomp = Decomposition(tensor, method=m, max_rr=6)
            if i==0: m_track = tracker(tensor,track_runtime=True)
            else:
                m_track.load('./'+dirname+'/' + m.__name__ + '-track')
                m_track.new()
            
            # run imputation
            tstart = time.time()
            decomp.Q2X_entry(drop=drop, repeat=impute_reps, callback=m_track, init=inits)       # TODO: adjust for entry/chord

            # save runs
            if i==0: m_decomp = MultiDecomp(decomp,'entry')                                     # TODO: adjust for entry/chord
            else: m_decomp(decomp)


            m_decomp.save('./'+dirname+'/' + m.__name__ + '-decomp')
            m_track.save('./'+dirname+'/' + m.__name__ + '-track')

        if printRuntime and (i+1)%10==0: print(f"Tensor {str(i+1)}, {m.__name__}: {str(time.time()-tstart)}")


    # plot components vs imputed/fitted error
    for methodID,m in enumerate(methods):
        m_decomp.load('./'+dirname+'/' + m.__name__ + '-decomp')
        m_track.load('./'+dirname+'/' + m.__name__ + '-track')
        m_track.combine()

        # plot graphs
        plotID = methodID
        q2x_plot(ax[plotID],decomp.imputed_entry_error,decomp.fitted_entry_error,m.__name)      # TODO: adjust for entry/chord
        plotID = methodID + 3
        m_track.plot_iteration(ax[plotID], methodname=m.__name__)
    
    subplotLabel(ax)
    return f
    