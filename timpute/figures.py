import numpy as np
import tensorly as tl
from timpute.decomposition import Decomposition
from timpute.tracker import tracker
from timpute.common import *
from timpute.plot import *
import time
import os

from timpute.test.simulated_tensors import createKnownRank, createUnknownRank, createNoise
from tensordata.atyeo import data as atyeo
from tensordata.zohar import data as zohar
from tensordata.alter import data as alter
from timpute.direct_opt import perform_DO
from timpute.cmtf import perform_CLS
from timpute.tensorly_als import perform_ALS

methods = [perform_DO,perform_ALS,perform_CLS]

def generateTensor(type=None, r=6, shape=(10,20,25), scale=2, distribution='gamma', par=2, missingness=0.2, noise_scale=50):
    if type == "zohar": return zohar().to_numpy()
    elif type == "atyeo": return atyeo().to_numpy()
    elif type == "alter": return alter()['Fc'].to_numpy()
    elif type == "unknown":
            temp = createUnknownRank(drop_perc=missingness, size=shape, distribution=distribution, scale=scale, par=par)
            return createNoise(temp,noise_scale)
    elif type == "known":
            temp = createKnownRank(drop_perc=missingness, size=shape, rank=r, distribution=distribution, scale=scale, par=par)
            return createNoise(temp,noise_scale)
    else:
            temp = createKnownRank(drop_perc=missingness, size=shape, rank=r, distribution=distribution, scale=scale, par=par)
            return createNoise(temp,noise_scale)


def compare_imputation(tensor=None, init='svd', reg=NotImplemented,
                       impute_type='entry', impute_r=6, impute_reps=5, impute_perc=0.25, impute_mode=0,
                       save=None, graphComp=True, graphIter=True):
    # run all methods
    if tensor==None: tensor = generateTensor()
    dirname = os.getcwd()+'/methodruns/'
    methods = [perform_DO,perform_ALS,perform_CLS]
    ax, f = getSetup((16,9), (2,3))
    methodID = 0

    for m in methods:
        # instantiate objects
        track = tracker(tensor,track_runtime=True)
        decomp = Decomposition(tensor, method=m, max_rr=impute_r)

        # run imputation
        # TODO: update with regularization when reg=True
        if impute_type=='entry':
            drop = int(impute_perc*np.sum(np.isfinite(tensor)))
            decomp.Q2X_entry(drop=drop, repeat=impute_reps, callback=track)
        elif impute_type=='chord':
            drop = int(impute_perc*tensor.shape[impute_mode])
            decomp.Q2X_chord(drop=drop, repeat=impute_reps, callback=track)
        track.combine()

        # plot components vs imputed/fitted error
        plotID = methodID
        comps = decomp.rrs
        if impute_type == 'entry':
            imputed_df = pd.DataFrame(decomp.imputed_entry_error).T
            fitted_df = pd.DataFrame(decomp.fitted_entry_error).T

            imputed_df.index = comps
            imputed_df['mean'] = imputed_df.mean(axis=1)
            imputed_df['sem'] = imputed_df.sem(axis=1)
            TR2X = imputed_df['mean']
            TErr = imputed_df['sem']
            ax[plotID].plot(comps + 0.05, TR2X, ".", label=m.__name__+' Imputed Error')
            ax[plotID].errorbar(comps + 0.05, TR2X, yerr=TErr, fmt='none', ecolor='b')
            
            fitted_df.index = comps
            fitted_df['mean'] = fitted_df.mean(axis=1)
            fitted_df['sem'] = fitted_df.sem(axis=1)
            TR2X = fitted_df['mean']
            TErr = fitted_df['sem']
            ax[plotID].plot(comps, TR2X, ".", label=m.__name__+' Imputed Error')
            ax[plotID].errorbar(comps, TR2X, yerr=TErr, fmt='none', ecolor='b')

            ax[plotID].set_ylabel("Entry Imputation Error")
            ax[plotID].set_xlabel("Number of Components")
            ax[plotID].set_xticks([x for x in comps])
            ax[plotID].set_xticklabels([x for x in comps])
            ax[plotID].set_ylim(0, 1)
            ax[plotID].legend(loc=4)
        
        if impute_type == 'chord':
            imputed_df = pd.DataFrame(decomp.imputed_chord_error).T
            fitted_df = pd.DataFrame(decomp.fitted_chord_error).T

            imputed_df.index = comps
            imputed_df['mean'] = imputed_df.mean(axis=1)
            imputed_df['sem'] = imputed_df.sem(axis=1)
            Q2Xchord = imputed_df['mean']
            Q2Xerrors = imputed_df['sem']
            ax.scatter(comps, Q2Xchord, s=10)
            ax.errorbar(comps, Q2Xchord, yerr=Q2Xerrors, fmt='none')

            comps = decomp.rrs
            fitted_df.index = comps
            fitted_df['mean'] = fitted_df.mean(axis=1)
            fitted_df['sem'] = fitted_df.sem(axis=1)
            Q2Xchord = fitted_df['mean']
            Q2Xerrors = fitted_df['sem']
            ax.scatter(comps, Q2Xchord, s=10)
            ax.errorbar(comps, Q2Xchord, yerr=Q2Xerrors, fmt='none')

            ax.set_ylabel("Chord Imputation Error")
            ax.set_xlabel("Number of Components")
            ax.set_xticks([x for x in comps])
            ax.set_xticklabels([x for x in comps])
            ax.set_ylim(bottom=0.0, top=1.0)         
        
        # plot iteration vs imputed/fitted error
        if graphIter:
            plotID = methodID + 3
            track.plot_iteration(ax[plotID], methodname=m.__name__)

        methodID = methodID + 1

        # save for inspection
        if save is not None:
            decomp.save(dirname + save + "-" + m.__name__ + '-imputations')
            track.save(dirname + save + "-" + m.__name__ + '-iters')
            f.savefig(dirname + save + "-" + "imputation_results.pdf", format="pdf", bbox_inches="tight")

    return f 
        
    
