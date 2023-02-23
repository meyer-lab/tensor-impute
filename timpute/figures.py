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


def compare_entry_imputation(tensor=None, save=False, savename="test", impute_r=6, impute_reps=5, impute_drop=0.25, init='svd', reg=False):
    # run all methods
    if tensor==None: tensor = generateTensor()
    dirname = os.getcwd()+'/methodruns/'
    methods = [perform_DO,perform_ALS,perform_CLS]
    for m in methods:
        track = tracker(tensor, track_runtime=True)
        decomp = Decomposition(tensor, method=m, max_rr=impute_r)
        # update with initialization type, init=init
        # update with regularization when reg=True
        decomp.entryQ2X(drop=impute_drop, repeat=impute_reps, callback=track)
        track.combine()
        track.save(dirname + savename + m.__name__ + '-iters')
        decomp.save(dirname + savename + m.__name__ + '-imputations')    

    # graphing logic
    figures = []
    for m in methods:
        counter = 0
        track.load(dirname + savename + m.__name__ + '-iters')
        decomp.load(dirname + savename + m.__name__ + '-imputations')
        ax,f = getSetup((10,10),(2,4))
        q2xentry(ax[0], decomp, methodname=m.__name__)
        track.plot_iteration(ax[4+counter], methodname=m.__name__)
        # add comp vs fitted and imputed error

        counter+=1


    if save==False: pass
