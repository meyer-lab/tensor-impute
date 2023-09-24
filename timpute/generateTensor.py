import numpy as np
import tensorly as tl
import xarray as xr
import os
from tensorly.cp_tensor import CPTensor
from .impute_helper import entry_drop

from tensordata.atyeo import data as atyeo
from tensordata.zohar import data as zohar
from tensordata.alter import data as alter
from .import_hmsData import hms_tensor

def generateTensor(type=None, r=6, shape=(10,10,10), scale=2, distribution='gamma', par=2, missingness=0.0, noise_scale=50):
    """
    Tensor options: 'known', 'unknown', 'zohar', 'atyeo', 'alter', 'hms', 'coh_receptor', or 'coh response'.
    Defaults to 'known'
    """
    if type == 'known':
        temp, _ = createKnownRank(drop_perc=missingness, size=shape, rank=r, distribution=distribution, scale=scale, par=par)
        return createNoise(temp,noise_scale)
    elif type == 'unknown':
        temp = createUnknownRank(drop_perc=missingness, size=shape, distribution=distribution, scale=scale, par=par)
        return createNoise(temp,noise_scale)
    elif type == 'zohar': return zohar().to_numpy().copy()
    elif type == 'atyeo': return atyeo().to_numpy().copy()
    elif type == 'alter': return alter()['Fc'].to_numpy().copy()
    elif type == 'hms':   return hms_tensor().to_numpy().copy()
    elif type == 'coh_receptor':
        receptor = xr.open_dataarray(f"{os.getcwd()}/timpute/data/CoH/CoH_Rec.nc")
        return receptor.to_numpy().copy()
    elif type == 'coh_response':
        response = xr.open_dataarray(f"{os.getcwd()}/timpute/data/CoH/CoH_Tensor_DataSet.nc")
        return response.to_numpy().copy()
    else:
        temp, _ = createKnownRank(drop_perc=missingness, size=shape, rank=r, distribution=distribution, scale=scale, par=par)
        return createNoise(temp,noise_scale)


def createNoise(tensor,scale=1.0):
    """ adds noise in-place """
    noise = np.random.normal(0, scale, tensor.shape)
    noisyTensor = noise+np.copy(tensor)
    return noisyTensor

def createUnknownRank(drop_perc=0.0, size=(10, 20, 25), distribution="gamma", scale=1, par=1):
    if distribution == "gamma": tensor = np.random.gamma(par, scale, size=size)
    if distribution == "chisquare": tensor = np.random.chisquare(size=size)
    if distribution == "logistic": tensor = np.random.logistic(size=size)
    if distribution == "exponential": tensor = np.random.exponential(size=size)
    if distribution == "uniform": tensor = np.random.uniform(size=size)
    if distribution == "normal": tensor = np.random.normal(size=size)

    if scale != 1: tensor *= scale

    entry_drop(tensor, int(drop_perc*tensor.size), dropany=True)
    return tensor

def createKnownRank(drop_perc=0.0, size=(10,10,10), rank=6, distribution="gamma", scale=1, par=1, noise=True):
    r"""
    Creates a random tensor following a set of possible distributions:
    "gamma", "chisquare", "logistic", "exponential", "uniform", "normal"
    """
    rng = np.random.default_rng(10)

    factors = []
    for i in size:
        if distribution == "gamma": factors.append(rng.gamma(par, scale, size=(i,rank)))
        if distribution == "chisquare": factors.append(rng.chisquare(par, size=(i,rank)))
        if distribution == "logistic": factors.append(rng.logistic(size=(i,rank)))
        if distribution == "exponential": factors.append(rng.exponential(size=(i,rank)))
        if distribution == "uniform": factors.append(rng.uniform(size=(i,rank)))
        if distribution == "normal": factors.append(rng.normal(size=(i,rank)))

    if scale != 1:
        for i in factors: i *= scale
    temp = tl.cp_to_tensor(CPTensor((None, factors)))
    if noise: 
        tensor = np.add(np.random.normal(0.5,0.15, size),temp)
    entry_drop(tensor, int(drop_perc*tensor.size), dropany=True)
    return tensor, factors