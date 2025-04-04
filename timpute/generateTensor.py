import os

import numpy as np
import tensorly as tl
import xarray as xr
from tensordata.alter import data as alter
from tensordata.atyeo import data as atyeo
from tensordata.zohar import data as zohar
from tensorly.cp_tensor import CPTensor, cp_to_tensor
from tensorly.random import random_cp

from .data.import_hmsData import hms_tensor
from .impute_helper import entry_drop


def generateTensor(
    tensor_type="known",
    r=6,
    shape=(10, 10, 10),
    scale=2,
    distribution="gamma",
    par=2,
    missingness=0.0,
    noise_scale=50,
):
    """
    Tensor options: 'known', 'unknown', 'zohar', 'atyeo',
                    'alter', 'hms', 'coh_receptor', or 'coh response'.
    """
    if tensor_type == "known":
        temp, factors = createKnownRank(
            drop_perc=missingness,
            size=shape,
            rank=r,
            distribution=distribution,
            scale=scale,
            par=par,
        )
        return createNoise(temp, noise_scale), factors

    elif tensor_type == "tensorly":
        factors = random_cp(
            shape=shape,
            rank=r,
            full=False,
        )
        temp = cp_to_tensor(factors)
        return createNoise(temp, noise_scale), factors

    elif tensor_type == "zohar":
        return zohar().to_numpy().copy()
    elif tensor_type == "atyeo":
        return atyeo().to_numpy().copy()
    elif tensor_type == "alter":
        return alter()["Fc"].to_numpy().copy()
    elif tensor_type == "hms":
        return np.swapaxes(hms_tensor().to_numpy().copy(), 0, 2)
    elif tensor_type == "coh_receptor":
        receptor = xr.open_dataarray(f"{os.getcwd()}/timpute/data/CoH/CoH_Rec.nc")
        return receptor.to_numpy().copy()
    elif tensor_type == "coh_response":
        response = xr.open_dataarray(
            f"{os.getcwd()}/timpute/data/CoH/CoH_Tensor_DataSet.nc"
        )
        return response.to_numpy().copy()

    else:
        ValueError(
            f"Tensor type {tensor_type} not recognized. Please use one of\
            the following: 'known', 'unknown', 'zohar', 'atyeo', 'alter',\
            'hms','coh_receptor', or 'coh response'."
        )


def createNoise(tensor, scale=1.0):
    """adds noise in-place"""
    noise = np.random.normal(0, scale, tensor.shape)
    noisyTensor = noise + np.copy(tensor)
    return noisyTensor


def createKnownRank(
    drop_perc=0.0,
    size=(10, 10, 10),
    rank=6,
    distribution="gamma",
    scale=1,
    par=1,
    noise=True,
):
    r"""
    Creates a random tensor following a set of possible distributions:
    "gamma", "chisquare", "logistic", "exponential", "uniform", "normal"
    may also a tuple of distributions w/ same order as `size`
    """
    rng = np.random.default_rng(10)

    factors = []

    if isinstance(distribution, str):
        for i in size:
            if distribution == "gamma":
                factors.append(rng.gamma(par, scale, size=(i, rank)))
            if distribution == "chisquare":
                factors.append(rng.chisquare(par, size=(i, rank)))
            if distribution == "logistic":
                factors.append(rng.logistic(size=(i, rank)))
            if distribution == "exponential":
                factors.append(rng.exponential(size=(i, rank)))
            if distribution == "uniform":
                factors.append(rng.uniform(size=(i, rank)))
            if distribution == "normal":
                factors.append(rng.normal(size=(i, rank)))

    else:
        assert len(distribution) == len(size)
        for i in size:
            if distribution[i] == "gamma":
                factors[1] = rng.gamma(par, scale, size=(i, rank))
            if distribution[i] == "chisquare":
                factors[1] = rng.chisquare(par, size=(i, rank))
            if distribution[i] == "logistic":
                factors[1] = rng.logistic(size=(i, rank))
            if distribution[i] == "exponential":
                factors[1] = rng.exponential(size=(i, rank))
            if distribution[i] == "uniform":
                factors[1] = rng.uniform(size=(i, rank))
            if distribution[i] == "normal":
                factors[1] = rng.normal(size=(i, rank))

    if scale != 1:
        for i in factors:
            i *= scale
    temp = tl.cp_to_tensor(CPTensor((None, factors)))
    if noise:
        tensor = np.add(np.random.normal(0.5, 0.15, size), temp)
    entry_drop(tensor, int(drop_perc * tensor.size), dropany=True)
    return tensor, factors
