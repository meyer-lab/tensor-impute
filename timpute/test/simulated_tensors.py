import numpy as np
import tensorly as tl
from tensorly.cp_tensor import CPTensor
from ..impute_helper import create_missingness

def createNoise(tensor,scale=1.0):
    noise = np.random.normal(0, scale, tensor.shape)
    noisyTensor = noise+np.copy(tensor)
    return noisyTensor


def createUnknownRank(drop_perc=0.0, size=(10, 20, 25), distribution="gamma", scale=1, par=1):
    if distribution == "gamma": tensor = np.random.gamma(par, scale, np.prod(size))
    if distribution == "chisquare": tensor = np.random.chisquare(size=size)
    if distribution == "logistic": tensor = np.random.logistic(size=size)
    if distribution == "exponential": tensor = np.random.exponential(size=size)
    if distribution == "uniform": tensor = np.random.uniform(size=size)
    if distribution == "normal": tensor = np.random.normal(size=size)

    if scale != 1: tensor *= scale

    create_missingness(tensor, int(drop_perc*tensor.size))
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
    create_missingness(tensor, int(drop_perc*tensor.size))
    return tensor