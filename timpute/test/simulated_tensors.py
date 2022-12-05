import numpy as np
import tensorly as tl
from tensorly.cp_tensor import CPTensor
from impute_helper import create_missingness

def createCube(missing=0.0, size=(10, 20, 25)):
    s = np.random.gamma(2, 2, np.prod(size))
    tensor = s.reshape(*size)
    if missing > 0.0:
        tensor[np.random.rand(*size) < missing] = np.nan
    return tensor

def createTensor(drop_perc=0.0, size=(10,10,10), rank=6, distribution="gamma", scale=1, *args):
    rng = np.random.default_rng(10)

    factors = []
    for i in size:
        if distribution == "gamma": factors.append(rng.gamma(args, scale=1, size=(i,rank)))
        if distribution == "logistic": factors.append(np.random.logistic(size=(i,rank)))
        if distribution == "exponential": factors.append(np.random.exponential((i,rank)))
        if distribution == "uniform": factors.append(np.random.gamma[(i,rank)])
        if distribution == "normal": factors.append(np.random.normal(size=(i,rank)))

    if scale != 1:
        for i in factors: i = i * scale

    tensor = tl.cp_to_tensor(CPTensor((None, factors)))
    create_missingness(tensor, int(drop_perc*tensor.size))
    
    return tensor