import numpy as np
import tensorly as tl
from tensorly.random import random_cp
from ..decomposition import Decomposition
from ..generateTensor import generateTensor
from ..impute_helper import entry_drop

def test_impute_alter():
    np.random.seed(5)
    tensor = generateTensor(type='alter')
    test = Decomposition(tensor)
    test.imputation(type='chord', drop=0.05, repeat=1)
    assert min(test.chord_total[0]) <= 1-.8
    test.imputation(type='entry', drop=0.05,repeat=3)
    assert min(test.entry_total[0]) <= 1-.85

def test_impute_zohar():
    np.random.seed(5)
    tensor = generateTensor(type='zohar')
    test = Decomposition(tensor)
    test.imputation(type='chord', drop=0.05, repeat=1)
    assert min(test.chord_total[0]) <= 1-.4
    test.imputation(type='entry', drop=0.05, repeat=1)
    assert min(test.entry_total[0]) <= 1-.5

def test_impute_hms():
    np.random.seed(5)
    test = Decomposition(generateTensor('hms'))
    test.imputation(type='chord', drop=0.05, repeat=1)
    assert min(test.chord_total[0]) <= 1-.95
    test.imputation(type='entry', drop=0.05, repeat=1)
    assert min(test.entry_total[0]) <= 1-.95

def test_impute_random():
    np.random.seed(5)
    shape = (10,10,10)
    test = Decomposition(tl.cp_to_tensor(random_cp(shape, 10)))
    test.imputation(type='chord', drop=0.1, repeat=1)
    assert min(test.chord_total[0]) <= 1-.95
    test.imputation(type='entry', drop=0.1, repeat=1)
    assert min(test.entry_total[0]) <= 1-.95

def test_impute_noise_missing():
    np.random.seed(5)
    shape = (10,10,10)
    tensor = tl.cp_to_tensor(random_cp(shape, 10))
    entry_drop(tensor,300)
    noise = np.random.normal(0.5,0.15, shape)
    tensor_2 = np.add(tensor,noise)

    test = Decomposition(tensor_2)
    test.imputation(type='chord', drop=0.1, repeat=1)
    assert min(test.chord_total[0]) <= 1-.95
    test.imputation(type='entry', drop=0.1, repeat=1)
    assert min(test.entry_total[0]) <= 1-.95
