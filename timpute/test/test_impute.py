import numpy as np
import tensorly as tl
from tensorly.random import random_cp
from ..decomposition import Decomposition
from ..impute_helper import *
from tensordata.alter import data as alter
from tensordata.zohar import data as zohar

def test_impute_alter():
    np.random.seed(5)
    test = Decomposition(alter()['Fc'].to_numpy())
    test.Q2X_chord(drop=30, repeat=1)
    assert max(test.chord_error[0]) >= 1-.8
    test.Q2X_entry(drop=9000,repeat=3)
    assert max(test.entry_error[0]) >= 1-.85

def test_impute_zohar():
    np.random.seed(5)
    test = Decomposition(zohar().to_numpy())
    test.Q2X_chord(drop=5, repeat=1)
    assert max(test.chord_error[0]) >= 1-.4
    test.Q2X_entry(drop=3000, repeat=1)
    assert max(test.entry_error[0]) >= 1-.5

def test_impute_random():
    np.random.seed(5)
    shape = (10,10,10)
    test = Decomposition(tl.cp_to_tensor(random_cp(shape, 10)))
    test.Q2X_chord(drop=10, repeat=1)
    assert max(test.chord_error[0]) >= 1-.95
    test.Q2X_entry(drop=100, repeat=1)
    assert max(test.entry_error[0]) >= 1-.95

def test_impute_noise_missing():
    np.random.seed(5)
    shape = (10,10,10)
    tensor = tl.cp_to_tensor(random_cp(shape, 10))
    entry_drop(tensor,300)
    noise = np.random.normal(0.5,0.15, shape)
    tensor_2 = np.add(tensor,noise)

    test = Decomposition(tensor_2)
    test.Q2X_chord(drop=10, repeat=1)
    assert max(test.chord_error[0]) >= 1-.95
    test.Q2X_entry(drop=100, repeat=1)
    assert max(test.entry_error[0]) >= 1-.95
