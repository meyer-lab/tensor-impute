import numpy as np
import tensorly as tl
from tensorly.random import random_cp
from ..decomposition import Decomposition
from ..impute_helper import create_missingness
from tensordata.alter import data as alter
from tensordata.zohar import data as zohar

def test_impute_alter():
    np.random.seed(5)
    test = Decomposition(alter()['Fc'].to_numpy())
    test.Q2X_chord(drop=30, repeat=1)
    assert max(test.chordQ2X[0]) >= .8
    test.Q2X_entry(drop=9000,repeat=3)
    assert max(test.entryQ2X[0]) >= .85

def test_impute_random():
    np.random.seed(5)
    shape = (10,10,10)
    test = Decomposition(tl.cp_to_tensor(random_cp(shape, 10)))
    test.Q2X_chord(drop=10, repeat=1)
    assert max(test.chordQ2X[0]) >= .95
    test.Q2X_entry(drop=100, repeat=1)
    assert max(test.entryQ2X[0]) >= .95

def test_impute_noise_missing():
    np.random.seed(5)
    shape = (10,10,10)
    tensor = tl.cp_to_tensor(random_cp(shape, 10))
    create_missingness(tensor,300)
    noise = np.random.normal(0.5,0.15, shape)
    tensor_2 = np.add(tensor,noise)

    test = Decomposition(tensor_2)
    test.Q2X_chord(drop=10, repeat=1)
    assert max(test.chordQ2X[0]) >= .95
    test.Q2X_entry(drop=100, repeat=1)
    assert max(test.entryQ2X[0]) >= .95
