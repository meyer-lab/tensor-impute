import numpy as np
from ..method_DO import perform_DO
from ..impute_helper import calcR2X
from ..generateTensor import generateTensor
from timpute.tracker import Tracker
from timpute.decomposition import Decomposition
from timpute.common import getSetup
from timpute.plot import q2xentry

def test_decomp_dopt(plot=False, method = perform_DO):
    tensor = generateTensor('known', shape = (10,10,10))
    decomp = Decomposition(tensor, max_rr = 6)
    track = Tracker(tensor)

    # average Q2X for components 1-6 (n=10) for a single test tensor with 10% imputation
    decomp.imputation(drop=0.1, repeat=10, callback=track, type='entry')
    print("average Q2X: " + np.array2string(np.average(decomp.entry_error, axis=0)))
    print("average fitted Q2X: " + np.array2string(np.average(decomp.fitted_entry_error, axis=0)))
    print("average imputed Q2X: " + np.array2string(np.average(decomp.imputed_entry_error, axis=0)))

    if plot:
        ax, f = getSetup((5,3),(1,1))
        q2xentry(ax, decomp, method.__name__, detailed=False)
        return f
    
    return decomp, track


def test_unit_dopt():
    tensor = generateTensor('known', shape = (10,10,10))
    tFac = perform_DO(tensor, rank=6)
    q2x = calcR2X(tFac, tensor)
    print(q2x)