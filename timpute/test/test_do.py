import numpy as np

from timpute.decomposition import Decomposition
from timpute.tracker import Tracker

from ..generateTensor import generateTensor
from ..impute_helper import calcR2X
from ..method_DO import perform_DO


def test_decomp_dopt(plot=False, method=perform_DO):
    tensor = generateTensor("known", shape=(10, 10, 10))
    decomp = Decomposition(tensor, max_rr=6)
    track = Tracker(tensor, track_runtime=False)

    # average Q2X for components 1-6 (n=10) for a single test tensor with 10% imputation
    decomp.imputation(drop=0.1, repeat=10, callback=track, type="entry")
    print("average Q2X: " + np.array2string(np.average(decomp.entry_total, axis=0)))
    print(
        "average fitted Q2X: "
        + np.array2string(np.average(decomp.entry_fitted, axis=0))
    )
    print(
        "average imputed Q2X: "
        + np.array2string(np.average(decomp.entry_imputed, axis=0))
    )


def test_unit_dopt():
    tensor = generateTensor("known", shape=(10, 10, 10))
    tFac = perform_DO(tensor, rank=6)
    q2x = calcR2X(tFac, tensor)
    print(q2x)
