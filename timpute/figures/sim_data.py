from multiprocessing import Pool

import numpy as np

from . import DROPS, METHODS
from .figure_data import sim_data

if __name__ == "__main__":
    CORES = 6

    shape = (50, 50, 50)
    decomp_ranks = (1, 30)
    reps = 20
    drops = DROPS
    methods = METHODS
    nonmissing = True
    savedir = "timpute/figures/revision_cache/simulated"

    total_ranks = np.arange(5, 25 + 1)
    true_ranks_list = [
        (int(i.min()), int(i.max())) for i in np.array_split(total_ranks, CORES)
    ]

    funcArgs = [
        (
            true_ranks,
            shape,
            decomp_ranks,
            reps,
            drops,
            methods,
            nonmissing,
            savedir,
        )
        for true_ranks in true_ranks_list
    ]

    # print(funcArgs)

    with Pool(processes=len(true_ranks_list)) as pool:
        results = pool.starmap(sim_data, funcArgs)
