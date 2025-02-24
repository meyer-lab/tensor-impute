from .figure_data import real_data
from . import METHODS, METHODNAMES, SAVENAMES, DROPS
from multiprocessing import Pool

if __name__ == "__main__":

    datalist = SAVENAMES
    reps = 20
    drops = DROPS
    methods = METHODS
    nonmissing = True
    savedir = "timpute/figures/revision_cache"

    start_comps = [1]
    max_comps = [50]
    continued = False

    funcArgs = [
        (
            [dataset],
            start_comps,
            max_comps,
            reps,
            drops,
            methods,
            nonmissing,
            savedir,
            continued,
        )
        for dataset in SAVENAMES
    ]
    with Pool(processes=4) as pool:
        results = pool.starmap(real_data, funcArgs)
