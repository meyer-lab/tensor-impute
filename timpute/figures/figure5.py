import numpy as np
from .runImputation import *
from ..plot import *
from ..common import *
from .realdatafigs import bestComps

# poetry run python -m timpute.figures.figure4
drops = (0.05,0.1,0.2,0.3,0.4,0.5)

def figure5():
    ax, f = getSetup((15,10), (2,3))
                    # (w,h),  (r,c)

    for d,data in enumerate(["zohar", "alter", "hms", "coh_response"]):
        folder = f"timpute/figures/cache/{data}/nonmissing/"

        for mID, m in enumerate(methods):
            _, tracker = loadImputation("entry", m, folder)

            label = f"{m.__name__} Total Error"
            totErr = tracker.total_array[list(tracker.total_array.keys())[-1]][:,:-1]
            total_errbar = np.vstack((-(np.percentile(totErr,25,0) - np.nanmedian(totErr,0)),
                                        np.percentile(totErr,75,0) - np.nanmedian(totErr,0)))
            ax[d].errorbar(np.arange(totErr.shape[1]), np.nanmedian(totErr,0), label=label, color=rgbs(mID,0.7),
                                yerr=total_errbar, ls = 'solid', errorevery=5)

            ax[d].legend()
            ax[d].set_xlim((0, totErr.shape[1]-1))
            ax[d].set_xlabel('Iteration')
            ax[d].set_ylabel('Error')
            ax[d].set_title(f"{data} dataset, median error by iteration")
            ax[d].set_yscale('log')

    # ////////////////////////////////////////////////////////
    
    plotIter(ax[4], "zohar", 'entry', 0.1)
    plotIter(ax[5], "alter", 'entry', 0.2)
    plotIter(ax[6], "coh_response", 'chord', 0.3)
    plotIter(ax[7], "hms", 'chord', 0.4)

    # ////////////////////////////////////////////////////////

    subplotLabel(ax)
    f.savefig('timpute/figures/img/figure4.png', bbox_inches="tight", format='png')


def plotIter(ax, data, impType, drop):
    folder = f"timpute/figures/cache/{data}/drop_{drop}/"
    comps = bestComps(drop=drop, datalist=[data])

    for mID, m in enumerate(methods):
        _, tracker = loadImputation(impType, m, folder)

        label = f"{m.__name__} Imputed Error"
        impErr = tracker.imputed_array[str(comps[data][mID])][:,:-1]
        imputed_errbar = np.vstack((-(np.percentile(impErr,25,0) - np.nanmedian(impErr,0)),
                                    np.percentile(impErr,75,0) - np.nanmedian(impErr,0),))
        e = ax.errorbar(np.arange(impErr.shape[1]) + (0.1-mID*0.1), np.nanmedian(impErr,0), label=label, color=rgbs(mID,0.7),
                            yerr = imputed_errbar, ls='dashed', errorevery=5)
        e[-1][0].set_linestyle('dashed')

        label = f"{m.__name__} Total Error"
        totErr = tracker.total_array[str(comps[data][mID])][:,:-1]
        total_errbar = np.vstack((-(np.percentile(totErr,25,0) - np.nanmedian(totErr,0)),
                                    np.percentile(totErr,75,0) - np.nanmedian(totErr,0)))
        ax.errorbar(np.arange(totErr.shape[1]), np.nanmedian(totErr,0), label=label, color=rgbs(mID,0.7),
                            yerr=total_errbar, ls = 'solid', errorevery=5)

        ax.legend()
        ax.set_xlim((0, totErr.shape[1]-1))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Error')
        ax.set_title(f"{data} dataset, median error by iteration")
        ax.set_yscale('log')


def figure5_exp(datalist=["zohar", "alter", "hms", "coh_response"]):
    ax, f = getSetup((48,40), (6,8))

    for d, drop in enumerate(drops):
        for t, impType in enumerate(['entry', 'chord']): 
            comps = bestComps(drop=drop, impType=impType, datalist=datalist)
            for dat, data in enumerate(datalist):
                folder = f"timpute/figures/cache/{data}/drop_{drop}/"
                for mID, m in enumerate(methods):
                    _, tracker = loadImputation(impType, m, folder)

                    label = f"{m.__name__} Imputed Error"
                    impErr = tracker.imputed_array[str(comps[data][mID]+1)][:,:-1]
                    imputed_errbar = np.vstack((-(np.percentile(impErr,25,0) - np.nanmedian(impErr,0)),
                                                np.percentile(impErr,75,0) - np.nanmedian(impErr,0),))
                    e = ax[dat+d*8+t*4].errorbar(np.arange(impErr.shape[1]) + (0.1-mID*0.1), np.nanmedian(impErr,0), label=label, color=rgbs(mID,0.7),
                                        yerr = imputed_errbar, ls='dashed', errorevery=5)
                    e[-1][0].set_linestyle('dashed')

                    label = f"{m.__name__} Total Error"
                    totErr = tracker.total_array[str(comps[data][mID]+1)][:,:-1]
                    total_errbar = np.vstack((-(np.percentile(totErr,25,0) - np.nanmedian(totErr,0)),
                                                np.percentile(totErr,75,0) - np.nanmedian(totErr,0)))
                    ax[dat+d*8+t*4].errorbar(np.arange(totErr.shape[1]), np.nanmedian(totErr,0), label=label, color=rgbs(mID,0.7),
                                        yerr=total_errbar, ls = 'solid', errorevery=5)

                    ax[dat+d*8+t*4].legend()
                    ax[dat+d*8+t*4].set_xlim((0, totErr.shape[1]))
                    ax[dat+d*8+t*4].set_xlabel('Iteration')
                    ax[dat+d*8+t*4].set_ylabel('Error')
                    ax[dat+d*8+t*4].set_title(f"{data} dataset, median error by iteration")

    # subplotLabel(ax)
    f.savefig('timpute/figures/img/figure4-exp.png', bbox_inches="tight", format='png')

figure5()
# figure5_exp()