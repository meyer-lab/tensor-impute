import numpy as np
from .runImputation import *
from ..plot import *
from ..common import *

# poetry run python -m timpute.figures.figure1

methods = (perform_DO, perform_ALS, perform_CLS)
methodname = ["DO","ALS","CLS"]
datanames = ['Zohar', 'Alter', 'Mills', 'CoH']
linestyles = ('dashdot', (0,(1,1)), 'solid', (3,(3,1,1,1,1,1)), 'dotted', (0,(5,1)))
drops = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)


def figure2(datalist=["zohar", "alter", "hms", "coh_response"]):
    ax, f = getSetup((12,12), (2,2))
    
    for i, data in enumerate(datalist):
        folder = f"timpute/figures/cache/{data}/nonmissing/"
        impType = 'entry'
        for mID, m in enumerate(methods):
            run, _ = loadImputation(impType, m, folder)

            comps = np.arange(1,run.entry_imputed.shape[1]+1)
            label = f"{methodname[mID]}"
            total_errbar = np.vstack((abs(np.percentile(run.entry_total,25,0) - np.nanmedian(run.entry_total,0)),
                                    abs(np.percentile(run.entry_total,75,0) - np.nanmedian(run.entry_total,0))))
            e = ax[i].errorbar(comps, np.median(run.entry_total,0), yerr=total_errbar,label=label, ls='solid', color=rgbs(mID, 0.7), alpha=0.5)

            ax[i].legend()
            ax[i].set_xlabel("Number of Components")
            ax[i].set_ylabel("Error")
            ax[i].set_xticks([x for x in comps])
            ax[i].set_xticklabels([x for x in comps])
            ax[i].set_title(f"{data} dataset, decomposition vs component")
            # ax[i].set_yscale("log")

    subplotLabel(ax)
    f.savefig('timpute/figures/img/figure2.png', bbox_inches="tight", format='png')

figure2()