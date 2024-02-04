import numpy as np
from .runImputation import *
from ..plot import *
from ..common import *

# poetry run python -m timpute.figures.figure2

def figure2(datalist=SAVENAMES):
    ax, f = getSetup((12,12), (2,2))
    
    for i, data in enumerate(datalist):
        folder = f"timpute/figures/cache/{data}/nonmissing/"
        impType = 'entry'
        for mID, m in enumerate(METHODS):
            run, _ = loadImputation(impType, m, folder)

            comps = np.arange(1,run.entry_imputed.shape[1]+1)
            label = f"{METHODNAMES[mID]}"
            total_errbar = np.vstack((abs(np.percentile(run.entry_total,25,0) - np.nanmedian(run.entry_total,0)),
                                    abs(np.percentile(run.entry_total,75,0) - np.nanmedian(run.entry_total,0))))
            e = ax[i].errorbar(comps, np.median(run.entry_total,0), yerr=total_errbar,label=label, ls='solid', color=rgbs(mID, 0.7), alpha=0.5)
            
            h,l = ax[i].get_legend_handles_labels()
            h = [a[0] for a in h]
            ax[i].legend(h, l, loc='best', handlelength=2)
            ax[i].set_xlabel("Number of Components")
            ax[i].set_ylabel("Error")
            ax[i].set_xticks([x for x in comps])
            ax[i].set_xticklabels([x for x in comps])
            ax[i].set_title(f"{DATANAMES[i]}")
            # ax[i].set_yscale("log")

    subplotLabel(ax)
    f.savefig('timpute/figures/img/svg/figure2.svg', bbox_inches="tight", format='svg')
    f.savefig('timpute/figures/img/figure2.png', bbox_inches="tight", format='png')

if __name__ == "__main__":
    figure2()