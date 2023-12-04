import numpy as np
from .runImputation import *
from ..plot import *
from ..common import *

# poetry run python -m timpute.figures.figure3
drops = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4)

def figure4(datalist=["zohar", "alter", "hms", "coh_response"]):
    ax, f = getSetup((20,10), (2,4))
    dirname = f"timpute/figures/img"
    stdout = open(f"{dirname}/figure3.txt", 'w')

    stdout.write(f"{drops}")

    # Figure 1, a)-d)
    for i, data in enumerate(datalist):
        impType = 'entry'
        for mID, m in enumerate(methods):
            ImpErr = list()
            ImpErrIQR = list()
            TotErr = list()
            TotErrIQR = list()
            stdout.write(f"\n{data}, {impType} {m.__name__}: ")
            for d in drops:
                folder = f"timpute/figures/cache/{data}/drop_{d}/"
                run, _ = loadImputation(impType, m, folder)
                comp = np.median(run.entry_imputed,0).argmin() # best imp error
                ImpErr.append(np.median(run.entry_imputed[comp]))
                ImpErrIQR.append(np.vstack((-(np.percentile(run.entry_imputed[comp],25,0) - np.median(run.entry_imputed[comp],0)),
                                          np.percentile(run.entry_imputed[comp],75,0) - np.median(run.entry_imputed[comp],0))))
                TotErr.append(np.median(run.entry_total[comp]))
                TotErrIQR.append(np.vstack((-(np.percentile(run.entry_total[comp],25,0) - np.median(run.entry_total[comp],0)),
                                          np.percentile(run.entry_total[comp],75,0) - np.median(run.entry_total[comp],0))))
                stdout.write(f"{comp}, ")

            label = f"{methodname[mID]} (best imputed)"
            ax[i].errorbar([str(x*100) for x in drops], np.array(ImpErr), fmt='d-', label=label, color=rgbs(mID, 0.7), yerr=np.hstack(tuple(ImpErrIQR)))
            label = "        (matching total)"
            ax[i].errorbar([str(x*100) for x in drops], np.array(TotErr), fmt='.-', label=label, color=rgbs(mID, 0.7), yerr=np.hstack(tuple(TotErrIQR)))

            ax[i].legend()
            ax[i].set_xlabel("Drop Percent")
            ax[i].set_ylabel("Median Error")
            ax[i].set_title(f"{datanames[i]} dataset\nlowest imputed median error by {impType} missingness")

        ax[1].set_ylim(top=0.15)
        ax[3].set_ylim(top=0.35)

        impType = 'chord'
        for mID, m in enumerate(methods):
            ImpErr = list()
            ImpErrIQR = list()
            TotErr = list()
            TotErrIQR = list()
            stdout.write(f"\n{data}, {impType} {m.__name__}: ")
            for d in drops:
                folder = f"timpute/figures/cache/{data}/drop_{d}/"
                run, _ = loadImputation(impType, m, folder)
                comp = np.median(run.chord_imputed,0).argmin() # best imp error
                ImpErr.append(np.median(run.chord_imputed[comp]))
                ImpErrIQR = np.vstack((-(np.percentile(run.chord_imputed[comp],25,0) - np.median(run.chord_imputed[comp],0)),
                                          np.percentile(run.chord_imputed[comp],75,0) - np.median(run.chord_imputed[comp],0)))
                TotErr.append(np.median(run.chord_total[comp]))
                TotErrIQR = np.vstack((-(np.percentile(run.chord_total[comp],25,0) - np.median(run.chord_total[comp],0)),
                                          np.percentile(run.chord_total[comp],75,0) - np.median(run.chord_total[comp],0)))
                stdout.write(f"{comp}, ")

            label = f"{methodname[mID]} (best imputed)"
            ax[i].errorbar([str(x*100) for x in drops], np.array(ImpErr), fmt='d-', label=label, color=rgbs(mID, 0.7), yerr=np.hstack(tuple(ImpErrIQR)))
            label = "        (matching total)"
            ax[i].errorbar([str(x*100) for x in drops], np.array(TotErr), fmt='.-', label=label, color=rgbs(mID, 0.7), yerr=np.hstack(tuple(TotErrIQR)))
            
            ax[i+4].legend()
            if (ax[i+4].get_ylim()[1] > 1):
                ax[i+4].set_ylim(0, top=1)
            ax[i+4].set_xlabel("Drop Percent")
            ax[i+4].set_ylabel("Median Error")
            ax[i+4].set_title(f"{datanames[i]} dataset\nlowest imputed median error by {impType} missingness")
    
    stdout = open("\n\n* values are indices, add 1 for component")

    subplotLabel(ax)
    f.savefig('timpute/figures/img/figure4.png', bbox_inches="tight", format='png')

figure4()