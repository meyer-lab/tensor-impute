import numpy as np
from .runImputation import *
from ..plot import *
from ..common import *

# poetry run python -m timpute.figures.figure4
drops = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4)

def figure4(datalist=SAVENAMES, errors=True):
    ax, f = getSetup((20,10), (2,4))
    dirname = f"timpute/figures/img"
    stdout = open(f"{dirname}/figure4.txt", 'w')
    stdout.write(f"{drops}")
    
    for i, data in enumerate(datalist):

        # Figure 4, a)-d)
        impType = 'entry'
        for mID, m in enumerate(METHODS):
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

            if errors is True:
                ax[i].errorbar([str(x*100) for x in drops], np.array(ImpErr), ls='dashed', color=rgbs(mID, 0.7), yerr=np.hstack(tuple(ImpErrIQR)))
            else:
                ax[i].plot([str(x*100) for x in drops], np.array(ImpErr), ls='dashed', color=rgbs(mID, 0.7))

            label = f"{METHODNAMES[mID]}"
            if errors is True:
                ax[i].errorbar([str(x*100) for x in drops], np.array(TotErr), ls='solid', label=label, color=rgbs(mID, 0.7), yerr=np.hstack(tuple(TotErrIQR)))
            else:
                ax[i].plot([str(x*100) for x in drops], np.array(TotErr), ls='solid', label=label, color=rgbs(mID, 0.7))

        if errors is True:
            ax[i].errorbar([],[], label="Best Imputed Error", ls='dashed', color='black')
            ax[i].errorbar([],[], label="Total Error", ls='solid', color='black')
            h,l = ax[i].get_legend_handles_labels()
            h = [a[0] for a in h]
            ax[i].legend(h, l, loc='best', handlelength=2)
        else:
            ax[i].plot([],[], label="Best Imputed Error", ls='dashed', color='black')
            ax[i].plot([],[], label="Total Error", ls='solid', color='black')
            ax[i].legend(loc='best', handlelength=2)

        ax[i].set_xlabel("Drop Percent")
        ax[i].set_ylabel("Median Error")
        ax[i].set_title(f"{DATANAMES[i]}\nBest imputed error by {impType} masking percent")
        
        # Figure 4, e)-h)
        impType = 'chord'
        for mID, m in enumerate(METHODS):
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
                ImpErrIQR.append(np.vstack((-(np.percentile(run.chord_imputed[comp],25,0) - np.median(run.chord_imputed[comp],0)),
                                          np.percentile(run.chord_imputed[comp],75,0) - np.median(run.chord_imputed[comp],0))))
                TotErr.append(np.median(run.chord_total[comp]))
                TotErrIQR.append(np.vstack((-(np.percentile(run.chord_total[comp],25,0) - np.median(run.chord_total[comp],0)),
                                          np.percentile(run.chord_total[comp],75,0) - np.median(run.chord_total[comp],0))))
                stdout.write(f"{comp}, ")

            if errors is True:
                ax[i+4].errorbar([str(x*100) for x in drops], np.array(ImpErr), ls='dashed', color=rgbs(mID, 0.7), yerr=np.hstack(tuple(ImpErrIQR)))
            else:
                ax[i+4].plot([str(x*100) for x in drops], np.array(ImpErr), ls='dashed', color=rgbs(mID, 0.7))

            label = f"{METHODNAMES[mID]}"
            if errors is True:
                ax[i+4].errorbar([str(x*100) for x in drops], np.array(TotErr), ls='solid', label=label, color=rgbs(mID, 0.7), yerr=np.hstack(tuple(TotErrIQR)))
            else:
                ax[i+4].plot([str(x*100) for x in drops], np.array(TotErr), ls='solid', label=label, color=rgbs(mID, 0.7))
            

        if errors is True:
            ax[i+4].errorbar([],[], label="Best Imputed Error", ls='dashed', color='black')
            ax[i+4].errorbar([],[], label="Total Error", ls='solid', color='black')
            h,l = ax[i+4].get_legend_handles_labels()
            h = [a[0] for a in h]
            ax[i+4].legend(h, l, loc='best', handlelength=2)
        else:
            ax[i+4].plot([],[], label="Best Imputed Error", ls='dashed', color='black')
            ax[i+4].plot([],[], label="Total Error", ls='solid', color='black')
            ax[i+4].legend(loc='best', handlelength=2)

        if (ax[i+4].get_ylim()[1] > 1):
            ax[i+4].set_ylim(0, top=1)
        ax[i+4].set_xlabel("Drop Percent")
        ax[i+4].set_ylabel("Median Error")
        ax[i+4].set_title(f"{DATANAMES[i]}\nBest imputed error by {impType} masking percent")
    
    stdout.write("\n\n* values are indices, add 1 for component")

    subplotLabel(ax)
    f.savefig('timpute/figures/img/svg/figure4.svg', bbox_inches="tight", format='svg')
    f.savefig('timpute/figures/img/figure4.png', bbox_inches="tight", format='png')

if __name__ == "__main__":
    figure4(errors=False)