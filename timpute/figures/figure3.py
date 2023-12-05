import numpy as np
from .runImputation import *
from ..plot import *
from ..common import *
from matplotlib.lines import Line2D

# poetry run python -m timpute.figures.figure3

methods = (perform_DO, perform_ALS, perform_CLS)
methodname = ["DO","ALS","CLS"]
datanames = ['Zohar', 'Alter', 'Mills', 'CoH']
linestyles = ('dashdot', (0,(1,1)), 'solid', (3,(3,1,1,1,1,1)), 'dotted', (0,(5,1)))
drops = (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)

def figure3(datalist=["zohar", "alter", "hms", "coh_response"]):
    ax, f = getSetup((20,10), (2,4))

    dirname = f"timpute/figures/img"
    stdout = open(f"{dirname}/figure3.txt", 'w')

    # Figure 1, a)-d)
    for i, data in enumerate(datalist):
        folder = f"timpute/figures/cache/{data}/drop_0.1/"
        impType = 'entry' 
        for mID, m in enumerate(methods):
            run, _ = loadImputation(impType, m, folder)

            comps = np.arange(1,run.entry_imputed.shape[1]+1)
            label = f"{methodname[mID]} Total"
            total_errbar = np.vstack((abs(np.percentile(run.entry_total,25,0) - np.nanmedian(run.entry_total,0)),
                                    abs(np.percentile(run.entry_total,75,0) - np.nanmedian(run.entry_total,0))))
            ax[i].errorbar(comps, np.median(run.entry_total,0), yerr=total_errbar, label=label, ls='solid', color=rgbs(mID, 0.7), alpha=0.5)

            label = f"{methodname[mID]} Imputed"
            imp_errbar = np.vstack((abs(np.percentile(run.entry_imputed,25,0) - np.nanmedian(run.entry_imputed,0)),
                                    abs(np.percentile(run.entry_imputed,75,0) - np.nanmedian(run.entry_imputed,0))))
            e = ax[i].errorbar(comps, np.median(run.entry_imputed,0), yerr=imp_errbar, label=label, ls='dashed', color=rgbs(mID, 0.7), alpha=0.5)
            e[-1][0].set_linestyle('dashed')

            ax[i].legend()
            ax[i].set_title(f"{data} dataset, 10% entry missingness vs component")
            ax[i].set_xlabel("Number of Components")
            ax[i].set_ylabel("Error")
            ax[i].set_xticks([x for x in comps])
            ax[i].set_xticklabels([x for x in comps])
            # ax[1+row].set_yscale("log")


    # Figure 1, e)-h)
    drop = 0.1

    # --- ENTRY ---
    plot_data = dict()
    for m in methods: plot_data[m.__name__] = list()

    for i, data in enumerate(datalist):
        folder = f"timpute/figures/cache/{data}/drop_{drop}/"
        impType = 'entry'
        stdout.write(f"{data}, {impType} {drop}: ")
        for mID, m in enumerate(methods):
            run, _ = loadImputation(impType, m, folder)
            impMatrix = run.entry_imputed
            impDist = impMatrix[:, np.median(impMatrix, axis=0).argmin()]
            plot_data[m.__name__].append(impDist)
            stdout.write(f"{np.median(impMatrix, axis=0).argmin()}, ")
        stdout.write("\n")

    bar_spacing = 0.5
    bar_width = 0.4
    exp_spacing = 2
    for mID, m in enumerate(methods):
        bp = ax[4].boxplot(plot_data[m.__name__], positions=np.array(range(len(plot_data[m.__name__]))) * exp_spacing - bar_spacing + bar_spacing*mID, sym='', widths=bar_width)
        set_boxplot_color(bp, rgbs(mID))
    ax[4].set_xticks(range(0, len(datanames) * exp_spacing, exp_spacing), datanames)

    ax[4].set_title(f"best imputation error by dataset ({int(drop*100)}% {impType} imputation)")
    ax[4].set_xlabel("Dataset")
    ax[4].set_ylabel("Error")
    ax[4].set_xlim(right=7)
    ax[4].set_yscale('log')
    ax[4].legend(handles=[Line2D([0], [0], label=m.__name__, color=rgbs(i)) for i,m in enumerate(methods)])



    # --- CHORD ---
    plot_data = dict()
    for m in methods: plot_data[m.__name__] = list()

    for i, data in enumerate(datalist):
        folder = f"timpute/figures/cache/{data}/drop_{drop}/"
        impType = 'chord'
        stdout.write(f"\n{data}, {impType} {drop}: ")
        for mID, m in enumerate(methods):
            run, _ = loadImputation(impType, m, folder)
            impMatrix = run.chord_imputed
            impDist = impMatrix[:, np.median(impMatrix, axis=0).argmin()]
            plot_data[m.__name__].append(impDist)
            stdout.write(f"{np.median(impMatrix, axis=0).argmin()}, ")

    bar_spacing = 0.5
    bar_width = 0.4
    exp_spacing = 2
    for mID, m in enumerate(methods):
        bp = ax[5].boxplot(plot_data[m.__name__], positions=np.array(range(len(plot_data[m.__name__]))) * exp_spacing - bar_spacing + bar_spacing*mID, sym='', widths=bar_width)
        set_boxplot_color(bp, rgbs(mID))
    ax[5].set_xticks(range(0, len(datanames) * exp_spacing, exp_spacing), datanames)

    ax[5].set_title(f"best imputation error by dataset ({int(drop*100)}% {impType} imputation)")
    ax[5].set_xlabel("Dataset")
    ax[5].set_ylabel("Error")
    ax[5].set_xlim(right=7)
    ax[5].set_yscale('log')
    ax[5].legend(handles=[Line2D([0], [0], label=m.__name__, color=rgbs(i)) for i,m in enumerate(methods)])
    
    stdout.write("\n\n* values are indices, add 1 for component")

    subplotLabel(ax)
    f.savefig('timpute/figures/img/figure3.png', bbox_inches="tight", format='png')


def figure3_exp(datalist=["zohar", "alter", "hms", "coh_response"]):
    ax, f = getSetup((48,40), (7,8))
    dirname = f"timpute/figures/img"

    # Figure 1, a)-d)
    for d,drop in enumerate(drops):
        for i, data in enumerate(datalist):
            folder = f"timpute/figures/cache/{data}/drop_{drop}/"
            for mID, m in enumerate(methods):
                run, _ = loadImputation("entry", m, folder)

                comps = np.arange(1,run.entry_imputed.shape[1]+1)
                label = f"{methodname[mID]}"
                total_errbar = np.vstack((abs(np.percentile(run.entry_total,25,0) - np.nanmedian(run.entry_total,0)),
                                        abs(np.percentile(run.entry_total,75,0) - np.nanmedian(run.entry_total,0))))
                ax[i+d*8].errorbar(comps, np.median(run.entry_total,0), yerr=total_errbar,label=label, ls='solid', color=rgbs(mID, 0.7), alpha=0.5)

                label = f"{methodname[mID]} Imputed Error"
                imp_errbar = np.vstack((abs(np.percentile(run.entry_imputed,25,0) - np.nanmedian(run.entry_imputed,0)),
                                        abs(np.percentile(run.entry_imputed,75,0) - np.nanmedian(run.entry_imputed,0))))
                e = ax[i+d*8].errorbar(comps, np.median(run.entry_imputed,0), yerr=imp_errbar, label=label, ls='dashed', color=rgbs(mID, 0.7), alpha=0.5)
                e[-1][0].set_linestyle('dashed')

                # ax[i].legend()
                ax[i+d*8].set_title(f"{data} dataset, {drop}% entry missingness vs component")
                ax[i+d*8].set_xlabel("Number of Components")
                ax[i+d*8].set_ylabel("Error")
                ax[i+d*8].set_xticks([x for x in comps])
                ax[i+d*8].set_xticklabels([x for x in comps])
                if (ax[i+d*8].get_ylim()[1] > 1):
                    ax[i+d*8].set_ylim(0, top=1)

                ## CHORD

                run, _ = loadImputation("chord", m, folder)

                comps = np.arange(1,run.chord_imputed.shape[1]+1)
                label = f"{methodname[mID]}"
                total_errbar = np.vstack((abs(np.percentile(run.chord_total,25,0) - np.nanmedian(run.chord_total,0)),
                                        abs(np.percentile(run.chord_total,75,0) - np.nanmedian(run.chord_total,0))))
                ax[i+4+d*8].errorbar(comps, np.median(run.chord_total,0), yerr=total_errbar,label=label, ls='solid', color=rgbs(mID, 0.7), alpha=0.5)

                label = f"{methodname[mID]} Imputed Error"
                imp_errbar = np.vstack((abs(np.percentile(run.chord_imputed,25,0) - np.nanmedian(run.chord_imputed,0)),
                                        abs(np.percentile(run.chord_imputed,75,0) - np.nanmedian(run.chord_imputed,0))))
                e = ax[i+4+d*8].errorbar(comps, np.median(run.chord_imputed,0), yerr=imp_errbar, label=label, ls='dashed', color=rgbs(mID, 0.7), alpha=0.5)
                e[-1][0].set_linestyle('dashed')

                # ax[i].legend()
                ax[i+4+d*8].set_title(f"{data} dataset, {drop}% chord missingness vs component")
                ax[i+4+d*8].set_xlabel("Number of Components")
                ax[i+4+d*8].set_ylabel("Error")
                ax[i+4+d*8].set_xticks([x for x in comps])
                ax[i+4+d*8].set_xticklabels([x for x in comps])

                if (ax[i+4+d*8].get_ylim()[1] > 1):
                    ax[i+4+d*8].set_ylim(0, top=1)
                
    f.savefig('timpute/figures/img/figure3-exp.png', bbox_inches="tight", format='png')

figure3()
# figure3_exp()