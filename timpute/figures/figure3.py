import numpy as np
from .runImputation import *
from ..plot import *
from ..common import *

# poetry run python -m timpute.figures.figure3

def figure3(datalist=SAVENAMES):
    ax, f = getSetup((20,10), (2,4), multz={4:1, 6:1}, empts=[5,7])

    # Figure 3, a)-d)
    for i, data in enumerate(datalist):
        folder = f"timpute/figures/cache/{data}/drop_0.1/"
        impType = 'entry' 
        for mID, m in enumerate(METHODS):
            run, _ = loadImputation(impType, m, folder)

            comps = np.arange(1,run.entry_imputed.shape[1]+1)
            imp_errbar = np.vstack((abs(np.percentile(run.entry_imputed,25,0) - np.nanmedian(run.entry_imputed,0)),
                                    abs(np.percentile(run.entry_imputed,75,0) - np.nanmedian(run.entry_imputed,0))))
            e = ax[i].errorbar(comps, np.median(run.entry_imputed,0), yerr=imp_errbar, ls='dashed', color=rgbs(mID, 0.7), alpha=0.5)
            # e[-1][0].set_linestyle('dashed')

            label = f"{METHODNAMES[mID]}"
            total_errbar = np.vstack((abs(np.percentile(run.entry_total,25,0) - np.nanmedian(run.entry_total,0)),
                                    abs(np.percentile(run.entry_total,75,0) - np.nanmedian(run.entry_total,0))))
            ax[i].errorbar(comps, np.median(run.entry_total,0), yerr=total_errbar, label=label, ls='solid', color=rgbs(mID, 0.7), alpha=0.5)

            ax[i].set_title(f"{DATANAMES[i]}, 10% entry masking")
            ax[i].set_xlabel("Number of Components")
            ax[i].set_ylabel("Error")
            ax[i].set_xticks([x for x in comps])
            ax[i].set_xticklabels([x for x in comps])
            # ax[1+row].set_yscale("log")

        ax[i].errorbar([],[], label="Imputed Error", ls='dashed', color='black')
        ax[i].errorbar([],[], label="Total Error", ls='solid', color='black')
        
        h,l = ax[i].get_legend_handles_labels()
        h = [a[0] for a in h]
        ax[i].legend(h, l, loc='best', handlelength=2)


    # Figure 3, e)-h)
    drop = 0.1

    # --- ENTRY ---
    plot_data = dict()
    comp_data = dict()
    for m in METHODS:
        plot_data[m.__name__] = list()
        comp_data[m.__name__] = list()

    for i, data in enumerate(datalist):
        folder = f"timpute/figures/cache/{data}/drop_{drop}/"
        impType = 'entry'
        for mID, m in enumerate(METHODS):
            run, _ = loadImputation(impType, m, folder)
            impMatrix = run.entry_imputed
            impDist = impMatrix[:, np.median(impMatrix, axis=0).argmin()]
            plot_data[m.__name__].append(impDist)
            comp_data[m.__name__].append(np.median(impMatrix, axis=0).argmin())

    bar_spacing = 0.5
    bar_width = 0.4
    exp_spacing = 2
    for mID, m in enumerate(METHODS):
        box = ax[4].boxplot(plot_data[m.__name__], positions=np.array(range(len(plot_data[m.__name__]))) * exp_spacing - bar_spacing + bar_spacing*mID, sym='', widths=bar_width)
        set_boxplot_color(box, rgbs(mID))
        for l,line in enumerate([1,3,5,7]):
            x = box['caps'][line].get_xdata().mean()
            y = box['caps'][line].get_ydata()[0]
            ax[4].text(x, y*1.03, comp_data[m.__name__][l], ha='center', va='bottom')
    ax[4].set_xticks(range(0, len(DATANAMES) * exp_spacing, exp_spacing), DATANAMES)

    ax[4].set_title(f"Best imputation error by dataset\n{int(drop*100)}% {impType} masking")
    ax[4].set_xlabel("Dataset")
    ax[4].set_ylabel("Error")
    ax[4].set_xlim(right=7)
    ax[4].set_ylim(top=ax[4].get_ylim()[1]*1.5)
    ax[4].set_yscale('log')
    ax[4].legend(loc='lower right', handles=[Line2D([0], [0], label=m, color=rgbs(i)) for i,m in enumerate(METHODNAMES)])



    # --- CHORD ---
    plot_data = dict()
    comp_data = dict()
    for m in METHODS:
        plot_data[m.__name__] = list()
        comp_data[m.__name__] = list()

    for i, data in enumerate(datalist):
        folder = f"timpute/figures/cache/{data}/drop_{drop}/"
        impType = 'chord'
        for mID, m in enumerate(METHODS):
            run, _ = loadImputation(impType, m, folder)
            impMatrix = run.chord_imputed
            impDist = impMatrix[:, np.median(impMatrix, axis=0).argmin()]
            plot_data[m.__name__].append(impDist)
            comp_data[m.__name__].append(np.median(impMatrix, axis=0).argmin())

    bar_spacing = 0.5
    bar_width = 0.4
    exp_spacing = 2
    for mID, m in enumerate(METHODS):
        box = ax[5].boxplot(plot_data[m.__name__], positions=np.array(range(len(plot_data[m.__name__]))) * exp_spacing - bar_spacing + bar_spacing*mID, sym='', widths=bar_width)
        set_boxplot_color(box, rgbs(mID))
        for l,line in enumerate([1,3,5,7]):
            x = box['caps'][line].get_xdata().mean()
            y = box['caps'][line].get_ydata()[0]
            ax[5].text(x, y*1.03, comp_data[m.__name__][l], ha='center', va='bottom')
    ax[5].set_xticks(range(0, len(DATANAMES) * exp_spacing, exp_spacing), DATANAMES)

    ax[5].set_title(f"Best imputation error by dataset\n{int(drop*100)}% {impType} masking")
    ax[5].set_xlabel("Dataset")
    ax[5].set_ylabel("Error")
    ax[5].set_xlim(right=7)
    ax[5].set_ylim(top=ax[5].get_ylim()[1]*1.5)
    ax[5].set_yscale('log')
    ax[5].legend(loc='lower right', handles=[Line2D([0], [0], label=m, color=rgbs(i)) for i,m in enumerate(METHODNAMES)])

    subplotLabel(ax)

    
    f.savefig('timpute/figures/img/svg/figure3.svg', bbox_inches="tight", format='svg')
    f.savefig('timpute/figures/img/figure3.png', bbox_inches="tight", format='png')


def figure3_exp(datalist=["zohar", "alter", "hms", "coh_response"]):
    ax, f = getSetup((48,40), (7,8))
    dirname = f"timpute/figures/img"

    # Figure 1, a)-d)
    for d,drop in enumerate(DROPS):
        for i, data in enumerate(datalist):
            folder = f"timpute/figures/cache/{data}/drop_{drop}/"
            for mID, m in enumerate(METHODS):
                run, _ = loadImputation("entry", m, folder)

                comps = np.arange(1,run.entry_imputed.shape[1]+1)
                label = f"{METHODNAMES[mID]}"
                total_errbar = np.vstack((abs(np.percentile(run.entry_total,25,0) - np.nanmedian(run.entry_total,0)),
                                        abs(np.percentile(run.entry_total,75,0) - np.nanmedian(run.entry_total,0))))
                ax[i+d*8].errorbar(comps, np.median(run.entry_total,0), yerr=total_errbar,label=label, ls='solid', color=rgbs(mID, 0.7), alpha=0.5)

                label = f"{METHODNAMES[mID]} Imputed"
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
                label = f"{METHODNAMES[mID]}"
                total_errbar = np.vstack((abs(np.percentile(run.chord_total,25,0) - np.nanmedian(run.chord_total,0)),
                                        abs(np.percentile(run.chord_total,75,0) - np.nanmedian(run.chord_total,0))))
                ax[i+4+d*8].errorbar(comps, np.median(run.chord_total,0), yerr=total_errbar,label=label, ls='solid', color=rgbs(mID, 0.7), alpha=0.5)

                label = f"{METHODNAMES[mID]} Imputed"
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
                
    f.savefig('timpute/figures/img/svg/figure3-exp.svg', bbox_inches="tight", format='svg')
    f.savefig('timpute/figures/img/figure3-exp.png', bbox_inches="tight", format='png')

if __name__ == "__main__":
    figure3()
    # figure3_exp()