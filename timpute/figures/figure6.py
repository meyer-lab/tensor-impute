import pandas as pd
from .figure_helper import *
from .common import getSetup, subplotLabel, rgbs
from . import METHODNAMES,DROPS

# poetry run python -m timpute.figures.figure6

SUBTITLE_FONTSIZE = 14
TEXT_FONTSIZE = 6

def figure6():
    TITLE_FONTSIZE = 14
    SUBTITLE_FONTSIZE = 10
    TEXT_FONTSIZE = 8

    ax, f = getSetup((7,10), (3,2), multz={2:1,4:1}, empts=[3,5])
    df = pd.read_excel("./timpute/figures/revision_img/iterTime.xlsx", index_col=[0,1], header=[0,1])

    # a) 10% entry DyeDrop barplot (runtime per method) - CLS much slower (general pattern)
    ax[0].tick_params(axis='both', which='major', labelsize=TEXT_FONTSIZE)
    ax[0].bar(METHODNAMES,df.loc[('BC cytokine','entry'),'10%'], color=[rgbs(i) for i in [0,1,2]])
    ax[0].set_title(f"Time per Iteration\nBC cytokine 10% entry masking", fontsize=TITLE_FONTSIZE)
    ax[0].set_xlabel("Method", fontsize=SUBTITLE_FONTSIZE)
    ax[0].set_ylabel("Time (s)", fontsize=SUBTITLE_FONTSIZE)

    # b) 20% entry & chord SARS-COV-19 serology barplot (runtime per method) - ALS & DO similar -> C-ALS much improved
    ax[1].tick_params(axis='both', which='major', labelsize=TEXT_FONTSIZE)
    ax[1].bar([0,0.25,0.5],df.loc[('SARS-COV-19 serology','entry'),'20%'], 0.25, color=[rgbs(i) for i in [0,1,2]])
    ax[1].bar([1,1.25,1.5],df.loc[('SARS-COV-19 serology','chord'),'20%'], 0.25, color=[rgbs(i) for i in [0,1,2]])
    ax[1].set_xticks([0.25,1.25], ['Entry', 'Chord'])
    ax[1].set_title(f"Time per Iteration\nSARS-COV-19 serology 20% masking", fontsize=TITLE_FONTSIZE)
    ax[1].set_xlabel("Masking Type", fontsize=SUBTITLE_FONTSIZE)
    ax[1].set_ylabel("Time (s)", fontsize=SUBTITLE_FONTSIZE)

    # c) HIV serology lineplot (each method runtime vs drop %) - C-ALS can improve over DO
    tmp = df.reorder_levels([1,0], 1)
    tmp = tmp[tmp.columns.sort_values()]
    width = 0.25
    sep = 1
    ax[2].tick_params(axis='both', which='major', labelsize=TEXT_FONTSIZE)

    for m,method in enumerate(METHODNAMES):
        x = [sep*i + width*m for i in range(len(DROPS))]
        y = [tmp.loc[('DyeDrop profiling', 'chord'),(method,f"{int(d*100)}%")] for d in DROPS]
        ax[2].bar(x, y, color=rgbs(m), width=0.25)
    ax[2].set_xticks([sep*i+0.25 for i in range(len(DROPS))], [f"{int(d*100)}%" for d in DROPS])
    ax[2].set_title(f"Time per Iteration by Chord Masking Percentage, DyeDrop profiling", fontsize=TITLE_FONTSIZE)
    ax[2].set_xlabel("Drop Percentage", fontsize=SUBTITLE_FONTSIZE)
    ax[2].set_ylabel("Time (s)", fontsize=SUBTITLE_FONTSIZE)
    # ax[2].set_ylim((0,0.1))

    # d) BC cytokine entry & chord lineplot (ALS & DO x-fold diff of CLS vs drop %) - ALS & DO doesn't vary, but C-ALS decreases over time
    linestyles = ['solid','dashed']
    ax[3].tick_params(axis='both', which='major', labelsize=TEXT_FONTSIZE)

    for t,impType in enumerate(['entry','chord']):
        for m,method in enumerate(METHODNAMES[:2]):
            x = [f"{int(d*100)}%" for d in DROPS]
            y = [tmp.loc[('HIV serology', impType),('C-ALS',f"{int(d*100)}%")] /
                tmp.loc[('HIV serology', impType),(method,f"{int(d*100)}%")]
                for d in DROPS]
            ax[3].plot(x, y, color=rgbs(m), ls=linestyles[t], label=f"{method}, {impType}")
    ax[3].set_title(f"Time per Iteration Decreases Relative to C-ALS\nby Drop Percentage, HIV serology", fontsize=TITLE_FONTSIZE)
    ax[3].set_xlabel("Drop Percentage", fontsize=SUBTITLE_FONTSIZE)
    ax[3].set_ylabel("Fold Decrease in Time", fontsize=SUBTITLE_FONTSIZE)
    ax[3].legend(handlelength=2, fontsize=TEXT_FONTSIZE)
    
    subplotLabel(ax)