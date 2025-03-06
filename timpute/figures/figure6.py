import pandas as pd
from .figure_helper import *
from .common import getSetup, subplotLabel, rgbs
from . import METHODNAMES, DATANAMES, LINE_WIDTH

# poetry run python -m timpute.figures.figure6

DROPS = (0.05, 0.1, 0.2, 0.3, 0.4)

def figure6():
    TITLE_FONTSIZE = 14
    SUBTITLE_FONTSIZE = 10
    TEXT_FONTSIZE = 8

    ax, f = getSetup((10, 7), (2, 3), multz={0: 1, 3: 1}, empts=[1, 4])
    df = pd.read_excel(
        "./timpute/figures/revision_img/iterTime.xlsx", index_col=[0, 1], header=[0, 1]
    )
    tmp = df.reorder_levels([1, 0], 1)
    tmp = tmp[tmp.columns.sort_values()]
    print(tmp)


    # A) chord lineplot (each method runtime vs drop %) - C-ALS can improve over DO
    dataset_a = DATANAMES[2]
    width = 0.25
    sep = 1
    ax[0].tick_params(axis="both", which="major", labelsize=TEXT_FONTSIZE)

    for m, method in enumerate(METHODNAMES):
        x = [sep * i + width * m for i in range(len(DROPS))]
        y = [
            tmp.loc[(dataset_a, "chord"), (method, f"{int(d*100)}%")]
            for d in DROPS
        ]
        ax[0].bar(x, y, color=rgbs(m), width=0.25)

    ax[0].set_xticks(
        [sep * i + 0.25 for i in range(len(DROPS))], [f"{int(d*100)}%" for d in DROPS]
    )
    ax[0].set_title(
        f"Time per Iteration by Chord Masking Percentage\n{dataset_a}",
        fontsize=TITLE_FONTSIZE,
    )
    ax[0].set_xlabel("Drop Percentage", fontsize=SUBTITLE_FONTSIZE)
    ax[0].set_ylabel("Time (s)", fontsize=SUBTITLE_FONTSIZE)
    # ax[0].set_ylim((0,0.1))


    # B) 10% entry barplot (runtime per method) - CLS much slower (general pattern)
    dataset_b = DATANAMES[3]
    ax[1].tick_params(axis="both", which="major", labelsize=TEXT_FONTSIZE)
    ax[1].bar(
        METHODNAMES,
        df.loc[(dataset_b, "entry"), "10%"],
        color=[rgbs(i) for i in [0, 1, 2]],
    )
    ax[1].set_title(
        f"Time per Iteration\n{dataset_b} 10% entry masking", fontsize=TITLE_FONTSIZE
    )
    ax[1].set_xlabel("Method", fontsize=SUBTITLE_FONTSIZE)
    ax[1].set_ylabel("Time (s)", fontsize=SUBTITLE_FONTSIZE)


    # C) entry & chord lineplot (ALS & DO x-fold diff of CLS vs drop %) - ALS & DO doesn't vary, but C-ALS decreases over time
    dataset_c = DATANAMES[0]
    linestyles = ["solid", "dashed"]
    ax[2].tick_params(axis="both", which="major", labelsize=TEXT_FONTSIZE)
    ax[2].axhline(1,color='black',lw=LINE_WIDTH-1)

    for t, impType in enumerate(["entry", "chord"]):
        for m, method in enumerate(METHODNAMES[:2]):
            x = [f"{int(d*100)}%" for d in DROPS]
            y = [
                tmp.loc[(dataset_c, impType), ("C-ALS", f"{int(d*100)}%")]
                / tmp.loc[(dataset_c, impType), (method, f"{int(d*100)}%")]
                for d in DROPS
            ]
            ax[2].plot(
                x, y, color=rgbs(m), ls=linestyles[t], label=f"{method}, {impType}", lw=LINE_WIDTH
            )

    ax[2].set_title(
        f"Time per Iteration Fold-Speedup Versus C-ALS\nby Drop Percentage, {dataset_c}",
        fontsize=TITLE_FONTSIZE,
    )
    ax[2].set_xlabel("Drop Percentage", fontsize=SUBTITLE_FONTSIZE)
    ax[2].set_ylabel("Fold Decrease in Time", fontsize=SUBTITLE_FONTSIZE)
    ax[2].legend(handlelength=2, fontsize=TEXT_FONTSIZE)
    ax[2].set_yscale('log')

    ticks = (np.arange(8)-2).astype(float)
    ax[2].set_yticks(ticks=np.power(2,ticks),
                     labels=["$2^{" + f"{int(i)}" + "}$" for i in ticks])


    # D) 20% entry & chord barplot (runtime per method) - ALS & DO similar -> C-ALS much improved
    dataset_d = DATANAMES[1]
    ax[3].tick_params(axis="both", which="major", labelsize=TEXT_FONTSIZE)
    ax[3].bar(
        [0, 0.25, 0.5],
        df.loc[(dataset_d, "entry"), "20%"],
        0.25,
        color=[rgbs(i) for i in [0, 1, 2]],
    )
    ax[3].bar(
        [1, 1.25, 1.5],
        df.loc[(dataset_d, "chord"), "20%"],
        0.25,
        color=[rgbs(i) for i in [0, 1, 2]],
    )
    ax[3].set_xticks([0.25, 1.25], ["Entry", "Chord"])
    ax[3].set_title(
        f"Time per Iteration\n{dataset_d} 20% masking", fontsize=TITLE_FONTSIZE
    )
    ax[3].set_xlabel("Masking Type", fontsize=SUBTITLE_FONTSIZE)
    ax[3].set_ylabel("Time (s)", fontsize=SUBTITLE_FONTSIZE)
    
    subplotLabel(ax)

    f.savefig(
        "timpute/figures/revision_img/svg/figure6.svg",
        bbox_inches="tight",
        format="svg",
    )
    f.savefig(
        "timpute/figures/revision_img/figure6.png",
        bbox_inches="tight",
        format="png",
    )

if __name__ == "__main__":
    figure6()