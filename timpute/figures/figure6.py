import numpy as np

from . import (
    DATANAMES,
    LINE_WIDTH,
    METHODNAMES,
    METHODS,
    SAVENAMES,
    SUBTITLE_FONTSIZE,
)
from .common import getSetup, rgbs, subplotLabel
from .figure_data import bestComps
from .figure_helper import loadImputation

# from matplotlib.legend_handler import HandlerErrorbar

# poetry run python -m timpute.figures.figure6

drops = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)


def figure6():
    ax, f = getSetup((16, 8), (2, 4))
    # (w,h),  (r,c)

    plotIter(ax[0], 0, "entry", 0)
    print("completed figure 6a")
    plotIter(ax[1], 1, "entry", 0)
    print("completed figure 6b")
    plotIter(ax[2], 2, "entry", 0)
    print("completed figure 6c")
    plotIter(ax[3], 3, "entry", 0)
    print("completed figure 6d")
    plotIter(ax[4], 0, "entry", 0.1)
    print("completed figure 6e")
    plotIter(ax[5], 1, "entry", 0.2)
    print("completed figure 6f")
    plotIter(ax[6], 2, "chord", 0.3)
    print("completed figure 6g")
    plotIter(ax[7], 3, "chord", 0.4)
    print("completed figure 6h")

    subplotLabel(ax)
    f.savefig(
        "timpute/figures/revision_img/svg/figure6.svg",
        bbox_inches="tight",
        format="svg",
    )
    f.savefig(
        "timpute/figures/revision_img/figure6.png", bbox_inches="tight", format="png"
    )


def plotTime(ax, dataN, impType, drop):
    data = SAVENAMES[dataN]
    if drop == 0:
        folder = f"timpute/figures/revision_cache/{data}/nonmissing/"
    else:
        folder = f"timpute/figures/revision_cache/{data}/drop_{drop}/"

    comps = bestComps(drop=drop, impType=impType, datalist=[data])
    for mID, m in enumerate(METHODS):
        _, tracker = loadImputation(impType, m, folder)
        timepoints = tracker.time_array[str(comps[data][METHODNAMES[mID]])]
        # for i in range(timepoints.shape[0]):
        #     timepoints[i] -= timepoints[i,0]
        if drop != 0:
            impErr = tracker.imputed_array[str(comps[data][METHODNAMES[mID]])]
            ax.errorbar(
                np.nanmedian(timepoints, 0),
                np.nanmean(impErr, 0),
                label=METHODNAMES[mID],
                color=rgbs(mID, 0.7),
                ls="dashed",
                errorevery=5,
                lw=LINE_WIDTH,
            )
        totErr = tracker.total_array[str(comps[data][METHODNAMES[mID]])]
        ax.errorbar(
            np.nanmedian(timepoints, 0),
            np.nanmean(totErr, 0),
            label=METHODNAMES[mID],
            color=rgbs(mID, 0.7),
            ls="solid",
            errorevery=5,
            lw=LINE_WIDTH,
        )

    ax.set_xlim(left=0)
    ax.set_xlabel("Iteration", size=SUBTITLE_FONTSIZE)
    ax.set_ylabel("Error", size=SUBTITLE_FONTSIZE)
    ax.set_title(
        f"{DATANAMES[dataN]}\n{int(drop * 100)}% {impType} masking",
        size=SUBTITLE_FONTSIZE * 1.1,
    )
    ax.set_xscale("symlog")


def plotIter(ax, dataN, impType, drop, legend=False):
    data = SAVENAMES[dataN]
    if drop == 0:
        folder = f"timpute/figures/revision_cache/{data}/nonmissing/"
    else:
        folder = f"timpute/figures/revision_cache/{data}/drop_{drop}/"
    comps = bestComps(drop=drop, impType=impType, datalist=[data])
    # print(comps)

    for mID, m in enumerate(METHODS):
        _, tracker = loadImputation(impType, m, folder)

        # print(str(comps[SAVENAMES[dataN]]))
        impErr = tracker.imputed_array[str(comps[SAVENAMES[dataN]][METHODNAMES[mID]])][
            :, :-1
        ]
        imputed_errbar = np.vstack(
            (
                -(np.percentile(impErr, 25, 0) - np.nanmedian(impErr, 0)),
                np.percentile(impErr, 75, 0) - np.nanmedian(impErr, 0),
            )
        )
        ax.errorbar(
            np.arange(impErr.shape[1]) + (0.1 - mID * 0.1),
            np.nanmedian(impErr, 0),
            color=rgbs(mID, 0.7),
            yerr=imputed_errbar,
            ls="dashed",
            errorevery=5,
            lw=LINE_WIDTH,
        )

        label = f"{METHODNAMES[mID]}"
        totErr = tracker.total_array[str(comps[SAVENAMES[dataN]][METHODNAMES[mID]])][
            :, :-1
        ]
        total_errbar = np.vstack(
            (
                -(np.percentile(totErr, 25, 0) - np.nanmedian(totErr, 0)),
                np.percentile(totErr, 75, 0) - np.nanmedian(totErr, 0),
            )
        )
        ax.errorbar(
            np.arange(totErr.shape[1]),
            np.nanmedian(totErr, 0),
            label=label,
            color=rgbs(mID, 0.7),
            yerr=total_errbar,
            ls="solid",
            errorevery=5,
            lw=LINE_WIDTH,
        )

    ax.errorbar([], [], label="Imputed Error", ls="dashed", color="black")
    ax.errorbar([], [], label="Total Error", ls="solid", color="black")
    if legend is True:
        handles, labels = ax.get_legend_handles_labels()
        handles = [a[0] for a in handles]
        ax.legend(handles, labels, loc="best", handlelength=2)

    ax.set_xlim((0, totErr.shape[1] - 1))
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Error")
    ax.set_title(f"{DATANAMES[dataN]}, {int(drop * 100)}% {impType} masking")
    ax.set_yscale("log")


def figure6_exp(datalist=SAVENAMES):
    ax, f = getSetup((48, 40), (6, 8))

    for d, drop in enumerate(drops):
        for t, impType in enumerate(["entry", "chord"]):
            comps = bestComps(drop=drop, impType=impType, datalist=datalist)
            for dat, data in enumerate(datalist):
                folder = f"timpute/figures/revision_cache/{data}/drop_{drop}/"
                for mID, m in enumerate(METHODS):
                    _, tracker = loadImputation(impType, m, folder)

                    label = f"{m.__name__} Imputed"
                    impErr = tracker.imputed_array[str(comps[data][mID] + 1)][:, :-1]
                    imputed_errbar = np.vstack(
                        (
                            -(np.percentile(impErr, 25, 0) - np.nanmedian(impErr, 0)),
                            np.percentile(impErr, 75, 0) - np.nanmedian(impErr, 0),
                        )
                    )
                    e = ax[dat + d * 8 + t * 4].errorbar(
                        np.arange(impErr.shape[1]) + (0.1 - mID * 0.1),
                        np.nanmedian(impErr, 0),
                        label=label,
                        color=rgbs(mID, 0.7),
                        yerr=imputed_errbar,
                        ls="dashed",
                        errorevery=5,
                    )
                    e[-1][0].set_linestyle("dashed")

                    label = f"{m.__name__} Total"
                    totErr = tracker.total_array[str(comps[data][mID] + 1)][:, :-1]
                    total_errbar = np.vstack(
                        (
                            -(np.percentile(totErr, 25, 0) - np.nanmedian(totErr, 0)),
                            np.percentile(totErr, 75, 0) - np.nanmedian(totErr, 0),
                        )
                    )
                    ax[dat + d * 8 + t * 4].errorbar(
                        np.arange(totErr.shape[1]),
                        np.nanmedian(totErr, 0),
                        label=label,
                        color=rgbs(mID, 0.7),
                        yerr=total_errbar,
                        ls="solid",
                        errorevery=5,
                    )

                    ax[dat + d * 8 + t * 4].legend()
                    ax[dat + d * 8 + t * 4].set_xlim((0, totErr.shape[1]))
                    ax[dat + d * 8 + t * 4].set_xlabel("Iteration")
                    ax[dat + d * 8 + t * 4].set_ylabel("Error")
                    ax[dat + d * 8 + t * 4].set_title(
                        f"{data} dataset, median error by iteration"
                    )

    # subplotLabel(ax)
    f.savefig(
        "timpute/figures/revision_img/svg/figure6-exp.svg",
        bbox_inches="tight",
        format="svg",
    )
    f.savefig(
        "timpute/figures/revision_img/figure6-exp.png",
        bbox_inches="tight",
        format="png",
    )


if __name__ == "__main__":
    print("\nbuilding figure 6...")
    figure6()
    # figure6_exp()
