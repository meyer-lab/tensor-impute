import math
import numpy as np
from .figure_helper import loadImputation
from .common import getSetup, subplotLabel, rgbs
from . import METHODS, METHODNAMES, SAVENAMES, DATANAMES, LINE_WIDTH

# poetry run python -m timpute.figures.figure2

SUBTITLE_FONTSIZE = 16
TEXT_FONTSIZE = 13


def figure2(datalist=SAVENAMES):
    ax, f = getSetup((16, 4), (1, 4))

    for i, data in enumerate(datalist):
        folder = f"timpute/figures/revision_cache/{data}/nonmissing/"
        impType = "entry"
        for mID, m in enumerate(METHODS):
            run, _ = loadImputation(impType, m, folder)

            comps = np.arange(1, run.entry_imputed.shape[1] + 1)
            label = f"{METHODNAMES[mID]}"
            total_errbar = np.vstack(
                (
                    abs(
                        np.percentile(run.entry_total, 25, 0)
                        - np.nanmedian(run.entry_total, 0)
                    ),
                    abs(
                        np.percentile(run.entry_total, 75, 0)
                        - np.nanmedian(run.entry_total, 0)
                    ),
                )
            )
            e = ax[i].errorbar(
                comps,
                np.median(run.entry_total, 0),
                yerr=total_errbar,
                label=label,
                ls="solid",
                color=rgbs(mID, 0.7),
                lw=LINE_WIDTH
            )

            h, l = ax[i].get_legend_handles_labels()
            h = [a[0] for a in h]

            ax[i].set_title(DATANAMES[i], fontsize=SUBTITLE_FONTSIZE * 1.1)
            ax[i].set_xlabel("Number of Components", fontsize=SUBTITLE_FONTSIZE)
            ax[i].set_ylabel("Fitting Error", fontsize=SUBTITLE_FONTSIZE)
            ax[i].tick_params(axis="both", which="major", labelsize=TEXT_FONTSIZE)
            ax[i].set_xticks(
                np.arange(max(comps) / 10, max(comps) + 1, max(comps) / 10)
            )
            ax[i].set_ylim(
                top=math.ceil(max(np.median(run.entry_total, 0)) * 5) / 5, bottom=0
            )

    # ax[0].legend(handles=h, labels=l, loc='best', handlelength=2, fontsize=TEXT_FONTSIZE)
    subplotLabel(ax)
    f.savefig(
        "timpute/figures/revision_img/svg/figure2.svg",
        bbox_inches="tight",
        format="svg",
    )
    f.savefig(
        "timpute/figures/revision_img/figure2.png", bbox_inches="tight", format="png"
    )


if __name__ == "__main__":
    figure2()
