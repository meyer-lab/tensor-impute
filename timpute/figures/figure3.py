import math

import numpy as np
from .figure_helper import loadImputation
from .common import getSetup, subplotLabel, rgbs
from . import METHODS, METHODNAMES, SAVENAMES, DATANAMES, LINE_WIDTH, SUBTITLE_FONTSIZE, TEXT_FONTSIZE

# poetry run python -m timpute.figures.figure3


def figure3(datalist=SAVENAMES):
    ax, f = getSetup((16, 4), (1, 4))

    # a-d) C-ALS outperforms other algorithms without imputation
    # 0% missingness, component vs total error

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
            ax[i].errorbar(
                comps,
                np.median(run.entry_total, 0),
                yerr=total_errbar,
                label=label,
                ls="solid",
                color=rgbs(mID, 0.7),
                lw=LINE_WIDTH,
            )

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

        print(f"completed figure 3{chr(ord('a') + i)}")

    subplotLabel(ax)
    f.savefig(
        "timpute/figures/revision_img/svg/figure3.svg",
        bbox_inches="tight",
        format="svg",
    )
    f.savefig(
        "timpute/figures/revision_img/figure3.png", bbox_inches="tight", format="png"
    )


if __name__ == "__main__":
    print("\nbuilding figure 3...")
    figure3()
