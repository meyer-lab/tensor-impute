import math
import numpy as np
from matplotlib.lines import Line2D
from .figure_helper import loadImputation
from .common import getSetup, subplotLabel, rgbs, set_boxplot_color
from . import (
    METHODS,
    METHODNAMES,
    SAVENAMES,
    DATANAMES,
    DROPS,
    SUBTITLE_FONTSIZE,
    TEXT_FONTSIZE,
    LINE_WIDTH,
)

# poetry run python -m timpute.figures.figure4


def plot_entry_decomp(ax, data, name, drop=0.1):
    folder = f"timpute/figures/revision_cache/{data}/drop_{drop}/"
    impType = "entry"
    maxErr = 0
    for mID, m in enumerate(METHODS):
        run, _ = loadImputation(impType, m, folder)
        maxErr = max(maxErr, math.ceil(max(np.median(run.entry_total, 0)) * 10) / 10)

        comps = np.arange(1, run.entry_imputed.shape[1] + 1)
        imp_errbar = np.vstack(
            (
                abs(
                    np.percentile(run.entry_imputed, 25, 0)
                    - np.nanmedian(run.entry_imputed, 0)
                ),
                abs(
                    np.percentile(run.entry_imputed, 75, 0)
                    - np.nanmedian(run.entry_imputed, 0)
                ),
            )
        )
        ax.errorbar(
            comps,
            np.median(run.entry_imputed, 0),
            yerr=imp_errbar,
            ls="dashed",
            color=rgbs(mID, 0.7),
            alpha=0.5,
            lw=LINE_WIDTH,
        )
        # e[-1][0].set_linestyle('dashed')

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
        ax.errorbar(
            comps,
            np.median(run.entry_total, 0),
            yerr=total_errbar,
            label=label,
            ls="solid",
            color=rgbs(mID, 0.7),
            lw=LINE_WIDTH,
        )

        ax.tick_params(axis="both", which="major", labelsize=TEXT_FONTSIZE)
        ax.set_xticks(np.arange(max(comps) / 10, max(comps) + 1, max(comps) / 10))

    ax.errorbar([], [], label="Imputed Error", ls="dashed", color="black")
    ax.errorbar([], [], label="Total Error", ls="solid", color="black")
    ax.set_title(f"{name}", fontsize=SUBTITLE_FONTSIZE * 1.1)
    ax.set_xlabel("Number of Components", fontsize=SUBTITLE_FONTSIZE)
    ax.set_ylabel("Error", fontsize=SUBTITLE_FONTSIZE)
    ax.set_ylim(top=min(maxErr, 1), bottom=0)


def plot_chord_decomp(ax, data, name, drop=0.1):
    folder = f"timpute/figures/revision_cache/{data}/drop_{drop}/"
    impType = "chord"
    maxErr = 0
    for mID, m in enumerate(METHODS):
        run, _ = loadImputation(impType, m, folder)
        maxErr = max(maxErr, math.ceil(max(np.median(run.chord_total, 0)) * 10) / 10)

        comps = np.arange(1, run.chord_imputed.shape[1] + 1)
        imp_errbar = np.vstack(
            (
                abs(
                    np.percentile(run.chord_imputed, 25, 0)
                    - np.nanmedian(run.chord_imputed, 0)
                ),
                abs(
                    np.percentile(run.chord_imputed, 75, 0)
                    - np.nanmedian(run.chord_imputed, 0)
                ),
            )
        )
        ax.errorbar(
            comps,
            np.median(run.chord_imputed, 0),
            yerr=imp_errbar,
            ls="dashed",
            color=rgbs(mID, 0.7),
            alpha=0.5,
            lw=LINE_WIDTH,
        )
        # e[-1][0].set_linestyle('dashed')

        label = f"{METHODNAMES[mID]}"
        total_errbar = np.vstack(
            (
                abs(
                    np.percentile(run.chord_total, 25, 0)
                    - np.nanmedian(run.chord_total, 0)
                ),
                abs(
                    np.percentile(run.chord_total, 75, 0)
                    - np.nanmedian(run.chord_total, 0)
                ),
            )
        )
        ax.errorbar(
            comps,
            np.median(run.chord_total, 0),
            yerr=total_errbar,
            label=label,
            ls="solid",
            color=rgbs(mID, 0.7),
            lw=LINE_WIDTH,
        )

        ax.tick_params(axis="both", which="major", labelsize=TEXT_FONTSIZE)
        ax.set_xticks(np.arange(max(comps) / 10, max(comps) + 1, max(comps) / 10))

    ax.errorbar([], [], label="Imputed Error", ls="dashed", color="black")
    ax.errorbar([], [], label="Total Error", ls="solid", color="black")

    ax.set_title(f"{name}", fontsize=SUBTITLE_FONTSIZE * 1.1)
    ax.set_xlabel("Number of Components", fontsize=SUBTITLE_FONTSIZE)
    ax.set_ylabel("Error", fontsize=SUBTITLE_FONTSIZE)
    ax.set_ylim(top=min(maxErr, 1), bottom=0)


def best_imputed_boxplot(ax, impType, drop=0.1, datalist=SAVENAMES):
    plot_data = dict()
    comp_data = dict()
    for m in METHODS:
        plot_data[m.__name__] = list()
        comp_data[m.__name__] = list()

    for data in datalist:
        folder = f"timpute/figures/revision_cache/{data}/drop_{drop}/"
        for mID, m in enumerate(METHODS):
            run, _ = loadImputation(impType, m, folder)

            if impType == "entry":
                impMatrix = run.entry_imputed
            elif impType == "chord":
                impMatrix = run.chord_imputed
            else:
                raise ValueError(f"{impType} not a valid impType arg")

            impDist = impMatrix[:, np.median(impMatrix, axis=0).argmin()]
            plot_data[m.__name__].append(impDist)
            comp_data[m.__name__].append(np.median(impMatrix, axis=0).argmin())

    bar_spacing = 0.5
    bar_width = 0.4
    exp_spacing = 2
    for mID, m in enumerate(METHODS):
        box = ax.boxplot(
            plot_data[m.__name__],
            positions=np.array(range(len(plot_data[m.__name__]))) * exp_spacing
            - bar_spacing
            + bar_spacing * mID,
            sym="",
            widths=bar_width,
            boxprops=dict(linewidth=LINE_WIDTH - 1),
            medianprops=dict(linewidth=LINE_WIDTH - 1),
            whiskerprops=dict(linewidth=LINE_WIDTH - 1),
            flierprops=dict(markersize=LINE_WIDTH - 1),
        )
        set_boxplot_color(box, rgbs(mID))
        for i, line in enumerate([1, 3, 5, 7]):
            x = box["caps"][line].get_xdata().mean()
            y = box["caps"][line].get_ydata()[0]
            ax.text(
                x,
                y * 1.03,
                comp_data[m.__name__][i] + 1,
                ha="center",
                va="bottom",
                size=TEXT_FONTSIZE,
            )
    ax.tick_params(axis="both", which="major", labelsize=TEXT_FONTSIZE)
    ax.set_xticks(range(0, len(DATANAMES) * exp_spacing, exp_spacing), DATANAMES)

    ax.set_title(
        f"Best imputation error by dataset, {int(drop*100)}% {impType} masking",
        fontsize=SUBTITLE_FONTSIZE * 1.1,
    )
    ax.set_xlabel("Dataset", fontsize=SUBTITLE_FONTSIZE)
    ax.set_ylabel("Imputed Error", fontsize=SUBTITLE_FONTSIZE)
    ax.set_xlim(right=7)
    # ax.set_ylim(top=ax.get_ylim()[1]*1.5)
    ax.set_ylim(top=1, bottom=0.01)
    ax.set_yscale("log")


def figure4(datalist=SAVENAMES, legend=False):
    ax, f = getSetup((16, 8), (2, 4), multz={4: 1, 6: 1}, empts=[5, 7])

    # a-d) C-ALS outperforms other algorithms in many cases, but not all.
    # DO has stable imputed error at higher components
    # 10% missingness, component vs imputed & total error
    for i, data in enumerate(datalist):
        plot_entry_decomp(ax[i], data, DATANAMES[i])
        print(f"completed figure 4{chr(ord('a') + i)}")

    # e)-f) Best imputed error identifies best rank for each algorithm
    # C-ALS outperforms other algorithms in many cases, but not all (DyeDrop, BC cytokine, Covid chord drop)
    # component vs imputed & total error
    # e) 10% missingness, all datasets ENTRY
    # f) 10% missingness, all datasets CHORD

    # --- ENTRY ---
    best_imputed_boxplot(ax[4], "entry")
    print("completed figure 4e")

    # --- CHORD ---
    best_imputed_boxplot(ax[5], "chord")
    print("completed figure 4f")

    subplotLabel(ax)

    if legend is True:
        ax.legend(
            loc="lower right",
            handlelength=2,
            fontsize=TEXT_FONTSIZE,
            handles=[
                Line2D([0], [0], label=m, color=rgbs(i))
                for i, m in enumerate(METHODNAMES)
            ],
        )
        f.savefig(
            "timpute/figures/revision_img/svg/figure4_legend.svg",
            bbox_inches="tight",
            format="svg",
        )
        f.savefig(
            "timpute/figures/revision_img/figure4_legend.png",
            bbox_inches="tight",
            format="png",
        )
    else:
        f.savefig(
            "timpute/figures/revision_img/svg/figure4.svg",
            bbox_inches="tight",
            format="svg",
        )
        f.savefig(
            "timpute/figures/revision_img/figure4.png",
            bbox_inches="tight",
            format="png",
        )

    return f


def figure4_exp(datalist=["zohar", "alter", "hms", "coh_response"]):
    ax, f = getSetup((48, 40), (7, 8))

    # Figure 1, a)-d)
    for d, drop in enumerate(DROPS):
        for i, data in enumerate(datalist):
            for mID, m in enumerate(METHODS):
                ## ENTRY
                plot_entry_decomp(ax[i + d * 8], data, DATANAMES[i], drop)

                ## CHORD
                plot_chord_decomp(ax[i + 4 + d * 8], data, DATANAMES[i], drop)

    f.savefig(
        "timpute/figures/revision_img/svg/figure4-exp.svg",
        bbox_inches="tight",
        format="svg",
    )
    f.savefig(
        "timpute/figures/revision_img/figure4-exp.png",
        bbox_inches="tight",
        format="png",
    )


if __name__ == "__main__":
    print("\nbuilding figure 4...")
    # figure4(legend=True)
    figure4(legend=False)
    # figure4_exp()
