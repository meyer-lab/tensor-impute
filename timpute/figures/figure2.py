import numpy as np
from sklearn.metrics import root_mean_squared_error as rmse

from . import DROPS, LINE_WIDTH, METHODNAMES, METHODS, SUBTITLE_FONTSIZE, TEXT_FONTSIZE
from .common import getSetup, rgbs
from .figure_data import bestSimComps
from .figure_helper import loadImputation

# poetry run python -m timpute.figures.figure2


def plot_simulated_data(ax, rank, drop, impType="entry", methods=METHODS):
    folder = f"timpute/figures/revision_cache/simulated/rank_{rank}/drop_{drop}/"
    ax.axvline(rank, color="black")

    for mID, m in enumerate(methods):
        run, _ = loadImputation(impType, m, folder)

        comp_info = bestSimComps(drop=drop, impType=impType)[rank]
        ax.axvline(comp_info[METHODNAMES[mID]] - 0.05 + mID * 0.05, color=rgbs(mID))

        if impType == "entry":
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
            ax.errorbar(
                comps,
                np.median(run.entry_total, 0),
                yerr=total_errbar,
                label=label,
                ls="solid",
                color=rgbs(mID, 0.7),
                alpha=0.5,
                lw=LINE_WIDTH,
            )

            label = f"{METHODNAMES[mID]} Imputed"
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
            e = ax.errorbar(
                comps,
                np.median(run.entry_imputed, 0),
                yerr=imp_errbar,
                label=label,
                ls="dashed",
                color=rgbs(mID, 0.7),
                alpha=0.5,
                lw=LINE_WIDTH,
            )
            e[-1][0].set_linestyle("dashed")

        elif impType == "chord":
            comps = np.arange(1, run.chord_imputed.shape[1] + 1)
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
                alpha=0.5,
                lw=LINE_WIDTH,
            )

            label = f"{METHODNAMES[mID]} Imputed"
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
            e = ax.errorbar(
                comps,
                np.median(run.chord_imputed, 0),
                yerr=imp_errbar,
                label=label,
                ls="dashed",
                color=rgbs(mID, 0.7),
                alpha=0.5,
                lw=LINE_WIDTH,
            )
            e[-1][0].set_linestyle("dashed")

        else:
            raise ValueError(f"{impType} is not a valid impType arg")

    ax.set_title(
        f"True Rank {rank} dataset, {int(drop*100)} {impType} masking",
        fontsize=SUBTITLE_FONTSIZE * 1.1,
    )
    ax.set_xlabel("Number of Components", fontsize=SUBTITLE_FONTSIZE)
    ax.set_ylabel("Error", fontsize=SUBTITLE_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TEXT_FONTSIZE)
    ax.set_xticks([x for x in comps if x % 5 == 0])
    ax.set_xticklabels([x for x in comps if x % 5 == 0])
    if ax.get_ylim()[1] > 1:
        ax.set_ylim(bottom=ax.get_ylim()[0], top=1)

    return


def figure2():
    ax, f = getSetup((24, 12), (2, 4))

    # a) imputation framework is a good estimate for determining true rank
    # component vs imputed & total error
    # i) 10% missingness true rank 10 ENTRY
    # ii) 10% missingness true rank 10 CHORD
    plot_simulated_data(ax[0], 10, 0.1, "entry")
    print("completed figure 2a")
    plot_simulated_data(ax[4], 20, 0.5, "chord")
    print("completed figure 2b")

    # c) as true rank increases, best imputed rank underestimates true rank
    # scatter plot, per method - best rank vs true rank
    # line of best fit, per method
    # i) 10% missingness all true ranks ENTRY
    # ii) 10% missingness all true ranks CHORD

    for n, impType in enumerate(["entry", "chord"]):
        comp_info = bestSimComps(drop=0.1, impType=impType)
        plot_limits = (4.7, 25.3)
        ax[n * 4 + 1].plot(
            np.linspace(*plot_limits),
            np.linspace(*plot_limits),
            color="black",
            lw=LINE_WIDTH,
        )

        for mID, m in enumerate(METHODNAMES):
            true_ranks = np.arange(5, 25 + 1)
            best_imputed_ranks = [
                comp_info[rank][m] - 0.05 + mID * 0.05 for rank in true_ranks
            ]
            ax[n * 4 + 1].scatter(
                true_ranks,
                best_imputed_ranks,
                color=rgbs(mID),
                alpha=0.7,
                label=(
                    "RMSE ="
                    f" {round(rmse(true_ranks, best_imputed_ranks),4)}"
                ),
            )

        ax[n * 4 + 1].legend()
        ax[n * 4 + 1].set_title(
            f"True vs Best Imputed Rank\n10% {impType} masking",
            fontsize=SUBTITLE_FONTSIZE * 1.1,
        )
        ax[n * 4 + 1].set_xlabel("True Rank", fontsize=SUBTITLE_FONTSIZE)
        ax[n * 4 + 1].set_ylabel("Best Imputed Rank", fontsize=SUBTITLE_FONTSIZE)
        ax[n * 4 + 1].set_xlim(*plot_limits)
        ax[n * 4 + 1].set_ylim(*plot_limits)
        ax[n * 4 + 1].set_xticks([int(x) for x in true_ranks if x % 5 == 0])
        ax[n * 4 + 1].set_xticklabels([int(x) for x in true_ranks if x % 5 == 0])
        ax[n * 4 + 1].set_yticks([int(x) for x in true_ranks if x % 5 == 0])
        ax[n * 4 + 1].set_yticklabels([int(x) for x in true_ranks if x % 5 == 0])

    print("completed figure 2c")

    # d) as artifical missingness increases, best imputed rank increasingly underestimates #noqa E501
    # scatter plot, per method - best rank vs true rank
    # i) all missingness true rank 25 ENTRY
    # ii) all missingness true rank 25 CHORD

    for n, impType in enumerate(["entry", "chord"]):
        rank = 25
        spacing = 2
        for mID, m in enumerate(METHODNAMES):
            best_imputed_ranks = [
                bestSimComps(drop=drop, impType=impType)[rank][m] for drop in DROPS
            ]

            ax[n * 4 + 2].plot(
                np.arange(len(DROPS)) * 2,
                best_imputed_ranks,
                color=rgbs(mID),
                lw=LINE_WIDTH,
            )

        ax[n * 4 + 2].tick_params(axis="both", which="major", labelsize=TEXT_FONTSIZE)
        ax[n * 4 + 2].set_xticks(range(0, len(DROPS) * spacing, spacing), DROPS)

        ax[n * 4 + 2].set_title(
            f"Best Imputed Rank by {(impType.capitalize())} Masking Percentage\nTrue"
            " Rank 25",
            fontsize=SUBTITLE_FONTSIZE * 1.1,
        )
        ax[n * 4 + 2].set_xlabel("Best Imputed Rank", fontsize=SUBTITLE_FONTSIZE)
        ax[n * 4 + 2].set_ylabel("Masking Percentage", fontsize=SUBTITLE_FONTSIZE)

        ax[n * 4 + 2].axhline(
            rank,
            *ax[n * 4 + 2].set_xlim(),
            color="black",
            lw=LINE_WIDTH,
        )

    print("completed figure 2d")

    # e) C-ALS generally performs best
    # i) bargraph of # of cases each method is a) correct (ENTRY & CHORD)
    # ii) bargraph of # of cases each method is a) closest/tied (ENTRY & CHORD)

    correct_comp = dict()
    closest_comp = dict()
    ALL_RANKS = np.arange(5, 25 + 1)

    for n, impType in enumerate(["entry", "chord"]):
        for m in METHODNAMES:
            correct_comp[m] = 0
            closest_comp[m] = 0

        for drop in DROPS:
            comp_info = bestSimComps(drop=drop, impType=impType)

            for rank in ALL_RANKS:
                # finds the distance of best imputed from true rank
                closest = np.abs(np.array(list(comp_info[rank].values())) - rank).min()

                for m in METHODNAMES:
                    if comp_info[rank][m] == rank:
                        correct_comp[m] += 1
                    if np.abs(comp_info[rank][m] - rank) == closest:
                        closest_comp[m] += 1

        for m in METHODNAMES:
            correct_comp[m] = correct_comp[m] / (len(DROPS) * len(ALL_RANKS))
            closest_comp[m] = closest_comp[m] / (len(DROPS) * len(ALL_RANKS))

        bar_width = 0.4
        group_spacing = 2

        ax[n * 4 + 3].bar(
            np.arange(3) * bar_width + group_spacing * 0,
            [correct_comp[m] for m in correct_comp],
            width=bar_width,
            color=[rgbs(i) for i in range(3)],
        )

        ax[n * 4 + 3].bar(
            np.arange(3) * bar_width + group_spacing * 1,
            [closest_comp[m] for m in closest_comp],
            width=bar_width,
            color=[rgbs(i) for i in range(3)],
        )

        ax[n * 4 + 3].tick_params(axis="both", which="major", labelsize=TEXT_FONTSIZE)
        ax[n * 4 + 3].set_xticks(
            np.arange(2) * group_spacing + bar_width, ["Correct Rank", "Closest Rank"]
        )
        ax[n * 4 + 3].set_title(
            "Best Imputed Rank\nAll-Case Correctness & Closeness",
            fontsize=SUBTITLE_FONTSIZE * 1.1,
        )
        ax[n * 4 + 3].set_ylabel("Percentage", fontsize=SUBTITLE_FONTSIZE)

    print("completed figure 2e")

    # SAVE FIGURE
    # plt.tight_layout(pad=2)
    f.set_constrained_layout_pads()

    f.savefig(
        "timpute/figures/revision_img/svg/figure2.svg",
        bbox_inches="tight",
        format="svg",
    )

    f.savefig(
        "timpute/figures/revision_img/figure2.png",
        bbox_inches="tight",
        format="png",
    )

    return f


def figure2_exp(ranks=(5, 10, 15, 20, 25)):
    ax, f = getSetup((60, 36), (6, 10))

    # Figure 2, a)-d)
    for d, drop in enumerate(DROPS):
        for i, rank in enumerate(ranks):
            plot_simulated_data(ax[i + d * 10], rank, drop, "entry")
            plot_simulated_data(ax[i + 5 + d * 10], rank, drop, "chord")

    # f.savefig(
    #     "timpute/figures/revision_img/svg/figure2-exp.svg",
    #     bbox_inches="tight",
    #     format="svg",
    # )

    f.savefig(
        "timpute/figures/revision_img/figure2-exp.png",
        bbox_inches="tight",
        format="png",
    )

    return f


if __name__ == "__main__":
    print("\nbuilding figure 2...")
    # figure2_exp()
    figure2()
