import pickle

import numpy as np
import pandas as pd

from ..decomposition import Decomposition
from ..tracker import Tracker
from . import DATANAMES, DROPS, METHODNAMES, METHODS, SAVENAMES
from .common import getSetup, rgbs
from .figure_data import bestComps, bestSimComps

# poetry run python -m timpute.figures.supplements


def tableS1() -> None:
    # Supplemental 1
    DROPS = (0.05, 0.1, 0.2, 0.3, 0.4)

    # create dataframe
    col = [
        [f"{int(d * 100)}%" for d in DROPS for _ in METHODNAMES],
        METHODNAMES * len(DROPS),
    ]
    row = [
        sorted(SAVENAMES * 3 + ["coh_response"]),
        [i for j in [3, 4, 3, 3] for i in range(1, j + 1)],
    ]
    colIndex = pd.Series(np.zeros(len(col[0])), index=col).index
    rowIndex = pd.Series(np.zeros(len(row[0])), index=row).index
    df = pd.DataFrame(
        np.zeros((len(rowIndex), len(colIndex))), index=rowIndex, columns=colIndex
    )

    # fill dataframe
    data = Decomposition()
    for method in METHODNAMES:
        for s in SAVENAMES:
            for d in (0.05, 0.1, 0.2, 0.3, 0.4):
                modes = (0, 1, 2, 3) if (s == "coh_response") else (0, 1, 2)
                for m in modes:
                    data.load(
                        f"./timpute/figures/revision_cache/modeComparison/{s}\
                        /drop_{d}/mode_{m}/chord-perform_{method}.decomposition"
                    )
                    df.loc[(s, m + 1), (f"{int(d * 100)}%", method)] = np.min(
                        np.mean(data.chord_fitted, axis=0)
                    )

    # cleaning
    df = df.reindex([i for name in SAVENAMES for i in df.index if i[0] == name])
    df.index = df.index.set_levels(
        [DATANAMES[SAVENAMES.index(i)] for i in df.index.levels[0]], level=0
    )
    df.index = df.index.set_names(["Dataset", "Mode"])
    df = df.style.set_caption(
        "Chord Imputation by Mode per Dataset by Masking Percentage"
    )

    df.to_excel(
        "./timpute/figures/revision_img/chordModes.xlsx",
        sheet_name="Chord Imputation by Mode per Dataset by Masking Percentage",
    )


def figureS1() -> None:
    # Supplemental 2
    ax, f = getSetup((24, 12), (2, 4))
    width = 0.3
    ind = np.arange(4)

    for i, filename in enumerate([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):
        # print(filename)
        with open(
            "timpute/figures/revision_cache/dataUsage/" + str(filename) + ".pickle",
            "rb",
        ) as handle:
            memory = pickle.load(handle)
        for n, m in enumerate(METHODNAMES):
            ax[i].bar(
                ind + width * n,
                [np.median(memory[i][m]) for i in SAVENAMES],
                width,
                label=m,
                color=rgbs(n),
                edgecolor="black",
            )
            errs = np.array(
                [
                    (
                        abs(np.percentile(memory[i][m], 25) - np.median(memory[i][m])),
                        abs(np.percentile(memory[i][m], 25) - np.median(memory[i][m])),
                    )
                    for i in SAVENAMES
                ]
            )
            ax[i].errorbar(
                ind + width * n,
                [np.median(memory[i][m]) for i in SAVENAMES],
                yerr=errs.T,
                color="black",
                ls="none",
            )

        ax[i].set_xticks(ind + width, DATANAMES)
        ax[i].set_xlabel("Dataset")
        ax[i].set_ylabel("Peak RAM Usage")
        ax[i].legend()

    f.savefig(
        "timpute/figures/revision_img/svg/RAM_Usage.svg",
        bbox_inches="tight",
        format="svg",
    )


def tableS2_S3(imputed=True, outputData=False):
    # Supplemental 2/3
    df_list = []
    for i in ["entry", "chord"]:
        data = dict()
        for drop_perc in DROPS:
            data[drop_perc] = bestSimComps(
                drop=drop_perc,
                impType=i,
                first=5,
                last=25,
                imputed=imputed,
                outputData=outputData,
            )

        df = pd.DataFrame(
            columns=["true rank", "method"] + [f"{int(d * 100)}%" for d in DROPS]
        )

        for rank in np.arange(5, 25 + 1):
            for m in METHODNAMES:
                df.loc[len(df.index)] = [int(rank), m] + [
                    data[drop_perc][rank][m] for drop_perc in DROPS
                ]

        df = df.set_index(["true rank", "method"])
        df_list.append(df)

        if outputData is False:
            df.to_excel(
                f"./timpute/figures/revision_img/bestSimComps_{i}\
                    {'_total' if imputed is False else ''}.xlsx",
                sheet_name=(
                    f"Factorization Rank with Lowest Median Imputation Error, by\
                    {i} Masking Percentage"
                ),
            )
        else:
            df.to_excel(
                f"./timpute/figures/revision_img/bestSimError_{i}\
                    {'_total' if imputed is False else ''}.xlsx",
                sheet_name=f"Lowest Median Imputation Error, \
                    by {i} Masking Percentage",
            )

    return df_list


def tableS4_S5(imputed=True, outputData=False):
    # Supplemental 4/5
    df_list = []
    for i in ["entry", "chord"]:
        data = dict()
        for drop_perc in DROPS:
            data[drop_perc] = bestComps(
                drop=drop_perc, impType=i, imputed=imputed, datalist=SAVENAMES
            )

        df = pd.DataFrame(
            columns=["dataset", "method"] + [f"{int(d * 100)}%" for d in DROPS]
        )

        for n, name in enumerate(SAVENAMES):
            for m in METHODNAMES:
                df.loc[len(df.index)] = [DATANAMES[n], m] + [
                    data[drop_perc][name][m] for drop_perc in DROPS
                ]
        df = df.set_index(["dataset", "method"])
        df_list.append(df)
        if outputData is False:
            df.to_excel(
                f"./timpute/figures/revision_img/bestComps_{i}\
                    {'_total' if imputed is False else ''}.xlsx",
                sheet_name=(
                    "Factorization Rank with Lowest Median Imputation Error, by"
                    f" {i} Masking Percentage"
                ),
            )
        else:
            df.to_excel(
                f"./timpute/figures/revision_img/bestError_{i}\
                    {'_total' if imputed is False else ''}.xlsx",
                sheet_name=f"Lowest Median Imputation Error, by {i} Masking Percentage",
            )

    return df_list


def tableS6() -> None:
    impType = "chord"
    DROPS = (0.05, 0.1, 0.2, 0.3, 0.4)

    # create dataframe
    col = [
        [f"{int(d * 100)}%" for d in DROPS for _ in METHODNAMES],
        METHODNAMES * len(DROPS),
    ]
    row = [sorted(SAVENAMES * 2), ["entry", "chord"] * len(SAVENAMES)]
    colIndex = pd.Series(np.zeros(len(col[0])), index=col).index
    rowIndex = pd.Series(np.zeros(len(row[0])), index=row).index
    df_iter = pd.DataFrame(
        np.zeros((len(rowIndex), len(colIndex))), index=rowIndex, columns=colIndex
    )
    df_init = pd.DataFrame(
        np.zeros((len(rowIndex), len(colIndex))), index=rowIndex, columns=colIndex
    )

    # fill dataframe
    data = Tracker()
    for mID, m in enumerate(METHODS):
        for d in (0.05, 0.1, 0.2, 0.3, 0.4):
            rank_data = bestComps(d, impType, SAVENAMES)
            for s in SAVENAMES:
                for impType in ["entry", "chord"]:
                    data.load(
                        f"./timpute/figures/revision_cache/{s}/drop_{d}/{impType}-{m.__name__}.tracker"
                    )
                    rr = str(rank_data[s][METHODNAMES[mID]])

                    # take mean across samples of (ti+1 - ti) from 1 to maxIter
                    df_iter.loc[
                        (s, impType), (f"{int(d * 100)}%", METHODNAMES[mID])
                    ] = np.median(
                        np.nanmedian(np.diff(data.time_array[rr][:, 1:]), axis=0)
                    )
                    # take mean across samples of (t1 - t0)
                    df_init.loc[
                        (s, impType), (f"{int(d * 100)}%", METHODNAMES[mID])
                    ] = np.nanmedian(np.diff(data.time_array[rr][:, 0:2]))

    # cleaning
    df_iter = df_iter.reindex(
        [i for name in SAVENAMES for i in df_iter.index if i[0] == name]
    )
    df_iter.index = df_iter.index.set_levels(
        [DATANAMES[SAVENAMES.index(i)] for i in df_iter.index.levels[0]], level=0
    )

    # print(df_iter)

    df_iter.index = df_iter.index.set_names(["Dataset", "Imputation Type"])
    df_iter = df_iter.style.set_caption(
        "Median Time per Iteration for Optimal Rank Imputation"
    )

    df_init = df_init.reindex(
        [i for name in SAVENAMES for i in df_init.index if i[0] == name]
    )
    df_init.index = df_init.index.set_levels(
        [DATANAMES[SAVENAMES.index(i)] for i in df_init.index.levels[0]], level=0
    )
    df_init.index = df_init.index.set_names(["Dataset", "Imputation Type"])
    df_init = df_init.style.set_caption(
        "Median Time for Data Processing, Prior to First Iteration for Optimal Rank"
        " Imputation"
    )

    df_iter.to_excel(
        "./timpute/figures/revision_img/iterTime.xlsx",
        sheet_name="Median Time per Iteration for Optimal Rank Imputation",
    )
    df_init.to_excel(
        "./timpute/figures/revision_img/initTime.xlsx",
        sheet_name=(
            "Median Time for Data Processing, Prior to First Iteration for Optimal Rank"
            " Imputation"
        ),
    )


if __name__ == "__main__":
    print("\nbuilding supplements...")
    # print("building figure S1...")
    # figureS1()
    # print("building table S1...")
    # tableS1()
    print("building table S2 & S3...")
    tableS2_S3(False, True)
    print("building table S4 & S5...")
    tableS4_S5()
    print("building table S6...")
    tableS6()
