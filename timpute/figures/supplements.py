import pickle
from ..plot import *
from ..common import * 
import pandas as pd
from .figure_helper import *

# poetry run python -m timpute.figures.supplements

def figureS1() -> None:
    # Supplemental 1
    DROPS = (0.05, 0.1, 0.2, 0.3, 0.4)

    # create dataframe
    col = [[f'{int(d*100)}%' for d in DROPS for _ in METHODNAMES], METHODNAMES * len(DROPS)]
    row = [sorted(SAVENAMES*3+['coh_response']),[i for j in [3,4,3,3] for i in range(1,j+1)]]
    colIndex = pd.Series(np.zeros(len(col[0])), index=col).index
    rowIndex = pd.Series(np.zeros(len(row[0])), index=row).index
    df = pd.DataFrame(np.zeros((len(rowIndex),len(colIndex))), index=rowIndex, columns=colIndex)

    # fill dataframe 
    data = Decomposition()
    for method in METHODNAMES:
        for s in SAVENAMES:
            for d in (0.05, 0.1, 0.2, 0.3, 0.4):
                modes = (0,1,2,3) if (s == 'coh_response') else (0,1,2)
                for m in modes:
                    data.load(f'./timpute/figures/cache/modeComparison/{s}/drop_{d}/mode_{m}/chord-perform_{method}.decomposition')
                    df.loc[(s,m+1),(f'{int(d*100)}%',method)] = np.min(np.mean(data.chord_fitted,axis=0))

    # cleaning
    df = df.reindex([i for name in SAVENAMES for i in df.index if i[0] == name])
    df.index = df.index.set_levels([DATANAMES[SAVENAMES.index(i)] for i in df.index.levels[0]],level=0)
    df.index = df.index.set_names(['Dataset', 'Mode'])
    df = df.style.set_caption(f"Chord Imputation by Mode per Dataset by Masking Percentage")

    df.to_excel("./timpute/figures/img/chordModes.xlsx", sheet_name=f'Chord Imputation by Mode per Dataset by Masking Percentage')


def figureS2() -> None:
    # Supplemental 2
    ax, f = getSetup((24,12), (2,4))
    width = 0.3
    ind = np.arange(4)

    for i,filename in enumerate([0,0.05,0.1,0.2,0.3,0.4,0.5]):
        print(filename)
        with open('timpute/figures/cache/dataUsage/'+str(filename)+'.pickle', 'rb') as handle:
            memory = pickle.load(handle)
        for n,m in enumerate(METHODNAMES):
            ax[i].bar(ind + width*n, [np.median(memory[i][m]) for i in SAVENAMES], width, label=m, color=rgbs(n), edgecolor='black')
            errs = np.array([(abs(np.percentile(memory[i][m],25) - np.median(memory[i][m])),
                            abs(np.percentile(memory[i][m],25) - np.median(memory[i][m])))
                            for i in SAVENAMES])
            ax[i].errorbar(ind + width*n, [np.median(memory[i][m]) for i in SAVENAMES], yerr=errs.T, color='black', ls = 'none')

        ax[i].set_xticks(ind+width, DATANAMES)
        ax[i].set_xlabel("Dataset")
        ax[i].set_ylabel("Peak RAM Usage")
        ax[i].legend()

    f.savefig('timpute/figures/img/svg/RAM_Usage.svg', bbox_inches="tight", format='svg')


def figureS3() -> None:
    # Supplemental 3
    df_list = []
    for i in ['entry','chord']:
        with open(f'./timpute/figures/cache/bestComps_{i}.pickle', 'rb') as handle:
            data = pickle.load(handle)
        df = pd.DataFrame(columns=['dataset','method']+[f'{int(d*100)}%' for d in DROPS])
        for n,name in enumerate(SAVENAMES):
            for m in METHODNAMES:
                df.loc[len(df.index)] = [DATANAMES[n],m]+[data[d][name][m] for d in DROPS]
        df = df.set_index(['dataset', 'method'])
        # df = df.style.set_caption(f"Factorization Rank with Lowest Median Imputation Error, by {i} Masking Percentage")
        df_list.append(df)
        df.to_excel(f"./timpute/figures/img/bestComps_{i}.xlsx", sheet_name=f'Factorization Rank with Lowest Median Imputation Error, by {i} Masking Percentage')


if __name__ == "__main__":
    figureS1()
    # figureS2()
    # figureS3()