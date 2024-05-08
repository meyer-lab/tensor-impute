import pickle
from ..plot import *
from ..common import * 
import pandas as pd
from .runImputation import *

# poetry run python -m timpute.figures.supplements

if __name__ == "__main__":
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

