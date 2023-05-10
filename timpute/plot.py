"""
This file makes all standard plots for tensor analysis. Requires a Decomposition object after running relevant values.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from .decomposition import Decomposition
from matplotlib.lines import Line2D

def tfacr2x(ax, decomp:Decomposition):
    """
    Plots R2X for tensor factorizations for all components up to decomp.max_rr.

    Parameters
    ----------
    ax : axis object
        Plot information for a subplot of figure f. See getSetup() in tensorpack.test.common.py for more detail.
    decomp : Decomposition
        Takes a Decomposition object that has successfully run decomp.perform_tfac().
    """
    comps = decomp.rrs
    ax.scatter(comps, decomp.TR2X, s=10)
    ax.set_ylabel("Tensor Fac R2X")
    ax.set_xlabel("Number of Components")
    ax.set_title("Variance explained by tensor decomposition")
    ax.set_xticks([x for x in comps])
    ax.set_xticklabels([x for x in comps])
    ax.set_ylim(0, 1)
    ax.set_xlim(0.5, np.amax(comps) + 0.5)


def reduction(ax, decomp):
    """
    Plots size reduction for tensor factorization versus PCA for all components up to decomp.max_rr.

    Parameters
    ----------
    ax : axis object
        Plot information for a subplot of figure f.
    decomp : Decomposition
        Takes a Decomposition object that has successfully run decomp.perform_tfac() and decomp.perform_PCA().
    """
    CPR2X, PCAR2X, sizeTfac, sizePCA = np.asarray(decomp.TR2X), np.asarray(decomp.PCAR2X), decomp.sizeT, decomp.sizePCA
    ax.set_xscale("log", base=2)
    ax.plot(sizeTfac, 1.0 - CPR2X, ".", label="TFac")
    ax.plot(sizePCA, 1.0 - PCAR2X, ".", label="PCA")
    ax.set_ylabel("Normalized Unexplained Variance")
    ax.set_xlabel("Size of Reduced Data")
    ax.set_title("Data reduction, TFac vs. PCA")
    ax.set_ylim(bottom=0.0)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.legend()


def q2xchord(ax, decomp, methodname = "CP", detailed=True):
    """
    Plots Q2X for tensor factorization when removing chords from a single mode for all components up to decomp.max_rr.
    Requires multiple runs to generate error bars.

    Parameters
    ----------
    ax : axis object
        Plot information for a subplot of figure f.
    decomp : Decomposition
        Takes a Decomposition object that has successfully run decomp.Q2X_chord().
    methodname : str
        Name of method on graph
    detailed : bool
        Plots fitted and imputed error if True, compined Q2X if False
    """
    if not detailed: q2x_plot(ax, methodname, total_arr=decomp.chordQ2X, detailed=False)
    else: q2x_plot(ax, methodname, imputed_arr=decomp.imputed_chord_error, fitted_arr=decomp.fitted_chord_error, detail=True)


def q2xentry(ax, decomp, methodname = "CP", detailed=True):
    """
    Plots Q2X for tensor factorization versus PCA when removing entries for all components up to decomp.max_rr.
    Requires multiple runs to generate error bars.

    Parameters
    ----------
    ax : axis object
        Plot information for a subplot of figure f.
    decomp : Decomposition
        Takes a Decomposition object that has successfully run decomp.entry().
    methodname : str
        Allows for proper tensor method when naming graph axes. 
    """
    if not detailed: q2x_plot(ax, methodname, total_arr=decomp.entryQ2X, detailed=False)
    else: q2x_plot(ax, methodname, imputed_arr=decomp.imputed_entry_error, fitted_arr=decomp.fitted_entry_error, detail=True)


def q2x_plot(ax, methodname:str = None, imputed_arr:np.ndarray = None, fitted_arr:np.ndarray = None, total_arr:np.ndarray = None,
             detailed = True, plot_total = False, showLegend=False, offset = 0,
             log = True, logbound = -3.5, endbound=1, color='blue'):

    if not detailed:
        assert(total_arr is not None)
        comps = np.arange(1,total_arr.shape[1]+1)
        entry_df = pd.DataFrame(total_arr).T
        entry_df.index = comps
        entry_df['mean'] = entry_df.median(axis=1)
        entry_df['sem'] = entry_df.iqr(axis=1)
        TR2X = entry_df['mean']
        TErr = entry_df['sem']
        ax.plot(comps, TR2X, ".", label=methodname)
        ax.errorbar(comps - 0.05, TR2X, yerr=TErr, fmt='none', ecolor='b')
        ax.set_ylabel("Q2X of Entry Imputation")
    
    else:
        assert imputed_arr is not None and fitted_arr is not None
        if plot_total: assert total_arr is not None
        comps = np.arange(1,imputed_arr.shape[1]+1)

        imputed_errbar = [np.percentile(imputed_arr,25,0),np.percentile(imputed_arr,75,0)]
        fitted_errbar = [np.percentile(fitted_arr,25,0),np.percentile(fitted_arr,75,0)]
        label = None
        if showLegend: label = f"{methodname} Imputed Error" 
        e1 = ax.errorbar(comps+0.025+offset*0.05, np.median(imputed_arr,0), label=label, yerr=imputed_errbar, fmt='^', color=color)
        if showLegend: label = f"{methodname} Fitted Error"
        e2 = ax.errorbar(comps-0.125+offset*0.05, np.median(fitted_arr,0), label=label, yerr=fitted_errbar, fmt='.', color=color)
        e1[-1][0].set_linestyle('-.')
        # e2[-1][0].set_linestyle('-.')

        if plot_total:
            total_errbar = [np.percentile(total_arr,25,0),np.percentile(total_arr,75,0)]
            if showLegend: label = f"{methodname} Fitted Error"
            e3 = ax.errorbar(comps, np.median(total_arr,0), yerr=total_errbar,label=label, fmt='D', color=color)
            e3[-1][0].set_linestyle('.')
        
        
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Imputation Error")
    ax.set_xticks([x for x in comps])
    ax.set_xticklabels([x for x in comps])

    if log:
        ax.set_yscale("log")
        ax.set_ylim(10**logbound,endbound)
    else:
        ax.set_ylim(0,1)


def l2_plot(ax, decomp, alpha, methodname = "CP", comp=None):
    """ Plots the error for a CLS factorization at a given component number with a given regularization term alpha.
    Must be run multiple times to graph every alpha level of interest

    """

    if comp is None: comp = max(decomp.rrs)
    imputed_df = pd.DataFrame(decomp.imputed_entry_error[:,comp])
    fitted_df = pd.DataFrame(decomp.fitted_entry_error[:,comp])

    fitted_df.index = alpha
    imputed_mean = imputed_df.mean()
    imputed_sem = imputed_df.sem()
    ax.scatter(alpha + 0.05, imputed_mean, color='C0', s=10, label=methodname+' Imputed Error')
    ax.errorbar(alpha + 0.05, imputed_mean, yerr=imputed_sem, fmt='none', ecolor='C0')
    
    fitted_df.index = alpha
    fitted_mean = fitted_df.mean()
    fitted_sem = fitted_df.sem()
    ax.scatter(alpha, fitted_mean, color='C1', s=10, label=methodname+' Fitted Error')
    ax.errorbar(alpha, fitted_mean, yerr=fitted_sem, fmt='none', ecolor='C1')

    ax.set_xlabel("L2 Regularization Term")
    ax.set_ylabel("Imputation Error at Component "+str(comp))
    ax.xscale("log")
    ax.yscale("log")
    ax.legend(loc="upper right")

def tucker_reduced_Dsize(tensor, ranks:list):
    """ Output the error (1 - r2x) for each size of the data at each component # for tucker decomposition.
    This forms the x-axis of the error vs. data size plot.

    Parameters
    ----------
    tensor : xarray or numpy.ndarray
        the multi-dimensional input data
    ranks : list
        the list of minimum-error Tucker fits for each component-combinations.

    Returns
    -------
    sizes : list
        the size of reduced data by Tucker for each error.
    """
    # if tensor is xarray...
    if type(tensor) is not np.ndarray:
        tensor = tensor.to_numpy()

    sizes = []
    for rank in ranks:
        sum_comps = 0
        for i in range(len(tensor.shape)):
            sum_comps += rank[i] * tensor.shape[i]
        sizes.append(sum_comps)

    return sizes

def tucker_reduction(ax, decomp:Decomposition, cp_decomp:Decomposition):
    """ Error versus data size for minimum error combination of rank from Tucker decomposition versus CP decomposition.
    The error for those combinations that are the same dimensions, ie., for a 3-D tensor, [1, 1, 1], [2, 2, 2], etc
    are shown by a different marker shape and color.
    
    Parameters
    ----------
    ax : axis object
        Plot information for a subplot of figure f.
    decomp : Decomposition
        Takes a Decomposition object to run perform_tucker().
    cp_decomp : Decomposition
        Takes a Decomposition object to run perform_CP().

    Example
    -------
    from tensorpack.tucker import tucker_decomp
    from tensorpack.plot import tucker_reduced_Dsize, tucker_reduction
    from tensordata.zohar import data3D as zohar
    from tensorpack.decomposition import Decomposition
    b = Decomposition(zohar().tensor, method=tucker_decomp)
    c = Decomposition(zohar().tensor)
    import matplotlib.pyplot as plt
    f = plt.figure()
    ax = f.add_subplot()
    fig = tucker_reduction(ax, b, c)
    plt.savefig("tucker_cp.svg")
    """
    # tucker decomp
    decomp.perform_tucker()
    sizes = tucker_reduced_Dsize(decomp.data, decomp.TuckRank)

    # CP decomp
    cp_decomp.perform_tfac()
    CPR2X, sizeTfac = np.asarray(cp_decomp.TR2X), cp_decomp.sizeT

    ax.plot(sizes, decomp.TuckErr, label="Tucker", color='C0', lw=3)
    ax.plot(sizeTfac, 1.0 - CPR2X, ".", label="CP", color='C1', markersize=12)
    ax.set_ylim((0.0, 1.0))
    ax.set_xscale("log", base=2)
    ax.set_title('Data Reduction Comparison')
    ax.set_ylabel('Normalized Unexplained Variance')
    ax.set_xlabel('Size of Reduced Data')
    ax.legend()

def plot_weight_mode(ax, factor, labels=False, title = ""):
    """
    Plots heatmaps for a single mode factors.

    Parameters
    ----------
    ax : axis object
        Plot information for a subplot of figure f.
    factor: numpy array
        Factorized mode
    labels: list of string or False
        Labels for each of the elements
    title" String
        Figure title
    """
    rank = np.shape(factor)[1]
    components = [str(ii + 1) for ii in range(rank)]
    facs = pd.DataFrame(factor, columns=[f"Cmp. {i}" for i in np.arange(1, rank + 1)],
                        index=labels if labels is not False else list(range(np.shape(factor)[0])))

    sns.heatmap(facs, cmap="PiYG", center=0, xticklabels=components, yticklabels=labels, cbar=True, vmin=-1.0,
                vmax=1.0, ax=ax)

    ax.set_xlabel("Components")
    ax.set_title(title)
