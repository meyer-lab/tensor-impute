"""
This file makes all standard plots for tensor analysis. Requires a Decomposition object after running relevant values.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from .decomposition import Decomposition
from matplotlib.lines import Line2D

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
             detailed = True, plot_total = False, showLegend = False, offset = 0,
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

        imputed_errbar = np.vstack(( -(np.percentile(imputed_arr,25,0) - np.nanmedian(imputed_arr,0)),
                                   np.percentile(imputed_arr,75,0) - np.nanmedian(imputed_arr,0) ))
        fitted_errbar = np.vstack((-(np.percentile(fitted_arr,25,0) - np.nanmedian(fitted_arr,0)),
                                   np.percentile(fitted_arr,75,0) - np.nanmedian(fitted_arr,0)))
        e1 = ax.errorbar(comps+0.025+offset*0.05, np.median(imputed_arr,0), label=f"{methodname} Imputed Error" , yerr=imputed_errbar, fmt='^', color=color)
        e2 = ax.errorbar(comps-0.125+offset*0.05, np.median(fitted_arr,0), label=f"{methodname} Fitted Error", yerr=fitted_errbar, fmt='.', color=color)
        print(np.median(imputed_arr,0))
        print(np.median(fitted_arr,0))
        print(np.median(total_arr,0))
        print("\n")
        e1[-1][0].set_linestyle('-.')
        # e2[-1][0].set_linestyle('-.')

        if plot_total:
            total_errbar = np.vstack((-(np.percentile(total_arr,25,0) - np.nanmedian(total_arr,0)),
                                      np.percentile(total_arr,75,0) - np.nanmedian(total_arr,0)))
            e3 = ax.errorbar(comps, np.median(total_arr,0), yerr=total_errbar,label=f"{methodname} Fitted Error", fmt='D', color=color)
            e3[-1][0].set_linestyle('--')

    # else:
    #     assert imputed_arr is not None and fitted_arr is not None
    #     if plot_total: assert total_arr is not None
    #     ax.errorbar(comps+0.1, np.median(imputed_arr,0), label=f"{methodname} Imputed Error", fmt='^', markersize=6, color=color)
    #     ax.errorbar(comps-0.1, np.median(fitted_arr,0), label=f"{methodname} Fitted Error", fmt='o', markersize=5, color=color)
    #     print(np.median(imputed_arr,0))
    #     print(np.median(fitted_arr,0))
    #     print(np.median(total_arr,0))
    #     print("\n")

    #     if plot_total:
    #         e3 = ax.errorbar(comps, np.median(total_arr,0), label=f"{methodname} Fitted Error", fmt='D', markersize=10, color=color)
        
    if not showLegend: ax.legend().remove()
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Error")
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
