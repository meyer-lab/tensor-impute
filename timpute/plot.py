"""
This file makes all standard plots for tensor analysis. Requires a Decomposition object after running relevant values.
"""

import numpy as np
import pandas as pd
from .decomposition import Decomposition
from .tracker import Tracker
from .common import getSetup
import matplotlib.ticker as mtick

def rgbs(color = 0, transparency = None):
    color_rgbs = [(0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
                  (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
                  (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
                  (0.8352941176470589, 0.3686274509803922, 0.0),
                  (0.8, 0.47058823529411764, 0.7372549019607844),
                  (0.792156862745098, 0.5686274509803921, 0.3803921568627451),
                  (0.984313725490196, 0.6862745098039216, 0.8941176470588236),
                  (0.5803921568627451, 0.5803921568627451, 0.5803921568627451),
                  (0.9254901960784314, 0.8823529411764706, 0.2),
                  (0.33725490196078434, 0.7058823529411765, 0.9137254901960784)]
    if transparency is not None:
        preTran = list(color_rgbs[color])
        preTran.append(transparency)
        return tuple(preTran)
    else: return color_rgbs[color]


def q2xchord(ax, decomp:Decomposition, methodname = "CP", detailed=True):
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
    if not detailed: q2x_plot(ax, methodname, total_arr=decomp.chord_total, detailed=False)
    else: q2x_plot(ax, methodname, imputed_arr=decomp.chord_imputed, fitted_arr=decomp.chord_fitted, detailed=True)


def q2xentry(ax, decomp:Decomposition, methodname = "CP", detailed=True):
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
    if not detailed: q2x_plot(ax, methodname, total_arr=decomp.entry_total, detailed=False)
    else: q2x_plot(ax, methodname, imputed_arr=decomp.entry_imputed, fitted_arr=decomp.entry_fitted, detailed=True)


def q2x_plot(ax,
             methodname:str,
             imputed_arr:np.ndarray = None, fitted_arr:np.ndarray = None, total_arr:np.ndarray = None,
             detailed = True,
             plot_total = False,
             showLegend = False,
             offset = 0,
             log = True, logbound = -3.5, endbound = 1,
             color='blue', s = 5,
             printvalues = False):

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
        e1 = ax.errorbar(comps+offset, np.median(imputed_arr,0), label=f"{methodname} Imputed Error" , yerr=imputed_errbar, fmt='^', color=color, markersize=s)
        e2 = ax.errorbar(comps+offset, np.median(fitted_arr,0), label=f"{methodname} Fitted Error", yerr=fitted_errbar, fmt='.', color=color, markersize=s*2)
        if printvalues:
            print(np.median(imputed_arr,0))
            print(np.median(fitted_arr,0))
            print(np.median(total_arr,0))
        e1[-1][0].set_linestyle('solid')
        e1[-1][0].set_linestyle('solid')
        # e2[-1][0].set_linestyle('-.')

        if plot_total:
            total_errbar = np.vstack((-(np.percentile(total_arr,25,0) - np.nanmedian(total_arr,0)),
                                      np.percentile(total_arr,75,0) - np.nanmedian(total_arr,0)))
            e3 = ax.errorbar(comps, np.median(total_arr,0), yerr=total_errbar,label=f"{methodname} Fitted Error", fmt='D', color=color, markersize=s/2.5)
            e3[-1][0].set_linestyle('--')

    if showLegend: ax.legend(loc="upper right")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Error")
    ax.set_xticks([x for x in comps])
    ax.set_xticklabels([x for x in comps])

    if log:
        ax.set_yscale("log")
        ax.set_ylim(10**logbound,endbound)
    else:
        ax.set_ylim(0,1)


def iteration_plot(ax,
                   methodname:str,
                   tracker:Tracker,
                   plot_total=False, 
                   showLegend=False,
                   offset=0,
                   log=True, logbound=-3.5,
                   color='blue'):
    """ Plots are designed to track the error of the method for the highest rank imputation of tOrig """
    
    if not tracker.combined: tracker.combine()
    imputed_errbar = np.vstack((-(np.percentile(tracker.imputed_array,25,0) - np.nanmedian(tracker.imputed_array,0)),
                                np.percentile(tracker.imputed_array,75,0) - np.nanmedian(tracker.imputed_array,0),))
    fitted_errbar = np.vstack((-(np.percentile(tracker.fitted_array,25,0) - np.nanmedian(tracker.fitted_array,0)),
                                np.percentile(tracker.fitted_array,75,0) - np.nanmedian(tracker.fitted_array,0)))

    e1 = ax.errorbar(np.arange(tracker.imputed_array.shape[1])+0.1-offset*0.1+1, np.nanmedian(tracker.imputed_array,0), label=f"{methodname} Imputed Error", color=color,
                        yerr = imputed_errbar, ls='--', errorevery=5)
    e2 = ax.errorbar(np.arange(tracker.fitted_array.shape[1])+0.1-offset*0.1+1, np.nanmedian(tracker.fitted_array,0), label=f"{methodname} Fitted Error", color=color,
                        yerr = fitted_errbar, errorevery=(1,5))
    e1[-1][0].set_linestyle('--')
    # e2[-1][0].set_linestyle('dotted')

    if plot_total:
        total_errbar = np.vstack((-(np.percentile(tracker.total_array,25,0) - np.nanmedian(tracker.total_array,0)),
                                    np.percentile(tracker.total_array,75,0) - np.nanmedian(tracker.total_array,0)))
        e3 = ax.errorbar(np.arange(tracker.total_array.shape[1]), np.nanmedian(tracker.total_array,0), label=f"{methodname} Total Error", color=color,
                            yerr=total_errbar, ls = 'dotted', errorevery=5)
        # e3[-1][0].set_linestyle('dotted')

    if not showLegend: ax.legend().remove()
    ax.set_xlim((0, 52))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Error')
    if log:
        ax.set_yscale("log")
        ax.set_ylim(10**logbound,1)
    else:
        ax.set_ylim(0,1)


def runtime_plot(ax,
                 methodname:str,
                 tracker:Tracker,
                 threshold = 0.1,
                 timebound = (0,0.1),
                 color='blue',
                 printvalues=True):

    thresholds = tracker.time_thresholds(threshold)
    ax.hist(thresholds, label=f"{methodname} ({len(thresholds)})", fc=color, edgecolor=color, bins=50, range=timebound)
    if printvalues:
        print(f"mean: {np.mean(thresholds)}")
        print(f"max: {np.max(thresholds)}")
    ax.axvline(np.mean(thresholds), color=color, linestyle='dashed', linewidth=1)

    ax.legend(loc='upper right')
    ax.set_xlabel('Runtime')
    ax.set_ylabel('Count')


# def l2_plot(ax, decomp, alpha, methodname = "CP", comp=None):
#     """ Plots the error for a CLS factorization at a given component number with a given regularization term alpha.
#     Must be run multiple times to graph every alpha level of interest

#     """

#     if comp is None: comp = max(decomp.rrs)
#     imputed_df = pd.DataFrame(decomp.imputed_entry_error[:,comp])
#     fitted_df = pd.DataFrame(decomp.fitted_entry_error[:,comp])

#     fitted_df.index = alpha
#     imputed_mean = imputed_df.mean()
#     imputed_sem = imputed_df.sem()
#     ax.scatter(alpha + 0.05, imputed_mean, color='C0', s=10, label=methodname+' Imputed Error')
#     ax.errorbar(alpha + 0.05, imputed_mean, yerr=imputed_sem, fmt='none', ecolor='C0')
    
#     fitted_df.index = alpha
#     fitted_mean = fitted_df.mean()
#     fitted_sem = fitted_df.sem()
#     ax.scatter(alpha, fitted_mean, color='C1', s=10, label=methodname+' Fitted Error')
#     ax.errorbar(alpha, fitted_mean, yerr=fitted_sem, fmt='none', ecolor='C1')

#     ax.set_xlabel("L2 Regularization Term")
#     ax.set_ylabel("Imputation Error at Component "+str(comp))
#     ax.xscale("log")
#     ax.yscale("log")
#     ax.legend(loc="upper right")
