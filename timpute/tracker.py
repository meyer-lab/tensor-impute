import numpy as np
import time
import pickle
from .impute_helper import calcR2X

class tracker():
    """ Tracks next unfilled entry & runtime, holds tracked name for plotting """
        
    def __init__(self, tOrig = [0], mask=None, track_runtime=False):
        """ self.data should be the original tensor (e.g. prior to running imputation) """
        self.data = tOrig
        self.mask = mask   # mask represents untouched (1) vs dropped (0) entries
        self.track_runtime = track_runtime
        self.rep = 0
        self.total_error = [np.full((1, 0), 0)]
        self.fitted_error = [np.full((1, 0), 0)]
        self.imputed_error = [np.full((1, 0), 0)]
        if self.track_runtime:
            self.timer = [np.full((1, 0), 0)]
        
        self.combined = False

    def __call__(self, tFac, error=None):
        """ Takes a CP tensor object """
        self.total_error[self.rep] = np.append(self.fitted_error[self.rep], calcR2X(tFac, self.data, True))
        if self.mask is not None:
            self.fitted_error[self.rep] = np.append(self.fitted_error[self.rep], calcR2X(tFac, self.data, True, self.mask))
            self.imputed_error[self.rep] = np.append(self.imputed_error[self.rep], calcR2X(tFac, self.data, True, np.ones_like(self.data) - self.mask))
        if self.track_runtime:
            assert self.start
            self.timer[self.rep] = np.append(self.timer[self.rep], time.time() - self.start)

    def begin(self):
        """ Must call to track runtime """
        self.start = time.time()
    
    def set_mask(self, mask):
        self.mask = mask

    def new(self):
        """ Call to start tracking a repetition of the same method """
        self.total_error.append([np.full((1, 0), 0)])
        self.fitted_error.append([np.full((1, 0), 0)])
        self.imputed_error.append([np.full((1, 0), 0)])
        if self.track_runtime:
            self.timer.append([np.full((1, 0), 0)])
        self.rep += 1

    def reset(self):
        self.total_error = [np.full((1, 0), 0)]
        self.fitted_error = [np.full((1, 0), 0)]
        self.imputed_error = [np.full((1, 0), 0)]
        if self.track_runtime: self.timer = [np.full((1, 0), 0)]

    def combine(self, remove_outliers=False):
        """ Combines all runs into a single np.ndarray."""

        # in case any run doesn't hit maximum iterations, make them all the same size
        max = 0
        for i in range(self.rep+1):
            assert(self.fitted_error[i].size == self.fitted_error[i].size)
            current = self.fitted_error[i].size
            if remove_outliers and np.max(self.fitted_error[i]) > 1: self.fitted_error[i][:] = np.nan

            if (current > max):
                for j in range(i):
                    self.total_error[j] = np.append(self.total_error[j], np.full((current-max), np.nan))
                    self.fitted_error[j] = np.append(self.fitted_error[j], np.full((current-max), np.nan))
                    self.imputed_error[j] = np.append(self.imputed_error[j], np.full((current-max), np.nan))
                    if self.track_runtime: self.timer[j] = np.append(self.timer[j], np.full((current-max), np.nan))
                max = current
            else:
                self.total_error[i] = np.append(self.total_error[i], np.full((max-current), np.nan))
                self.fitted_error[i] = np.append(self.fitted_error[i], np.full((max-current), np.nan))
                self.imputed_error[i] = np.append(self.imputed_error[i], np.full((max-current), np.nan))
                if self.track_runtime: self.timer[i] = np.append(self.timer[i], np.full((max-current), np.nan))
        
        self.total_array = np.vstack(tuple(self.total_error))
        self.fitted_array = np.vstack(tuple(self.fitted_error))
        self.imputed_array = np.vstack(tuple(self.imputed_error))
        if self.track_runtime: self.time_array = np.vstack(tuple(self.timer))
        self.combined = True

    """ Plots are designed to track the error of the method for the highest rank imputation of tOrig """
    def plot_iteration(self, ax, methodname=None, grouped=True, showLegend=False,
                       rep=None, plot_total=False, offset=0,
                       log=True, logbound=-3.5, color='blue'):
        if not self.combined: self.combine()
        if grouped:
            imputed_errbar = np.vstack((-(np.percentile(self.imputed_array,25,0) - np.nanmedian(self.imputed_array,0)),
                                        np.percentile(self.imputed_array,75,0) - np.nanmedian(self.imputed_array,0),))
            fitted_errbar = np.vstack((-(np.percentile(self.fitted_array,25,0) - np.nanmedian(self.fitted_array,0)),
                                       np.percentile(self.fitted_array,75,0) - np.nanmedian(self.fitted_array,0)))

            e1 = ax.errorbar(np.arange(self.imputed_array.shape[1])+0.1-offset*0.1+1, np.nanmedian(self.imputed_array,0), label=f"{methodname} Imputed Error", color=color,
                             yerr = imputed_errbar, ls='--', errorevery=5)
            e2 = ax.errorbar(np.arange(self.fitted_array.shape[1])+0.1-offset*0.1+1, np.nanmedian(self.fitted_array,0), label=f"{methodname} Fitted Error", color=color,
                             yerr = fitted_errbar, errorevery=(1,5))
            e1[-1][0].set_linestyle('--')
            # e2[-1][0].set_linestyle('dotted')

            if plot_total:
                total_errbar = np.vstack((-(np.percentile(self.total_array,25,0) - np.nanmedian(self.total_array,0)),
                                          np.percentile(self.total_array,75,0) - np.nanmedian(self.total_array,0)))
                e3 = ax.errorbar(np.arange(self.total_array.shape[1]), np.nanmedian(self.total_array,0), label=f"{methodname} Total Error", color=color,
                                 yerr=total_errbar, ls = 'dotted', errorevery=5)
                # e3[-1][0].set_linestyle('dotted')
        elif rep == None: pass

        if not showLegend: ax.legend().remove()
        ax.set_xlim((0, 52))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Error')
        if log:
            ax.set_yscale("log")
            ax.set_ylim(10**logbound,1)
        else:
            ax.set_ylim(0,1)
    
    def time_thresholds(self, threshold = 0.25):
        if not self.combined: self.combine()
        thres_arr = self.imputed_array <= threshold
        met_rows = np.argwhere(np.sum(thres_arr, axis=1)).flatten()
        thres_arr = thres_arr[met_rows,:]

        thres_iter = np.argmax(thres_arr, axis=1)
        thres_time = [self.time_array[i,thres_iter[ID]] for ID,i in enumerate(met_rows)]
        return thres_time
    
    def unmet_thresholds(self, threshold = 0.25):
        if not self.combined: self.combine()
        return np.sum(np.amin(self.imputed_array,axis=1) > threshold)


    def save(self, pfile):
        with open(pfile, "wb") as output_file:
            pickle.dump(self.__dict__, output_file)

    def load(self, pfile):
        with open(pfile, "rb") as input_file:
            tmp_dict = pickle.load(input_file)
            self.__dict__.update(tmp_dict)