import numpy as np
import time
import pickle
from .cmtf import calcR2X
import matplotlib.patches as mpatches

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

    def combine(self, remove_outliers=False):
        """ Combines all runs into a single np.ndarray. MUST RUN BEFORE PLOTTING """
        max = 0

        # in case any run doesn't hit maximum iterations, make them all the same size
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

    def reset(self):
        self.total_error = [np.full((1, 0), 0)]
        self.fitted_error = [np.full((1, 0), 0)]
        self.imputed_error = [np.full((1, 0), 0)]
        if self.track_runtime: self.timer = [np.full((1, 0), 0)]


    """ Plots are designed to track the error of the method for the highest rank imputation of tOrig """
    def plot_iteration(self, ax, methodname='Method', grouped=True,
                       rep=None, plot_total=False, offset=0,
                       log=True, logbound=-3.5, color='blue'):
        if not self.combined: self.combine()
        if grouped:
            fitted_errbar = [np.percentile(self.fitted_array,25,0),np.percentile(self.fitted_array,75,0)]
            imputed_errbar = [np.percentile(self.imputed_array,25,0),np.percentile(self.imputed_array,75,0)]
            e1 = ax.errorbar(np.arange(self.imputed_array.shape[1])-0.5, np.nanmedian(self.imputed_array,0), color=color,
                             yerr=imputed_errbar, label=methodname+' Imputed Error', ls='-.', errorevery=5)
            e2 = ax.errorbar(np.arange(self.fitted_array.shape[1])+0.5, np.nanmedian(self.fitted_array,0), color=color,
                             yerr=fitted_errbar, label=methodname+' Fitted Error', errorevery=5)
            e1[-1][0].set_linestyle('dotted')
            e2[-1][0].set_linestyle('dotted')

            if plot_total:
                total_errbar = [np.percentile(self.total_array,25,0),np.percentile(self.total_array,75,0)]
                e3 = ax.errorbar(np.arange(self.total_array.shape[1]), np.nanmedian(self.total_array,0), color=color,
                                 yerr=total_errbar, label=methodname+' Total Error', ls = '--', errorevery=5)
                e3[-1][0].set_linestyle('dotted')

            ax.legend(loc='upper right')
        elif rep == None: pass

        ax.set_xlim((-0.5, self.fitted_array.shape[1]))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Error')
        if log:
            ax.set_yscale("log")
            ax.set_ylim(10**logbound,2)
        else:
            ax.set_ylim(0,1)

        #     for i in range(self.rep+1):
        #         ax.plot(np.arange(self.fitted_array.shape[1]), self.fitted_array[i], color='blue')
        #         ax.plot(np.arange(self.imputed_array.shape[1]), self.imputed_array[i], color='green')
        #         if plot_total: ax.plot(np.arange(self.total_array.shape[1]), self.total_array[i], color='red')
        #     leg1 = mpatches.Patch(color='blue', label=methodname+'Fitted Error')
        #     leg2 = mpatches.Patch(color='green', label=methodname+'Imputation Error')
        #     if plot_total: leg3 = mpatches.Patch(color='red', label=methodname+'Total Error')
        #     ax.legend(loc='upper right', handles=[leg1, leg2])
        # elif isinstance(rep, int):
        #     assert rep < self.rep + 1
        #     ax.plot(np.arange(self.fitted_array.shape[1]), self.fitted_array[rep-1], color='blue')
        #     ax.plot(np.arange(self.imputed_array.shape[1]), self.imputed_array[rep-1], color='green')
        #     leg1 = mpatches.Patch(color='blue', label=methodname+'Fitted Error'+str(rep))
        #     leg2 = mpatches.Patch(color='green', label=methodname+'Imputed Error'+str(rep))
        #     if plot_total:
        #         ax.plot(np.arange(self.total_array.shape[1]), self.total_array[i], color='red')
        #         leg3 = mpatches.Patch(color='red', label=methodname+'Total Error'+str(rep))
        #     ax.legend(loc='upper right', handles=[leg1, leg2])
    
    def time_thresholds(self, threshold = 0.25):
        if not self.combined: self.combine()
        thres_iter = np.argmax(self.imputed_array <= threshold, axis=1)
        thres_time = [self.time_array[i,thres_iter[i]] for i in range(self.imputed_array.shape[0])]
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