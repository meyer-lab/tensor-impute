import numpy as np
import time
import pickle
from .cmtf import calcR2X
import matplotlib.patches as mpatches

def calc_impute_error(mask, data, tFac):
    if mask is not None:
        assert mask.all() == False, "Mask indicates no removed entries"
        tensorImp = np.copy(data)
        tensorImp[mask] = np.nan
        return calcR2X(tFac, tensorImp, calcError=True)
    else:
        return np.nan

class tracker():
    """
    Tracks next unfilled entry & runtime, holds tracked name for plotting
    """
        
    def __init__(self, tOrig, mask=None, track_runtime=False):
        """ self.data should be the original tensor (e.g. prior to running imputation) """
        self.data = tOrig
        self.mask = mask
        self.track_runtime = track_runtime
        self.rep = 0
        self.fitted_error = [np.full((1, 0), 0)]
        self.impute_error = [np.full((1, 0), 0)]
        if self.track_runtime:
            self.timer = [np.full((1, 0), 0)]

    def __call__(self, tFac, error=None):
        """ Takes a CP tensor object """
        error = None # Issue with how error is currently calculated in tensorly
        if error is None:
            if self.mask is not None: # Assure error is calculated with non-removed values 
                mask_data = np.copy(self.data)
                mask_data[~self.mask] = np.nan
                error = calcR2X(tFac, mask_data, calcError=True)
            else:
                error = calcR2X(tFac, self.data, calcError=True)
        self.fitted_error[self.rep] = np.append(self.fitted_error[self.rep], error)
        self.impute_error[self.rep] = np.append(self.impute_error[self.rep], calc_impute_error(self.mask, self.data, tFac))
            
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
        self.fitted_error.append([np.full((1, 0), 0)])
        self.impute_error.append([np.full((1, 0), 0)])
        if self.track_runtime:
            self.timer.append([np.full((1, 0), 0)])
        self.rep += 1

    def combine(self):
        """ Combines all runs into a single np.ndarray. MUST RUN BEFORE PLOTTING """
        max = 0
        for i in range(self.rep+1):
            assert(self.fitted_error[i].size == self.fitted_error[i].size)
            current = self.fitted_error[i].size
            if (current > max):
                for j in range(i):
                    self.fitted_error[j] = np.append(self.fitted_error[j], np.full((current-max), np.nan))
                    self.impute_error[j] = np.append(self.impute_error[j], np.full((current-max), np.nan))
                    if self.track_runtime: self.timer[j] = np.append(self.timer[j], np.full((current-max), np.nan))
                max = current
            else:
                self.fitted_error[i] = np.append(self.fitted_error[i], np.full((max-current), np.nan))
                self.impute_error[i] = np.append(self.impute_error[i], np.full((max-current), np.nan))
                if self.track_runtime: self.timer[i] = np.append(self.timer[i], np.full((max-current), np.nan))
                
        
        self.fitted_array = np.vstack(tuple(self.fitted_error)) 
        self.impute_array = np.vstack(tuple(self.impute_error))    
        if self.track_runtime: self.time_array = np.vstack(tuple(self.timer))

    def reset(self):
        self.fitted_error = [np.full((1, 0), 0)]
        self.impute_error = [np.full((1, 0), 0)]
        if self.track_runtime: self.timer = [np.full((1, 0), 0)]

    """ Plots are designed to track the error of the method for the highest rank imputation of tOrig """
    def plot_iteration(self, ax, methodname='Method', average=True, rep=None):
        if average:
            ax.plot(range(self.fitted_array.shape[1]), np.nanmean(self.fitted_array,0), label=methodname+' Fitted Error')
            ax.plot(range(self.impute_array.shape[1]), np.nanmean(self.impute_array,0), label=methodname+' Imputation Error')
            ax.legend(loc='upper right')
        elif rep == None:
            for i in range(self.rep+1):
                ax.plot(range(self.fitted_array.shape[1]), self.fitted_array[i], color='blue')
                ax.plot(range(self.impute_array.shape[1]), self.impute_array[i], color='green')
            leg1 = mpatches.Patch(color='blue', label=methodname+'Fitted Error')
            leg2 = mpatches.Patch(color='green', label=methodname+'Imputation Error')
            ax.legend(loc='upper right', handles=[leg1, leg2])
        else:
            assert rep < self.rep + 1
            ax.plot(range(self.fitted_array.shape[1]), self.fitted_array[rep-1], color='blue')
            ax.plot(range(self.impute_array.shape[1]), self.impute_array[rep-1], color='green')
            leg1 = mpatches.Patch(color='blue', label=methodname+'Fitted Error'+str(rep))
            leg2 = mpatches.Patch(color='green', label=methodname+'Imputation Error'+str(rep))
            ax.legend(loc='upper right', handles=[leg1, leg2])
        ax.set_ylim((0.0, 1.0))
        ax.set_xlim((0, self.fitted_array.shape[1]))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Error')

    def plot_runtime(self, ax, methodname='Method', rep=None):
        assert self.track_runtime
        maxlen = self.fitted_array.shape[1]
        if rep == None:
            for i in range(self.rep+1):
                ax.plot(self.time_array[i,1:maxlen], self.fitted_array[i,1:maxlen], color='blue')
                ax.plot(self.time_array[i,1:maxlen], self.impute_array[i,1:maxlen], color='green')
            ax.set_xlim((0, np.nanmax(self.time_array) * 1.2))
            leg1 = mpatches.Patch(color='blue', label=methodname+' Fitted Error')
            leg2 = mpatches.Patch(color='green', label=methodname+' Imputation Error')
        else:
            assert rep < self.rep + 1
            ax.plot(self.time_array[rep-1,1:maxlen], self.fitted_array[rep-1,1:maxlen], color='blue')
            ax.plot(self.time_array[rep-1,1:maxlen], self.impute_array[rep-1,1:maxlen], color='green')
            ax.set_xlim((0, np.nanmax(self.time_array[rep-1]) * 1.2))
            leg1 = mpatches.Patch(color='blue', label=methodname+' Fitted Error'+str(rep))
            leg2 = mpatches.Patch(color='green', label=methodname+' Imputation Error'+str(rep))
        ax.legend(loc='upper right', handles=[leg1, leg2])
        ax.set_ylim((0.0, 1.0))
        ax.set_xlabel('Runtime (s)')
        ax.set_ylabel('Error')
    
    def save(self, pfile):
        with open(pfile, "wb") as output_file:
            pickle.dump(self.__dict__, output_file)

    def load(self, pfile):
        with open(pfile, "rb") as input_file:
            tmp_dict = pickle.load(input_file)
            self.__dict__.update(tmp_dict)