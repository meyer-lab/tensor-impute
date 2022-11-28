import numpy as np
import time
import pickle
from .cmtf import calcR2X

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
    Creates an fitted_array, tracks next unfilled entry & runtime, holds tracked name for plotting
    """
        
    def __init__(self, tOrig, mask=None, track_runtime=False):
        """ self.data should be the original tensor (e.g. prior to running imputation) """

        self.data = tOrig
        self.mask = mask
        self.track_runtime = track_runtime
        self.rep = 0
        self.fitted_array = [np.full((1, 0), 0)]
        self.impute_array = [np.full((1, 0), 0)]
        if self.track_runtime:
            self.time_array = [np.full((1, 0), 0)]

    def __call__(self, tFac, error=None):
        """ Takes a CP tensor object """
        if error is None:
            if self.mask is not None: # Assure error is calculated with non-removed values 
                mask_data = np.copy(self.data)
                mask_data[~self.mask] = np.nan
                error = calcR2X(tFac, mask_data, calcError=True)
            else:
                error = calcR2X(tFac, self.data, calcError=True)
        self.fitted_array[self.rep] = np.append(self.fitted_array[self.rep], error)
        self.impute_array[self.rep] = np.append(self.impute_array[self.rep], calc_impute_error(self.mask, self.data, tFac))
        if self.track_runtime:
            assert self.start
            self.time_array = np.append(self.time_array, time.time() - self.start)

    def begin(self):
        """ Must call to track runtime """
        self.start = time.time()
    
    def set_mask(self, mask):
        self.mask = mask

    def new(self):
        """ Call to start tracking a repetition of the same method """
        self.fitted_array.append([np.full((1, 0), 0)])
        self.impute_array.append([np.full((1, 0), 0)])
        if self.track_runtime:
            self.time_array.append([np.full((1, 0), 0)])
        self.rep += 1

    def combine(self):
        max = 0
        for i in range(self.rep+1):
            assert(self.fitted_array[i].size == self.fitted_array[i].size)
            current = self.fitted_array[i].size
            if (current > max):
                for j in range(self.rep):
                    self.fitted_array[j] = np.append(self.fitted_array[j], np.full((current-max), np.nan))
                    self.impute_array[j] = np.append(self.impute_array[j], np.full((current-max), np.nan))
                    if self.track_runtime: self.time_array[j] = np.append(self.time_array[j], np.full((current-max), np.nan))
                max = current
        
        self.fitted_array = np.vstack(tuple(self.fitted_array)) 
        self.impute_array = np.vstack(tuple(self.impute_array))    
        if self.track_runtime: self.time_array = np.vstack(tuple(self.time_array))

    def reset(self):
        self.fitted_array = [np.full((1, 0), 0)]
        self.impute_array = [np.full((1, 0), 0)]
        if self.track_runtime:
            self.time_array = [np.full((1, 0), 0)]

    """ Plots are designed to track the error of the method for the highest rank imputation of tOrig """
    def plot_iteration(self, ax, methodname='Method', average=True):
        if average:
            ax.plot(range(1, self.fitted_array.shape[1] + 1), self.fitted_array, label=methodname+' Fitted Error')
            ax.plot(range(1, self.impute_array.shape[1] + 1), self.impute_array[0], label=methodname+' Imputation Error')
        else:
            for i in range(self.rep+1):
                ax.plot(range(1, self.fitted_array[i].size + 1), self.fitted_array[i], label=methodname+' Fitted Error')
                ax.plot(range(1, self.impute_array[i].size + 1), self.impute_array[i], label=methodname+' Imputation Error')
        ax.legend(loc='upper right')
        ax.set_ylim((0.0, 1.0))
        ax.set_xlim((1, self.fitted_array.size))
        ax.set_xlabel('Iteration')
        ax.set_ylabel(self.metric)

    def plot_runtime(self, ax, methodname='Method', average=True):
        assert self.track_runtime
        self.time_array
        ax.plot(self.time_array[0], self.fitted_array[0], label=methodname+' Fitted Error')
        ax.plot(self.time_array[0], self.impute_array[0], label=methodname+' Imputation Error')
        ax.legend(loc='upper right')
        ax.set_ylim((0.0, 1.0))
        ax.set_xlim((0, np.max(self.time_array) * 1.2))
        ax.set_xlabel('Runtime')
        ax.set_ylabel(self.metric)
    
    def save(self, pfile):
        with open(pfile, "wb") as output_file:
            pickle.dump(self.__dict__, output_file)

    def load(self, pfile):
        with open(pfile, "rb") as input_file:
            tmp_dict = pickle.load(input_file)
            self.__dict__.update(tmp_dict)