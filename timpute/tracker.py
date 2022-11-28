import numpy as np
import time
from .cmtf import calcR2X

class tracker():
    """
    Creates an fitted_array, tracks next unfilled entry & runtime, holds tracked name for plotting
    """
        
    def __init__(self, tOrig, mask=None, entry_type='Error', track_runtime=False):
        """ self.data should be the original tensor (e.g. prior to running imputation) """

        self.data = tOrig
        self.mask = mask
        self.metric = entry_type
        self.track_runtime = track_runtime
        self.fitted_array = np.full((1, 0), 0)
        if self.track_runtime:
            self.time_array = np.full((1, 0), 0)

        self.fitted_array = np.full((1, 0), 0)
        self.impute_array = np.full((1, 0), 0)
        if self.track_runtime:
            self.time_array = np.full((1, 0), 0)

    def __call__(self, tFac, error=None):
        """ Takes a CP tensor object """
        if error is None:
            if self.mask is not None: # Assure error is calculated with non-removed values 
                mask_data = np.copy(self.data)
                mask_data[~self.mask] = np.nan
                error = calcR2X(tFac, mask_data, calcError=True)
            else:
                error = calcR2X(tFac, self.data, calcError=True)
        self.fitted_array = np.append(self.fitted_array, error)
        self.impute_array = np.append(self.impute_array, self.calc_impute_error(tFac))
        if self.track_runtime:
            assert self.start
            self.time_array = np.append(self.time_array, time.time() - self.start)

    def begin(self):
        """ Must call to track runtime """
        self.start = time.time()
    
    def reset(self):
        self.fitted_array = np.full((1, 0), 0)
        if self.track_runtime:
            self.time_array = np.full((1, 0), 0)

    def set_mask(self, mask):
        self.mask = mask    
    
    def calc_impute_error(self, tFac):
        if self.mask is not None:
            assert self.mask.all() == False, "Mask indicates no removed entries"
            tensorImp = np.copy(self.data)
            tensorImp[self.mask] = np.nan
            return calcR2X(tFac, tensorImp, calcError=True)
        else:
            return np.nan

    """ Plots are designed to track the error of the method for the highest rank imputation of tOrig """
    def plot_iteration(self, ax, methodname='Method'):
        ax.plot(range(1, self.fitted_array.size + 1), self.fitted_array, label=methodname+' Fitted Error')
        ax.plot(range(1, self.impute_array.size + 1), self.impute_array, label=methodname+' Imputation Error')
        ax.legend(loc='upper right')
        ax.set_ylim((0.0, 1.0))
        ax.set_xlim((1, self.fitted_array.size))
        ax.set_xlabel('Iteration')
        ax.set_ylabel(self.metric)

    def plot_runtime(self, ax, methodname='Method'):
        assert self.track_runtime
        self.time_array
        ax.plot(self.time_array, self.fitted_array, label=methodname+' Fitted Error')
        ax.plot(self.time_array, self.impute_array, label=methodname+' Imputation Error')
        ax.legend(loc='upper right')
        ax.set_ylim((0.0, 1.0))
        ax.set_xlim((0, np.max(self.time_array) * 1.2))
        ax.set_xlabel('Runtime')
        ax.set_ylabel(self.metric)