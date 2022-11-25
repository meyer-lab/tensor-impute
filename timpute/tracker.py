import numpy as np
from timpute.cmtf import calcR2X
import time
from .cmtf import calcR2X

class tracker():
    """
    Creates an array, tracks next unfilled entry & runtime, holds tracked name for plotting
    """
        
    def __init__(self, tOrig, mask=None, entry_type='R2X', track_runtime=False):
        """ self.data should be the original tensor (e.g. prior to running imputation) """

        self.data = tOrig
        self.mask = mask
        self.metric = entry_type
        self.track_runtime = track_runtime

        self.array = np.full((1, 0), 0)
        self.impute_array = np.full((1, 0), 0)
        if self.track_runtime:
            self.time_array = np.full((1, 0), 0)

    def __call__(self, tFac, error=None):
        """ Takes a CP tensor object """
        error = None # Issue with how error is currently calculated in tensorly
        if error is None:
            if self.mask is not None: # Assure error is calcualted with non-removed values 
                mask_data = np.copy(self.data)
                mask_data[~self.mask] = np.nan
                error = calcR2X(tFac, mask_data)
            else:
                error = calcR2X(tFac, self.data)
        self.array = np.append(self.array, error)
        self.impute_array = np.append(self.impute_array, self.calc_impute_error(tFac))
        if self.track_runtime:
            assert self.start
            self.time_array = np.append(self.time_array, time.time() - self.start)

    def calc_impute_error(self, tFac):
        if self.mask is not None:
            assert self.mask.all() == False, "Mask indicates no removed enteries"
            tensorImp = np.copy(self.data)
            tensorImp[self.mask] = np.nan
            return calcR2X(tFac, tensorImp)
        else:
            return np.nan

    def begin(self):
        """ Must call to track runtime """
        self.start = time.time()
    
    def reset(self):
        self.array = np.full((1, 0), 0)
        self.impute_array = np.full((1, 0), 0)
        if self.track_runtime:
            self.time_array = np.full((1, 0), 0)

    """ Plots are designed to track the R2X of the method for the highest rank imputation of tOrig """
    def plot_iteration(self, ax):
        ax.plot(range(1, self.array.size + 1), self.array, label='Fitted Error')
        ax.plot(range(1, self.impute_array.size + 1), self.impute_array, label='Imputation Error')
        ax.legend(loc='upper right')
        ax.set_ylim((0.0, 1.0))
        ax.set_xlim((1, self.array.size))
        ax.set_xlabel('Iteration')
        ax.set_ylabel(self.metric)
        ax.legend(loc=4)

    def plot_runtime(self, ax):
        assert self.track_runtime
        self.time_array
        ax.plot(self.time_array, self.array, label='Fitted Error')
        ax.plot(self.time_array, self.impute_array, label='Imputation Error')
        ax.legend(loc='upper right')
        ax.set_ylim((0.0, 1.0))
        ax.set_xlim((0, np.max(self.time_array) * 1.2))
        ax.set_xlabel('Runtime')
        ax.set_ylabel(self.metric)
        ax.legend(loc=4)