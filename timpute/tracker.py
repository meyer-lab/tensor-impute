import numpy as np
from timpute.cmtf import calcR2X
import time

class tracker():
    """
    Creates an array, tracks next unfilled entry & runtime, holds tracked name for plotting
    """

    def __init__(self, tOrig, mask=None, entry_type='R2X', track_runtime=False):

        self.tensor = tOrig
        self.mask = mask
        self.metric = entry_type
        self.track_runtime = track_runtime

        self.array = []           # Array containing fit error  
        self.impute_array = []    # Array containing impute error
        self.time_array = []      # Array containing time points 

    def __call__(self, tFac, error):
        self.array = np.append(self.array, 1 - error)
        self.impute_array = np.append(self.impute_array, 1 - self.calc_impute_error(tFac))
        if self.track_runtime:
            self.time_array = np.append(self.time_array, time.time() - self.start)

    def calc_impute_error(self, tFac):
        if self.mask is not None:
            assert self.mask.all() == False
            tensorImp = np.copy(self.tensor)
            tensorImp[self.mask] = np.nan
            return calcR2X(tFac, tensorImp)
        else:
            return np.nan

    def begin(self):
        """ Must run to track runtime """
        self.start = time.time()

    def plot_iteration(self, ax):
        ax.plot(range(1, self.array.size + 1), self.array, label='Fitted Error')
        ax.plot(range(1, self.impute_array.size + 1), self.impute_array, label='Imputation Error')
        ax.legend(loc='upper right')
        ax.set_ylim((0.0, 1.0))
        ax.set_xlim((0, self.array.size))
        ax.set_xlabel('Iteration')
        ax.set_ylabel(self.metric)

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