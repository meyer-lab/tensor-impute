import numpy as np
import time
from .cmtf import calcR2X

class tracker():
    """
    Creates an array, tracks next unfilled entry & runtime, holds tracked name for plotting
    """

    def __init__(self, data, entry_type='R2X', track_runtime=False):
        self.data = data
        self.metric = entry_type
        self.track_runtime = track_runtime
        self.array = np.full((1, 0), 0)
        if self.track_runtime:
            self.time_array = np.full((1, 0), 0)

    def __call__(self, tFac):
        self.array = np.append(self.array, calcR2X(tFac, self.data))
        if self.track_runtime:
            assert self.start
            self.time_array = np.append(self.time_array, time.time() - self.start)

    def begin(self):
        """ Must run to track runtime """
        self.start = time.time()
    
    def reset(self):
        self.array = np.full((1, 0), 0)
        if self.track_runtime:
            self.time_array = np.full((1, 0), 0)

    def plot_iteration(self, ax, methodname):
        ax.plot(range(1, self.array.size + 1), self.array, label=methodname)
        ax.set_ylim((0.0, 1.0))
        ax.set_xlim((1, self.array.size))
        ax.set_xlabel('Iteration')
        ax.set_ylabel(self.metric)
        ax.legend(loc=4)

    def plot_runtime(self, ax, methodname):
        assert self.track_runtime
        self.time_array
        ax.plot(self.time_array, self.array, label=methodname)
        ax.set_ylim((0.0, 1.0))
        ax.set_xlim((0, np.max(self.time_array) * 1.2))
        ax.set_xlabel('Runtime')
        ax.set_ylabel(self.metric)
        ax.legend(loc=4)