import numpy as np
import time
import pickle
from .impute_helper import calcR2X

class Tracker():
    """ Tracks next unfilled entry & runtime, holds tracked name for plotting """
        
    def __init__(self, tOrig = [0], mask=None, track_runtime = True):
        """ self.data should be the original tensor (e.g. prior to running imputation) """
        self.data = tOrig.copy()
        self.mask = mask   # mask represents untouched (1) vs dropped (0) entries
        self.track_runtime = track_runtime
        self.rep = 0
        self.total_error = [np.full((1, 0), 0)]
        self.fitted_error = [np.full((1, 0), 0)]
        self.imputed_error = [np.full((1, 0), 0)]
        if self.track_runtime:
            self.timer = [np.full((1, 0), 0)]
        
        self.combined = False

    def __call__(self, tFac, **kwargs):
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
        self.mask = None
        self.start = None
        self.rep = 0

    def combine(self, remove_outliers=False):
        """ Combines all runs into a single np.ndarray."""

        # in case any run doesn't hit maximum iterations, make them all the same size
        max = 0
        for i in range(self.rep+1):
            current = self.total_error[i].size
            if remove_outliers and np.max(self.total_error[i]) > 1: self.total_error[i][:] = np.nan

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
    
    def time_thresholds(self, threshold = 0.25, total = False):
        if not self.combined: self.combine()
        if total:
            thres_arr = self.total_array <= threshold
        else:
            thres_arr = self.imputed_array <= threshold
        met_rows = np.argwhere(np.sum(thres_arr, axis=1)).flatten()
        thres_arr = thres_arr[met_rows,:]

        thres_iter = np.argmax(thres_arr, axis=1)
        thres_time = [self.time_array[i,thres_iter[ID]] for ID,i in enumerate(met_rows)]
        return thres_time
    
    def unmet_threshold_count(self, threshold = 0.25):
        if not self.combined: self.combine()
        return np.sum(np.nanmin(self.imputed_array,axis=1) > threshold)


    def save(self, pfile):
        with open(pfile, "wb") as output_file:
            pickle.dump(self.__dict__, output_file)

    def load(self, pfile):
        with open(pfile, "rb") as input_file:
            tmp_dict = pickle.load(input_file)
            self.__dict__.update(tmp_dict)


class MultiTracker():
    '''
    Saves tracker results for many Tracker objects.
    '''
    def __init__(self, track_runtime = True):
        self.initialized = False
        self.track_runtime = track_runtime
        self.combined = True


    def __call__(self, tracker:Tracker):
        if self.initialized is True:
            self.total_array = np.vstack((self.total_array,tracker.total_array))
            self.imputed_array = np.vstack((self.imputed_array,tracker.imputed_array))
            self.fitted_array = np.vstack((self.fitted_array,tracker.fitted_array))
            if self.track_runtime:
                self.time_array = np.vstack((self.time_array,tracker.time_array))
        else:
            self.initialized = True
            self.total_array = tracker.total_array
            self.imputed_array = tracker.imputed_array
            self.fitted_array = tracker.fitted_array
            if self.track_runtime:
                self.time_array = tracker.time_array

    

    def time_thresholds(self, threshold = 0.25, total = False):
        if total:
            thres_arr = self.total_array <= threshold
        else:
            thres_arr = self.imputed_array <= threshold
        met_rows = np.argwhere(np.sum(thres_arr, axis=1)).flatten()
        thres_arr = thres_arr[met_rows,:]

        thres_iter = np.argmax(thres_arr, axis=1)
        thres_time = [self.time_array[i,thres_iter[ID]] for ID,i in enumerate(met_rows)]
        return thres_time
    
    def unmet_threshold_count(self, threshold = 0.25):
        return np.sum(np.nanmin(self.imputed_array,axis=1) > threshold)

    def save(self, pfile):
        with open(pfile, "wb") as output_file:
            pickle.dump(self.__dict__, output_file)

    def load(self, pfile):
        with open(pfile, "rb") as input_file:
            tmp_dict = pickle.load(input_file)
            self.__dict__.update(tmp_dict)