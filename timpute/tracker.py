import pickle
import time

import numpy as np

from .impute_helper import calcR2X


class Tracker():
    """ Tracks next unfilled entry & runtime, holds tracked name for plotting """
        
    def __init__(self, tOrig = [0], mask=None, track_runtime = True):
        """ self.data should be the original tensor (e.g. prior to running imputation) """
        self.data = tOrig.copy()
        self.mask = mask   # mask represents untouched (1) vs dropped (0) entries
        self.track_runtime = track_runtime
        self.rank = None
        self.rep = 0
        self.total_error = dict()
        self.fitted_error = dict()
        self.imputed_error = dict()
        if self.track_runtime:
            self.timer = dict()
        # [np.full((1, 0), 0)]
        
        self.combined = False

    def __call__(self, tFac):
        """ Takes a CP tensor object """
        self.total_error[self.rank][self.rep] = np.append(self.fitted_error[self.rank][self.rep],
                                                          calcR2X(tFac, self.data, True))
        if self.mask is not None:
            self.fitted_error[self.rank][self.rep] = np.append(self.fitted_error[self.rank][self.rep],
                                                               calcR2X(tFac, self.data, True, self.mask))
            self.imputed_error[self.rank][self.rep] = np.append(self.imputed_error[self.rank][self.rep],
                                                                calcR2X(tFac, self.data, True, np.ones_like(self.data) - self.mask))
        if self.track_runtime:
            if self.start is None:
                self.timer[self.rank][self.rep] = np.append(self.timer[self.rank][self.rep], 0.0)
            else:
                self.timer[self.rank][self.rep] = np.append(self.timer[self.rank][self.rep], time.time() - self.start)

    def begin(self):
        """ Must call to track runtime """
        self.start = time.time()
    
    def set_mask(self, mask):
        self.mask = mask

    def new_rank(self, rank):
        """ Call to start tracking a repetition of the same method """
        self.rank = str(rank)
        self.rep = 0

        self.total_error[self.rank] = [np.full((1, 0), 0)]
        self.fitted_error[self.rank] = [np.full((1, 0), 0)]
        self.imputed_error[self.rank] = [np.full((1, 0), 0)]
        if self.track_runtime:
            self.timer[self.rank] = [np.full((1, 0), 0)]
            self.start = None
    
    def existing_rank(self, rank):
        """ Call to continue tracking a repetition of the same method """
        self.rank = str(rank)
        self.rep = len(self.total_error[self.rank])-1

        self.total_error[self.rank].append([np.full((1, 0), 0)])
        self.fitted_error[self.rank].append([np.full((1, 0), 0)])
        self.imputed_error[self.rank].append([np.full((1, 0), 0)])
        if self.track_runtime:
            self.timer[self.rank].append([np.full((1, 0), 0)])
            self.start = None
        self.rep += 1

    def combine(self):
        """ Combines all runs into a single np.ndarray."""

        # in case any run doesn't hit maximum iterations, make them all the same size
        max = 0
        for r in self.total_error:
            for i in range(len(self.total_error[r])):
                current = self.total_error[r][i].size

                if (current > max):
                    for j in range(i):
                        self.total_error[r][j] = np.append(self.total_error[r][j], np.full((current-max), np.nan))
                        self.fitted_error[r][j] = np.append(self.fitted_error[r][j], np.full((current-max), np.nan))
                        self.imputed_error[r][j] = np.append(self.imputed_error[r][j], np.full((current-max), np.nan))
                        if self.track_runtime: self.timer[r][j] = np.append(self.timer[r][j], np.full((current-max), np.nan))
                    max = current
                else:
                    self.total_error[r][i] = np.append(self.total_error[r][i], np.full((max-current), np.nan))
                    self.fitted_error[r][i] = np.append(self.fitted_error[r][i], np.full((max-current), np.nan))
                    self.imputed_error[r][i] = np.append(self.imputed_error[r][i], np.full((max-current), np.nan))
                    if self.track_runtime: self.timer[r][i] = np.append(self.timer[r][i], np.full((max-current), np.nan))
        
        self.total_array = self.total_error.copy()
        self.fitted_array = self.fitted_error.copy()
        self.imputed_array = self.imputed_error.copy()
        if self.track_runtime: self.time_array = self.timer.copy()

        for r in self.total_error: 
            self.total_array[r] = np.vstack(tuple(self.total_array[r]))
            self.fitted_array[r] = np.vstack(tuple(self.fitted_array[r]))
            self.imputed_array[r] = np.vstack(tuple(self.imputed_array[r]))
            if self.track_runtime: self.time_array[r] = np.vstack(tuple(self.time_array[r]))
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