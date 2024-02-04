import pickle
import numpy as np
import tensorly as tl

from .tracker import Tracker
from .initialization import initialize_fac
from .impute_helper import entry_drop, chord_drop
from .impute_helper import calcR2X, corcondia
from .method_CLS import perform_CLS

from copy import deepcopy
from tqdm import tqdm


class Decomposition():
    def __init__(self, data=np.ndarray([0]), max_rr=5):
        """
        Decomposition object designed for plotting. Capable of handling a single tensor.

        Parameters
        ----------
        data : ndarray
            Takes a tensor of any shape.
        matrix : ndarray (optional)
            Takes a matrix of any shape.
        max_rr : int
            Defines the maximum component to consider during factorization.
        method : function
            Takes a factorization method. Default set to perform_CLS() from cmtf.py
            other methods include: tucker_decomp
        """
        self.data = data.copy()
        self.rrs = np.arange(1,max_rr+1)

    def imputation(self,
                   type:str='chord',
                   repeat:int=3, 
                   drop:int=0.05,
                   chord_mode:int=0, 
                   method=perform_CLS,
                   tol=1e-6,
                   init='random',
                   maxiter:int=50,
                   seed = 1,
                   callback:Tracker=None, 
                   trackCoreConsistency=False,
                   printRuntime=False):
        """
        Performs imputation (chord or entry) from the [self.data] using [method] for factor decomposition,
        comparing each component. Drops in Q2X from one component to the next may signify overfitting.

        Parameters
        ----------
        type : str
            'chord' or 'entry' imputation
        repeat : int
            Number of repetitions to run imputation (for every component up to self.max_rr). Defaults to 3.
        drop : int
            Percent dropped from tensor during imputation (rounded to int). Defaults to 5%.
        chord_mode : 0 ≤ int < self.data.ndim
            Mode to drop chords along (ignore for entry drop). Defaults to 0.
        method : function
        init : str // CPTensor // list of list of CPTensors
            Valid strings include 'svd' and 'random'. Otherwise, an initial guess for the CPTensor must be provided.
        maxiter : int
            Max iterations to cap method at. Defaults to 50.
        callback : tracker class.
            Optional callback class to track R2X over iteration/runtime for the factorization with max_rr components.



        Returns
        -------
        self.Q2X : ndarray of size (repeat, max_rr)
            Each value in a row represents the Q2X of the tensor calculated for components 1 to max_rr.
            Each row represents a single repetition.
        self.imputed_chord_error / self.imputed_entry_error : ndarray of size (repeat, max_rr)
            Each value in a row represents error of the IMPUTED artificial dropped values of the tensor
            calculated for components 1 to max_rr.
        self.fitted_chord_error / self.fitted_entry_error : ndarray of size (repeat, max_rr)
            Each value in a row represents error of the FITTED (not dropped, not missing) values of the tensor
            calculated for components 1 to max_rr.
        """
        assert(chord_mode >= 0 and chord_mode < self.data.ndim)
        assert(drop < 1 and drop >= 0)
        if type=='entry':
            drop = int(drop*np.sum(np.isfinite(self.data)))
        elif type=='chord': 
            drop = int(drop*self.data.size/self.data.shape[0])
        else:
            raise ValueError('invalid imputation type')

        error = np.zeros((repeat,self.rrs[-1]))
        imputed_error = np.zeros((repeat,self.rrs[-1]))
        fitted_error = np.zeros((repeat,self.rrs[-1]))
        if trackCoreConsistency is True:
            assert drop == 0
            corcon = np.zeros((repeat,self.rrs[-1]))
        
        if printRuntime:
            missingpatterns = tqdm(range(repeat), desc=f"Decomposing {repeat} tensors using {method.__name__}")
        else:
            missingpatterns = range(repeat)

        """ drop values (in-place)
        - `tImp` is a copy of data, used a reference for imputation accuracy
        - `missingCube` is where values are dropped
        """
        tImp = self.data.copy()           # avoid editing in-place of data
        np.moveaxis(tImp,chord_mode,0)      # reshaping for correct chord dropping

        for x in missingpatterns:
            missingCube = tImp.copy()

            """ track masks 
            - `imputed_vals` has a 1 where values were artifically dropped
            - `fitted_vals` has a 1 where values were not artifically dropped (considers non-imputed values)
            """
            if type=='entry':
                mask = entry_drop(missingCube, drop)
            elif type=='chord':
                mask = chord_drop(missingCube, drop)

            if callback is not None:
                callback.set_mask(mask)
            imputed_vals = np.ones_like(missingCube) - mask
            fitted_vals = np.ones_like(missingCube) - imputed_vals
            
            # for each component up to max, run method
            for rr in self.rrs:
                if isinstance(init, str):
                    np.random.seed(int(x*seed))
                    CPinit = initialize_fac(missingCube.copy(), rr, init)
                elif isinstance(init, tl.cp_tensor.CPTensor):
                    CPinit = deepcopy(init)
                else:
                    raise ValueError(f'Initialization method "{init}" not recognized')

                # run method
                if isinstance(callback, Tracker):
                    # track rank & repetition
                    if str(rr) in callback.total_error:
                        callback.existing_rank(rr)
                    else:
                        callback.new_rank(rr)

                    if callback.track_runtime:
                        callback.begin()
                    callback(CPinit)

                tFac = method(missingCube.copy(), rank=rr, n_iter_max=maxiter, mask=mask, init=CPinit, callback=callback, tol=tol)
                
                # update error metrics
                error[x,rr-1] = calcR2X(tFac, tIn=tImp, calcError=True)
                if drop > 0:
                    imputed_error[x,rr-1] = calcR2X(tFac, tIn=tImp, mask=imputed_vals, calcError=True)
                    fitted_error[x,rr-1] = calcR2X(tFac, tIn=tImp, mask=fitted_vals, calcError=True)
                if trackCoreConsistency is True:
                    corcon[x,rr-1] = corcondia(tFac)
        
        # save objects
        if type == 'entry':
            self.entry_total = error
            self.entry_imputed = imputed_error
            self.entry_fitted = fitted_error

        elif type == 'chord': 
            self.chord_total = error
            self.chord_imputed = imputed_error
            self.chord_fitted = fitted_error
        
        if trackCoreConsistency is True:
            self.corcondia = corcon

    def profile_imputation(self,
                   type:str='chord',
                   drop:int=0.05,
                   chord_mode:int=0, 
                   method=perform_CLS,
                   tol=1e-6,
                   init='random',
                   maxiter:int=50,
                   seed = 1,
                   callback:Tracker=None):
        """
        Performs imputation (chord or entry) from the [self.data] using [method] for factor decomposition,
        comparing each component. Drops in Q2X from one component to the next may signify overfitting.

        Parameters
        ----------
        type : str
            'chord' or 'entry' imputation
        repeat : int
            Number of repetitions to run imputation (for every component up to self.max_rr). Defaults to 3.
        drop : int
            Percent dropped from tensor during imputation (rounded to int). Defaults to 5%.
        chord_mode : 0 ≤ int < self.data.ndim
            Mode to drop chords along (ignore for entry drop). Defaults to 0.
        method : function
        init : str // CPTensor // list of list of CPTensors
            Valid strings include 'svd' and 'random'. Otherwise, an initial guess for the CPTensor must be provided.
        maxiter : int
            Max iterations to cap method at. Defaults to 50.
        callback : tracker class.
            Optional callback class to track R2X over iteration/runtime for the factorization with max_rr components.



        Returns
        -------
        self.Q2X : ndarray of size (repeat, max_rr)
            Each value in a row represents the Q2X of the tensor calculated for components 1 to max_rr.
            Each row represents a single repetition.
        self.imputed_chord_error / self.imputed_entry_error : ndarray of size (repeat, max_rr)
            Each value in a row represents error of the IMPUTED artificial dropped values of the tensor
            calculated for components 1 to max_rr.
        self.fitted_chord_error / self.fitted_entry_error : ndarray of size (repeat, max_rr)
            Each value in a row represents error of the FITTED (not dropped, not missing) values of the tensor
            calculated for components 1 to max_rr.
        """
        assert(chord_mode >= 0 and chord_mode < self.data.ndim)
        assert(drop < 1 and drop >= 0)
        """ drop values (in-place)
        - `tImp` is a copy of data, used a reference for imputation accuracy
        - `missingCube` is where values are dropped
        """

        tImp = self.data.copy()           # avoid editing in-place of data
        np.moveaxis(tImp,chord_mode,0)      # reshaping for correct chord dropping
        missingCube = tImp.copy()

        """ track masks 
        - `imputed_vals` has a 1 where values were artifically dropped
        - `fitted_vals` has a 1 where values were not artifically dropped (considers non-imputed values)
        """
        if type=='entry':
            mask = entry_drop(missingCube, drop)
        elif type=='chord':
            mask = chord_drop(missingCube, drop)
        
        # RUN ONLY THE MAX COMPONENT
        rr = self.rrs.max()

        if isinstance(init, str):
            np.random.seed(int(seed))
            CPinit = initialize_fac(missingCube.copy(), rr, init)
        elif isinstance(init, tl.cp_tensor.CPTensor):
            CPinit = deepcopy(init)
        else:
            raise ValueError(f'Initialization method "{init}" not recognized')

        # run method
        if isinstance(callback, Tracker):
            # track rank & repetition
            if str(rr) in callback.total_error:
                callback.existing_rank(rr)
            else:
                callback.new_rank(rr)

            if callback.track_runtime:
                callback.begin()
            callback(CPinit)

        tFac = method(missingCube.copy(), rank=rr, n_iter_max=maxiter, mask=mask, init=CPinit, callback=callback, tol=tol)

        return tFac    

            
    def save(self, pfile):
        with open(pfile, "wb") as output_file:
            pickle.dump(self.__dict__, output_file)

    def load(self, pfile):
        with open(pfile, "rb") as input_file:
            tmp_dict = pickle.load(input_file)
            self.__dict__.update(tmp_dict)

class MultiDecomp():
    '''
    Saves decomposition results for many Decomposition objects.
    '''
    def __init__(self, entry = True, chord = True):
        assert entry or chord
        self.hasEntry = entry
        self.hasChord = chord
        self.initialized = False


    def __call__(self, decomp:Decomposition):
        if self.initialized:
            if self.hasEntry:
                self.entry_total = np.vstack((self.entry_total,decomp.entry_total))
                self.entry_imputed = np.vstack((self.entry_imputed,decomp.entry_imputed))
                self.entry_fitted = np.vstack((self.entry_fitted,decomp.entry_fitted))
            if self.hasChord:
                self.chord_total = np.vstack((self.chord_total,decomp.chord_total))
                self.chord_imputed = np.vstack((self.chord_imputed,decomp.chord_imputed))
                self.chord_fitted = np.vstack((self.chord_fitted,decomp.chord_fitted))
        else:
            self.initialized = True
            self.rr = decomp.rrs[-1]
            if self.hasEntry:
                self.entry_total = decomp.entry_total
                self.entry_imputed = decomp.entry_imputed
                self.entry_fitted = decomp.entry_fitted
            if self.hasChord:
                self.chord_total = decomp.chord_total
                self.chord_imputed = decomp.chord_imputed
                self.chord_fitted = decomp.chord_fitted


    def save(self, pfile):
        with open(pfile, "wb") as output_file:
            pickle.dump(self.__dict__, output_file)

    def load(self, pfile):
        with open(pfile, "rb") as input_file:
            tmp_dict = pickle.load(input_file)
            self.__dict__.update(tmp_dict)