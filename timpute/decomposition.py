import pickle
import numpy as np
import tensorly as tl
from .tracker import tracker
from .cmtf import perform_CLS, calcR2X
from tensorly.tenalg import svd_interface
from .initialize_fac import initialize_fac
from .SVD_impute import IterativeSVD
from .impute_helper import entry_drop, chord_drop
from copy import deepcopy
from time import process_time


class Decomposition():
    def __init__(self, data, max_rr=5):
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
        self.data = data
        self.rrs = np.arange(1,max_rr+1)

    def perform_tfac(self, callback=None):
        """
        Paramters
        ---------
        callback : tracker Class
            Optional callback class to track R2X over iteration/runtime for the factorization with max_rr components.      

        Returns
        -------
        self.tfac : list of length max_rr
            Each value represents the tensor factorization calculated for components 1 to max_rr.
        self.TR2X : list of length max_rr
            Each value represents the R2X of factorizations calculated for components 1 to max_rr.
        self.sizeT : list of length max_rr
            Each value represents the size of factorizations for components 1 to max_rr.
        """
        self.tfac = [self.method(self.data, rank=rr) for rr in np.delete(self.rrs, max(self.rrs)-1)]
        self.tfac.append(self.method(self.data, rank=max(self.rrs), callback=callback))
        self.TR2X = [calcR2X(c, tIn=self.data) for c in self.tfac]
        self.sizeT = [rr * sum(self.tfac[0].shape) for rr in self.rrs]

    def perform_PCA(self, flattenon=0):
        dataShape = self.data.shape
        flatData = np.reshape(np.moveaxis(self.data, flattenon, 0), (dataShape[flattenon], -1))
        if not np.all(np.isfinite(flatData)):
            flatData = IterativeSVD(rank=1, random_state=1).fit_transform(flatData)

        U, S, V = svd_interface(matrix=flatData, n_eigenvecs=max(self.rrs))
        recon = [U[:, :rr] @ np.diag(S) @ V[:rr, :] for rr in self.rrs]
        self.PCAR2X = [1.0 - np.linalg.norm(c - flatData) / np.linalg.norm(flatData) for c in recon]
        self.sizePCA = [sum(flatData.shape) * rr for rr in self.rrs]

    def imputation(self, type:str='chord', drop:int=0.05, repeat:int=3, maxiter:int=50,
                   mode:int=0, alpha=None, method=perform_CLS,
                   single:bool=False, init='svd',
                   callback:tracker=None, callback_r:int=None, printRuntime=False):
        """
        Calculates Q2X when dropping chords along axis = mode from the data using self.method for factor decomposition,
        comparing each component. Drops in Q2X from one component to the next may signify overfitting.

        Parameters
        ----------
        type : str
            'chord' or 'entry' imputation type
        drop : int
            Percent dropped from tensor during imputation.
        repeat : int
        maxiter : int
        mode : int
            Defaults to mode corresponding to axis = 0. Can be set to any mode of the tensor.
        alpha : int
            Describes ridge regression alpha value for CLS.
        single : boolean
            Looks at only the final component (useful for plotting)
        callback : tracker Class
            Optional callback class to track R2X over iteration/runtime for the factorization with max_rr components.
        callback_r : int
            Optional component to look atl; defaulted to the max component if not specified.



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
        if callback_r is None: callback_r = max(self.rrs)
        if callback_r is not None: assert(callback_r >= 0 and callback_r <= np.max(self.rrs))
        assert(mode >= 0 and mode < self.data.ndim)
        assert(drop <= 1 and drop >= 0)

        if type=='entry':
            drop = int(drop*np.sum(np.isfinite(self.data)))
        elif type=='chord': 
            drop = int(drop*self.data.size/self.data.shape[0])
        else:
            raise ValueError('invalid imputation type')

        error = np.zeros((repeat,self.rrs[-1]))
        imputed_error = np.zeros((repeat,self.rrs[-1]))
        fitted_error = np.zeros((repeat,self.rrs[-1]))


        # Calculate Q2X for each number of components
        if isinstance(init,list):
            assert(len(init) == np.max(self.rrs))
            assert(len(init[0]) == repeat)
            assert(isinstance(init[0][0],tl.cp_tensor.CPTensor) or isinstance(init[0],tl.cp_tensor.CPTensor))
        
        for x in range(repeat):
            # drop values
            tImp = np.copy(self.data)
            np.moveaxis(tImp,mode,0)
            missingCube = np.copy(tImp)
            if type=='entry':
                mask = entry_drop(missingCube, drop)
            elif type=='chord':
                mask = chord_drop(missingCube, drop)
            
            # track masks
            if callback: callback.set_mask(mask)
            imputed_vals = np.ones_like(missingCube) - mask
            fitted_vals = np.ones_like(missingCube) - imputed_vals
            
            # for each component up to max, run method
            for rr in self.rrs:
                # tracking iteration
                if callback and rr == callback_r:
                    # handle initialization
                    if isinstance(init,tl.cp_tensor.CPTensor):
                        CPinit = deepcopy(init)
                    elif isinstance(init,list):
                        CPinit = deepcopy(init[rr-1][x])
                    else:
                        np.random.seed(x)
                        CPinit = initialize_fac(missingCube, rr, init)
                    
                    # run method
                    if callback.track_runtime:
                        callback.begin()
                    callback(CPinit)
                    if alpha is not None:
                        tFac = method(missingCube, rank=rr, n_iter_max=maxiter, mask=mask, callback=callback, init=CPinit, alpha=alpha)
                    else:
                        tFac = method(missingCube, rank=rr, n_iter_max=maxiter, mask=mask, callback=callback, init=CPinit)
                    
                # not tracking iteration
                else:
                    # handle initialization
                    if isinstance(init,tl.cp_tensor.CPTensor):
                        CPinit = deepcopy(init)
                    elif isinstance(init,list):
                        CPinit = deepcopy(init[rr-1][x])
                    else:
                        np.random.seed(x)
                        CPinit = initialize_fac(missingCube, rr, init)

                    # run method
                    if alpha is not None:
                        tFac = method(missingCube, rank=rr, n_iter_max=maxiter, mask=mask, init=CPinit, alpha=alpha)
                    else:
                        tFac = method(missingCube, rank=rr, n_iter_max=maxiter, mask=mask, init=CPinit)

                # save error/Q2X
                error[x,rr-1] = calcR2X(tFac, tIn=tImp, calcError=True)
                imputed_error[x,rr-1] = calcR2X(tFac, tIn=tImp, mask=imputed_vals, calcError=True)
                fitted_error[x,rr-1] = calcR2X(tFac, tIn=tImp, mask=fitted_vals, calcError=True)

            if (printRuntime and (x+1)%round(repeat*0.2) == 0):
                print(f"({method.__name__}) Average runtime for {x+1} tensors: {(process_time())/(x+1)} seconds")

            if callback:
                if x+1 < repeat: callback.new()
                
                # # Calculate Q2X for each number of principal components using PCA for factorization as comparison
                # if comparePCA:
                #     Q2XPCA = np.zeros((repeat,self.rrs[-1]))
                #     si = IterativeSVD(rank=max(self.rrs), random_state=1)
                #     missingMat = np.reshape(np.moveaxis(missingCube, 0, 0), (missingCube.shape[0], -1))
                #     mImp = np.reshape(np.moveaxis(tImp, 0, 0), (tImp.shape[0], -1))

                #     missingMat = si.fit_transform(missingMat)
                #     U, S, V = svd_interface(matrix=missingMat, n_eigenvecs=max(self.rrs))
                #     scores = U @ np.diag(S)
                #     loadings = V
                #     recon = [scores[:, :rr] @ loadings[:rr, :] for rr in self.rrs]
                #     Q2XPCA[x,:] = [calcR2X(c, mIn = mImp) for c in recon]
                #     self.entryQ2XPCA = Q2XPCA
        
        if type=='entry':
            self.entry_error = error
            self.imputed_entry_error = imputed_error
            self.fitted_entry_error = fitted_error
        elif type=='chord': 
            self.chord_error = error
            self.imputed_chord_error = imputed_error
            self.fitted_chord_error = fitted_error

        if single:
            #     for x in range(repeat):
            #         # drop values
            #         tImp = np.copy(self.data)
            #         np.moveaxis(tImp,mode,0)
            #         missingCube = np.copy(tImp)
            #         mask = chord_drop(missingCube, drop)
                    
            #         # track masks
            #         if callback: callback.set_mask(mask)
            #         imputed_vals = np.ones_like(missingCube) - mask
            #         fitted_vals = np.isfinite(tImp) - imputed_vals

            #         # method chunk
            #         if callback:
            #             # handle initialization
            #             if isinstance(init,tl.cp_tensor.CPTensor): CPinit = init
            #             else: CPinit = initialize_fac(missingCube, rr, init)

            #             # run method
            #             if callback.track_runtime:
            #                 callback.begin()
            #             callback(CPinit)
            #             if alpha is not None: tFac = method(missingCube, rank=rr, n_iter_max=maxiter, mask=mask, callback=callback, init=CPinit, alpha=alpha)
            #             else: tFac = method(missingCube, rank=rr, n_iter_max=maxiter, mask=mask, callback=callback, init=CPinit)
            #         else:   # not tracking iteration
            #             # handle initialization
            #             if isinstance(init,tl.cp_tensor.CPTensor): CPinit = init
            #             else: CPinit = initialize_fac(missingCube, rr, init)

            #             # run method
            #             if alpha is not None: tFac = method(missingCube, rank=rr, n_iter_max=maxiter, mask=mask, callback=callback, init=CPinit, alpha=alpha)
            #             else: tFac = method(missingCube, rank=rr, n_iter_max=maxiter, mask=mask, callback=callback, init=CPinit)

            #         # save error/Q2X
            #         Q2X[x,max(self.rrs)-1] = calcR2X(tFac, tIn=tImp)
            #         imputed_error[x,max(self.rrs)-1] = calcR2X(tFac, tIn=tImp, mask=imputed_vals, calcError=True)
            #         fitted_error[x,max(self.rrs)-1] = calcR2X(tFac, tIn=tImp, mask=fitted_vals, calcError=True)
            pass
            
    def save(self, pfile):
        with open(pfile, "wb") as output_file:
            pickle.dump(self.__dict__, output_file)

    def load(self, pfile):
        with open(pfile, "rb") as input_file:
            tmp_dict = pickle.load(input_file)
            self.__dict__.update(tmp_dict)


class MultiDecomp():
    def __init__(self, decomp:Decomposition = None, entry = True, chord=True):
        if decomp is not None:
            assert entry or chord
            self.rr = decomp.rrs[-1]
            self.hasEntry=entry
            self.hasChord=chord

            if entry:
                self.entry_total = decomp.entry_error
                self.entry_imputed = decomp.imputed_entry_error
                self.entry_fitted = decomp.fitted_entry_error

            if chord:
                self.chord_total = decomp.chord_error
                self.chord_imputed = decomp.imputed_chord_error
                self.chord_fitted = decomp.fitted_chord_error

    def __call__(self, decomp:Decomposition):
        if self.hasEntry:
            self.entry_total = np.vstack((self.entry_total,decomp.entry_error))
            self.entry_imputed = np.vstack((self.entry_imputed,decomp.imputed_entry_error))
            self.entry_fitted = np.vstack((self.entry_fitted,decomp.fitted_entry_error))
        if self.hasChord:
            self.chord_total = np.vstack((self.chord_total,decomp.chord_error))
            self.chord_imputed = np.vstack((self.chord_imputed,decomp.imputed_chord_error))
            self.chord_fitted = np.vstack((self.chord_fitted,decomp.fitted_chord_error))

    def save(self, pfile):
        with open(pfile, "wb") as output_file:
            pickle.dump(self.__dict__, output_file)

    def load(self, pfile):
        with open(pfile, "rb") as input_file:
            tmp_dict = pickle.load(input_file)
            self.__dict__.update(tmp_dict)