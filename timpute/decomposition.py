import pickle
from re import A
import numpy as np
from .cmtf import perform_CLS, calcR2X
from tensorly.tenalg import svd_interface
from .SVD_impute import IterativeSVD
from .impute_helper import entry_drop, chord_drop
from .initialize_fac import initialize_fac

class Decomposition():
    def __init__(self, data, max_rr=5, method=perform_CLS):
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
        self.method = method
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
        scores = U @ np.diag(S)
        loadings = V
        recon = [scores[:, :rr] @ loadings[:rr, :] for rr in self.rrs]
        self.PCAR2X = [calcR2X(c, mIn=flatData) for c in recon]
        self.sizePCA = [sum(flatData.shape) * rr for rr in self.rrs]

    def Q2X_chord(self, drop=5, repeat=3, maxiter=50, mode=0, callback=None, single=False, init='svd'):
        """
        Calculates Q2X when dropping chords along axis = mode from the data using self.method for factor decomposition,
        comparing each component. Drops in Q2X from one component to the next may signify overfitting.

        Parameters
        ----------
        drop : int
            To set a percentage, tensor.shape[mode] and multiply by the percentage 
            to find the relevant drop value, rounding to nearest int.
        repeat : int
        maxiter : int
        mode : int
            Defaults to mode corresponding to axis = 0. Can be set to any mode of the tensor.
        callback : tracker Class
            Optional callback class to track R2X over iteration/runtime for the factorization with max_rr components.
        single : boolean
            Looks at only the final component (useful for plotting)



        Returns
        -------
        self.Q2X : ndarray of size (repeat, max_rr)
            Each value in a row represents the Q2X of the tensor calculated for components 1 to max_rr.
            Each row represents a single repetition.
        """
        Q2X = np.zeros((repeat,self.rrs[-1]))
        fitted_Q2X = np.zeros((repeat,self.rrs[-1]))

        # Calculate Q2X for each number of components

        if single:
            for x in range(repeat):
                tImp = np.copy(self.data)
                np.moveaxis(tImp,mode,0)
                missingCube = np.copy(tImp)
                mask = chord_drop(missingCube, drop)
                if callback: callback.set_mask(mask)
                imputed_vals = np.ones_like(missingCube) - mask
                fitted_vals = np.isfinite(tImp) - imputed_vals

                if callback:
                    if callback.track_runtime: callback.begin()
                    CPinit = initialize_fac(missingCube, max(self.rrs), init)
                    tFac = self.method(missingCube, rank=max(self.rrs), n_iter_max=maxiter, mask=mask, callback=callback, init=CPinit)
                else:
                    CPinit = initialize_fac(missingCube, max(self.rrs), init)
                    tFac = self.method(missingCube, rank=max(self.rrs), n_iter_max=maxiter, mask=mask, init=CPinit)
                Q2X[x,max(self.rrs)-1] = calcR2X(tFac, tIn=tImp, mask=imputed_vals)
                fitted_Q2X[x,max(self.rrs)-1] = calcR2X(tFac, tIn=tImp, mask=fitted_vals)

        else:
            for x in range(repeat):
                tImp = np.copy(self.data)
                np.moveaxis(tImp,mode,0)
                missingCube = np.copy(tImp)
                mask = chord_drop(missingCube, drop)
                if callback: callback.set_mask(mask)
                imputed_vals = np.ones_like(missingCube) - mask
                fitted_vals = np.isfinite(tImp) - imputed_vals

                for rr in self.rrs:
                    if callback and rr == max(self.rrs):
                        if callback.track_runtime: callback.begin()
                        CPinit = initialize_fac(missingCube, rr, init)
                        tFac = self.method(missingCube, rank=rr, n_iter_max=maxiter, mask=mask, callback=callback, init=CPinit)
                    else:
                        CPinit = initialize_fac(missingCube, rr, init)
                        tFac = self.method(missingCube, rank=rr, n_iter_max=maxiter, mask=mask, init=CPinit)
                    Q2X[x,rr-1] = calcR2X(tFac, tIn=tImp, mask=imputed_vals)
                    fitted_Q2X[x,rr-1] = calcR2X(tFac, tIn=tImp, mask=fitted_vals)

                if callback:
                    if x+1 < repeat: callback.new()
        
        self.chordQ2X = Q2X
        self.fitted_entryQ2X = fitted_Q2X
            

    def Q2X_entry(self, drop=20, repeat=3, maxiter=50, comparePCA=False, callback=None, single=False, init='svd'):
        """
        Calculates Q2X when dropping entries from the data using self.method for factor decomposition,
        comparing each component. Drops in Q2X from one component to the next may signify overfitting.

        Parameters
        ----------
        drop : int
            To set a percentage, multiply np.sum(np.isfinite(tensor)) by the percentage 
            to find the relevant drop value, rounding to nearest int.
        repeat : int
        maxiter : int
        comparePCA : boolean
            Defaulted to calculate Q2X for respective principal components using PCA for factorization
            to compare against self.method.
        callback : tracker Class
            Optional callback class to track R2X over iteration/runtime for the factorization with max_rr components.
        single : boolean
            Looks at only the final component (useful for plotting)

        Returns
        -------
        self.Q2X : ndarray of size (repeat, max_rr)
            Each value in a row represents the Q2X of the tensor calculated for components 1 to max_rr using self.method.
            Each row represents a single repetition.
        self.Q2XPCA : ndarray of size (repeat, max_rr)
            Each value in a row represents the Q2X of the tensor calculated for components 1 to max_rr using PCA after
            SVD imputation. Each row represents a single repetition.
        """
        Q2X = np.zeros((repeat,self.rrs[-1]))
        fitted_Q2X = np.zeros((repeat,self.rrs[-1]))

        if single:
            for x in range(repeat):
                tImp = np.copy(self.data)
                missingCube = np.copy(tImp)
                mask = entry_drop(missingCube, drop)
                if callback: callback.set_mask(mask)
                imputed_vals = np.ones_like(missingCube) - mask
                fitted_vals = np.isfinite(tImp) - imputed_vals

                if callback:
                    if callback.track_runtime: callback.begin()
                    CPinit = initialize_fac(missingCube, max(self.rrs), init)
                    tFac = self.method(missingCube, rank=max(self.rrs), n_iter_max=maxiter, mask=mask, callback=callback, init=CPinit)
                else:
                    CPinit = initialize_fac(missingCube, max(self.rrs), init)
                    tFac = self.method(missingCube, rank=max(self.rrs), n_iter_max=maxiter, mask=mask, init=CPinit)
                Q2X[x,max(self.rrs)-1] = calcR2X(tFac, tIn=tImp, mask=imputed_vals)
                fitted_Q2X[x,max(self.rrs)-1] = calcR2X(tFac, tIn=tImp, mask=fitted_vals)

        else:
            for x in range(repeat):
                tImp = np.copy(self.data)
                missingCube = np.copy(tImp)
                mask = entry_drop(missingCube, drop)
                if callback: callback.set_mask(mask)
                imputed_vals = np.ones_like(missingCube) - mask
                fitted_vals = np.isfinite(tImp) - imputed_vals

                for rr in self.rrs:
                    if callback and rr == max(self.rrs):
                        if callback.track_runtime: callback.begin()
                        CPinit = initialize_fac(missingCube, rr, init)
                        tFac = self.method(missingCube, rank=rr, n_iter_max=maxiter, mask=mask, callback=callback, init=CPinit)
                    else:
                        CPinit = initialize_fac(missingCube, rr, init)
                        tFac = self.method(missingCube, rank=rr, n_iter_max=maxiter, mask=mask, init=CPinit)
                    Q2X[x,rr-1] = calcR2X(tFac, tIn=tImp, mask=imputed_vals)
                    fitted_Q2X[x,rr-1] = calcR2X(tFac, tIn=tImp, mask=fitted_vals)

                if callback:
                    if x+1 < repeat: callback.new()
                
                # Calculate Q2X for each number of principal components using PCA for factorization as comparison
                if comparePCA:
                    Q2XPCA = np.zeros((repeat,self.rrs[-1]))
                    si = IterativeSVD(rank=max(self.rrs), random_state=1)
                    missingMat = np.reshape(np.moveaxis(missingCube, 0, 0), (missingCube.shape[0], -1))
                    mImp = np.reshape(np.moveaxis(tImp, 0, 0), (tImp.shape[0], -1))

                    missingMat = si.fit_transform(missingMat)
                    U, S, V = svd_interface(matrix=missingMat, n_eigenvecs=max(self.rrs))
                    scores = U @ np.diag(S)
                    loadings = V
                    recon = [scores[:, :rr] @ loadings[:rr, :] for rr in self.rrs]
                    Q2XPCA[x,:] = [calcR2X(c, mIn = mImp) for c in recon]
                    self.entryQ2XPCA = Q2XPCA
        
            self.entryQ2X = Q2X
            self.fitted_entryQ2X = fitted_Q2X
    

    def save(self, pfile):
        with open(pfile, "wb") as output_file:
            pickle.dump(self.__dict__, output_file)

    def load(self, pfile):
        with open(pfile, "rb") as input_file:
            tmp_dict = pickle.load(input_file)
            self.__dict__.update(tmp_dict)
