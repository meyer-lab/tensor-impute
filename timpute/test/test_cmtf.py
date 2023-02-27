"""
Unit test file.
"""
import numpy as np
import tensorly as tl
import warnings
from tensorly.cp_tensor import _validate_cp_tensor
from tensorly.random import random_cp
from ..cmtf import delete_component, calcR2X, buildMat, sort_factors, perform_CLS, censored_lstsq
from tensordata.alter import data as alter
from tensorly.tenalg import khatri_rao
from ..initialize_fac import initialize_fac
from sklearn.linear_model import Ridge


def createCube(missing=0.0, size=(10, 20, 25)):
    s = np.random.gamma(2, 2, np.prod(size))
    tensor = s.reshape(*size)
    if missing > 0.0:
        tensor[np.random.rand(*size) < missing] = np.nan
    return tensor
    

def test_cp():
    # Test that the CP decomposition code works.
    tensor = createCube(missing=0.2, size=(10, 20, 25))
    fac3 = perform_CLS(tensor, rank=3)
    fac6 = perform_CLS(tensor, rank=6)
    assert fac3.R2X < fac6.R2X
    assert fac3.R2X > 0.0
    if fac3.R2X < 0.67:
        warnings.warn("CP (rank=3) with 20% missingness, R2X < 0.67 (expected)" + str(fac3.R2X))

    # test case where mode size < rank
    tensor2 = createCube(missing=0.2, size=(10, 4, 50))
    fac23 = perform_CLS(tensor2, rank=3)
    fac26 = perform_CLS(tensor2, rank=6)
    assert fac23.R2X < fac26.R2X
    assert fac23.R2X > 0.0


def test_sort():
    """ Test that sorting does not affect anything. """
    tOrig = createCube(missing=0.2, size=(10, 20, 25))
    mOrig = createCube(missing=0.2, size=(10, 15))

    tFac = random_cp(tOrig.shape, 3)
    tFac.mFactor = np.random.randn(mOrig.shape[1], 3)
    tFac.mWeights = np.ones(3)

    R2X = calcR2X(tFac, tOrig, mOrig)
    tRec = tl.cp_to_tensor(tFac)
    mRec = buildMat(tFac)

    tFac = sort_factors(tFac)
    sR2X = calcR2X(tFac, tOrig, mOrig)
    stRec = tl.cp_to_tensor(tFac)
    smRec = buildMat(tFac)

    np.testing.assert_allclose(R2X, sR2X)
    np.testing.assert_allclose(tRec, stRec)
    np.testing.assert_allclose(mRec, smRec)

def test_ridge():
    """Testing ridge solver is equal to numpy least squares"""
    tOrig = createCube(missing=0.2)
    tFac = initialize_fac(tOrig.copy(), 5)
    unfolded = tl.unfold(tOrig, 0)
    kr = khatri_rao(tFac.factors, skip_matrix=0)
    
    A = kr
    B = unfolded.T

    X1 = censored_lstsq(A, B)
    X2 = censored_lstsq(A, B, alpha=0)

    np.testing.assert_allclose(X1, X2)

def test_colbycol():
    """Testing solver is equal column by column and when unique grouping applied"""
    tOrig = createCube(missing=0.2, size=(100,3,3))
    tFac = initialize_fac(tOrig.copy(), 3)
    unfolded = tl.unfold(tOrig, 0)
    kr = khatri_rao(tFac.factors, skip_matrix=0)

    A = kr
    B = unfolded.T
    X1 = np.empty((A.shape[1], B.shape[1]))

    unique, uIDX = np.unique(np.isfinite(B), axis=1, return_inverse=True)
    for i in range(0, B.shape[1]):
        uu = np.squeeze(unique[:, uIDX[i]])
        Bx = B[uu,:]
        clf = Ridge(alpha=0, fit_intercept=False)
        clf.fit(A[uu,:], Bx[:, i])
        X1[:, i] = clf.coef_.T

    X2 = censored_lstsq(A, B)

    np.testing.assert_allclose(X1.T, X2)