import numpy as np
import tensorly as tl
from tensorly.cp_tensor import cp_flip_sign, cp_normalize
from tensorly.tenalg import khatri_rao
from tqdm import tqdm

from .impute_helper import calcR2X
from .initialization import initialize_fac


def perform_PM(
    tOrig: np.ndarray,
    rank: int = 6,
    n_iter_max: int = 50,
    tol=1e-6,
    callback=None,
    init=None,
    verbose=None,
) -> tl.cp_tensor.CPTensor:
    # function [A, B, C, LFT, M]=PARAFACM(XIJK, Fac, epsilon)
    # % Input
    # % XIJK is a three-way data array with missing values
    # % Fac is the number of components
    # % epsilon is tolerance
    # %
    # % Output
    # % A(I, N),B(J, N),C(K, N) are the decomposed underlying profile matrices
    # % LFT is the loss function
    # % M is the iterative number
    # % ______________________________________________________________

    # % -----------STEP 0--------------
    # % decompose the cube X along I,J,K direction respectively
    # [I, J, K] = size(XIJK);        % X_IJK with size of IxJxK
    # % I is the number of rows
    # % J is the number of columns
    # % K is the number of channels
    # XJKI = shiftdim(XIJK, 1);      % X_JKI with size of JxKxI
    # XKIJ = shiftdim(XJKI, 1);      % X_KIJ with size of KxIxJ
    # XIxJK = reshape(XIJK, I, J*K); % X_IxJK with size of IxJK
    # XJxKI = reshape(XJKI, J, K*I); % X_JxKI with size of JxKI
    # XKxIJ = reshape(XKIJ, K, I*J); % X_KxIJ with size of KxIJ

    unfolded = [tl.unfold(tOrig, i) for i in range(tOrig.ndim)]

    # % -----------STEP 1---------------
    # % initialize A & B and compute C
    # [A, B, C] = DTLD_nan(XIJK, Fac);
    # % A = rand(I, Fac);
    # % B = rand(J, Fac);

    # % estimation of C
    # AB = krao(B, A);
    # for k = 1 : K
    #     ABK = AB;
    #     XK = XKxIJ(k, :);
    #     ABK(isnan(XK), :) = [];
    #     XK(isnan(XK)) = [] ;
    #     C(k, :) = XK*ABK*pinv(ABK'*ABK);
    # end

    if init is None:
        tFac = initialize_fac(tOrig, rank)
    else:
        tFac = init

    # % ------------STEP 2---------------
    # % start to caculate LFT and do iteration
    # TOL = 10;
    # M = 0;
    # LFT = [];
    # LF = 0.01;
    # while TOL > epsilon && M < 500
    # % estimation of A
    #     BC = krao(C, B);
    #     for i = 1 : I
    #         BCI = BC;
    #         XI = XIxJK(i, :);
    #         BCI(isnan(XI), :) = [];
    #         XI(isnan(XI)) = [] ;
    #         A(i, :) = XI*BCI*pinv(BCI'*BCI);
    #     end
    #      % normalization of A columnwisely
    #      A = A*diag(1./diag(sqrt(A'*A)));
    #     end
    # % caculate loss function
    #     LFTT = 0;
    #     for k = 1 : K
    #         XX = XIJK(:, :, k);
    #         XXX = A*diag(C(k, :))*B';
    #         XXX(isnan(XX)) = 0;
    #         XX(isnan(XX)) = 0;
    #         LFTT = LFTT+trace((XX-XXX)'*(XX-XXX));
    #     end
    #     TOL = abs((LFTT-LF)/LF);
    #     LFT = [LFT, LFTT];
    #     LF = LFTT;
    #     M = M+1;
    # end

    R2X_last = -np.inf
    tFac.R2X = calcR2X(tFac, tOrig)

    tq = tqdm(range(n_iter_max), disable=(not verbose))
    for _ in tq:
        for m in range(len(tFac.factors)):
            kr = khatri_rao(tFac.factors, skip_matrix=m)
            for i in range(tFac.factors[m].shape[0]):
                mIDs = np.isfinite(unfolded[m][i])
                X_miss = unfolded[m][i, mIDs]
                kr_miss = kr[mIDs, :]
                tFac.factors[m][i] = (
                    X_miss @ kr_miss @ np.linalg.pinv(kr_miss.T @ kr_miss)
                )

        R2X_last = tFac.R2X
        tFac.R2X = calcR2X(tFac, tOrig)
        tq.set_postfix(R2X=tFac.R2X, delta=tFac.R2X - R2X_last, refresh=False)
        # assert tFac.R2X > 0.0
        if callback:
            callback(tFac)

        if tFac.R2X - R2X_last < tol:
            break

        # % ------------STEP 3---------------
    # % post-processing to keep sign convention
    # [maxa, inda] = max(abs(A));
    # [maxb, indb] = max(abs(B));
    # asign = ones(Fac, 1);
    # bsign = ones(Fac, 1);
    # for n = 1 : Fac
    #     asign(n) = sign(A(inda(n), n));
    #     bsign(n) = sign(B(indb(n), n));
    # end
    # A = A*diag(asign);
    # B = B*diag(bsign);
    # C = C*diag(asign)*diag(bsign);

    tFac = cp_normalize(tFac)
    tFac = cp_flip_sign(tFac)
    tFac.R2X = calcR2X(tFac, tOrig)

    return tFac
