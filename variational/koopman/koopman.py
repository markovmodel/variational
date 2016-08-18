import numpy as np
import scipy.linalg as scl

import variational.solvers.direct as vsd

def stationary_vector_koopman(C0, Ct, ex, ey, ep=1e-10):
    """
    The goal of this function is to estimate an approximation of the ratio of stationary over empirical
    distribution from the basis. To this end, the basis is expanded by adding the constant function and
    transformed to be orthonormal w.r.t. the empirical distribution.

    Parameters:
    -----------
    C0 : ndarray(N, N)
        instantaneous correlation matrix of the basis.
    Ct : ndarray(N, N)
        time lagged correlation matrix of the basis.
    ex : ndarray(N,)
        vector of empirical expectation values of the basis functions, over the first T-tau steps.
    ey : ndarray(N,)
        vector of empirical expectation values of the basis functions, over the last T-tau steps.
    ep : float,
        threshold for truncation of singular values of C0.

    Returns:
    --------
    u : ndarray(M,)
        coefficients of the ratio stationary / empirical dist. from the expanded and transformed basis.
    v : ndarray(M,)
        coefficients of the constant fucntion from the expanded and transformed basis.
    st: ndarray(M,)
        eigenvalues of transformed correlation matrix.
    Vt: ndarray(M, M)
        matrix of right eigenvectors of transformed correlation matrix.
    U : ndarray(N+1, M)
        transformation matrix.
    Ctr: ndarray(M, M)
        correlation matrix of transformed basis.

    """
    # Pad the correlation matrices by the expectation values:
    C0 = np.hstack((C0, ex[:, None]))
    C0 = np.vstack((C0, np.concatenate((ex, np.ones(1, dtype=float)))))
    Ct = np.hstack((Ct, ex[:, None]))
    Ct = np.vstack((Ct, np.concatenate((ey, np.ones(1, dtype=float)))))
    # Perform whitening transformation:
    s, U = scl.eigh(C0)
    # Remove close-to-zero eigenvalues:
    ind = s > ep
    s = s[ind]
    U = np.dot(U[:, ind], np.diag(s**-0.5))
    # Transform Ct:
    Ctr = np.dot(U.T, np.dot(Ct, U))
    # Compute right and left eigenvectors:
    st, Vt = scl.eig(Ctr)
    _, Vt = vsd.sort_by_norm(st, Vt)
    st, Ut = scl.eig(Ctr.T)
    st, Ut = vsd.sort_by_norm(st, Ut)
    # Extract those with eigenvalue 1 and normalize:
    u = np.real(Ut[:, 0])
    v = np.real(Vt[:, 0])
    u = u / np.dot(u, v)
    return u, v, st, Vt, U, Ctr

def reweight_trajectory(X, est, U, u, add_constant=False):
    """
    This function re-weights a trajectory using the stationary weights computed by approximating the
    Koopman operator.

    Parameters:
    -----------
    X : ndarray (T, N)
      evaluation of N basis function over T time steps.
    est : RunningCovar estimator,
        the trajectory will be added to this estimator with new weights.
    U : ndarray (N, M)
        transformation of the basis to orthogonal basis.
    u, ndarray (M,)
        expansion coefficients of ratio stat. / empirical dist. from transformed basis.
    add_constant : bool
        if True, a constant column is added to X.

    Returns:
    --------
    est,
        the RunningCovar estimator after adding X.

    """
    # Add the constant if required:
    if add_constant:
        X = np.hstack((X, np.ones((X.shape[0], 1))))
    # Transform the basis:
    X = np.dot(X, U)
    # Determine the weights from u:
    w = np.dot(X, u)
    # Add to the estimator:
    est.add(X, weights=w)
    return est

def equilibrium_correlation(est, K):
    """
    Compute equilibrium correlation matrices from re-weighted data.

    Parameters:
    -----------
    est : RunningCovar estimator,
        containing the instantaneous correlation matrix from re-weighted data.
    K : ndarray (N, N)
        correlation matrix of transformed basis.

    Returns:
    --------
    C0, ndarray (N, N)
        the equilibrium instantaneous correlation matrix.
    Ct, ndarray (N, N)
        the equilibrium time-lagged correlation matrix.

    """
    # Get Sigma:
    Sigma = est.moments_XX() / est.weight_XX()
    # Compute Sigma*K
    SK = np.dot(Sigma, K)
    # Symmetrize
    Ctau_mu = 0.5*(SK + SK.T)
    # Also return K_eq for reference:
    Sigma_inv = scl.pinv(Sigma)
    K_new = np.dot(Sigma_inv, Ctau_mu)
    return Sigma, Ctau_mu, K_new


