import numpy as np
import scipy.linalg as scl

import variational.solvers.direct as vsd

def stationary_vector_koopman(C0, Ct, ex, ey, ep=1e-10):
    """
    Return approximation of the stationary density through the matrix approximation of the Koopman operator.

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
    u : ndarray(N+1,)
        coefficients of approximation to the stationary density from the basis.
    v : ndarray(N+1,)
        right eigenvector of Koopman matrix, s.t. np.dot(u, v) = 1.
    Ct: ndarray(N+1, N+1)
        expanded correlation matrix, including the constant function.
    st: ndarray(N+1,)
        eigenvalues of Koopman matrix.
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
    _, Ut = vsd.sort_by_norm(st, Ut)
    # Extract those with eigenvalue 1 and normalize:
    u = np.real(Ut[:, 0])
    v = np.real(Vt[:, 0])
    u = u / np.dot(u, v)
    v = np.dot(U, v)
    u = np.dot(U, u)
    return u, v, Ct, st

def reweight_trajectory(X, est, u, add_constant=False):
    """
    This function re-weights a trajectory using the stationary weights computed by approximating the
    Koopman operator.

    Parameters:
    -----------
    X : ndarray (T, N)
      evaluation of N basis function over T time steps.
    est : RunningCovar estimator,
        the trajectory will be added to this estimator with new weights.
    u, ndarray (N,)
        expansion coefficients of stat. density from basis (i.e. left eigenvector of Koopman matrix.
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
    # Determine the weights from u:
    w = np.dot(X, u)
    # Add to the estimator:
    est.add(X, weights=w)
    return est

def equilibrium_correlation(est, K, rcond=1e-12):
    """
    Compute equilibrium correlation matrices from re-weighted data.

    Parameters:
    -----------
    est : RunningCovar estimator,
        containing the instantaneous correlation matrix from re-weighted data.
    K : ndarray (N, N)
        the approximation of the Koopman operator from the basis.

    Returns:
    --------
    C0, ndarray (N, N)
        the equilibrium instantaneous correlation matrix.
    Ct, ndarray (N, N)
        the equilibrium time-lagged correlation matrix.

    """
    # Get C0:
    C0 = est.cov_XX()
    # Compute its inverse:
    C0_inv = scl.pinv(C0, rcond=rcond)
    SK = np.dot(C0, K)
    Ktilde = 0.5*np.dot(C0_inv, SK + SK.T)
    Ct = np.dot(C0, Ktilde)
    return C0, Ct


