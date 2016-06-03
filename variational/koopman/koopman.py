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
    e_x : ndarray(N,)
        vector of empirical expectation values of the basis functions, over the first T-tau steps.
    e_y : ndarray(N,)
        vector of empirical expectation values of the basis functions, over the last T-tau steps.
    ep : float,
        threshold for truncation of singular values of C0.

    Returns:
    --------
    u : ndarray(N+1,)
        coefficients of approximation to the stationary density from the basis.
    v : ndarray(N+1,)
        right eigenvector of Koopman matrix, s.t. np.dot(u, v) = 1.
    K: ndarray(N+1, N+1)
        expanded and transformed correlation matrix.
    """
    # Get the number of features:
    N = C0.shape[0]
    # Pad the correlation matrices by the expectation values:
    C0 = np.hstack((C0, ex[:, None]))
    C0 = np.vstack((C0, np.concatenate((ex, np.ones(1, dtype=float)))))
    Ct = np.hstack((C0, ey[:, None]))
    Ct = np.vstack((Ct, np.concatenate((ex, np.ones(1, dtype=float)))))
    # Perform whitening transformation:
    s, U = scl.eigh(C0)
    s, U = vsd.sort_by_norm(s, U)
    # Remove close-to-zero eigenvalues:
    ind = s < ep
    s = s[ind]
    U = U[:, ind]
    # Transform Ct:
    Ct = np.dot(U.T, np.dot(Ct, U))
    # Compute right and left eigenvectors:
    st, Ut, Vt = scl.eig(Ct, left=True)
    _, Ut = vsd.sort_by_norm(st, Ut)
    _, Vt = vsd.sort_by_norm(st, Vt)
    # Extract those with eigenvalue 1 and normalize:
    v = Vt[:, 0]
    u = Ut[:, 0]
    u = u / np.dot(u, v)
    return u, v, Ct

def reweight_trajectory(X, est, u):
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

    Returns:
    --------
    est,
        the RunningCovar estimator after adding X.

    """
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
    # Compute Ct:
    C0_inv = scl.pinv(C0)
    K0 = np.dot(C0, K)
    Ct = 0.5*np.dot(C0_inv, K0 + K0.T)
    return C0, Ct


