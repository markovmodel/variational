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