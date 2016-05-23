import numpy as np


def stationary_vector_koopman(C0, Ct, ex, ey):
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

    Returns:
    --------
    u : ndarray(N+1,)
        coefficients of approximation to the stationary density from the basis.
    u : ndarray(N+1,)
        right eigenvector of Koopman matrix, s.t. np.dot(u, v) = 1.
    V: ndarray(N+1, N+1)
        whitening transformation of the basis expanded by the constant function.
    """