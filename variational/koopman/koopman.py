import numpy as np
import scipy.linalg as scl

import variational.estimators.running_moments as vrm
import variational.solvers.direct as vsd

def Koopman_Estimation(trajs, tau, ep0=1e-10):
    NT = len(trajs)
    est1 = vrm.RunningCovar(compute_XY=True, time_lagged=True, lag=tau)
    for i in range(NT):
        est1.add(trajs[i])
    # Extract expectations:
    ex = est1.mean_X()
    # Compute K0, u
    R, K0 = compute_K0(est1)
    u = compute_u(K0)
    # Re-weight:
    est_rw = vrm.RunningCovar(compute_XY=True, time_lagged=True, lag=tau)
    for i in range(NT):
        est_rw = reweight_trajectory(trajs[i], ex, est_rw, R, u)
    # Reversibility:
    Ks, Rs = equilibrium_correlation(est_rw, K0)
    return K0, R, u, Ks, Rs, ex


def compute_K0(est, ep0=1e-10):
    """
    Compute K0 for data padded with the constant function.

    Parameters:
    -----------
    est, RunningCovar estimator
        estimator for correlations
    ep0, float
        tolerance for eigenvalue of covariance matrix.

    Returns:
    --------
    R, ndarray(N, M)
        whitening transformation of basis functions.
    K0, ndarray(M+1, M+1)
        time-lagged correlation matrix of whitened data, padded with the constant.

    """
    # Get the correlations from the data:
    C0 = est.cov_XX()
    Ct = est.cov_XY()
    # Get the expectations:
    ex = est.mean_X()
    ey = est.mean_Y()
    # Perform whitening transformation:
    s, Q = scl.eigh(C0 - np.outer(ex, ex))
    # Determine negative magnitudes:
    evmin = np.min(s)
    if evmin < 0:
        ep0 = np.maximum(ep0, -evmin)
    # Cut-off small or negative eigenvalues:
    s, Q = vsd.sort_by_norm(s, Q)
    ind = np.where(np.abs(s) > ep0)[0]
    s = s[ind]
    Q = Q[:, ind]
    # Compute the whitening transformation:
    R = np.dot(Q, np.diag(s**-0.5))
    # Set the new correlation matrix:
    M = R.shape[1]
    K0 = np.dot(R.T, np.dot((Ct - np.outer(ex, ey)), R))
    K0 = np.vstack((K0, np.dot((ey - ex), R)))
    ex1 = np.zeros((M+1, 1))
    ex1[M, 0] = 1.0
    K0 = np.hstack((K0, ex1))
    return R, K0

def compute_u(K0):
    """
    Estimate an approximation of the ratio of stationary over empirical distribution from the basis.

    Parameters:
    -----------
    K0, ndarray(M+1, M+1),
        time-lagged correlation matrix for the whitened and padded data set.

    Returns:
    --------
    u : ndarray(M,)
        coefficients of the ratio stationary / empirical dist. from the whitened and expanded basis.

    """
    M = K0.shape[0] - 1
    # Compute right and left eigenvectors:
    l, U = scl.eig(K0.T)
    l, U = vsd.sort_by_norm(l, U)
    # Extract the eigenvector for eigenvalue one and normalize:
    u = np.real(U[:, 0])
    v = np.zeros(M+1)
    v[M] = 1.0
    u = u / np.dot(u, v)
    return u

def reweight_trajectory(X, ex, est, R, u):
    """
    This function re-weights a trajectory using the stationary weights computed by approximating the
    Koopman operator.

    Parameters:
    -----------
    X : ndarray (T, N)
        evaluation of N basis function over T time steps.
    ex: ndarray(N,)
        mean-values of original basis functions.
    est : RunningCovar estimator,
        the trajectory will be added to this estimator with new weights.
    R : ndarray (N, M)
        whitening transformation of the basis.
    u, ndarray (M+1,)
        expansion coefficients of ratio stat. / empirical dist. from whitened and padded basis set.

    Returns:
    --------
    est,
        the RunningCovar estimator after adding X.

    """
    # Transform the basis:
    X = np.dot(X - ex[None, :], R)
    # Pad by ones:
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    # Determine the weights from u:
    w = np.dot(X, u)
    # Add to the estimator:
    est.add(X, weights=w)
    return est

def equilibrium_correlation(est, K0, ep0=1e-10):
    """
    Compute equilibrium correlation matrices from re-weighted data.

    Parameters:
    -----------
    est : RunningCovar estimator,
        containing the instantaneous correlation matrix from re-weighted data.
    K0 : ndarray (M+1, M+1)
        correlation matrix of whitened and padded basis.

    Returns:
    --------
    Ks, ndarray(M',M')
        Reversibility modification of Koopman matrix.
    Rs, ndarray(M+1, M')
        whitening transformation based on Sigma_0

    """
    # Get Sigma_0:
    Sigma_0 = est.cov_XX()
    # Remove small eigenvalues:
    # Perform whitening transformation:
    s, Q = scl.eigh(Sigma_0)
    # Determine negative magnitudes:
    evmin = np.min(s)
    if evmin < 0:
        ep0 = np.maximum(ep0, -evmin)
    # Cut-off small or negative eigenvalues:
    s, Q = vsd.sort_by_norm(s, Q)
    ind = np.where(np.abs(s) > ep0)[0]
    s = s[ind]
    Q = Q[:, ind]
    # Determine whitening transformation at equilibrium:
    Rs = np.dot(Q, np.diag(s**-0.5))
    # Compute Ks:
    Ks = np.dot(Rs.T, np.dot(Sigma_0, np.dot(K0, Rs)))
    # Reversibility modification:
    Ks = 0.5 * (Ks + Ks.T)
    return Ks, Rs

def Dimensionality_Reduction_Ks(Ks, m=2):
    """
    Perform dimensionality reduction of Ks based on kinetic distance:

    Parameters:
    -----------
    Ks: ndarray(M', M')
        reversibility modification of Koopman matrix.

    """
    # Perform SVD:
    Us, sig_s, _ = scl.svd(Ks, full_matrices=False)
    # Transformation:
    Rd = np.dot(Us[:, :m], np.diag(sig_s[:m]))
    return Rd