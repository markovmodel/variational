import numpy as np
import scipy.linalg as scl


def filter_eigenvalues(l, R, ep=0.36, ep1=1e-2, return_indices=False):
    """
    Filter out meaningful eigenvalues:

    Parameters:
    -----------
    l, ndarray(M,):
        eigenvalues.
    R, ndarray(N, M):
        right eigenvectors.
    ep, float:
        tolerance to accept only eigenvalues larger than ep.
    ep1, float:
        tolerance to accept eigenvalues which are smaller 1 + ep1.
    return_indices, bool:
        return the sorting indices after sorting the eigenvalues
    """
    # Remove complex or meaningless eigenvalues:
    ind1 = np.where(np.logical_and((l <= 1 + ep1), np.logical_and((l >= ep), np.isreal(l))))[0]
    l = np.real(l[ind1])
    R = np.real(R[:, ind1])
    # Sort the eigenvalues:
    ind2 = np.argsort(l)[::-1]
    l = l[ind2]
    R = R[:, ind2]
    if return_indices:
        return l, R, ind1[ind2]
    else:
        return l, R



def eigenvalue_estimation(Ct, C2t, ep=0.36):
    """
    Perform estimation of dominant system eigenvalues from short time simulations.

    Parameters:
    -----------
    Ct, ndarray (N, N)
        correlation matrix between N basis functions at lag time tau.
    C2t, ndarray (N, N)
        correlation matrix between N basis functions at lag time 2*tau.
    ep, float:
        tolerance to use only eigenvalues greater than ep.

    Returns:
    --------
    l, ndarray(M,)
        dominant eigenvalues of the system.
    VEq, ndarray(N, M)
        matrix of overlaps between basis functions and dominant left eigenfunctions.
        ONLY CORRECT UP TO A DIAGONAL MATRIX OF SCALING FACTORS!
    """
    # Get the SVD of Ctau:
    U, s, V = scl.svd(Ct, full_matrices=False)
    # Discard close-to-zero singular values:
    ind = np.where(s > 1e-16)[0]
    s = s[ind]
    U = U[:, ind]
    V = V[ind, :].transpose()
    # Define transformation matrices:
    F1 = np.dot(U, np.diag(s**(-0.5)))
    F2 = np.dot(V, np.diag(s**(-0.5)))
    # Solve eigenvalue problem:
    W = np.dot(F1.transpose(), np.dot(C2t, F2))
    l, R = scl.eig(W)
    # Extract VEq:
    Rp = scl.inv(R)
    VEq = np.dot(np.diag(l**(-0.5)), np.dot(Rp, np.dot(F1.T, Ct)))
    VEq = np.real(VEq.T)
    # Remove complex or meaningless eigenvalues:
    l, R, ind = filter_eigenvalues(l, R, ep=ep, return_indices=True)
    VEq = VEq[:, ind]
    return l, VEq

def correct_stationary_vector(VEq):
    """
    Computes corrected stationary vector of discrete states from VEq matrix:

    Parameters:
    -----------
    VEq, ndarray(N, M):
        matrix of overlaps between N discrete states with M dominant eigenfunctions.

    Returns:
    --------
    pi, ndarray(N,):
        stationary vector of the N microstates.
    """
    return VEq[:, 0] / np.sum(VEq[:, 0])

def correct_transition_matrix(l, VEq, sigma, ptau):
    """
    Computes corrected equilibrium correlation matrix between N discrete states from local equilibrium data.

    Parameters:
    -----------
    l, ndarray(M,)
        dominant eigenvalues of the system at lag time tau.
    VEq, ndarray(N, M):
        matrix of overlaps between N discrete states with M dominant eigenfunctions.
    sigma, ndarray(N,):
        array of starting probabilities for discrete states.
    ptau, ndarray(N,):
        array of probabilities for discrete states at time tau.

    Returns:
    Ctau, ndarray(N, N):
        equilibrium correlation matrix at lag time tau.
    pi, ndarray(N,):
        stationary vector of the discrete states.
    """
    # Get the stationary vector:
    pi = correct_stationary_vector(VEq)
    # Construct the A-matrix:
    veq_row_sum = np.sum(np.dot(np.diag(sigma / pi), VEq), axis=0)
    A = np.dot(VEq, np.diag(veq_row_sum*l))
    # Solve the linear system:
    theta = scl.lstsq(A, ptau)[0]
    # Find negative entries:
    neg_ind = theta < 0
    if theta[neg_ind].shape[0] > 0:
        print "Warning: Negative entries in theta, maximal modulus is %.e"%(np.max(np.abs(theta[neg_ind])))
        theta[neg_ind] = -theta[neg_ind]
    # Correct:
    VEq_cor = np.dot(VEq, np.diag(np.sqrt(theta)))
    Ctau = np.dot(VEq_cor, np.dot(np.diag(l), VEq_cor.T))
    return Ctau, pi