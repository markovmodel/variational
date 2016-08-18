import numpy as np
import scipy.linalg as scl


def filter_eigenvalues(l, R=None, ep=0.36, ep1=1e-2, ep_im=0.0):
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
    ep_im, float:
        tolerance governing how much of an imaginary part is accepted to due statistical noise.
    """
    # Remove meaningless eigenvalues:
    ind1 = np.where(np.logical_and(np.real(l) <= 1 + ep1, np.real(l) >= ep))[0]
    l = l[ind1]
    if R is not None:
        R = R[:, ind1]
    # Remove eigenvalues with imaginary part greater than ep_im:
    #ind2 = np.where(np.abs(np.imag(l)) <= ep_im)[0]
    #l = l[ind2]
    #if R is not None:
    #    R = R[:, ind2]
    # Discard all imaginary parts:
    #l = np.real(l)
    #if R is not None:
    #    R = np.real(R)
    # Sort the eigenvalues:
    ind3 = np.argsort(np.real(l))[::-1]
    l = l[ind3]
    ind_final = ind1[[ind3]]
    if R is not None:
        R = R[:, ind3]
    if R is not None:
        return l, R, ind_final
    else:
        return l, ind_final



def eigenvalue_estimation(Ct, C2t, ep=0.36, ep1=1e-2, ep_im=1e-2, ep_svd=None):
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
    ep1, float:
        tolerance governing how much larger than one the eigenvalues are allowed to be.
    ep_im, float:
        tolerance governing how much of an imaginary part is accepted to due statistical noise.
    ep_svd, float:
        singular value cutoff for Ct.

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
    if ep_svd is None:
        ep_svd = s > 1e-16
    s = s[ep_svd]
    U = U[:, ep_svd]
    V = V[ep_svd, :].transpose()
    # Define transformation matrices:
    F1 = np.dot(U, np.diag(s**(-0.5)))
    F2 = np.dot(V, np.diag(s**(-0.5)))
    # Solve eigenvalue problem:
    #W1 = np.dot(U.transpose(), np.dot(Ct, V))
    W = np.dot(F1.transpose(), np.dot(C2t, F2))
    l, R = scl.eig(W)
    # Compute VEq:
    Rp = scl.pinv(R)
    VEq = np.dot(np.diag(l**(-0.5)), np.dot(Rp, np.dot(F1.T, Ct)))
    VEq = VEq.T
    # Remove complex or meaningless eigenvalues:
    l, R, ind = filter_eigenvalues(l, R, ep=ep, ep1=ep1, ep_im=ep_im)
    VEq = VEq[:, ind]
    return l, VEq

def oom_estimation(Ct, C2t, ep_svd=None, ep=0.36, ep1=1e-2):
    """
    Perform steps from OOM-estimation method.

    """
    # Get the number of states:
    N = Ct.shape[0]
    # Get the SVD of Ctau:
    U, s, V = scl.svd(Ct, full_matrices=False)
    # Discard close-to-zero singular values:
    if ep_svd is None:
        ep_svd = s > 1e-16
    s = s[ep_svd]
    U = U[:, ep_svd]
    V = V[ep_svd, :].transpose()
    # Define transformation matrices:
    F1 = np.dot(U, np.diag(s**(-0.5)))
    F2 = np.dot(V, np.diag(s**(-0.5)))
    # Get the number of slow processes:
    M = F1.shape[1]
    # Compute observable operators:
    E = np.zeros((N, M, M))
    for n in range(N):
        E[n, :, :] = np.dot(F1.T, np.dot(C2t[n, :, :], F2))
    E_Omega = np.sum(E, axis=0)
    # Compute evaluator:
    ci = np.sum(Ct, axis=1)
    sigma = np.dot(F1.T, ci)
    # Compute information state:
    l0, omega_0 = scl.eig(E_Omega.T)
    _, omega_0, _ = filter_eigenvalues(l0, omega_0, ep=0.8, ep1=ep1)
    omega_0 = omega_0[:, 0] / np.dot(omega_0[:, 0], sigma)
    return E, omega_0, sigma

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