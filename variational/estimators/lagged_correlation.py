__author__ = 'noe'

import numpy as np

class LaggedCorrelation:

    def __init__(self, output_dimension, tau=1):
        """ Computes correlation matrices C0 and Ctau from a bunch of trajectories

        Parameters
        ----------
        output_dimension: int
            Number of basis functions.
        tau: int
            Lag time

        """
        self.tau = tau
        self.output_dimension = output_dimension
        # Initialize the two correlation matrices:
        self.Ct = np.zeros((self.output_dimension, self.output_dimension))
        self.C0 = np.zeros((self.output_dimension, self.output_dimension))
        # Also initialize the mean:
        self.mu = np.zeros(self.output_dimension)
        # Create counter for the frames used for mu, C0, Ct:
        self.nmu = 0
        self.nC0 = 0
        self.nCt = 0

    def add(self, X):
        """ Adds trajectory to the running estimate for computing mu, C0 and Ct:

        Parameters
        ----------
        X: ndarray (T,N)
            basis function trajectory of T time steps for N basis functions.

        """
        # Raise an error if output dimension is wrong:
        
        # Get the trajectory length:
        TX = X.shape[0]
        # # Update the mean:
        self.mu += np.sum(X,axis=0)
        self.nmu += TX
        # Update the instantaneous correlation matrix:
        self.C0 += np.dot(X.T,X)
        self.nC0 += TX
        # Update time lagged correlation matrix:
        Y = X[self.tau:,:]
        self.Ct += np.dot(X.T,Y)
        self.nCt += TX - self.tau

    def mean(self):
        """ Returns the mean.

        Returns
        -------
        mu: ndarray (N,)
            array of mean values for each of the N basis function.

        """
        return self.mu/self.nmu

    def C0(self):
        """ Returns the current estimate of C0:

        Returns
        -------
        C0: ndarray (N,N)
            time instantaneous correlation matrix of N basis function.

        """
        return self.C0/self.nC0

    def Ctau(self):
        """ Returns the current estimate of Ctau

        Returns
        -------
        Ct: ndarray (N,N)
            time lagged correlation matrix of N basis function.

        """
        return self.Ct/self.nCt
