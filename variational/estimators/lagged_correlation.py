__author__ = 'noe'

class LaggedCorrelation:
    """ Computes correlation matrices C0 and Ctau from a bunch of trajectories
    """

    def __init__(self, lag=1):
        self.lag = lag

    def add(self, X):
        """ Adds trajectory X to the running estimate for computing mu, C0 and Ct
        """
        pass

    def mean(self):
        """ Returns the mean
        """
        pass

    def C0(self):
        """ Returns the current estimate of C0
        """
        pass

    def Ctau(self):
        """ Returns the current estimate of Ctau
        """
        pass
