__author__ = 'noe'

import numpy as np
import variational.estimators.moment_computation as _mc

class Chunked_Covar:

    def __init__(self, nsave=5):

        self.storage = _mc.MomentsStorage(nsave)

    def add_chunk(self, X, Y=None, weight=1.0):
        self.storage.store(_mc.compute_moments(X, Y, weight))

    @property
    def weight(self):
        return(self.storage.moments.w)

    @property
    def sum1(self):
        return(self.storage.moments.s)

    @property
    def mean(self):
        return(self.storage.moments.mean)

    @property
    def sum2(self):
        return(self.storage.moments.M)

    @property
    def covar(self):
        return(self.storage.moments.covar)


# TODO: automatic selection of nonconstant columns inside covariance detector.
# TODO: how does the API estimator look like?