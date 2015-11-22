__author__ = 'noe'

import warnings
import numpy as np
from moments import moments_XX, moments_XXXY


class Moments(object):

    def __init__(self, w, s, M):
        """
        Parameters
        ----------
        w : float
            statistical weight.
                w = \sum_t w_t
            In most cases, :math:`w_t=1`, and then w is just the number of samples that went into s1, S2.
        s : ndarray(n,)
            sum over samples:
            .. math:
                s = \sum_t w_t x_t
        M : ndarray(n, n)
            .. math:
                M = (X-s)^T (X-s)
        """
        self.w = float(w)
        self.s = s
        self.M = M

    def copy(self):
        return Moments(self.w, self.s.copy(), self.M.copy())

    def combine(self, other, mean_free=False):
        """
        References
        ----------
        [1] http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
        """
        w1 = self.w
        w2 = other.w
        w = w1 + w2
        ds = (w2/w1) * self.s - other.s
        # update
        self.w = w1 + w2
        self.s += other.s
        #
        if mean_free:
            self.M += other.M + (w1 / (w2 * w)) * np.outer(ds, ds)
        else:
            self.M += other.M
        return self

    @property
    def mean(self):
        return self.s / self.w

    @property
    def covar(self):
        """ Returns M / (w-1)

        Careful: The normalization w-1 assumes that we have counts as weights.

        """
        return self.M / (self.w-1)


class MomentsStorage(object):
    """
    """

    def __init__(self, nsave, remove_mean=False, rtol=1.5):
        """
        Parameters
        ----------
        rtol : float
            To decide when to merge two Moments. Ideally I'd like to merge two
            Moments when they have equal weights (i.e. equally many data points
            went into them). If we always add data chunks with equal weights,
            this can be achieved by using a binary tree, i.e. let M1 be the
            moment estimates from one chunk. Two of them are added to M2, Two
            M2 are added to M4, and so on. This way you need to store log2
            (n_chunks) number of Moment estimates.
            In practice you might get data in chunks of unequal length or weight.
            Therefore we need some heuristic when two Moment estimates should get
            merged. This is the role of rtol.

        """
        self.nsave = nsave
        self.storage = []
        self.rtol = rtol
        self.remove_mean = remove_mean

    def _can_merge_tail(self):
        """ Checks if the two last list elements can be merged
        """
        if len(self.storage) < 2:
            return False
        return self.storage[-2].w <= self.storage[-1].w * self.rtol

    def store(self, moments):
        """ Store object X with weight w
        """
        if len(self.storage) == self.nsave:  # merge if we must
            # print 'must merge'
            self.storage[-1].combine(moments, mean_free=self.remove_mean)
        else:  # append otherwise
            # print 'append'
            self.storage.append(moments)
        # merge if possible
        while self._can_merge_tail():
            # print 'merge: ',self.storage
            M = self.storage.pop()
            # print 'pop last: ',self.storage
            self.storage[-1].combine(M, mean_free=self.remove_mean)
            # print 'merged: ',self.storage

    @property
    def moments(self):
        """
        """
        # collapse storage if necessary
        while len(self.storage) > 1:
            # print 'collapse'
            M = self.storage.pop()
            self.storage[-1].combine(M, mean_free=self.remove_mean)
        # print 'return first element'
        return self.storage[0]


class RunningCovar(object):
    """
    """

    def __init__(self, compute_XX=True, compute_XY=False, compute_YY=False,
                 remove_mean=False, symmetrize=False,
                 nsave=5):
        # check input
        if not compute_XX and not compute_XY:
            raise ValueError('One of compute_XX or compute_XY must be True.')
        if symmetrize and not compute_XY:
            warnings.warn('symmetrize=True has no effect with compute_XY=False.')
        # storage
        self.compute_XX = compute_XX
        if compute_XX:
            self.storage_XX = MomentsStorage(nsave, remove_mean=remove_mean)
        self.compute_XY = compute_XY
        if compute_XY:
            self.storage_XY = MomentsStorage(nsave, remove_mean=remove_mean)
        self.compute_YY = compute_YY
        if compute_YY:
            raise NotImplementedError('Currently not implemented')
        # symmetry
        self.remove_mean = remove_mean
        self.symmetrize = symmetrize

    def add(self, X, Y=None):
        # check input
        T = X.shape[0]
        if Y is not None:
            assert Y.shape[0] == T, 'X and Y must have equal length'
        # estimate and add to storage
        if self.compute_XX and not self.compute_XY:
            s_X, C_XX = moments_XX(X, remove_mean=self.remove_mean)
            self.storage_XX.store(Moments(T, s_X, C_XX))
        elif self.compute_XX and self.compute_XY:
            assert Y is not None
            s_X, s_Y, C_XX, C_XY = moments_XXXY(X, Y, remove_mean=self.remove_mean, symmetrize=self.symmetrize)
            # make copy in order to get independently mergeable moments
            self.storage_XX.store(Moments(T, s_X.copy(), C_XX))
            self.storage_XY.store(Moments(T, s_X.copy(), C_XY))

    def sum_X(self):
        return self.storage_XX.moments.s

    def mean_X(self):
        return self.storage_XX.moments.mean

    def moments_XX(self):
        return self.storage_XX.moments.M

    def cov_XX(self):
        return self.storage_XX.moments.covar

    def moments_XY(self):
        return self.storage_XY.moments.M

    def cov_XY(self):
        return self.storage_XY.moments.covar

    def sum_Y(self):
        raise NotImplementedError('Currently not implemented')

    def moments_YY(self):
        raise NotImplementedError('Currently not implemented')
