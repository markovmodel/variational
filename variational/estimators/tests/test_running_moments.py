from __future__ import absolute_import
import unittest
import numpy as np
from variational.estimators import running_moments

__author__ = 'noe'

class TestMomentCombination(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        #cls.X = np.ones((2000, 2))
        cls.X = np.random.rand(2000, 2)
        cls.X[1000:] += 1.0
        cls.w = cls.X.shape[0]
        cls.xsum = cls.X.sum(axis=0)
        cls.xmean = cls.xsum / float(cls.X.shape[0])
        cls.X0 = cls.X - cls.xmean
        cls.M = np.dot(cls.X.T, cls.X)
        cls.M0 = np.dot(cls.X0.T, cls.X0)
        # Introduce weights:
        cls.we = np.random.rand(2000)
        # Compute weighted references:
        cls.w1 = cls.we.sum()
        cls.xsum_we = (cls.we[:, None] * cls.X).sum(axis=0)
        cls.xmean_we = cls.xsum_we / float(cls.w1)
        cls.X0_we = cls.X - cls.xmean_we
        cls.M_we = np.dot((cls.we[:, None] * cls.X).T, cls.X)
        cls.M0_we = np.dot((cls.we[:, None] * cls.X0_we).T, cls.X0_we)

    def test_combine_withmean(self):
        div = 1000
        w1 = div
        s1 = self.X[:div].sum(axis=0)
        C1 = np.dot(self.X[:div].T, self.X[:div])
        w2 = self.w - div
        s2 = self.X[div:].sum(axis=0)
        C2 = np.dot(self.X[div:].T, self.X[div:])

        # two passes
        m1 = running_moments.Moments(w1, s1, s1, C1)
        m2 = running_moments.Moments(w2, s2, s2, C2)
        m = m1.combine(m2, mean_free=False)

        assert np.allclose(m.w, self.w)
        assert np.allclose(m.sx, self.xsum)
        assert np.allclose(m.Mxy, self.M)

    def test_combine_meanfree(self):
        div = 1000
        # part 1
        X1 = self.X[:div]
        w1 = X1.shape[0]
        s1 = X1.sum(axis=0)
        C1 = np.dot((X1-X1.mean(axis=0)).T, X1-X1.mean(axis=0))
        # part 2
        X2 = self.X[div:]
        w2 = X2.shape[0]
        s2 = X2.sum(axis=0)
        C2 = np.dot((X2-X2.mean(axis=0)).T, X2-X2.mean(axis=0))

        # many passes
        m1 = running_moments.Moments(w1, s1, s1, C1)
        m2 = running_moments.Moments(w2, s2, s2, C2)
        m = m1.combine(m2, mean_free=True)

        assert np.allclose(m.w, self.w)
        assert np.allclose(m.sx, self.xsum)
        assert np.allclose(m.Mxy, self.M0)

    def test_combine_weighted_withmean(self):
        div = 1000
        w1 = self.we[:div].sum()
        s1 = (self.we[:div, None] * self.X[:div]).sum(axis=0)
        C1 = np.dot((self.we[:div, None] * self.X[:div]).T, self.X[:div])
        w2 = self.we[div:].sum()
        s2 = (self.we[div:, None] * self.X[div:]).sum(axis=0)
        C2 = np.dot((self.we[div:, None] * self.X[div:]).T, self.X[div:])

        # two passes
        m1 = running_moments.Moments(w1, s1, s1, C1)
        m2 = running_moments.Moments(w2, s2, s2, C2)
        m = m1.combine(m2, mean_free=False)

        assert np.allclose(m.w, self.w1)
        assert np.allclose(m.sx, self.xsum_we)
        assert np.allclose(m.Mxy, self.M_we)#

    def test_combine_weighted_meanfree(self):
        div = 1000
        w1 = self.we[:div].sum()
        s1 = (self.we[:div, None] * self.X[:div]).sum(axis=0)
        X1 = self.X[:div] - s1/float(w1)
        C1 = np.dot((self.we[:div, None] * X1).T, X1)
        w2 = self.we[div:].sum()
        s2 = (self.we[div:, None] * self.X[div:]).sum(axis=0)
        X2 = self.X[div:] - s2/float(w2)
        C2 = np.dot((self.we[div:, None] * X2).T, X2)

        # two passes
        m1 = running_moments.Moments(w1, s1, s1, C1)
        m2 = running_moments.Moments(w2, s2, s2, C2)
        m = m1.combine(m2, mean_free=True)

        assert np.allclose(m.w, self.w1)
        assert np.allclose(m.sx, self.xsum_we)
        assert np.allclose(m.Mxy, self.M0_we)


class TestRunningMoments(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X = np.random.rand(10000, 2)
        cls.Y = np.random.rand(10000, 2)
        cls.T = cls.X.shape[0]
        # Set a lag time for time-lagged tests:
        cls.lag = 50
        # bias the first part
        cls.X[:2000] += 1.0
        cls.Y[:2000] -= 1.0
        # direct calculation, moments of X and Y
        cls.w = np.shape(cls.X)[0]
        cls.wsym = 2*np.shape(cls.X)[0]
        cls.sx = cls.X.sum(axis=0)
        cls.sy = cls.Y.sum(axis=0)
        cls.Mxx = np.dot(cls.X.T, cls.X)
        cls.Mxy = np.dot(cls.X.T, cls.Y)
        cls.Myy = np.dot(cls.Y.T, cls.Y)
        cls.mx = cls.sx / float(cls.w)
        cls.my = cls.sy / float(cls.w)
        cls.X0 = cls.X - cls.mx
        cls.Y0 = cls.Y - cls.my
        cls.Mxx0 = np.dot(cls.X0.T, cls.X0)
        cls.Mxy0 = np.dot(cls.X0.T, cls.Y0)
        cls.Myy0 = np.dot(cls.Y0.T, cls.Y0)
        # direct calculation, symmetric moments
        cls.s_sym = cls.sx + cls.sy
        cls.Mxx_sym = np.dot(cls.X.T, cls.X) + np.dot(cls.Y.T, cls.Y)
        cls.Mxy_sym = np.dot(cls.X.T, cls.Y) + np.dot(cls.Y.T, cls.X)
        cls.m_sym = cls.s_sym / float(cls.wsym)
        cls.X0_sym = cls.X - cls.m_sym
        cls.Y0_sym = cls.Y - cls.m_sym
        cls.Mxx0_sym = np.dot(cls.X0_sym.T, cls.X0_sym) + np.dot(cls.Y0_sym.T, cls.Y0_sym)
        cls.Mxy0_sym = np.dot(cls.X0_sym.T, cls.Y0_sym) + np.dot(cls.Y0_sym.T, cls.X0_sym)
        # direct calculation, time-lagged moments of X:
        cls.CXX = np.zeros((2, 2))
        cls.CXY = np.zeros((2, 2))
        cls.s_lag = np.zeros(2)
        cls.CXX_sym = np.zeros((2, 2))
        cls.CXY_sym = np.zeros((2, 2))
        cls.s_lag_sym = np.zeros(2)
        cls.L = 1000
        cls.nchunks = cls.T / cls.L
        for i in range(0, cls.T, cls.L):
            # Non-symmetric version:
            iX = cls.X[i:i+cls.L, :]
            cls.CXX += np.dot(iX[:-cls.lag, :].T, iX[:-cls.lag, :])
            cls.CXY += np.dot(iX[:-cls.lag, :].T, iX[cls.lag:, :])
            cls.s_lag += np.sum(iX[:-cls.lag, :], axis=0)
            # Symmetric version:
            cls.CXX_sym += np.dot(iX[:-cls.lag, :].T, iX[:-cls.lag, :]) + np.dot(iX[cls.lag:, :].T, iX[cls.lag:, :])
            cls.CXY_sym += np.dot(iX[:-cls.lag, :].T, iX[cls.lag:, :]) + np.dot(iX[cls.lag:, :].T, iX[:-cls.lag, :])
            cls.s_lag_sym += np.sum(iX[:-cls.lag, :], axis=0) + np.sum(iX[cls.lag:, :], axis=0)
        # Compute the mean and substract it:
        cls.mean_lag = cls.s_lag / (cls.T - cls.nchunks*cls.lag)
        cls.mean_lag_sym = cls.s_lag_sym / (2*(cls.T - cls.nchunks*cls.lag))
        cls.X0_lag = cls.X - cls.mean_lag
        cls.X0_lag_sym = cls.X - cls.mean_lag_sym
        # direct calculation, mean free:
        cls.CXX0 = np.zeros((2, 2))
        cls.CXY0 = np.zeros((2, 2))
        cls.CXX0_sym = np.zeros((2, 2))
        cls.CXY0_sym = np.zeros((2, 2))
        for i in range(0, cls.T, cls.L):
            # Non-symmetric version:
            iX = cls.X0_lag[i:i+cls.L, :]
            cls.CXX0 += np.dot(iX[:-cls.lag, :].T, iX[:-cls.lag, :])
            cls.CXY0 += np.dot(iX[:-cls.lag, :].T, iX[cls.lag:, :])
            # Symmetric version:
            iX = cls.X0_lag_sym[i:i+cls.L, :]
            cls.CXX0_sym += np.dot(iX[:-cls.lag, :].T, iX[:-cls.lag, :]) + np.dot(iX[cls.lag:, :].T, iX[cls.lag:, :])
            cls.CXY0_sym += np.dot(iX[:-cls.lag, :].T, iX[cls.lag:, :]) + np.dot(iX[cls.lag:, :].T, iX[:-cls.lag, :])

        # Weighted references:
        cls.we = np.random.rand(10000)
        # direct calculation, time-lagged moments of X:
        cls.CXX_w = np.zeros((2, 2))
        cls.CXY_w = np.zeros((2, 2))
        cls.s_w = np.zeros(2)
        cls.CXX_sym_w = np.zeros((2, 2))
        cls.CXY_sym_w = np.zeros((2, 2))
        cls.s_sym_w = np.zeros(2)
        cls.we_sum = 0
        cls.we_sum_sym = 0
        for i in range(0, cls.T, cls.L):
            iX = cls.X[i:i+cls.L, :]
            iwe = cls.we[i:i+cls.L]
            # Non-symmetric version:
            cls.CXX_w += np.dot((iwe[:-cls.lag, None] * iX[:-cls.lag, :]).T, iX[:-cls.lag, :])
            cls.CXY_w += np.dot((iwe[:-cls.lag, None] * iX[:-cls.lag, :]).T, iX[cls.lag:, :])
            cls.s_w += np.sum(iwe[:-cls.lag, None] * iX[:-cls.lag, :], axis=0)
            cls.we_sum += np.sum(iwe[:-cls.lag])
            # Symmetric version:
            cls.CXX_sym_w += np.dot((iwe[:-cls.lag, None] * iX[:-cls.lag, :]).T, iX[:-cls.lag, :]) + \
                           np.dot((iwe[cls.lag:, None] * iX[cls.lag:, :]).T, iX[cls.lag:, :])
            cls.CXY_sym_w += np.dot((iwe[:-cls.lag, None] * iX[:-cls.lag, :]).T, iX[cls.lag:, :]) + \
                           np.dot((iwe[cls.lag:, None] * iX[cls.lag:, :]).T, iX[:-cls.lag, :])
            cls.s_sym_w += np.sum((iwe[:-cls.lag, None] * iX[:-cls.lag, :]), axis=0) + \
                             np.sum((iwe[cls.lag:, None] * iX[cls.lag:, :]), axis=0)
            cls.we_sum_sym += np.sum(iwe[:-cls.lag]) + np.sum(iwe[cls.lag:])
        # Compute the mean and substract it:
        cls.mean_w = cls.s_w / (cls.we_sum)
        cls.mean_sym_w = cls.s_sym_w / (cls.we_sum_sym)
        cls.X0_w = (cls.X - cls.mean_w).copy()
        cls.X0_w_sym = (cls.X - cls.mean_sym_w).copy()
        # direct calculation, mean free:
        cls.CXX0_w = np.zeros((2, 2))
        cls.CXY0_w = np.zeros((2, 2))
        cls.CXX0_sym_w = np.zeros((2, 2))
        cls.CXY0_sym_w = np.zeros((2, 2))
        for i in range(0, cls.T, cls.L):
            iX = cls.X0_w[i:i+cls.L, :]
            iwe = cls.we[i:i+cls.L]
            # Non-symmetric version:
            cls.CXX0_w += np.dot((iwe[:-cls.lag, None] * iX[:-cls.lag, :]).T, iX[:-cls.lag, :])
            cls.CXY0_w += np.dot((iwe[:-cls.lag, None] * iX[:-cls.lag, :]).T, iX[cls.lag:, :])
            # Symmetric version:
            cls.CXX0_sym_w += np.dot((iwe[:-cls.lag, None] * iX[:-cls.lag, :]).T, iX[:-cls.lag, :]) + \
                           np.dot((iwe[cls.lag:, None] * iX[cls.lag:, :]).T, iX[cls.lag:, :])
            cls.CXY0_sym_w += np.dot((iwe[:-cls.lag, None] * iX[:-cls.lag, :]).T, iX[cls.lag:, :]) + \
                           np.dot((iwe[cls.lag:, None] * iX[cls.lag:, :]).T, iX[:-cls.lag, :])

        return cls

    def test_XX_withmean(self):
        # many passes
        cc = running_moments.RunningCovar(remove_mean=False)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i+self.L])
        assert np.allclose(cc.sum_X(), self.sx)
        assert np.allclose(cc.moments_XX(), self.Mxx)

    def test_XX_meanfree(self):
        # many passes
        cc = running_moments.RunningCovar(remove_mean=True)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i+self.L])
        assert np.allclose(cc.sum_X(), self.sx)
        assert np.allclose(cc.moments_XX(), self.Mxx0)

    def test_XXXY_withmean(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False)
        for i in range(0, self.T, self.L):
            cc.add(self.X[i:i+self.L], self.Y[i:i+self.L])
        assert np.allclose(cc.sum_X(), self.sx)
        assert np.allclose(cc.moments_XX(), self.Mxx)
        assert np.allclose(cc.moments_XY(), self.Mxy)

    def test_XXXY_meanfree(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True)
        L = 1000
        for i in range(0, self.X.shape[0], L):
            cc.add(self.X[i:i+L], self.Y[i:i+L])
        assert np.allclose(cc.sum_X(), self.sx)
        assert np.allclose(cc.moments_XX(), self.Mxx0)
        assert np.allclose(cc.moments_XY(), self.Mxy0)

    def test_XXXY_time_lagged_withmean(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False, time_lagged=True,
                                          lag=self.lag)
        for i in range(0, self.T, self.L):
            iX = self.X[i:i+self.L, :]
            cc.add(iX)
        assert np.allclose(cc.sum_X(), self.s_lag)
        assert np.allclose(cc.moments_XX(), self.CXX)
        assert np.allclose(cc.moments_XY(), self.CXY)

    def test_XXXY_time_lagged_meanfree(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True, time_lagged=True,
                                          lag=self.lag)
        for i in range(0, self.T, self.L):
            iX = self.X[i:i+self.L, :]
            cc.add(iX)
        assert np.allclose(cc.sum_X(), self.s_lag)
        assert np.allclose(cc.moments_XX(), self.CXX0)
        assert np.allclose(cc.moments_XY(), self.CXY0)

    def test_XXXY_weighted_withmean(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False, time_lagged=True,
                                          lag=self.lag)
        for i in range(0, self.T, self.L):
            iX = self.X[i:i+self.L, :]
            iwe = self.we[i:i+self.L]
            cc.add(iX, weights=iwe)
        assert np.allclose(cc.sum_X(), self.s_w)
        assert np.allclose(cc.moments_XX(), self.CXX_w)
        assert np.allclose(cc.moments_XY(), self.CXY_w)

    def test_XXXY_weighted_meanfree(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True, time_lagged=True,
                                          lag=self.lag)
        for i in range(0, self.T, self.L):
            iX = self.X[i:i+self.L, :]
            iwe = self.we[i:i+self.L]
            cc.add(iX, weights=iwe)
        assert np.allclose(cc.sum_X(), self.s_w)
        assert np.allclose(cc.moments_XX(), self.CXX0_w)
        assert np.allclose(cc.moments_XY(), self.CXY0_w)

    def test_XXXY_sym_withmean(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False, symmetrize=True)
        L = 1000
        for i in range(0, self.X.shape[0], L):
            cc.add(self.X[i:i+L], self.Y[i:i+L])
        assert np.allclose(cc.sum_X(), self.s_sym)
        assert np.allclose(cc.moments_XX(), self.Mxx_sym)
        assert np.allclose(cc.moments_XY(), self.Mxy_sym)

    def test_XXXY_sym_time_lagged_withmean(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False, time_lagged=True,
                                          symmetrize=True, lag=self.lag)
        for i in range(0, self.T, self.L):
            iX = self.X[i:i+self.L, :]
            cc.add(iX)
        assert np.allclose(cc.sum_X(), self.s_lag_sym)
        assert np.allclose(cc.moments_XX(), self.CXX_sym)
        assert np.allclose(cc.moments_XY(), self.CXY_sym)

    def test_XXXY_sym_time_lagged_meanfree(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True, time_lagged=True,
                                          symmetrize=True, lag=self.lag)
        for i in range(0, self.T, self.L):
            iX = self.X[i:i+self.L, :]
            cc.add(iX)
        assert np.allclose(cc.sum_X(), self.s_lag_sym)
        assert np.allclose(cc.moments_XX(), self.CXX0_sym)
        assert np.allclose(cc.moments_XY(), self.CXY0_sym)

    def test_XXXY_sym_meanfree(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True, symmetrize=True)
        L = 1000
        for i in range(0, self.X.shape[0], L):
            cc.add(self.X[i:i+L], self.Y[i:i+L])
        assert np.allclose(cc.sum_X(), self.s_sym)
        assert np.allclose(cc.moments_XX(), self.Mxx0_sym)
        assert np.allclose(cc.moments_XY(), self.Mxy0_sym)

    def test_XXXY_sym_weighted_withmean(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False, time_lagged=True,
                                          lag=self.lag, symmetrize=True)
        for i in range(0, self.T, self.L):
            iX = self.X[i:i+self.L, :]
            iwe = self.we[i:i+self.L]
            cc.add(iX, weights=iwe)
        assert np.allclose(cc.sum_X(), self.s_sym_w)
        assert np.allclose(cc.moments_XX(), self.CXX_sym_w)
        assert np.allclose(cc.moments_XY(), self.CXY_sym_w)

    def test_XXXY_sym_weighted_meanfree(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True, time_lagged=True,
                                          lag=self.lag, symmetrize=True)
        for i in range(0, self.T, self.L):
            iX = self.X[i:i+self.L, :]
            iwe = self.we[i:i+self.L]
            cc.add(iX, weights=iwe)
        assert np.allclose(cc.sum_X(), self.s_sym_w)
        assert np.allclose(cc.moments_XX(), self.CXX0_sym_w)
        assert np.allclose(cc.moments_XY(), self.CXY0_sym_w)


if __name__ == "__main__":
    unittest.main()