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


class TestRunningMoments(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X = np.random.rand(10000, 2)
        cls.Y = np.random.rand(10000, 2)
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

        return cls


    def test_simple_withmean(self):
        # reference
        X = np.ones((2000, 2))
        X[1000:] += 1.0
        xsum = X.sum(axis=0)
        xmean = xsum / float(X.shape[0])
        X0 = X - xmean
        C = np.dot(X.T, X)

        # many passes
        rc = running_moments.RunningCovar(remove_mean=False)
        rc.add(X[:1000])
        rc.add(X[1000:])

        assert np.allclose(rc.sum_X(), xsum)
        assert np.allclose(rc.moments_XX(), C)

    def test_simple_meanfree(self):
        # reference
        X = np.ones((2000, 2))
        X[1000:] += 1.0
        xsum = X.sum(axis=0)
        xmean = xsum / float(X.shape[0])
        X0 = X - xmean
        C0 = np.dot(X0.T, X0)

        # many passes
        rc = running_moments.RunningCovar(remove_mean=True)
        rc.add(X[:1000])
        rc.add(X[1000:])

        assert np.allclose(rc.sum_X(), xsum)
        assert np.allclose(rc.moments_XX(), C0)

    def test_XX_withmean(self):
        # many passes
        cc = running_moments.RunningCovar(remove_mean=False)
        L = 1000
        for i in range(0, self.X.shape[0], L):
            cc.add(self.X[i:i+L])
        assert np.allclose(cc.sum_X(), self.sx)
        assert np.allclose(cc.moments_XX(), self.Mxx)

    def test_XX_meanfree(self):
        # many passes
        cc = running_moments.RunningCovar(remove_mean=True)
        L = 1000
        for i in range(0, self.X.shape[0], L):
            cc.add(self.X[i:i+L])
        assert np.allclose(cc.sum_X(), self.sx)
        assert np.allclose(cc.moments_XX(), self.Mxx0)

    def test_XXXY_withmean(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False)
        L = 1000
        for i in range(0, self.X.shape[0], L):
            cc.add(self.X[i:i+L], self.Y[i:i+L])
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

    def test_XXXY_sym_withmean(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=False, symmetrize=True)
        L = 1000
        for i in range(0, self.X.shape[0], L):
            cc.add(self.X[i:i+L], self.Y[i:i+L])
        assert np.allclose(cc.sum_X(), self.s_sym)
        assert np.allclose(cc.moments_XX(), self.Mxx_sym)
        assert np.allclose(cc.moments_XY(), self.Mxy_sym)

    def test_XXXY_sym_meanfree(self):
        # many passes
        cc = running_moments.RunningCovar(compute_XX=True, compute_XY=True, remove_mean=True, symmetrize=True)
        L = 1000
        for i in range(0, self.X.shape[0], L):
            cc.add(self.X[i:i+L], self.Y[i:i+L])
        assert np.allclose(cc.sum_X(), self.s_sym)
        assert np.allclose(cc.moments_XX(), self.Mxx0_sym)
        assert np.allclose(cc.moments_XY(), self.Mxy0_sym)


if __name__ == "__main__":
    unittest.main()